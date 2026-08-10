"""Stage A — full-pipeline RunPod serverless GPU worker.

Runs the ENTIRE infrascan video->data pipeline (not just stage 0): the CEO's
platform is cloned into the image (fix/da3-serpentine-pose-order branch), and
this handler drives it headlessly the same way the web app does on upload:

    create space -> place video -> stitch -> frames -> views
        -> 00b_da3_streaming (depth + poses + pointcloud.ply)  [GPU]
        -> pano_clean (operator removal, non-fatal)             [GPU]
        -> pano_lowres (viewer-size downsample, non-fatal)
    -> upload: scans/<slug>.zip (complete dataset, for train + archive) + the small
       viewer-only file set as individual S3 objects (panoramas, cameras, pointcloud)

    The object-search stages (01_propose -> 02_embed -> 02b_match_views ->
    03_backproject -> 03b_merge_groups -> 04_index: proposals -> embeddings ->
    cross-view matching -> a FAISS index) used to run here too, but their outputs
    were never uploaded anywhere a viewer reads from — pure wasted GPU time every
    scan. Dropped from pipeline.runner's STAGES; see that file's docstring.

Input JSON:
    {"input": {"video_url": "...", "slug": "scan1", "capture_type": "insta360",
               "every_n": 100}}
Output JSON:
    {"slug": "...", "result_url": "https://.../scan1.zip",
     "stages": {"00b_gen_da3": "ok", ...}, "num_views": N, "num_points": M}
  or {"error": "...", "failed_stage": "...", "stderr": "..."}

NOTE: this is v1 — the exact per-space layout for the pre-DA3 steps is validated
against a local run of the platform before trusting it. Every stage is a
subprocess with captured stderr, so the first failing stage names itself.
"""
import json, os, glob, shutil, subprocess, sys, tempfile, threading, traceback, urllib.request, uuid
from pathlib import Path

import runpod
from PIL import Image

# ---- where the CEO's platform lives in the image (cloned by the Dockerfile) ----
PLATFORM = os.environ.get("INFRASCAN_PLATFORM_DIR", "/app/pipeline")
# Per-run working data lives on the (optionally mounted) volume, else /workspace.
WORKROOT = os.environ.get("INFRASCAN_WORKROOT", "/workspace/runs")

# Bump when rebuilding so we can confirm (via a cheap maintenance:df call) that the
# endpoint is actually serving the NEW image before kicking off an expensive re-run.
HANDLER_VERSION = "2026-08-07-no-volume"

# The entrypoint is `python -m pipeline.runner --slug <slug>`, which now only runs
# the cosmetic/optional gen_topdown -> downsample_ply stages (00b_da3_streaming
# already ran directly, above, and the object-search stages were dropped — see
# runner.py's docstring). We call it directly rather than re-implementing the
# (now short) stage list.
PRE_STAGES = ["_00_stitch_insv", "00_video_to_img", "00a_sample_views", "00b_da3_streaming"]


def _meaningful_stderr(text, n=80):
    r"""DA3/tqdm floods stderr with progress bars (\r-redrawn), which otherwise
    bury the real Python traceback in the last-3000-chars tail — that's why every
    DA3 failure only showed 'Extracting features: 97%...' with no actual error.
    Split on \r and \n, drop the progress-bar fragments, keep the last n real
    lines (where the traceback lives)."""
    out = []
    for ln in (text or "").replace("\r", "\n").split("\n"):
        s = ln.strip()
        if not s:
            continue
        if "%|" in s or "it/s]" in s or s.startswith("Extracting features"):
            continue
        out.append(s)
    return "\n".join(out[-n:])


# Human-readable label per real pipeline stage, in run order. Reported live via
# runpod.serverless.progress_update() so our platform's poller (which already
# hits GET /status/<job_id> every few seconds) can show WHICH stage is running
# instead of one opaque "Cloud GPU ... elapsed Xs" blob for the whole job —
# RunPod relays the update through the same status response, no separate
# network path from this container back to our (Tailscale-only) server needed.
STAGE_LABELS = {
    "_00_stitch_insv":  "Stitching video",
    "00_video_to_img":  "Extracting frames",
    "00a_sample_views": "Sampling perspective views",
    "00b_da3_streaming": "Estimating depth + camera poses",
    "pipeline.runner":  "Building floor plan + point cloud",
    "pano_clean":       "Removing capture operator",
    "pano_lowres":      "Downsampling panoramas",
    "upload":           "Uploading scan to storage",
}
STAGE_ORDER = list(STAGE_LABELS)


def _report(job, stage: str) -> None:
    """Best-effort progress ping — must never fail or slow down the job."""
    try:
        runpod.serverless.progress_update(
            job, {"stage": stage, "text": STAGE_LABELS.get(stage, stage)})
    except Exception as e:
        print(f"[progress] update failed (non-fatal): {e}", flush=True)


def _run(cmd, cwd, env, stage):
    """Run one stage as a subprocess; raise with captured stderr on failure.

    Streams stdout line-by-line as the child produces it (RunPod's log tab
    ships each print() live), instead of the old `subprocess.run(capture_
    output=True)`, which silently buffers EVERYTHING and only prints once the
    whole stage exits — a stage with no output for its full duration looked
    identical to one hard-stuck at line 1, with no way to tell them apart from
    the log. stderr is still captured whole (not streamed) so DA3/tqdm's
    flood of \r-redrawn progress bars doesn't spam the console; it's only
    surfaced, tqdm-stripped, if the stage actually fails."""
    print(f"[stage {stage}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    proc = subprocess.Popen(cmd, cwd=cwd, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=1)

    stderr_lines = []

    def _drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    t.join()

    stderr_text = "".join(stderr_lines)
    tail = stderr_text[-3000:]
    if proc.returncode != 0:
        # Surface the REAL error: strip tqdm progress lines so the traceback shows.
        errtail = _meaningful_stderr(stderr_text) or tail
        raise RuntimeError(f"stage {stage} exited {proc.returncode}\n"
                           f"--- stderr (progress bars stripped) ---\n{errtail}")
    return tail


# da3_streaming loads three weight files by RELATIVE path (./weights/...) with
# cwd=da3_streaming/. They're gitignored (~6.6 GB), so not in the git tree, but the
# Dockerfile bakes them into the image at /app/da3_weights (DA3_WEIGHTS_DIR) at build
# time — this function just symlinks them in. If DA3_WEIGHTS_DIR ever points somewhere
# without them yet (e.g. local dev), it falls back to downloading them here instead.
DA3_WEIGHTS = {
    "config.json":
        "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/config.json",
    "model.safetensors":
        "https://huggingface.co/depth-anything/DA3NESTED-GIANT-LARGE-1.1/resolve/main/model.safetensors",
    "dino_salad.ckpt":
        "https://github.com/serizba/salad/releases/download/v1.0.0/dino_salad.ckpt",
}


def _ensure_da3_weights():
    da3_dir = Path(PLATFORM) / "pipeline" / "da3_streaming"
    wdir = da3_dir / "weights"
    cache = Path(os.environ.get("DA3_WEIGHTS_DIR",
                                str(Path(WORKROOT).parent / "da3_weights")))
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in DA3_WEIGHTS.items():
        dst = cache / name
        if dst.exists() and dst.stat().st_size > 0:
            continue
        print(f"[weights] downloading {name} (first cold start only) ...", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        urllib.request.urlretrieve(url, str(tmp))
        tmp.replace(dst)
        print(f"[weights] {name} -> {dst} ({dst.stat().st_size/1e6:.1f} MB)", flush=True)
    wdir.mkdir(parents=True, exist_ok=True)
    for name in DA3_WEIGHTS:
        link = wdir / name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(cache / name)
    print(f"[weights] linked {list(DA3_WEIGHTS)} into {wdir}", flush=True)


def _disk_report(tag=""):
    """Print volume free space + available RAM so logs show headroom per job.
    Lets us tell a disk-full failure (No space left on device) apart from a GPU
    OOM when a stage exits 1."""
    p = Path(WORKROOT)
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        du = shutil.disk_usage(str(p))
        print(f"[disk {tag}] {p}: free={du.free/1e9:.1f}GB "
              f"used={du.used/1e9:.1f}GB total={du.total/1e9:.1f}GB", flush=True)
    except Exception as e:
        print(f"[disk {tag}] usage failed: {e}", flush=True)
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable"):
                print(f"[mem {tag}] {line.strip()}", flush=True)
                break
    except Exception:
        pass


def _maintenance(action):
    """Volume maintenance, no pipeline run. Dispatch with e.g.
        {"input": {"maintenance": "df"}}     -> just report free space
        {"input": {"maintenance": "purge"}}  -> delete leftover per-run dirs under WORKROOT
    Every job already wipes its own run_root at start AND end (see handler), so on a
    healthy deployment `purge` should find little. It exists to reclaim the backlog of
    run dirs left by older handler versions that never cleaned up after upload. DA3
    weights live OUTSIDE WORKROOT (WORKROOT.parent/da3_weights) and are never touched."""
    root = Path(WORKROOT)
    _disk_report("before")
    removed = []
    if action == "purge" and root.exists():
        for child in sorted(root.iterdir()):
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
                removed.append(child.name)
        print(f"[maintenance] purged {len(removed)} run dir(s): {removed}", flush=True)
    _disk_report("after")
    return {"maintenance": action, "removed": removed, "count": len(removed),
            "root": str(root), "version": HANDLER_VERSION}


def handler(job):
    inp = job.get("input", {}) or {}
    if inp.get("maintenance"):
        return _maintenance(inp["maintenance"])
    video_url = inp.get("video_url")
    if not video_url:
        return {"error": "provide input.video_url"}
    slug = (inp.get("slug") or f"scan-{uuid.uuid4().hex[:8]}").lower()
    capture_type = inp.get("capture_type", "insta360")
    every_n = int(inp.get("every_n", 100))

    # Isolate this run's data/DB under the work root; point the platform config at it.
    run_root = Path(WORKROOT) / slug
    # A warm worker can be reused across multiple jobs for the SAME slug (a re-scan),
    # so run_root isn't guaranteed empty even though it's on the container's own disk
    # (not persisted across a full cold restart, but not guaranteed clean within one
    # either). A re-scan landing on a warm worker would otherwise inherit the previous
    # run's DA3 chunk files (_da3_streaming/pcd/*_pcd.ply) — merge_ply_files globs
    # those, mixing stale + fresh chunks into a corrupt pointcloud.ply whose header
    # count != its body (crashes training / silently drops points). Start every job
    # from a clean dir. The DA3 weights live at DA3_WEIGHTS_DIR (baked into the image,
    # outside run_root entirely), so they are never touched by this.
    shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)
    _disk_report("job-start")
    env = dict(os.environ)
    env["INFRASCAN_DB_PATH"] = str(run_root / "infrascan.db")
    env["INFRASCAN_DATA_ROOT"] = str(run_root / "data")
    env["INFRASCAN_OUT_ROOT"] = str(run_root / "out")
    env["PYTHONPATH"] = PLATFORM + os.pathsep + env.get("PYTHONPATH", "")
    py = sys.executable
    stages_status = {}

    try:
        # 1) bootstrap the platform DB + a space row (mirrors app.spaces.create_space)
        sys.path.insert(0, PLATFORM)
        for k in ("INFRASCAN_DB_PATH", "INFRASCAN_DATA_ROOT", "INFRASCAN_OUT_ROOT"):
            os.environ[k] = env[k]
        from app import config as cfg
        from app import spaces as space_repo
        from app import db as _appdb
        from app.db import init as db_init, get_conn
        from app.auth import create_user
        # RunPod reuses the Python process across jobs (warm workers). config.* paths
        # are import-time constants and get_conn() caches a thread-local connection,
        # so a 2nd job would otherwise bootstrap the space into the PREVIOUS job's
        # run_root DB — then the stage subprocesses (correct env path) open a fresh
        # empty DB and die with "Space not in the database". Re-pin the paths + drop
        # the cached connection so every job uses its own run_root DB.
        cfg.DB_PATH   = Path(env["INFRASCAN_DB_PATH"]).resolve()
        cfg.DATA_ROOT = Path(env["INFRASCAN_DATA_ROOT"]).resolve()
        cfg.OUT_ROOT  = Path(env["INFRASCAN_OUT_ROOT"]).resolve()
        if getattr(_appdb._local, "conn", None) is not None:
            try: _appdb._local.conn.close()
            except Exception: pass
            _appdb._local.conn = None
        cfg.ensure_dirs(); db_init()
        if not space_repo.by_slug(slug):
            # spaces.owner_id is a FK to users(id) (TEXT) — need a real user first.
            row = get_conn().execute("SELECT id FROM users LIMIT 1").fetchone()
            owner_id = row[0] if row else create_user(
                email=f"{slug}@worker.local", name="worker", password="worker-bootstrap-pw", role="admin")
            space_repo.create_space(slug=slug, title=slug, owner_id=owner_id, status="processing")
        data_dir = Path(space_repo.data_dir(slug))
        (data_dir / "uploads").mkdir(parents=True, exist_ok=True)

        # 2) download the video
        ext = os.path.splitext(video_url.split("?")[0])[1].lower() or ".mp4"
        vid = data_dir / "uploads" / f"input{ext}"
        print(f"[dl] {video_url} -> {vid}", flush=True)
        urllib.request.urlretrieve(video_url, str(vid))

        # 3) pre-DA3: stitch -> frames -> views  (writes into the space's data dir)
        P = Path(PLATFORM) / "pipeline"
        eq = data_dir / "equirect.mp4"
        frames = data_dir / "frames"; views = data_dir / "views"
        _report(job, "_00_stitch_insv")
        _run([py, str(P / "_00_stitch_insv.py"), "--input", str(vid), "--output", str(eq)],
             PLATFORM, env, "_00_stitch_insv"); stages_status["_00_stitch_insv"] = "ok"
        _report(job, "00_video_to_img")
        _run([py, str(P / "00_video_to_img.py"), "--video", str(eq),
              "--output_dir", str(frames), "--every_n", str(every_n)],
             PLATFORM, env, "00_video_to_img"); stages_status["00_video_to_img"] = "ok"
        # 3 pitches (0, +30, -30) like the original pipeline: DA3 poses all of them
        # (richer point cloud from up/down coverage) and the perspective viewer can
        # look up/down. Gaussian TRAINING stays single-pitch — build_hires_dataset.py
        # (train branch) filters to pz000 via its --pz 0 default, so only the eye-level
        # crops feed splatfacto. Encoded in filenames as pz000/pz030/pz330.
        _report(job, "00a_sample_views")
        _run([py, str(P / "00a_sample_views.py"), "--input_dir", str(frames),
              "--output_dir", str(views), "--pitches", "-30", "0", "30"],
             PLATFORM, env, "00a_sample_views")
        stages_status["00a_sample_views"] = "ok"

        # 3b) DA3 streaming: estimate camera POSES (+depth) from the views -> cameras.json.
        #     A fresh video has no poses; the runner's 00b_gen_da3 requires cameras.json,
        #     so this must run first. Ensure the ~6.6GB DA3+SALAD weights are on the
        #     volume + linked in before running it.
        _ensure_da3_weights()
        _report(job, "00b_da3_streaming")
        _run([py, str(P / "00b_da3_streaming.py"), "--space", slug],
             PLATFORM, env, "00b_da3_streaming"); stages_status["00b_da3_streaming"] = "ok"

        # 4) the proven entrypoint: runs 00b -> ... -> downsample_ply (through pointcloud)
        _report(job, "pipeline.runner")
        _run([py, "-m", "pipeline.runner", "--slug", slug], PLATFORM, env, "pipeline.runner")
        stages_status["pipeline.runner"] = "ok"

        # 4b) operator removal (pano_clean): erase the camera operator from the equirect
        #     panoramas via YOLO-seg + LaMa nadir reprojection, into data/<slug>/pano_clean/
        #     frames/. NON-FATAL — a bad inpaint or a crash here must never fail the ingest,
        #     so it runs in its own try/except and just records a skipped status. Weights are
        #     baked into the image (Dockerfile): YOLO at /app/weights, big-lama.pt under the
        #     TORCH_HOME we point at /app/lama_cache, so no runtime download and no volume use.
        pano_clean_dir = data_dir / "pano_clean" / "frames"
        try:
            _report(job, "pano_clean")
            pc_env = dict(env)
            pc_env["TORCH_HOME"] = os.environ.get("LAMA_TORCH_HOME", "/app/lama_cache")
            pc_env["YOLO_CONFIG_DIR"] = "/tmp/ultralytics"
            _run([py, "/app/pipeline_panoclean/pano_clean.py",
                  "--frames", str(frames), "--out", str(pano_clean_dir),
                  "--yolo", os.environ.get("PANO_CLEAN_YOLO", "/app/weights/yolo11x-seg.pt")],
                 "/app", pc_env, "pano_clean")
            stages_status["pano_clean"] = "ok"
        except Exception as e:
            stages_status["pano_clean"] = f"skipped: {type(e).__name__}: {e}"
            print(f"[pano_clean] non-fatal failure, panoramas keep the operator: {e}",
                  flush=True)

        # 4c) low-res panoramas (pano_lowres): the panorama-mode viewer streams these
        #     equirect frames at native capture res (commonly 7680x3840, several MB
        #     each) as a full-sphere texture -- heavy to pan/switch between. Downsample
        #     to a size that still holds up under that mode's zoom range (2560 wide is
        #     the balance point platform/pipeline/downsample_panoramas.py already uses
        #     for the same purpose in infrascan-onprem). Built from pano_clean's output
        #     when it produced frames (operator already removed, no point downsampling
        #     the version we're about to throw away), falling back to the raw frames
        #     otherwise. NON-FATAL for the same reason as pano_clean above.
        pano_lowres_dir = data_dir / "pano_lowres" / "frames"
        try:
            _report(job, "pano_lowres")
            lowres_src = pano_clean_dir if any(pano_clean_dir.glob("*.jpg")) else frames
            lowres_width = int(os.environ.get("PANO_LOWRES_WIDTH", "2560"))
            lowres_height = lowres_width // 2
            lowres_quality = int(os.environ.get("PANO_LOWRES_QUALITY", "88"))
            pano_lowres_dir.mkdir(parents=True, exist_ok=True)
            lowres_files = sorted(lowres_src.glob("*.jpg"))
            for p in lowres_files:
                im = Image.open(p)
                if im.size != (lowres_width, lowres_height):
                    im = im.resize((lowres_width, lowres_height), Image.LANCZOS)
                im.save(pano_lowres_dir / p.name, quality=lowres_quality)
            stages_status["pano_lowres"] = "ok" if lowres_files else "skipped: no source frames"
        except Exception as e:
            stages_status["pano_lowres"] = f"skipped: {type(e).__name__}: {e}"
            print(f"[pano_lowres] non-fatal failure, panorama mode keeps serving full-res: {e}",
                  flush=True)

        # 5) Two separate uploads, sized for two different consumers:
        #
        #    a) scans/<slug>.zip — the COMPLETE original dataset (raw frames, views,
        #       depth, raw pointcloud, cameras/intrinsics), one archive, one PUT. This
        #       is what train downloads (it wants everything anyway, no per-file access
        #       needed) and it doubles as the permanent archive of the original capture
        #       — if pano_clean/training ever needs redoing with a better model, the
        #       source is never lost. Always built, not a mode/toggle: views+depth are
        #       the vast majority of a scan's file count (~7,400 of ~7,600 files on a
        #       dense scan) and NEITHER the panorama viewer nor the 3D/scenegraph
        #       overlay ever reads either of them — only training does.
        #
        #    b) individual S3 objects under scans/<slug>/, pano_clean/<slug>/ and
        #       pano_lowres/<slug>/ — ONLY what the viewer actually touches: frames
        #       (raw, kept as the final fallback), pano_clean frames (operator
        #       removed), pano_lowres frames (operator removed + downsampled,
        #       default), cameras.json, intrinsics.json, pointcloud_downsampled.ply.
        #       Small (frames are ~200 files, not ~7,400), so individually-addressable
        #       S3 streaming stays fast — no local caching needed anywhere to serve it.
        import storage
        s3c = storage._client()
        bucket = os.environ["S3_BUCKET"]
        prefix = f"scans/{slug}"

        train_manifest = []   # -> bundled into scans/<slug>.zip
        for sub in ("views",):
            for p in sorted((data_dir / sub).glob("*")):
                if p.is_file():
                    train_manifest.append((p, f"{prefix}/{sub}/{p.name}"))
        for f in ("cameras.json", "intrinsics.json", "pointcloud.ply"):
            if (data_dir / f).exists():
                train_manifest.append((data_dir / f, f"{prefix}/{f}"))
        ro = data_dir / "_da3_streaming" / "results_output"
        if ro.is_dir():
            for npz in sorted(ro.glob("frame_*.npz")):
                train_manifest.append((npz, f"{prefix}/depth/{npz.name}"))
        for p in sorted((data_dir / "frames").glob("*")):
            if p.is_file():
                train_manifest.append((p, f"{prefix}/frames/{p.name}"))

        viewer_manifest = []   # -> individual S3 objects
        for p in sorted((data_dir / "frames").glob("*")):
            if p.is_file():
                viewer_manifest.append((p, f"{prefix}/frames/{p.name}"))
        for f in ("cameras.json", "intrinsics.json"):
            if (data_dir / f).exists():
                viewer_manifest.append((data_dir / f, f"{prefix}/{f}"))

        # downsample_ply (part of pipeline.runner, above) already voxel-downsamples
        # pointcloud.ply for the web topdown viewer — it just never leaves this worker.
        # A dense/3-pitch scan's raw pointcloud.ply can be 30M+ points, which is fine
        # for the topdown view but turns into 30M+ un-capped splatfacto Gaussians on
        # the train endpoint (OOMs the GPU before a single training step). Upload this
        # already-computed, already-cheap file too so train can prefer it. NOT anchored
        # under data_dir — downsample_ply.py (unlike the rest of this pipeline) still
        # writes to the fixed PLATFORM/ui/_spaces/<slug>/ path, not the per-job run_root.
        downsampled = Path(PLATFORM) / "ui" / "_spaces" / slug / "Data_" / "downsampled_web.ply"
        if downsampled.exists():
            viewer_manifest.append((downsampled, f"{prefix}/pointcloud_downsampled.ply"))

        # operator-removed panoramas (if the pano_clean step produced them). The viewer
        # requests pano_clean/<slug>/... , falling back server-side to the raw scan:
        #   pano_clean/<slug>/frames/*.jpg   cleaned equirect panos the viewer serves
        #   pano_clean/<slug>/cameras.json   copied so the pano viewer is self-contained
        n_clean = 0
        if pano_clean_dir.is_dir():
            for p in sorted(pano_clean_dir.glob("*.jpg")):
                viewer_manifest.append((p, f"pano_clean/{slug}/frames/{p.name}")); n_clean += 1
            if n_clean and (data_dir / "cameras.json").exists():
                viewer_manifest.append((data_dir / "cameras.json", f"pano_clean/{slug}/cameras.json"))

        # low-res panoramas (if the pano_lowres step produced them). Same layout as
        # pano_clean, under its own prefix -- the viewer requests pano_lowres/<slug>/...
        # first, falling back server-side to pano_clean/ then raw scans/ if absent.
        n_lowres = 0
        if pano_lowres_dir.is_dir():
            for p in sorted(pano_lowres_dir.glob("*.jpg")):
                viewer_manifest.append((p, f"pano_lowres/{slug}/frames/{p.name}")); n_lowres += 1
            if n_lowres and (data_dir / "cameras.json").exists():
                viewer_manifest.append((data_dir / "cameras.json", f"pano_lowres/{slug}/cameras.json"))

        _report(job, "upload")
        print(f"[s3] building train archive ({len(train_manifest)} files) + "
              f"uploading {len(viewer_manifest)} viewer files individually...", flush=True)

        import zipfile
        zip_path = run_root / f"{slug}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED) as zf:
            for i, (local, key) in enumerate(train_manifest, 1):
                zf.write(local, arcname=key)
                if i % 2000 == 0:
                    print(f"[zip] added {i}/{len(train_manifest)} files...", flush=True)
        zip_key = f"scans/{slug}.zip"
        print(f"[s3] uploading {zip_path.stat().st_size/1e6:.0f}MB archive "
              f"({len(train_manifest)} files) -> s3://{bucket}/{zip_key} ...", flush=True)
        s3c.upload_file(str(zip_path), bucket, zip_key)
        print(f"[s3] uploaded s3://{bucket}/{zip_key}", flush=True)

        nfiles = 0
        for local, key in viewer_manifest:
            s3c.upload_file(str(local), bucket, key)
            nfiles += 1
            if nfiles % 100 == 0:
                print(f"[s3] uploaded {nfiles}/{len(viewer_manifest)} viewer files so far...", flush=True)

        n_views = len(glob.glob(str(views / "*.jpg")))
        n_panos = len(glob.glob(str(frames / "*.jpg")))
        print(f"[s3] uploaded {nfiles} viewer files to s3://{bucket}/{prefix}/ "
              f"(+ {len(train_manifest)} files archived in {zip_key})", flush=True)

        return {
            "slug": slug, "scan_prefix": prefix + "/",
            "num_views": n_views, "num_panos": n_panos, "num_clean": n_clean,
            "num_lowres": n_lowres, "num_files": nfiles,
            "num_archived": len(train_manifest), "stages": stages_status,
        }
    except Exception as e:
        return {
            "error": f"{type(e).__name__}: {e}",
            "failed_stage": next((s for s in (PRE_STAGES + ["pipeline.runner"])
                                  if s not in stages_status), "?"),
            "stages_ok": stages_status,
            "trace": traceback.format_exc()[-1500:],
        }
    finally:
        # Reclaim the volume. On success the dataset is already on S3; on failure the
        # job retries from scratch (start-of-job rmtree). Leaving run_root behind — the
        # video, frames/, views/ (36x the frames), pointcloud, DA3 chunks, depth npz — is
        # what fills the volume over many scans, since each is keyed by slug and only ever
        # freed by re-scanning that same slug. Clean it here so every job nets to ~zero.
        # DA3 weights live OUTSIDE run_root (WORKROOT.parent/da3_weights) and persist.
        shutil.rmtree(run_root, ignore_errors=True)
        # downsample_ply.py writes outside run_root too (see the upload step above) —
        # clean it up here for the same net-zero-per-job reason.
        shutil.rmtree(Path(PLATFORM) / "ui" / "_spaces" / slug, ignore_errors=True)
        _disk_report("job-end")


runpod.serverless.start({"handler": handler})

# build trigger 65dfca5
