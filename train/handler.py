"""RunPod TRAINING endpoint — depth-supervised splatfacto (abai's recipe).

Runs AFTER the pipeline job. Input: {"slug": "<space>"}. It pulls the scan from
S3 (scans/<slug>/: frames, views, cameras.json, intrinsics.json, pointcloud.ply),
then replicates abai's exact chain:

  filter_bad_poses.py      drop pz000 crops w/ genuine rig-roll error (>15deg from consensus up)
  make_transforms.py       cameras.json -> nerfstudio transforms.json (504)
  build_hires_dataset.py   re-render perspective crops at 1024 (reuse poses)
  reproject_scanner_depth  OUR pointcloud.ply -> per-view depth  (depth_scanner_splat/)
  generate_person_masks    YOLO person masks -> masks/  (+ wire mask_path)
  [5-iter dry run]         measure THIS scene's DEPTH_SCALE (dataparser scale)
  ns_depthsup.py           splatfacto 30k + EdgeAwareLogL1 depth loss (+masks)
  ns_export_gs.py          checkpoint -> splat.ply
  ply2ksplat.mjs           splat.ply -> splat.ksplat

Then uploads scans/<slug>/splat.ksplat to S3. The home-server worker flips
has3d=true so the tri-viewer's 3D tab un-greys and streams the splat from S3.

Output: {"slug", "has3d": true, "splat_key", "depth_scale", "iters"} or {"error"}.
"""
import os, sys, json, glob, shutil, subprocess, traceback
from pathlib import Path

import runpod
import boto3

BUCKET = os.environ["S3_BUCKET"]
s3 = boto3.client(
    "s3",
    region_name=os.environ.get("S3_REGION", "us-east-1"),
    aws_access_key_id=os.environ.get("S3_ACCESS_KEY"),
    aws_secret_access_key=os.environ.get("S3_SECRET_KEY"),
)

VENDOR = Path("/app/train/vendor")
WORKROOT = Path(os.environ.get("TRAIN_WORKROOT", "/runpod-volume/train"))
PY = sys.executable

# Scene-graph stage (Approach A) runs inline after training on the same warm
# worker. It lives in its own venv (open3d + open_clip + boto3) inside the image.
SG_DIR = Path("/app/train/scenegraph")
PY_SG = os.environ.get("PY_SG", "/opt/venv-sg/bin/python")

# Bump on every image change so the endpoint's returned build can be identified.
HANDLER_VERSION = "2026-08-06-bad-pose-filter"

# abai's detailfloater-v2 recipe (factory13-pz000-detailfloater-30k-v2), replacing the
# earlier shinhan-pz000-hires-30k values. 7 knobs differ: 3 raise DETAIL (densify on a
# 4x lower grad threshold, split smaller gaussians, densify 5k steps longer) and 4
# suppress FLOATERS (cull faint / oversized / large-screen-space gaussians, and keep
# screen-size culling active twice as long). Measured on factory13: LPIPS 0.481 -> 0.396,
# and it replaced sharpen-100k as that space's deployed model.
#
# TWO CAVEATS, unresolved at the time of this change:
#  1. That measurement was made WITHOUT the EdgeAwareLogL1 depth loss this endpoint also
#     applies (ns_depthsup.py patches splatfacto at runtime, so it leaves no trace in the
#     saved config.yml and the pairing could not be confirmed either way). The combination
#     of these knobs + depth supervision is therefore UNVERIFIED.
#  2. More densification means more gaussians, so bigger checkpoints. This handler has
#     already hit disk-full once; watch for a regression.
# Judge the result on LPIPS + a sharpness proxy, NOT PSNR: on these scenes MSE-based PSNR
# rewards blur, so it can be improved by making the model worse.
MODEL_ARGS = [
    "--pipeline.model.num-downscales", "0",
    "--pipeline.model.camera-optimizer.mode", "SO3xR3",
    "--pipeline.model.rasterize-mode", "antialiased",
    "--pipeline.model.densify-grad-thresh", "0.0001",
    "--pipeline.model.cull-alpha-thresh", "0.01",
    "--pipeline.model.cull-scale-thresh", "0.2",
    "--pipeline.model.densify-size-thresh", "0.005",
    "--pipeline.model.split-screen-size", "0.05",
    "--pipeline.model.cull-screen-size", "0.08",
    "--pipeline.model.stop-screen-size-at", "8000",
    "--pipeline.model.stop-split-at", "20000",
    "--pipeline.model.warmup-length", "500",
    "--pipeline.model.refine-every", "100",
    "--pipeline.model.reset-alpha-every", "30",
    "--pipeline.model.use-absgrad", "True",
    "--pipeline.model.use-scale-regularization", "True",
    "--pipeline.model.use-bilateral-grid", "True",
    "--pipeline.model.background-color", "random",
    "--pipeline.model.sh-degree", "3",
    "--pipeline.model.sh-degree-interval", "1000",
    "--vis", "tensorboard",
    "--viewer.quit-on-train-completion", "True",
]


def data_args(data_dir: str):
    return ["nerfstudio-data", "--data", data_dir,
            "--orientation-method", "up", "--center-method", "poses",
            "--auto-scale-poses", "True", "--train-split-fraction", "0.9"]


def _sh(cmd, env=None, cwd=None):
    print("[train] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, env=env, cwd=cwd)


def _dl_scan(slug: str, dst: Path):
    """Download scans/<slug>/ from S3 into a 3d-data-style capture dir.

    If ingest ran with INFRASCAN_STORAGE_MODE=zip, the whole scan is one archive at
    scans/<slug>.zip instead of thousands of individual keys — download and extract
    that (one GET) instead of listing+downloading every key one at a time. Falls
    back to the per-file loop below for scans uploaded in the default unpacked mode,
    so this is automatic per-scan and needs no separate config on this endpoint."""
    dst.mkdir(parents=True, exist_ok=True)
    zip_key = f"scans/{slug}.zip"
    try:
        s3.head_object(Bucket=BUCKET, Key=zip_key)
        has_zip = True
    except Exception:
        has_zip = False

    if has_zip:
        import zipfile, tempfile
        prefix = f"scans/{slug}/"
        with tempfile.NamedTemporaryFile(suffix=".zip") as tmp:
            print(f"[train] downloading archive s3://{BUCKET}/{zip_key} ...", flush=True)
            s3.download_file(BUCKET, zip_key, tmp.name)
            n = 0
            with zipfile.ZipFile(tmp.name) as zf:
                for info in zf.infolist():
                    if not info.filename.startswith(prefix):
                        continue  # skip pano_clean/... entries — train doesn't need them
                    rel = info.filename[len(prefix):]
                    if not rel:
                        continue
                    target = dst / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as fsrc, open(target, "wb") as fout:
                        shutil.copyfileobj(fsrc, fout)
                    n += 1
        if n == 0:
            raise RuntimeError(f"archive s3://{BUCKET}/{zip_key} had no {prefix} entries")
        print(f"[train] extracted {n} files from archive -> {dst}", flush=True)
        return

    prefix = f"scans/{slug}/"
    n = 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
        for o in page.get("Contents", []):
            rel = o["Key"][len(prefix):]
            if not rel:
                continue
            local = dst / rel
            local.parent.mkdir(parents=True, exist_ok=True)
            s3.download_file(BUCKET, o["Key"], str(local))
            n += 1
    if n == 0:
        raise RuntimeError(f"no objects under s3://{BUCKET}/{prefix}")
    print(f"[train] downloaded {n} files -> {dst}", flush=True)


def _add_mask_paths(ds: Path):
    """nerfstudio uses per-frame mask_path; generate_person_masks writes masks/<stem>.png
    but doesn't edit transforms.json. Wire each frame to its mask if present."""
    tj_path = ds / "transforms.json"
    tj = json.loads(tj_path.read_text())
    masks = ds / "masks"
    wired = 0
    for fr in tj.get("frames", []):
        stem = Path(fr["file_path"]).stem
        m = masks / f"{stem}.png"
        if m.exists():
            fr["mask_path"] = f"masks/{stem}.png"
            wired += 1
    tj_path.write_text(json.dumps(tj))
    print(f"[train] wired {wired} mask_path entries", flush=True)


def _measure_depth_scale(ds: Path, probe_dir: Path) -> float:
    """5-iter dry run → read the dataparser's auto-scale-poses factor (per-scene).
    Reusing another scene's scale is the exact bug abai hit (PSNR plateau)."""
    _sh(["ns-train", "splatfacto", "--output-dir", str(probe_dir),
         "--experiment-name", "scaleprobe", "--max-num-iterations", "5",
         "--vis", "tensorboard", "--viewer.quit-on-train-completion", "True"]
        + data_args(str(ds)))
    dpt = sorted(probe_dir.glob("**/dataparser_transforms.json"))
    if not dpt:
        raise RuntimeError("dataparser_transforms.json not found after dry run")
    scale = float(json.loads(dpt[-1].read_text())["scale"])
    print(f"[train] measured DEPTH_SCALE = {scale}", flush=True)
    return scale


def _sg_extra_args(inp):
    """Optional per-job scene-graph knobs, passed through to run_scenegraph.py.
    All optional; omit for the documented defaults (yaw=0, surface=table)."""
    extra = []
    for key, flag in (("yaw_deg", "--yaw-deg"), ("surface_label", "--surface-label"),
                      ("max_long_m", "--max-long-m"), ("drop_labels", "--drop-labels")):
        if inp.get(key) is not None:
            extra += [flag, str(inp[key])]
    if inp.get("skip_masks"):
        extra.append("--skip-masks")
    if inp.get("skip_verify"):
        extra.append("--skip-verify")
    return extra


def handler(job):
    inp = job.get("input", {}) or {}
    slug = inp.get("slug")
    if not slug:
        return {"error": "provide input.slug"}

    # Fast path / manual trigger: re-run ONLY the scene-graph stage on the
    # already-trained splat in S3 (no retrain). Reads scans/<slug>/ (which now
    # includes splat.ply) and runs run_scenegraph on those local files.
    if inp.get("only_scenegraph"):
        root = WORKROOT / slug
        if root.exists():
            shutil.rmtree(root)
        src = root / "src"
        try:
            _dl_scan(slug, src)          # incl. splat.ply + pointcloud/cameras/intrinsics/views
            _sh([PY_SG, SG_DIR / "run_scenegraph.py", "--slug", slug,
                 "--splat", src / "splat.ply", "--pointcloud", src / "pointcloud.ply",
                 "--cameras", src / "cameras.json",
                 "--intrinsics", src / "intrinsics.json",
                 "--views", src / "views", "--workdir", root / "sg", "--bucket", BUCKET]
                + _sg_extra_args(inp))
            return {"slug": slug, "only_scenegraph": True, "has_scene_graph": True,
                    "handler_version": HANDLER_VERSION}
        except subprocess.CalledProcessError as e:
            return {"slug": slug, "only_scenegraph": True, "has_scene_graph": False,
                    "error": f"scenegraph rc={e.returncode} — see scans/{slug}/scenegraph_error.txt",
                    "handler_version": HANDLER_VERSION}
        finally:
            shutil.rmtree(root, ignore_errors=True)

    iters = int(inp.get("iters", 30000))
    use_masks = bool(inp.get("masks", True))

    root = WORKROOT / slug
    if root.exists():
        shutil.rmtree(root)
    src = root / "src"; ds504 = root / "ds504"; ds = root / "ds1024"
    out = root / "out"; export = root / "export"

    try:
        _dl_scan(slug, src)

        # 1) drop pz000 crops with a genuine rig-roll error (>15deg from this scan's own
        # consensus "up" direction) BEFORE building the training dataset, so the exclusion
        # propagates cleanly through make_transforms/build_hires_dataset/depth reprojection
        # with no separate patching downstream. Automates the diagnostic previously run by
        # hand on every space (see PSNR_PLATEAU_DIAGNOSTIC.md Sec 7): these are real pose
        # defects (raw pose wrong, not a training artifact), rare (~1-2%) but each one
        # renders catastrophically at its own viewpoint and can't be trained around.
        bad_pose_report = root / "bad_pose_report.json"
        _sh([PY, VENDOR / "filter_bad_poses.py", "--cameras", src / "cameras.json",
             "--report", bad_pose_report])
        bad_poses_dropped = json.loads(bad_pose_report.read_text())["pz000_dropped"]

        # 2) cameras.json -> transforms.json (504)  3) re-render at 1024
        _sh([PY, VENDOR / "make_transforms.py", "--src", src, "--out", ds504])
        _sh([PY, VENDOR / "build_hires_dataset.py", "--src", src,
             "--ref-data", ds504, "--out", ds, "--res", "1024"])

        # 4) reproject OUR pointcloud -> per-view depth. depth_scanner_splat (NOT the
        # legacy depth_scanner name): each point is splatted over its own angular
        # footprint (point_spacing/z), fixing a z-buffer occlusion bug where a near
        # surface's sparse pixels let a far surface leak through and win depth. Root
        # cause of the "wall reads as absent in free-fly" report -- alpha coverage was
        # always 100%, it was the wrong surface at the wrong depth, not missing geometry.
        # --point-spacing defaults to -1 (auto-measured from the cloud); do not pass 0,
        # which selects the legacy leaking behaviour.
        _sh([PY, VENDOR / "reproject_scanner_depth.py", "--data", ds,
             "--pointcloud", src / "pointcloud.ply", "--out", ds / "depth_scanner_splat"])

        # 5) YOLO person masks (+ wire mask_path)
        if use_masks:
            _sh([PY, VENDOR / "generate_person_masks.py", "--data", ds])
            _add_mask_paths(ds)

        # 6) per-scene DEPTH_SCALE (dry run)
        scale = _measure_depth_scale(ds, root / "scaleprobe")

        # 7) depth-supervised splatfacto (abai's recipe)
        env = os.environ.copy()
        env["DEPTH_DIR"] = str(ds / "depth_scanner_splat")
        env["DEPTH_W"] = "0.1"
        env["DEPTH_START_ITER"] = "500"
        env["DEPTH_SCALE"] = str(scale)
        # save checkpoints infrequently — each splatfacto ckpt is ~700MB-1GB (3M
        # gaussians); every 2000 filled the disk. 5000 -> ~5 ckpts (few GB, fits
        # the container disk) and guarantees a late checkpoint exists to export.
        _sh([PY, VENDOR / "ns_depthsup.py", "splatfacto",
             "--output-dir", out, "--experiment-name", slug,
             "--max-num-iterations", str(iters), "--steps-per-save", "5000"]
            + MODEL_ARGS + data_args(str(ds)), env=env)

        # 8) export checkpoint -> splat.ply
        cfg = sorted(out.glob("**/config.yml"))
        if not cfg:
            raise RuntimeError("no config.yml after training")
        _sh([PY, VENDOR / "ns_export_gs.py", "--load-config", cfg[-1],
             "--output-dir", export])

        # 9) save the trained splat.ply to S3 FIRST (so a 50-min run is never lost
        #    if a trailing step fails — we can re-convert without re-training).
        ply = export / "splat.ply"
        s3.upload_file(str(ply), BUCKET, f"scans/{slug}/splat.ply")
        print(f"[train] uploaded scans/{slug}/splat.ply", flush=True)

        # 10) splat.ply -> splat.ksplat -> S3 so the tri-viewer's 3D tab can stream it
        ksplat = root / "splat.ksplat"
        _sh(["node", VENDOR / "node" / "ply2ksplat.mjs", ply, ksplat, "3"])
        key = f"scans/{slug}/splat.ksplat"
        s3.upload_file(str(ksplat), BUCKET, key)
        print(f"[train] uploaded {key}", flush=True)

        # 11) scene graph (Approach A: OWLv2-on-splat) on the just-trained splat.
        #     Runs on the SAME warm GPU worker and reads the local files
        #     (export/splat.ply + src/{pointcloud.ply,cameras.json,intrinsics.json,
        #     views/}) — no re-download. Uploads scans/<slug>/scene_graph.json.
        #     NON-FATAL: the splat is already safely in S3, so a scene-graph
        #     failure must never fail the training job.
        has_scene_graph = False
        try:
            _sh([PY_SG, SG_DIR / "run_scenegraph.py", "--slug", slug,
                 "--splat", ply, "--pointcloud", src / "pointcloud.ply",
                 "--cameras", src / "cameras.json",
                 "--intrinsics", src / "intrinsics.json",
                 "--views", src / "views", "--workdir", root / "sg",
                 "--bucket", BUCKET]
                + _sg_extra_args(inp))
            has_scene_graph = True
            print(f"[train] uploaded scans/{slug}/scene_graph.json", flush=True)
        except Exception as e:
            print(f"[train] scenegraph stage failed (non-fatal): {e}", flush=True)

        return {"slug": slug, "has3d": True, "splat_key": key,
                "has_scene_graph": has_scene_graph,
                "depth_scale": scale, "iters": iters,
                "bad_poses_dropped": bad_poses_dropped,
                "handler_version": HANDLER_VERSION}
    except subprocess.CalledProcessError as e:
        return {"error": f"stage failed rc={e.returncode}: {e.cmd[:3]}",
                "trace": traceback.format_exc()[-1500:]}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}",
                "trace": traceback.format_exc()[-1500:]}
    finally:
        shutil.rmtree(root, ignore_errors=True)


runpod.serverless.start({"handler": handler})
