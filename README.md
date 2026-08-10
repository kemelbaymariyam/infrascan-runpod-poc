# InfraScan RunPod pipeline

A 360° video goes in; a searchable 3D digital twin (Gaussian splat + scene graph)
comes out. All the heavy GPU work runs on **two RunPod serverless endpoints**, each
built from a different branch of this repo:

```
website/worker (infrascan-server)
        │  1. upload video -> S3, dispatch job
        ▼
┌─────────────────────────┐        ┌──────────────────────────┐
│  INGEST endpoint          │        │  TRAIN endpoint            │
│  branch: main              │──────► │  branch: train              │
│  handler.py                │  S3    │  train/handler.py           │
│  video -> frames -> views  │  zip   │  splat.ply + splat.ksplat   │
│  -> DA3 depth/poses ->     │        │  + scene_graph.json         │
│  pointcloud -> pano_clean  │        │  (Approach-A OWLv2 scene    │
│  -> pano_lowres            │        │   graph, same GPU worker)   │
└─────────────────────────┘        └──────────────────────────┘
        │                                       │
        └───────────────► S3 (shared bucket) ◄──┘
                                  │
                                  ▼
                       viewer (infrascan-onprem) streams from S3
```

Both endpoints are dispatched by `worker/runpod_worker.py` in the
`infrascan-server` repo — this repo only holds what actually runs *on* RunPod.

## Files
- `handler.py` — **ingest** endpoint (deployed from the `main` branch): video → frames
  → views → DA3 depth/poses → point cloud → operator-removed panoramas → upload.
- `train/handler.py` — **train** endpoint (deployed from the `train` branch): downloads
  the ingest output, trains a depth-supervised Gaussian splat, runs the scene-graph
  stage, uploads results.
- `train/scenegraph/run_scenegraph.py` — the scene-graph sub-stage (own venv, see below).
- `pipeline/` — the vendored infrascan pipeline scripts the ingest handler calls.
- `Dockerfile` (root) / `train/Dockerfile` — one image per endpoint.
- `website/index.html` — a bare-bones manual tester (curl is more reliable, see below).

---

## Pipeline stages

### Ingest (main branch → `infrascan-runpod-ingest` endpoint)
One video in, per-view depth + poses + point cloud + viewer-ready panoramas out.

| # | Stage | What it does |
|---|---|---|
| 1 | `_00_stitch_insv.py` | Insta360 dual-fisheye → single equirectangular video (skipped for plain `.mp4`) |
| 2 | `00_video_to_img.py` | Extract equirect frames every `every_n`-th frame |
| 3 | `00a_sample_views.py` | Sample perspective crops at 3 pitches (`-30°, 0°, +30°`) per frame — richer point-cloud coverage; training later filters back down to pitch-0 only |
| 4 | `00b_da3_streaming.py` | DA3 (Depth Anything 3, streaming) estimates **camera poses + per-view depth** from the views — the only source of poses, everything downstream needs `cameras.json` |
| 5 | `pipeline.runner` (`gen_topdown`, `downsample_ply`) | Builds the point cloud from DA3's depth+poses, renders a floor-plan thumbnail, voxel-downsamples the cloud (train-only, keeps splatfacto from OOMing on a 30M+ point dense scan) |
| 6 | `pano_clean.py` | YOLO-seg + LaMa inpainting erases the camera operator from each equirect panorama. **Non-fatal** — a bad inpaint never fails the whole ingest, panoramas just keep the operator |
| 7 | pano_lowres (inline in `handler.py`) | Downsamples pano_clean's output to 2560×1280 — what the panorama viewer actually streams by default (full-res kept as fallback) |
| 8 | upload | Two uploads, see **Storage** below |

Stages 6–7 are wrapped in their own try/except — a failure there degrades gracefully
(operator stays visible, or full-res panoramas get served) rather than failing the scan.

### Train (train branch → `infrascan-runpod-train` endpoint)
Runs after ingest. Downloads the scan, trains a Gaussian splat, builds the scene graph.

| # | Stage | What it does |
|---|---|---|
| 1 | `filter_bad_poses.py` | Drops pitch-0 crops with genuine rig-roll error (>15° from this scan's own consensus "up") — rare (~1–2%) real pose defects that would otherwise render catastrophically at their own viewpoint |
| 2 | `make_transforms.py` | `cameras.json` → nerfstudio `transforms.json` at 504px |
| 3 | `build_hires_dataset.py` | Re-renders the same poses at 1024px for training resolution |
| 4 | `reproject_scanner_depth.py` | Reprojects the ingest point cloud into per-view depth maps (`depth_scanner_splat/`) — this is the depth *supervision* signal, distinct from DA3's own depth |
| 5 | `generate_person_masks.py` | YOLO person masks, so people walking through the capture don't get baked into the splat |
| 6 | depth-scale probe | A 5-iteration dry run measures this scene's own `DEPTH_SCALE` before the real run |
| 7 | `ns_depthsup.py` | **The training step** — splatfacto (nerfstudio) + EdgeAwareLogL1 depth loss, `iters` gaussians (default 30,000; caller-configurable) |
| 8 | `ns_export_gs.py` | Checkpoint → `splat.ply` — uploaded to S3 immediately, before the next step, so a late-stage failure never loses a training run |
| 9 | `ply2ksplat.mjs` (Node) | `splat.ply` → `splat.ksplat` — the format the web viewer's 3D tab actually streams |
| 10 | `run_scenegraph.py` | Scene graph ("Approach A": OWLv2 object detection projected onto the splat) — **non-fatal**, the splat is already safe in S3 by this point |

The scene-graph stage can also be re-run **on its own**, without retraining, by
passing `{"only_scenegraph": true, "slug": "..."}` — useful after a scene-graph-only
bugfix, since retraining is the expensive part (tens of minutes) and the scene graph
only reads already-uploaded splat/pointcloud/camera files.

---

## Storage: one archive for train, individual files for the viewer
Each ingest run uploads two things, sized for two different consumers:

- **`scans/<slug>.zip`** — the complete original dataset (raw frames, `views/`, `depth/`,
  raw `pointcloud.ply`, `cameras.json`, `intrinsics.json`). One archive, one PUT. This is
  what the train endpoint downloads (`train/handler.py`'s `_dl_scan()` auto-detects and
  extracts it) and doubles as the permanent archive of the original capture — if
  pano_clean or training ever need redoing with a better model, the source is never
  lost. Always built, not a mode/toggle: `views/`+`depth/` are the vast majority of a
  scan's file count (~7,400 of ~7,600 on a dense scan) and *neither* the panorama
  viewer nor the 3D/scene-graph overlay ever reads either of them — only training does.
- **Individual S3 objects**, all nested under `scans/<slug>/` alongside `frames/`,
  `splat.ksplat` and `scene_graph.json` — one scan, one place, instead of scattering
  related data across sibling top-level prefixes:
  - `scans/<slug>/frames/` — raw panoramas (kept as the final viewer fallback)
  - `scans/<slug>/pano_clean/frames/`, `cameras.json` — operator removed, full-res
  - `scans/<slug>/pano_lowres/frames/`, `cameras.json` — operator removed + downsampled (default)
  - `scans/<slug>/cameras.json`, `intrinsics.json`
  - `scans/<slug>/splat.ply`, `splat.ksplat`, `scene_graph.json` (uploaded by train)
  - `scans/<slug>/_history/<scan_id>/` — a **prior** scan, archived here in full
    (including its own `archive.zip`) the moment a re-scan starts, or a user deletes/
    switches away from it — see next section.

  Small (frames are ~200 files, not the ~7,400 that `views/`+`depth/` add up to), so
  the on-prem server streams this set straight from S3 with no local caching. Deliberately
  *not* any point cloud — the viewer's minimap uses a splat-derived `floorplan.json` or a
  pure-`cameras.json` fallback, never a point cloud.

`pointcloud_downsampled.ply` (train-only, inside the zip) is the same voxel-downsampled
cloud `pipeline.runner`'s `downsample_ply` stage already computes — the train endpoint
prefers it for splatfacto's initial Gaussians, since a dense/3-pitch scan's raw cloud can
exceed 30M points and OOM the GPU before training starts.

### Re-scans and "which scan is live" — both directions now handled
Only **one** scan per space is ever "live" at `scans/<slug>/` (+ `scans/<slug>.zip`) at
a time — that's the prefix the viewer reads by default. Every other scan for that space
lives under `scans/<slug>/_history/<scan_id>/`, including its own `archive.zip`.

- **A re-scan** (this repo, `handler.py`): before dispatching the new job, the on-prem
  worker archives the *entire* live prefix — folder tree **and** the sibling zip — into
  `_history/<prior_scan_id>/`, so the new run starts from a clean slate and nothing
  is silently overwritten.
- **Deleting the live scan, or switching "shown" to an older one** (on-prem app, not
  this repo): the same archive step runs in reverse — the outgoing scan's data moves
  into its own `_history/<id>/`, and the incoming scan's data gets promoted *back* to
  the live prefix. Without this, the DB would say one scan is active while the viewer
  kept serving whichever scan's files happened to physically be at `scans/<slug>/` —
  this was a real bug (fixed 2026-08-10) that could permanently orphan a deleted scan's
  files with no way to purge them.

---

## Deployed RunPod endpoints
Both are **serverless** (scale-to-zero, pay only while a job runs), built by RunPod
directly from GitHub (no local Docker push needed) — one endpoint per branch.

| | Ingest | Train |
|---|---|---|
| Endpoint name | `infrascan-runpod-ingest` | `infrascan-runpod-train` |
| Source branch | `main` | `train` |
| GPU (RunPod picks from this pool, cheapest available first) | RTX 6000 Ada / L40 / L40S / H100 (PCIe, 80GB HBM3, or NVL) | A40 / RTX A6000 / RTX 6000 Ada / L40 / L40S |
| Container disk | 250 GB | 60 GB |
| Execution timeout | 15,000,000 ms (**~250 min**) | 25,000,000 ms (**~417 min**) |
| Idle timeout (worker stays warm after a job) | 5 s | 5 s |
| Workers min / max / standby | 0 / 2 / 2 | 0 / 1 / 1 |
| Scaling | Queue-delay based (scales up once jobs wait ~4s) | Same |
| FlashBoot | off | off |

Pulled live from the RunPod REST API (`GET /v1/endpoints`), not hand-maintained — if
you change these in the RunPod console, this table will drift; re-check via the API
rather than trusting it blindly for anything time-sensitive.

A few things worth knowing:
- **`infrascan-server`'s own `JOB_TIMEOUT_MIN`** (`.env`, default 210 min) is the
  *client's* give-up timeout — separate from and shorter than RunPod's own execution
  timeout above. If a job is going to legitimately run longer than that on a very
  dense scan, raise `JOB_TIMEOUT_MIN` rather than the RunPod-side timeout.
- Ingest gets a **bigger container disk (250 GB)** than train (60 GB) because a dense,
  3-pitch scan's `views/`+`depth/` can be tens of GB before anything gets uploaded;
  train only ever holds one scan's zip + training intermediates at a time.
- Both templates have **AWS/S3 credentials baked in as plain environment variables**
  (RunPod's serverless templates don't have a secrets vault) — anyone with the RunPod
  API key for this account can read them back out via the API, same way this table was
  generated. Worth keeping in mind if the API key's exposure ever needs re-assessing.
- The ingest template still carries a leftover `INFRASCAN_STORAGE_MODE=zip` env var
  from an earlier design where zip-vs-individual-files was a toggle — `handler.py` no
  longer reads it (both uploads always happen now), so it's dead and harmless, not a
  bug to chase.
- Image tags follow the git commit that built them, e.g.
  `k-ailab-infrascan-runpod-train-train-dockerfile:26cc047da` — the endpoint always
  serves whatever was last pushed to its branch, so `git log <branch>` tells you
  exactly what's live.

### Deploying / updating an endpoint
Your dev box is aarch64 but RunPod GPUs are amd64, so let RunPod build the image —
don't try to build+push locally.

1. Push your change to the relevant branch (`main` for ingest, `train` for train).
2. RunPod auto-builds from GitHub on push (already wired up for both endpoints) — check
   the endpoint's **Builds** tab in the RunPod console for progress/errors.
3. Once it shows **ready**, new jobs pick up the new image automatically — no manual
   redeploy step.

To set up a **new** endpoint from scratch: RunPod console → Serverless → New Endpoint →
"Import Git Repository" → point at the branch → RunPod finds the `Dockerfile` and builds.

## Host a test video (RunPod must reach it by URL)
Local videos aren't reachable by RunPod. Put one somewhere with a public/presigned URL
(S3 presigned GET works fine). Trim a short clip first for fast iteration:
```bash
ffmpeg -i 8k_route1.mp4 -t 5 -c copy clip5s.mp4   # first 5 seconds
```

## Test an endpoint directly (curl — most reliable)
```bash
# submit (ingest example — train's payload is {"slug": "..."} instead)
curl -s -X POST https://api.runpod.ai/v2/ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"video_url":"https://.../clip5s.mp4","slug":"my-test-space","every_n":50}}'
# -> {"id":"...","status":"IN_QUEUE"}

# poll
curl -s https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
```

## Test from the website
`website/index.html` is a bare manual tester (endpoint ID + API key + video URL →
submit → poll → render). It predates the real ingest/train split and only exercises the
old inline-base64 stage-0 response shape — use the curl commands above for anything
against the current endpoints. (If the browser call is CORS-blocked, that's expected —
a real client always proxies through a backend so the API key stays server-side; see
`infrascan-server/worker/runpod_worker.py` for how the real one does it.)
