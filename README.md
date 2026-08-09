# InfraScan video→data — RunPod POC

Minimal proof of concept: a **website** sends a 360° video to a **RunPod serverless
worker** (a Docker image) that runs the infrascan **stage-0 pipeline**
(stitch → frames → perspective views) and returns the result. No depth/poses,
no reconstruction, no viewer — just the video→data front half, to learn the
RunPod + Docker loop.

```
website/index.html  ──►  RunPod endpoint (this Docker image)  ──►  handler.py
                                                                     stitch → frames → views
                         ◄── {num_frames, num_views, sample_view} ──┘
```

## Files
- `handler.py` — the worker: downloads the video, runs the 3 stage-0 scripts, returns counts + one sample view (base64).
- `pipeline/` — the 3 infrascan stage-0 scripts (vendored, unchanged).
- `Dockerfile` — python + ffmpeg + opencv + runpod SDK.
- `website/index.html` — enter endpoint + video URL, see the result.

## Upload mode: `INFRASCAN_STORAGE_MODE`
Set on this endpoint's own env (RunPod endpoint template, not a per-job input field):
- `unpacked` (default) — every file in the scan is its own S3 object under `scans/<slug>/`.
  The home server stays storage-free; the viewer streams each file from S3 on demand.
  Thousands of individual PUTs on a dense scan means per-request overhead dominates and
  the upload step alone can take many minutes.
- `zip` — bundles the whole scan into one `scans/<slug>.zip` object instead (one PUT).
  Much faster upload, at the cost of the home server needing to unpack it locally to
  serve it (see `infrascan-onprem`'s README, "Storage mode" section) and the train
  endpoint downloading+extracting the archive instead of listing individual keys —
  both auto-detect the archive's presence per scan, no other config needed. Fully
  reversible: set this back to `unpacked` and every later scan goes back to individual
  S3 objects.

`handler.py` also uploads `scans/<slug>/pointcloud_downsampled.ply` (the same
voxel-downsampled cloud `pipeline.runner`'s `downsample_ply` stage already computes for
the web topdown view) alongside the raw `pointcloud.ply` — the train endpoint prefers it
for splatfacto's initial Gaussians, since a dense/3-pitch scan's raw cloud can exceed 30M
points and OOM the GPU before training starts.

## Note on GPU
This stage-0 part is **CPU-only** (ffmpeg + OpenCV). You can deploy it on a small/CPU
worker. GPU is only needed *later* for DA3 depth+poses and splat training.

---

## Deploy (recommended: RunPod builds from GitHub — no local Docker)
Your dev box is aarch64 but RunPod GPUs are amd64, so let RunPod build the image.

1. **Put this folder in a GitHub repo** (private is fine):
   ```bash
   cd /home/mariyam/infrascan-runpod-poc
   git init && git add -A && git commit -m "infrascan video->data runpod poc"
   # create an empty repo on github, then:
   git remote add origin git@github-mariyam:YOURUSER/infrascan-runpod-poc.git
   git push -u origin main
   ```
2. **RunPod console → Serverless → New Endpoint → "Import Git Repository"** (GitHub).
   - Connect the repo. RunPod finds the `Dockerfile`, builds it (amd64), and deploys.
   - Worker: min 0 / max 1, a small GPU (or CPU), container disk ~10 GB (room for the video).
3. Wait until the endpoint shows **ready**. Copy its **Endpoint ID**.

## Host a test video (RunPod must reach it by URL)
Your test videos are local to the dev box; RunPod can't see them. Put one where it
has a public URL — e.g. a cloud bucket (S3/R2/GCS) with a presigned URL, or a temp
host. Tip: trim a short clip first so the download is fast:
```bash
ffmpeg -i 8k_route1.mp4 -t 5 -c copy clip5s.mp4   # first 5 seconds
```

## Test the endpoint (curl — most reliable)
```bash
# submit
curl -s -X POST https://api.runpod.ai/v2/ENDPOINT_ID/run \
  -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" \
  -d '{"input":{"video_url":"https://.../clip5s.mp4","every_n":50}}'
# -> {"id":"...","status":"IN_QUEUE"}

# poll
curl -s https://api.runpod.ai/v2/ENDPOINT_ID/status/JOB_ID \
  -H "Authorization: Bearer YOUR_API_KEY"
# -> when COMPLETED: {"output":{"num_frames":..,"num_views":..,"sample_view_jpg_base64":".."}}
```

## Test from the website
Open `website/index.html` in a browser, fill in Endpoint ID + API key + video URL,
click **Process**. It submits, polls, and renders the sample view.
(If the browser call is CORS-blocked, use the curl commands — a real site would add
a tiny backend proxy so the API key stays server-side.)

---

## How you'd extend this to the full product (later)
- **Add DA3 depth+poses:** bake the DA3 weights into the image, add a `00b` step in
  `handler.py`, switch the worker to a **GPU**. Now the output is a full posed dataset.
- **Add reconstruction:** append `ns-train`/`gsplat` to the same worker → return a `.ply`.
- **Big outputs:** stop returning inline base64; upload results to a bucket and return
  a URL (base64 is only fine for this small POC).
- **Real uploads:** website uploads the video to your bucket (presigned PUT), then
  passes that URL to the endpoint.
