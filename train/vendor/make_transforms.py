"""Convert a 3d-data/<space> capture (cameras.json + intrinsics.json + views/) into a
nerfstudio dataset (transforms.json), reusing the existing perspective crops.

Pose convention (validated against our known-good shinhan transforms.json):
    transform_matrix (c2w, OpenGL) = [ R_cam @ diag(1,-1,-1) | pos ]
where cameras.json R is the OpenCV c2w rotation and pos is the camera position.

Uses ALL views (all pitches) — MCMC handles the image count, and more pitches =
better floor/ceiling coverage. Init from pointcloud.ply.

Usage:
    python make_transforms.py --selfcheck                     # verify convention on shinhan
    python make_transforms.py --src <3d-data/space> --out <dataset dir>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

FLIP = np.diag([1.0, -1.0, -1.0])   # OpenCV c2w -> OpenGL c2w (negate Y,Z axes)

def c2w_from(cam):
    R = np.asarray(cam["R"], dtype=np.float64)          # [3,3] OpenCV c2w
    t = np.asarray(cam["pos"], dtype=np.float64)        # [3]
    T = np.eye(4)
    T[:3, :3] = R @ FLIP
    T[:3, 3] = t
    return T

def selfcheck():
    cams = json.load(open("/home/mariyam/3d-data/shinhan_space/cameras.json"))
    tj = json.load(open("/home/abai/splatfacto/data/shinhan-pz000-hires/transforms.json"))
    byid = {c["id"]: c for c in cams}
    # frame_NNNNNN -> the pz000_y* view; match each transforms frame to the cam whose
    # flipped R best matches T's rotation (rotation match is unambiguous).
    errs = []
    Rall = np.stack([np.asarray(c["R"]) @ FLIP for c in cams])   # [M,3,3]
    for f in tj["frames"][:200]:
        T = np.asarray(f["transform_matrix"]); Rt = T[:3, :3]
        d = np.linalg.norm(Rall - Rt[None], axis=(1, 2))
        errs.append(d.min())
    errs = np.array(errs)
    print(f"[selfcheck] rotation match error over 200 frames: "
          f"mean={errs.mean():.4f} max={errs.max():.4f}  (small => convention correct)")
    return errs.max() < 0.05

def convert(src, out):
    src, out = Path(src), Path(out)
    cams = json.load(open(src / "cameras.json"))
    intr = json.load(open(src / "intrinsics.json"))
    out.mkdir(parents=True, exist_ok=True)
    # images: symlink to the existing crops (views/), so nothing is duplicated
    views = src / "views"
    (out / "images").exists() or (out / "images").symlink_to(views)
    # init pointcloud — prefer the ingest-side voxel-downsampled cloud (built for the
    # web topdown view, but just as good an init cloud and orders of magnitude
    # smaller) over the raw one. A dense/3-pitch scan's raw pointcloud.ply can exceed
    # 30M points, which splatfacto turns into 30M+ uncapped Gaussians — enough to OOM
    # the GPU before a single training step runs. Older scans uploaded before this
    # existed won't have it, so fall back to the raw cloud.
    ply = src / "pointcloud_downsampled.ply"
    if not ply.exists():
        ply = src / "pointcloud.ply"
    print(f"[convert] init pointcloud: {ply.name}")
    if ply.exists():
        (out / "sparse_pc.ply").exists() or (out / "sparse_pc.ply").symlink_to(ply)

    frames = []
    cam_pos = []
    for c in cams:
        pano = c["pano"].split("/")[-1]            # panos/000000_..jpg -> 000000_..jpg
        T = c2w_from(c)
        cam_pos.append(T[:3, 3])
        frames.append({"file_path": f"images/{pano}", "transform_matrix": T.tolist()})
    out_json = {
        "camera_model": "OPENCV",
        "fl_x": intr["fx"], "fl_y": intr["fy"], "cx": intr["cx"], "cy": intr["cy"],
        "w": intr["width"], "h": intr["height"],
        "frames": frames,
    }
    if ply.exists():
        out_json["ply_file_path"] = "sparse_pc.ply"
    json.dump(out_json, open(out / "transforms.json", "w"))

    # --- CPU sanity: cameras should sit within the point-cloud footprint and look inward ---
    cam_pos = np.asarray(cam_pos)
    try:
        from plyfile import PlyData
        v = PlyData.read(str(ply))["vertex"]
        pc = np.column_stack((v["x"], v["y"], v["z"])).astype(np.float64)
        c_lo, c_hi = cam_pos.min(0), cam_pos.max(0)
        p_lo, p_hi = pc.min(0), pc.max(0)
        inside = np.all((cam_pos >= p_lo - 0.5) & (cam_pos <= p_hi + 0.5), axis=1).mean()
        # forward = -Z of OpenGL c2w; check it points toward the scene centroid on average
        ctr = pc.mean(0)
        fwd = -np.stack([c2w_from(c)[:3, 2] for c in cams])
        todir = ctr[None] - cam_pos; todir /= (np.linalg.norm(todir, axis=1, keepdims=True) + 1e-9)
        align = (fwd * todir).sum(1).mean()
        print(f"[convert] {len(frames)} frames | cams inside PC bounds: {100*inside:.0f}% | "
              f"mean forward·(toward-centroid): {align:+.2f}")
        print(f"[convert] cam bounds x[{c_lo[0]:.1f},{c_hi[0]:.1f}] vs PC x[{p_lo[0]:.1f},{p_hi[0]:.1f}]")
    except Exception as e:
        print("[convert] sanity skipped:", e)
    print(f"[convert] wrote {out/'transforms.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--src"); ap.add_argument("--out")
    a = ap.parse_args()
    if a.selfcheck:
        ok = selfcheck(); print("SELFCHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)
    if not (a.src and a.out): sys.exit("need --src and --out (or --selfcheck)")
    convert(a.src, a.out)
