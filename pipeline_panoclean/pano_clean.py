"""Operator removal (pano_clean) — erase the camera operator from equirect panoramas.

NADIR REPROJECTION (v2): reproject each panorama's bottom cone to a flat, undistorted
down-view, YOLO-seg the person + LaMa-inpaint there (clean, because the floor is no
longer stretched), then inverse-reproject the cleaned patch back into the panorama below
BAND_LAT with a per-row feathered seam. Ported from op_removal_test/nadir_proto.py, which
proved v1 (LaMa straight on the distorted equirect nadir) produced a grey smear — this
undistort-first approach does not. The equirect<->perspective math matches
pipeline/00a_sample_views.build_maps (yaw=0, pitch about X).

Runs as a post-pipeline step inside the ingest job (handler.py), on the RunPod GPU, so it
never uses the on-prem GB10. Reads the local equirect frames the pipeline already produced
and writes cleaned copies; handler.py uploads them to S3 under pano_clean/<slug>/frames/.

    python pano_clean.py --frames <dir> --out <dir> [--yolo yolo11x-seg.pt]

Resumable: skips output frames that already exist. Frames with no person detected are
copied through unchanged.
"""
import argparse
import glob
import os

import cv2
import numpy as np
from PIL import Image
from simple_lama_inpainting import SimpleLama
from ultralytics import YOLO

FOV = 150          # wide enough to catch the whole operator wedge (head -> nadir)
OUTSZ = 1200       # flat down-view resolution
DOWN_P = 90        # build_maps pitch that aims the view straight down (nadir)
BAND_LAT = -18     # replace panorama below this latitude (deg); the flat view covers it
FEATHER_DEG = 8    # soft blend over this many degrees at the seam

# A single YOLO pass on the down-view sometimes segments only *part* of the
# operator (e.g. torso + arm but not the head) even though overall detection
# confidence is fine — confirmed on a real frame (hong_kong_office scanpoint 32,
# 2026-08-10) where confidence swung from 0.32 to 0.65 on tiny (2-5deg) rotations
# of the exact same crop, i.e. this specific input sits in a genuinely unstable
# part of the model's decision surface, not a systematic issue (dark hair,
# motion blur, and hair/carpet contrast were all directly measured and ruled
# out). Detecting on a few rotated copies and unioning the masks (each rotated
# back) catches area a single pass misses; every tested rotation added new
# mask coverage regardless of whether its own confidence went up or down.
# Cheap: detection is ~2% of this function's cost, LaMa dominates and still
# runs once. Verified to add zero false positives on person-free crops at
# conf=0.05 (well below MASK_CONF below) across all angles.
MASK_ROTATIONS_DEG = (0, -5, -2, 2, 5)
MASK_CONF = 0.25


def build_maps(fov, pitch, ow, oh, eh, ew):
    """00a_sample_views convention (yaw=0): output pixel -> equirect (u,v) to sample."""
    fov = np.deg2rad(fov); pit = np.deg2rad(pitch)
    fx = (ow / 2) / np.tan(fov / 2); fy = fx; cx = ow / 2; cy = oh / 2
    xv, yv = np.meshgrid(np.arange(ow, dtype=np.float32), np.arange(oh, dtype=np.float32))
    x = (xv - cx) / fx; y = -(yv - cy) / fy; z = np.ones_like(x)
    n = np.sqrt(x * x + y * y + z * z); x /= n; y /= n; z /= n
    cp, sp = np.cos(pit), np.sin(pit)
    x2 = x; y2 = cp * y - sp * z; z2 = sp * y + cp * z          # pitch about X (yaw=0)
    lon = np.arctan2(x2, z2); lat = np.arcsin(np.clip(y2, -1, 1))
    return (((lon / (2 * np.pi) + 0.5) * ew).astype(np.float32),
            ((0.5 - lat / np.pi) * eh).astype(np.float32),
            (fx, fy, cx, cy, pit))


def inv_maps(eh, ew, row0, fx, fy, cx, cy, pit):
    """For equirect rows [row0, H): the flat-view (u,v) to sample + validity mask."""
    xv, yv = np.meshgrid(np.arange(ew, dtype=np.float32),
                         np.arange(row0, eh, dtype=np.float32))
    lon = (xv / ew - 0.5) * 2 * np.pi; lat = (0.5 - yv / eh) * np.pi
    X = np.cos(lat) * np.sin(lon); Y = np.sin(lat); Z = np.cos(lat) * np.cos(lon)
    cp, sp = np.cos(pit), np.sin(pit)
    x1 = X; y1 = cp * Y + sp * Z; z1 = -sp * Y + cp * Z         # undo the pitch rotation
    valid = z1 > 1e-6; zz = np.where(valid, z1, 1.0)
    u = cx + fx * (x1 / zz); v = cy - fy * (y1 / zz)
    inb = valid & (u >= 0) & (u < 2 * cx) & (v >= 0) & (v < 2 * cy)
    return u.astype(np.float32), v.astype(np.float32), inb


def _detect_mask(down, yolo):
    """Person mask for one down-view, unioned across MASK_ROTATIONS_DEG (see the
    comment on that constant for why: a single pass can under-segment even a
    confidently-detected person). Each rotated detection is rotated back before
    the union so all masks line up in the original down-view's pixel grid."""
    h, w = down.shape[:2]
    center = (w / 2, h / 2)
    m = np.zeros((h, w), np.uint8)
    for angle in MASK_ROTATIONS_DEG:
        if angle == 0:
            probe = down
        else:
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            probe = cv2.warpAffine(down, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        res = yolo.predict(probe, classes=[0], conf=MASK_CONF, retina_masks=True, verbose=False)[0]
        if res.masks is None:
            continue
        for mm in res.masks.data.cpu().numpy():
            pm = cv2.resize((mm * 255).astype(np.uint8), (w, h))
            if angle != 0:
                Minv = cv2.getRotationMatrix2D(center, -angle, 1.0)
                pm = cv2.warpAffine(pm, Minv, (w, h))
            m = np.maximum(m, pm)
    return m


def clean(img, yolo, lama):
    """Return (cleaned_image, 1 if an operator was removed else 0)."""
    H, W = img.shape[:2]
    mu, mv, cam = build_maps(FOV, DOWN_P, OUTSZ, OUTSZ, H, W)
    down = cv2.remap(img, mu, mv, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    m = _detect_mask(down, yolo)
    if m.max() == 0:
        return img, 0
    m = cv2.dilate(m, np.ones((15, 15), np.uint8), iterations=2)
    dc = cv2.cvtColor(
        np.array(lama(Image.fromarray(cv2.cvtColor(down, cv2.COLOR_BGR2RGB)),
                      Image.fromarray(m))),
        cv2.COLOR_RGB2BGR)
    if dc.shape[:2] != (OUTSZ, OUTSZ):
        dc = cv2.resize(dc, (OUTSZ, OUTSZ))
    row0 = int((0.5 - BAND_LAT / 180.0) * H)
    iu, iv, inb = inv_maps(H, W, row0, *cam)
    patch = cv2.remap(dc, iu, iv, cv2.INTER_LINEAR).astype(np.float32)
    out = img.copy(); band = out[row0:].astype(np.float32); rows = H - row0
    fr = max(1, int(FEATHER_DEG / 180.0 * H))
    alpha = np.clip(np.arange(rows) / fr, 0, 1).astype(np.float32)[:, None]
    a = (inb.astype(np.float32) * alpha)[:, :, None]
    out[row0:] = (patch * a + band * (1 - a)).astype(np.uint8)
    return out, 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True, help="dir of equirect panorama .jpg frames")
    ap.add_argument("--out", required=True, help="dir to write cleaned frames into")
    ap.add_argument("--yolo", default=os.environ.get("PANO_CLEAN_YOLO", "yolo11x-seg.pt"),
                    help="path to the YOLO person-seg weights")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    srcs = sorted(glob.glob(os.path.join(a.frames, "*.jpg"))
                  + glob.glob(os.path.join(a.frames, "*.png")))
    if not srcs:
        print(f"[pano_clean] no frames in {a.frames} — nothing to do", flush=True)
        return
    print(f"[pano_clean] frames={len(srcs)} out={a.out}", flush=True)
    print("[pano_clean] loading models (YOLO11x-seg + LaMa)...", flush=True)
    yolo = YOLO(a.yolo); lama = SimpleLama()

    removed = skipped = 0
    for i, f in enumerate(srcs):
        name = os.path.basename(f)
        dst = os.path.join(a.out, name)
        if os.path.exists(dst):
            skipped += 1
            continue
        img = cv2.imread(f)
        if img is None:
            print(f"[pano_clean] WARN unreadable {name}", flush=True)
            continue
        out, r = clean(img, yolo, lama); removed += r
        cv2.imwrite(dst, out, [cv2.IMWRITE_JPEG_QUALITY, 92])
        if i % 20 == 0:
            print(f"  {i}/{len(srcs)} (operator removed in {removed})", flush=True)
    print(f"[pano_clean] DONE frames={len(srcs)} removed={removed} skipped={skipped}",
          flush=True)


if __name__ == "__main__":
    main()
