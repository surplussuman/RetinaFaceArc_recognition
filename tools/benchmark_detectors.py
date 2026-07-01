"""
benchmark_detectors.py — Phase 2: measured detector comparison on YOUR frames.

Compares the production detector against lighter candidates on real frames and
reports, per model:
  - median detect ms (server speed)
  - average faces / frame
  - a RECALL PROXY bucketed by face size (small/medium/large), using the heaviest
    available model as ground truth — CCTV recall loss concentrates on small faces.

It does NOT change the production default. Decide on data, not vibes.

Models auto-included if their file exists in models/:
  - det_10g  (current SCRFD-10GF)         static 640  AND dynamic
  - scrfd_2.5g (SCRFD-2.5GF, drop-in)     static 640  AND dynamic   [run tools/download_scrfd_2.5g.py]
RetinaFace-MobileNet0.25 uses a DIFFERENT decode head, so it is only listed if a
file is present and a matching config is supplied (skipped with a note otherwise).

Usage
-----
  python tools/benchmark_detectors.py --frames path/to/frames_dir
  python tools/benchmark_detectors.py --video "uploads/test.mp4" --num-frames 40

Tip: fix the thread count for a fair comparison, e.g. ONNX_INTRA_OP=8.
"""
import os
import sys
import time
import copy
import tempfile
import argparse
import statistics
from pathlib import Path

import numpy as np
import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))

from core.detector import RetinaFaceDetector

BASE_CFG = ROOT / "config" / "detector_config.yaml"
MODELS = ROOT / "models"

SMALL_MAX = 40    # face bbox height (px) < 40 = small/distant
MEDIUM_MAX = 100  # 40..100 = medium; >100 = large


def build_detector(model_path: Path, dynamic: bool, tmpdir: Path) -> RetinaFaceDetector:
    """Instantiate RetinaFaceDetector pointing at model_path with dynamic on/off."""
    with open(BASE_CFG) as f:
        cfg = yaml.safe_load(f)
    cfg = copy.deepcopy(cfg)
    cfg["model"]["path"] = str(model_path)
    cfg["model"]["name"] = model_path.stem
    cfg["debug"]["log_inference_time"] = False
    cfg.setdefault("preprocessing", {}).setdefault("dynamic_input", {})
    cfg["preprocessing"]["dynamic_input"]["enabled"] = dynamic
    tmp = tmpdir / f"cfg_{model_path.stem}_{'dyn' if dynamic else 'stat'}.yaml"
    with open(tmp, "w") as f:
        yaml.safe_dump(cfg, f)
    # Force FP32 here regardless of global quant flag (compare architectures cleanly).
    prev = os.environ.get("USE_QUANTIZED_MODELS")
    os.environ["USE_QUANTIZED_MODELS"] = "0"
    try:
        det = RetinaFaceDetector(str(tmp))
    finally:
        if prev is None:
            os.environ.pop("USE_QUANTIZED_MODELS", None)
        else:
            os.environ["USE_QUANTIZED_MODELS"] = prev
    return det


def load_frames(args):
    frames = []
    if args.frames:
        d = Path(args.frames)
        files = sorted([p for p in d.iterdir()
                        if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")])
        for p in files[: args.num_frames]:
            img = cv2.imread(str(p))
            if img is not None:
                frames.append(img)
    elif args.video:
        cap = cv2.VideoCapture(args.video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
        if total > 0:
            idxs = np.linspace(0, max(total - 1, 0), args.num_frames, dtype=int)
            for i in idxs:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
                ok, fr = cap.read()
                if ok:
                    frames.append(fr)
        else:  # stream/unknown length: just grab sequentially
            for _ in range(args.num_frames):
                ok, fr = cap.read()
                if not ok:
                    break
                frames.append(fr)
        cap.release()
    return frames


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def bucket(h):
    if h < SMALL_MAX:
        return "small"
    if h < MEDIUM_MAX:
        return "medium"
    return "large"


def run_model(det, frames):
    """Return (median_ms, list_of_detections_per_frame)."""
    if frames:
        det.detect(frames[0])  # warmup
    times, dets = [], []
    for fr in frames:
        t = time.perf_counter()
        d = det.detect(fr)
        times.append((time.perf_counter() - t) * 1000)
        dets.append(d)
    med = statistics.median(times) if times else 0.0
    return med, dets


def recall_vs_gt(gt_dets, cand_dets):
    """Per-bucket recall: fraction of GT faces with an IoU>=0.5 match in candidate."""
    counts = {"small": 0, "medium": 0, "large": 0}
    matched = {"small": 0, "medium": 0, "large": 0}
    for gt_frame, cand_frame in zip(gt_dets, cand_dets):
        cand_boxes = [d["bbox"] for d in cand_frame]
        for g in gt_frame:
            gb = g["bbox"]
            b = bucket(gb[3] - gb[1])
            counts[b] += 1
            if any(iou(gb, cb) >= 0.5 for cb in cand_boxes):
                matched[b] += 1
    rec = {}
    for b in counts:
        rec[b] = (matched[b] / counts[b]) if counts[b] else None
    return rec, counts


def main():
    ap = argparse.ArgumentParser(description="Phase 2 detector comparison.")
    ap.add_argument("--frames", type=str, default=None, help="Directory of frame images.")
    ap.add_argument("--video", type=str, default=None, help="Video file to sample frames from.")
    ap.add_argument("--num-frames", type=int, default=30)
    args = ap.parse_args()

    if not args.frames and not args.video:
        ap.error("provide --frames <dir> or --video <path>")

    frames = load_frames(args)
    if not frames:
        print("No frames loaded. Check --frames / --video.")
        return 1
    print(f"Loaded {len(frames)} frames. Frame size example: {frames[0].shape}")
    print(f"ONNX threads: set via ONNX_INTRA_OP (current={os.environ.get('ONNX_INTRA_OP','auto')})\n")

    # Assemble candidate list from what's on disk.
    det10g = MODELS / "retinaface_resnet50.onnx"   # actually SCRFD-10GF (Phase 0)
    scrfd25 = MODELS / "scrfd_2.5g.onnx"
    scrfd05 = MODELS / "scrfd_500m.onnx"
    candidates = []  # (label, model_path, dynamic, gflops_rank)
    if det10g.exists():
        candidates.append(("det_10g  static640", det10g, False, 10))
        candidates.append(("det_10g  dynamic ", det10g, True, 10))
    if scrfd25.exists():
        candidates.append(("scrfd2.5 static640", scrfd25, False, 2.5))
        candidates.append(("scrfd2.5 dynamic ", scrfd25, True, 2.5))
    else:
        print("[note] models/scrfd_2.5g.onnx not found — run tools/download_scrfd_2.5g.py")
        print("       to include the SCRFD-2.5GF comparison.\n")
    if scrfd05.exists():
        candidates.append(("scrfd500 static640", scrfd05, False, 0.5))
        candidates.append(("scrfd500 dynamic ", scrfd05, True, 0.5))

    if not candidates:
        print("No detector models found in models/.")
        return 1

    tmpdir = Path(tempfile.mkdtemp(prefix="benchdet_"))
    results = []
    gt_dets = None
    gt_label = None
    # Ground truth = heaviest model, static (most faithful). It is the first entry.
    for label, mp, dyn, rank in candidates:
        print(f"running: {label}  ({mp.name})")
        det = build_detector(mp, dyn, tmpdir)
        med, dets = run_model(det, frames)
        avg_faces = np.mean([len(d) for d in dets]) if dets else 0.0
        results.append({"label": label, "ms": med, "avg_faces": avg_faces,
                        "dets": dets, "rank": rank, "dynamic": dyn})
        # pick GT: heaviest rank, static
        if gt_dets is None and not dyn:
            gt_dets = dets
            gt_label = label
        elif not dyn and rank > next(r["rank"] for r in results if r["label"] == gt_label):
            gt_dets = dets
            gt_label = label

    # Recompute GT as the highest-rank static model.
    static_results = [r for r in results if not r["dynamic"]]
    gt = max(static_results, key=lambda r: r["rank"])
    gt_dets = gt["dets"]
    gt_label = gt["label"]

    # ---------------------------------------------------------------- table
    print("\n" + "=" * 92)
    print(f"DETECTOR COMPARISON   (ground truth for recall = {gt_label.strip()})")
    print("=" * 92)
    print(f"{'model':20s} {'detect ms':>10s} {'faces/frm':>10s} "
          f"{'recall S':>9s} {'recall M':>9s} {'recall L':>9s}  {'vs GT speed':>11s}")
    print("-" * 92)
    gt_ms = gt["ms"]
    for r in results:
        rec, counts = recall_vs_gt(gt_dets, r["dets"])
        def fmt(b):
            v = rec[b]
            return "  n/a " if v is None else f"{v*100:6.1f}%"
        speed = f"{gt_ms / r['ms']:.2f}x" if r["ms"] > 0 else "n/a"
        print(f"{r['label']:20s} {r['ms']:10.1f} {r['avg_faces']:10.2f} "
              f"{fmt('small')} {fmt('medium')} {fmt('large')}  {speed:>11s}")
    print("-" * 92)
    # face-size population (from GT)
    _, gtcounts = recall_vs_gt(gt_dets, gt_dets)
    print(f"GT face population:  small={gtcounts['small']}  medium={gtcounts['medium']}  "
          f"large={gtcounts['large']}  (buckets by bbox height px: S<{SMALL_MAX}, "
          f"M<{MEDIUM_MAX}, L>= {MEDIUM_MAX})")
    print("=" * 92)

    print("\nHow to read this:")
    print("  - 'det_10g dynamic' vs 'det_10g static640' = the FREE Phase 1c win (same model,")
    print("    fewer padding FLOPs). Recall should be ~identical; speed should rise.")
    print("  - 'scrfd2.5 *' = the Phase 2 lighter model. Look at recall on SMALL faces — that")
    print("    is where a lighter detector loses CCTV recall. If small-face recall is")
    print("    acceptable for your cameras, scrfd2.5 is the win on this slow CPU.")
    print("  - Production default is unchanged. Pick based on the recall/speed trade above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
