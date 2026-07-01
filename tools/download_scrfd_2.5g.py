"""
download_scrfd_2.5g.py — fetch the SCRFD-2.5GF detector (InsightFace det_2.5g).

Why
---
Phase 0 confirmed the production detector is SCRFD-10GF (det_10g). SCRFD-2.5GF
(det_2.5g) is a near drop-in: IDENTICAL decode (same strides 8/16/32, 2 anchors,
LTRB), ~4x fewer FLOPs. This script downloads it so tools/benchmark_detectors.py
can compare them on YOUR frames.

It does NOT change any production config. It only writes the model file.

Output: models/scrfd_2.5g.onnx

If the automatic download fails (network/policy), the script prints manual steps.
No new pip dependency is required (uses urllib + zipfile from the stdlib).
"""
import io
import sys
import zipfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
TARGET = MODELS / "scrfd_2.5g.onnx"

# det_2.5g.onnx ships in the buffalo_M pack (NOT buffalo_s — that pack contains
# det_500m.onnx). The GitHub release zip is the reliable source; the insightface.ai
# store is a fallback (its DNS is blocked on some hosts). Verified 2026-07-01.
ZIP_URLS = [
    "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_m.zip",
    "http://storage.insightface.ai/files/models/buffalo_m.zip",
]
# Some mirrors host the bare onnx directly.
DIRECT_URLS = [
    "https://huggingface.co/immich-app/buffalo_m/resolve/main/det_2.5g.onnx",
]

MEMBER_NAMES = ("det_2.5g.onnx", "buffalo_s/det_2.5g.onnx")


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def try_direct():
    for url in DIRECT_URLS:
        try:
            print(f"  trying direct: {url}")
            data = _get(url)
            if data[:4] == b"\x08\x01\x12\x00" or len(data) > 1_000_000:
                TARGET.write_bytes(data)
                return True
        except Exception as e:
            print(f"    failed: {e}")
    return False


def try_zip():
    for url in ZIP_URLS:
        try:
            print(f"  trying zip: {url}")
            data = _get(url, timeout=120)
            zf = zipfile.ZipFile(io.BytesIO(data))
            names = zf.namelist()
            member = next((n for n in names
                           if n.split("/")[-1] == "det_2.5g.onnx"), None)
            if member is None:
                print(f"    det_2.5g.onnx not in zip (has: {names})")
                continue
            TARGET.write_bytes(zf.read(member))
            return True
        except Exception as e:
            print(f"    failed: {e}")
    return False


def main():
    MODELS.mkdir(exist_ok=True)
    if TARGET.exists() and TARGET.stat().st_size > 1_000_000:
        print(f"[OK] already present: {TARGET} ({TARGET.stat().st_size/1e6:.1f} MB)")
        return 0

    print("Downloading SCRFD-2.5GF (det_2.5g)...")
    ok = try_direct() or try_zip()

    if ok and TARGET.exists():
        print(f"\n[OK] saved {TARGET} ({TARGET.stat().st_size/1e6:.1f} MB)")
        print("Next: python tools/benchmark_detectors.py --frames <dir-of-frames>")
        return 0

    print("\n[FAILED] Could not download automatically.")
    print("Manual steps:")
    print("  1. Get the InsightFace 'buffalo_m' pack (contains det_2.5g.onnx), e.g.:")
    print("       pip install insightface  # then it auto-downloads on first use, OR")
    print("       download buffalo_m.zip from the InsightFace v0.7 GitHub release")
    print(f"  2. Place det_2.5g.onnx at: {TARGET}")
    print("  3. Re-run tools/benchmark_detectors.py")
    return 1


if __name__ == "__main__":
    sys.exit(main())
