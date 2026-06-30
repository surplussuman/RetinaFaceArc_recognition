"""
diagnose_env.py — Phase 0 environment diagnostic for the face-recognition pipeline.

Purpose
-------
Decide *why* the server is ~6x slower than the laptop BEFORE changing any model.
Splits the slowdown into recoverable vs genuine-silicon causes:

  (a) masked CPU instructions (no AVX2/FMA in the guest)  -> recoverable (provider issue)
  (b) un-tuned numpy BLAS / wrong core target             -> recoverable (library)
  (c) CPU steal / sustained-CPU cap under load            -> partly recoverable (instance)
  (d) co-located contention (web stack on same box)       -> recoverable (affinity / move)
  (e) genuine per-core silicon (low clock, no turbo)      -> NOT recoverable -> model swap

THE decisive measurement is the SINGLE-THREAD benchmark. Run this script on BOTH the
laptop and the server, then compare:
    server_single_thread / laptop_single_thread
  ~2.0-2.5x  -> mostly recoverable (throttle + library + contention)
  ~6-8x      -> genuine silicon -> a lighter model (Phase 2) is required

Usage
-----
  # On the laptop first:
  python tools/diagnose_env.py

  # Then on the server, feed the laptop numbers to get an automatic ratio + verdict:
  python tools/diagnose_env.py --baseline-matmul 6.1 --baseline-detect 95
  #                                              ^laptop ST matmul   ^laptop ST detect

  # Optional: use a real CCTV frame instead of a synthetic one
  python tools/diagnose_env.py --frame path/to/frame.jpg

This script changes NOTHING in the pipeline. It only reads and measures.
"""

# ---------------------------------------------------------------------------
# IMPORTANT: thread-limiting env vars MUST be set before numpy/BLAS is imported,
# otherwise the single-thread benchmark is not actually single-threaded.
# ---------------------------------------------------------------------------
import os

_THREAD_ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)
# Capture what the user actually had set (for reporting) before we override.
_ORIG_THREAD_ENV = {k: os.environ.get(k) for k in _THREAD_ENV_KEYS}
_ORIG_THREAD_ENV["OPENBLAS_CORETYPE"] = os.environ.get("OPENBLAS_CORETYPE")
_ORIG_THREAD_ENV["ONNX_INTRA_OP"] = os.environ.get("ONNX_INTRA_OP")

for _k in _THREAD_ENV_KEYS:
    os.environ[_k] = "1"
# Force the detector ONNX session to a single intra-op thread for the decisive test.
os.environ["ONNX_INTRA_OP"] = "1"
os.environ["ONNX_INTER_OP"] = "1"

import sys
import time
import platform
import argparse
import statistics
import subprocess
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
# Run from repo root so the relative model paths in the configs resolve.
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

IS_LINUX = sys.platform.startswith("linux")


# ===========================================================================
# small helpers
# ===========================================================================
def hr(title=""):
    line = "=" * 78
    if title:
        print(f"\n{line}\n{title}\n{line}")
    else:
        print(line)


def sub(title):
    print(f"\n--- {title} ---")


def median_ms(samples):
    return statistics.median(samples) * 1000.0


# ===========================================================================
# 1. CPU instruction flags + topology
# ===========================================================================
def report_cpu_flags():
    hr("1. CPU INSTRUCTION FLAGS (as exposed INSIDE the guest)")

    flags = set()
    source = None
    if IS_LINUX and Path("/proc/cpuinfo").exists():
        source = "/proc/cpuinfo"
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("flags"):
                        flags = set(line.split(":", 1)[1].split())
                        break
        except Exception as e:
            print(f"  (could not read /proc/cpuinfo: {e})")
    else:
        # Windows / other: try the optional py-cpuinfo package.
        try:
            import cpuinfo  # type: ignore
            info = cpuinfo.get_cpu_info()
            flags = set(info.get("flags", []))
            source = "py-cpuinfo"
        except Exception:
            source = None

    critical = ["avx", "avx2", "fma", "avx512f", "avx_vnni", "avx512_vnni", "f16c", "sse4_2"]
    if flags:
        print(f"  flag source: {source}")
        for c in critical:
            present = c in flags
            mark = "YES" if present else "no"
            note = ""
            if c == "avx2" and not present:
                note = "  <-- CRITICAL: AVX2 MASKED IN GUEST. This alone explains a huge slowdown."
            if c == "avx2" and present:
                note = "  (good — MLAS/BLAS can use 256-bit kernels)"
            if c in ("avx512_vnni", "avx_vnni") and not present:
                note = "  (expected absent on Zen3 — limits INT8 GEMM speedup to ~1.5-2.5x)"
            print(f"    {c:14s}: {mark}{note}")
    else:
        print("  Could not read CPU flags on this platform.")
        print("  (On Windows, `pip install py-cpuinfo` to enable. The CRITICAL check is")
        print("   on the SERVER anyway, where /proc/cpuinfo always works.)")

    sub("topology (lscpu)")
    if IS_LINUX:
        try:
            out = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=10).stdout
            keep = ("Model name", "Socket", "Core(s) per socket", "Thread(s) per core",
                    "CPU(s)", "NUMA node", "CPU max MHz", "CPU min MHz", "BogoMIPS",
                    "Hypervisor", "Virtualization")
            sockets = cores_per = None
            for ln in out.splitlines():
                if any(ln.strip().startswith(k) for k in keep):
                    print(f"    {ln.strip()}")
                if ln.strip().startswith("Socket(s)"):
                    sockets = ln.split(":")[1].strip()
                if ln.strip().startswith("Core(s) per socket"):
                    cores_per = ln.split(":")[1].strip()
            try:
                if sockets and int(sockets) >= 4:
                    print(f"    ** FRAGMENTED TOPOLOGY: {sockets} sockets x {cores_per} cores — "
                          f"hurts thread/cache placement on a single physical NUMA node. **")
            except ValueError:
                pass
        except Exception as e:
            print(f"  (lscpu unavailable: {e})")
    else:
        print(f"    platform: {platform.platform()}")
        print(f"    processor: {platform.processor()}")
        print(f"    os.cpu_count(): {os.cpu_count()}")


# ===========================================================================
# 2. numpy BLAS backend
# ===========================================================================
def report_blas():
    hr("2. numpy BLAS BACKEND")
    print(f"  numpy version: {np.__version__}")

    blas_name = None
    try:
        cfg = np.show_config(mode="dicts")  # numpy >= 1.25
        deps = cfg.get("Build Dependencies", {})
        blas = deps.get("blas", {})
        lapack = deps.get("lapack", {})
        blas_name = (blas.get("name") or "").lower()
        print(f"  BLAS name   : {blas.get('name')}  (version {blas.get('version')})")
        print(f"  LAPACK name : {lapack.get('name')}  (version {lapack.get('version')})")
    except Exception:
        sub("np.show_config() raw")
        try:
            np.show_config()
        except Exception as e:
            print(f"  (show_config failed: {e})")

    sub("threading env (as the user set it, before this script forced 1)")
    for k, v in _ORIG_THREAD_ENV.items():
        print(f"    {k:22s} = {v}")

    sub("verdict")
    if blas_name:
        if "openblas" in blas_name:
            print("  OpenBLAS detected. Generally well-tuned IF it picked the right core target.")
            print("  Check OPENBLAS_CORETYPE above: if empty, OpenBLAS auto-detects — usually fine")
            print("  on Zen3. If it shows a wrong/old target (e.g. PRESCOTT/generic), that is a")
            print("  recoverable library problem (Phase 1a).")
        elif "mkl" in blas_name:
            print("  Intel MKL detected — note MKL can run slow paths on AMD. On EPYC, OpenBLAS is")
            print("  often faster. Flag for Phase 1a.")
        elif blas_name in ("blas", "reference", "netlib", "generic"):
            print("  ** REFERENCE/GENERIC BLAS — UN-TUNED. This is recoverable and can be a big")
            print("     chunk of the slowdown. Phase 1a: install numpy with OpenBLAS. **")
        else:
            print(f"  BLAS = {blas_name}. Compare the single-thread matmul ms vs laptop to judge.")
    else:
        print("  Could not introspect BLAS name; rely on the single-thread matmul number below.")


# ===========================================================================
# 3. onnxruntime
# ===========================================================================
def report_ort(cpu_flags_have_avx2):
    hr("3. onnxruntime BUILD")
    try:
        import onnxruntime as ort
        print(f"  onnxruntime version : {ort.__version__}")
        print(f"  get_device()        : {ort.get_device()}")
        print(f"  available providers : {ort.get_available_providers()}")
        sub("MLAS kernel dispatch")
        print("  ORT's CPU math (MLAS) auto-selects kernels at runtime from CPU flags.")
        if cpu_flags_have_avx2 is True:
            print("  AVX2 present -> MLAS uses 256-bit AVX2 GEMM kernels (good).")
        elif cpu_flags_have_avx2 is False:
            print("  ** AVX2 ABSENT -> MLAS falls back to scalar/SSE kernels -> catastrophic. **")
        else:
            print("  AVX2 presence unknown on this platform (see section 1 on the server).")
    except Exception as e:
        print(f"  (onnxruntime import failed: {e})")


# ===========================================================================
# 4. SINGLE-THREAD pure-compute benchmark  (THE decisive test)
# ===========================================================================
def pin_to_one_core():
    """Pin this process to a single core; return the original affinity set (or None)."""
    if hasattr(os, "sched_setaffinity"):
        try:
            orig = os.sched_getaffinity(0)
            os.sched_setaffinity(0, {sorted(orig)[0]})
            return orig
        except Exception as e:
            print(f"  (could not set affinity: {e})")
    return None


def restore_affinity(orig):
    if orig and hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, orig)
        except Exception:
            pass


def bench_single_thread(detector, frame, iters):
    hr("4. SINGLE-THREAD BENCHMARK (decisive — compare across machines)")

    orig_aff = pin_to_one_core()
    if orig_aff is not None:
        print(f"  pinned to one core (was {len(orig_aff)} cores); BLAS forced to 1 thread.")
    else:
        print("  affinity pin unavailable on this platform; BLAS still forced to 1 thread.")

    # --- 4a. pure matmul ---
    N = 2048
    rng = np.random.RandomState(0)
    a = rng.rand(N, N).astype(np.float32)
    b = rng.rand(N, N).astype(np.float32)
    _ = a @ b  # warmup
    samples = []
    for _ in range(iters):
        t = time.perf_counter()
        c = a @ b
        samples.append(time.perf_counter() - t)
    matmul_ms = median_ms(samples)
    print(f"\n  SINGLE-THREAD matmul ms : {matmul_ms:8.1f}   ({N}x{N} f32, {iters} iters, median)")

    # --- 4b. single detector ONNX inference ---
    detect_ms = None
    if detector is not None:
        try:
            tensor, _scale, _pad = detector.preprocess(frame)
            inp = detector.input_name
            _ = detector.session.run(None, {inp: tensor})  # warmup
            samples = []
            for _ in range(iters):
                t = time.perf_counter()
                detector.session.run(None, {inp: tensor})
                samples.append(time.perf_counter() - t)
            detect_ms = median_ms(samples)
            print(f"  SINGLE-THREAD detect ms : {detect_ms:8.1f}   (ONNX inference only, "
                  f"intra_op=1, {iters} iters, median)")
        except Exception as e:
            print(f"  (detector benchmark failed: {e})")

    restore_affinity(orig_aff)
    return matmul_ms, detect_ms


# ===========================================================================
# 5. Steal / contention snapshot
# ===========================================================================
def _read_proc_stat_cpu():
    """Return (total, steal) jiffies for the aggregate 'cpu' line."""
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu "):
                parts = [int(x) for x in line.split()[1:]]
                # user nice system idle iowait irq softirq steal guest guest_nice
                total = sum(parts)
                steal = parts[7] if len(parts) > 7 else 0
                return total, steal
    return None, None


def steal_during(seconds):
    """Run an all-core matmul burst and measure steal% during it."""
    t0, s0 = _read_proc_stat_cpu()
    if t0 is None:
        return None
    # Restore full affinity so the burst can use every core.
    if hasattr(os, "sched_setaffinity"):
        try:
            os.sched_setaffinity(0, set(range(os.cpu_count())))
        except Exception:
            pass

    import threading
    stop = threading.Event()

    def worker():
        a = np.random.rand(512, 512).astype(np.float32)
        b = np.random.rand(512, 512).astype(np.float32)
        while not stop.is_set():
            # numpy @ releases the GIL during BLAS, so N threads load N cores
            # even with BLAS pinned to 1 thread each.
            _ = a @ b

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(os.cpu_count())]
    for t in threads:
        t.start()
    time.sleep(seconds)
    stop.set()
    for t in threads:
        t.join(timeout=1.0)

    t1, s1 = _read_proc_stat_cpu()
    dt, ds = (t1 - t0), (s1 - s0)
    return (ds / dt * 100.0) if dt > 0 else 0.0


def report_steal(burst_seconds):
    hr("5. STEAL / CONTENTION SNAPSHOT")
    if not IS_LINUX:
        print("  /proc/stat steal counters are Linux-only; skipped on this platform.")
        return None, None

    # idle steal: sample for 1s while doing nothing heavy
    t0, s0 = _read_proc_stat_cpu()
    time.sleep(1.0)
    t1, s1 = _read_proc_stat_cpu()
    idle_steal = ((s1 - s0) / (t1 - t0) * 100.0) if (t1 - t0) > 0 else 0.0
    print(f"  steal % at IDLE          : {idle_steal:6.1f}%")

    load_steal = steal_during(burst_seconds)
    print(f"  steal % under ALL-CORE LOAD ({burst_seconds:.0f}s burst): {load_steal:6.1f}%")
    if load_steal is not None:
        if load_steal >= 25:
            print("  ** High steal ONLY under load = provider is capping sustained CPU. **")
            print("     You can touch all cores briefly but not sustain them.")
        elif load_steal >= 10:
            print("  Moderate steal under load — some co-tenant contention / soft cap.")
        else:
            print("  Steal stays low even under load — not the main bottleneck.")

    sub("top CPU-consuming processes (confirm contention / stale services)")
    try:
        out = subprocess.run(
            ["ps", "-eo", "pid,comm,pcpu,etime", "--sort=-pcpu"],
            capture_output=True, text=True, timeout=10).stdout
        lines = out.splitlines()
        for ln in lines[:13]:
            print(f"    {ln}")
        flags = [ln for ln in lines if "runserver" in ln or "manage.py" in ln]
        if flags:
            print("    ** Found a Django dev server (manage.py runserver) — kill it; it should")
            print("       not run in production and it burns CPU. **")
    except Exception as e:
        print(f"  (ps unavailable: {e})")

    return idle_steal, load_steal


# ===========================================================================
# 6. Confirm which detector model is actually loaded
# ===========================================================================
def identify_model():
    hr("6. ACTUAL DETECTOR MODEL")
    import yaml
    cfg_path = REPO_ROOT / "config" / "detector_config.yaml"
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)
    model_path = REPO_ROOT / cfg["model"]["path"]
    config_name = cfg["model"]["name"]
    print(f"  config name  : {config_name}")
    print(f"  model file   : {model_path}")
    if not model_path.exists():
        print("  ** model file missing! **")
        return None

    size_mb = model_path.stat().st_size / 1e6
    print(f"  file size    : {size_mb:.1f} MB")

    try:
        import onnxruntime as ort
        so = ort.SessionOptions()
        sess = ort.InferenceSession(str(model_path), sess_options=so,
                                    providers=["CPUExecutionProvider"])
        ins = sess.get_inputs()
        outs = sess.get_outputs()
        sub("inputs")
        for i in ins:
            print(f"    {i.name:12s} shape={i.shape} dtype={i.type}")
        sub("outputs")
        for o in outs:
            print(f"    {o.name:12s} shape={o.shape} dtype={o.type}")
        n_out = len(outs)

        sub("family identification")
        # SCRFD det_10g signature: ~16-17MB, 9 outputs (3 score/3 bbox/3 kps),
        # 640x640 input, LTRB anchor decode (16800 anchors).
        verdict = "UNKNOWN"
        if n_out == 9 and size_mb < 30:
            verdict = ("InsightFace SCRFD-10GF (det_10g). The config NAME 'retinaface_resnet50' "
                       "is a MISNOMER — the file size, 9 FPN outputs and LTRB/anchor decode in "
                       "core/detector.py are pure SCRFD.")
            print(f"  => {verdict}")
            print("     Nominal cost: ~10 GFLOPs @ 640x640.")
            print("     ** Phase 2 drop-in: SCRFD-2.5GF (buffalo_s det_2.5g) — IDENTICAL decode,")
            print("        ~4x fewer FLOPs. This is the preferred lighter path. **")
        elif size_mb > 90:
            verdict = "Classic RetinaFace-ResNet50 (heavy backbone)."
            print(f"  => {verdict}")
            print("     Phase 2 fallback: RetinaFace-MobileNet0.25 (same RetinaFace decode).")
        else:
            print(f"  => {verdict} (outputs={n_out}, size={size_mb:.1f}MB). Inspect manually.")
        return verdict
    except Exception as e:
        print(f"  (could not load model for inspection: {e})")
        return None


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Phase 0 environment diagnostic.")
    ap.add_argument("--frame", type=str, default=None,
                    help="Path to a real frame (jpg/png) for the detect benchmark. "
                         "Defaults to a synthetic 720p frame.")
    ap.add_argument("--iters", type=int, default=20, help="Benchmark iterations (median).")
    ap.add_argument("--burst-seconds", type=float, default=4.0,
                    help="All-core burst duration for the steal-under-load measurement.")
    ap.add_argument("--baseline-matmul", type=float, default=None,
                    help="Laptop SINGLE-THREAD matmul ms, to auto-compute the ratio + verdict.")
    ap.add_argument("--baseline-detect", type=float, default=None,
                    help="Laptop SINGLE-THREAD detect ms, to auto-compute the ratio + verdict.")
    args = ap.parse_args()

    hr("PHASE 0 — ENVIRONMENT DIAGNOSTIC")
    print(f"  host      : {platform.node()}")
    print(f"  platform  : {platform.platform()}")
    print(f"  python    : {sys.version.split()[0]}")
    print(f"  cpu_count : {os.cpu_count()}")
    print(f"  run from  : {REPO_ROOT}")

    # Section 1
    flags = set()
    if IS_LINUX and Path("/proc/cpuinfo").exists():
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("flags"):
                        flags = set(line.split(":", 1)[1].split())
                        break
        except Exception:
            pass
    report_cpu_flags()
    have_avx2 = ("avx2" in flags) if flags else None

    # Sections 2, 3
    report_blas()
    report_ort(have_avx2)

    # Section 6 first (we need the detector for section 4)
    identify_model()

    # Build the detector once (single-thread via env set at top), for section 4.
    detector = None
    try:
        from core.detector import RetinaFaceDetector
        print("\n(loading detector for the single-thread inference benchmark...)")
        detector = RetinaFaceDetector()
    except Exception as e:
        print(f"  (could not construct detector: {e})")

    # Frame for the detect benchmark
    if args.frame:
        import cv2
        frame = cv2.imread(args.frame)
        if frame is None:
            print(f"  (could not read --frame {args.frame}; using synthetic)")
            frame = np.random.RandomState(1).randint(0, 255, (720, 1280, 3), dtype=np.uint8)
    else:
        frame = np.random.RandomState(1).randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    # Section 4 (decisive)
    matmul_ms, detect_ms = bench_single_thread(detector, frame, args.iters)

    # Section 5
    idle_steal, load_steal = report_steal(args.burst_seconds)

    # ----------------------------------------------------------------- verdict
    hr("VERDICT")
    print(f"  SINGLE-THREAD matmul ms : {matmul_ms:.1f}")
    if detect_ms is not None:
        print(f"  SINGLE-THREAD detect ms : {detect_ms:.1f}")

    if args.baseline_matmul:
        r = matmul_ms / args.baseline_matmul
        print(f"\n  matmul ratio vs laptop  : {r:.1f}x   (laptop {args.baseline_matmul} ms)")
        if args.baseline_detect and detect_ms:
            rd = detect_ms / args.baseline_detect
            print(f"  detect ratio vs laptop  : {rd:.1f}x   (laptop {args.baseline_detect} ms)")
        print()
        if r <= 2.5:
            print("  => Single-thread gap is SMALL (~2x). The big multi-thread gap you saw is")
            print("     mostly RECOVERABLE: throttle/steal + thread contention + maybe BLAS.")
            print("     Phase 1 (affinity, BLAS, quant, dynamic input) should close most of it.")
            print("     A model swap is likely NOT required yet.")
        elif r <= 4.5:
            print("  => Single-thread gap is MODERATE. Mix of silicon + recoverable factors.")
            print("     Do Phase 1 first; reassess before any model swap.")
        else:
            print("  => Single-thread gap is LARGE (>4.5x). This is GENUINE SILICON (low clock,")
            print("     no turbo). Phase 1 helps but cannot close it. Phase 2 (lighter model,")
            print("     e.g. SCRFD-2.5GF) WILL be required to hit real-time on this box.")
    else:
        print("\n  Run this on the LAPTOP too, then re-run here with:")
        print("    --baseline-matmul <laptop matmul ms> --baseline-detect <laptop detect ms>")
        print("  to get the automatic ratio + recoverable-vs-silicon verdict.")

    print("\n  Cause breakdown to fill in after comparing both machines:")
    print(f"    (a) masked instructions : {'NO (avx2 present)' if have_avx2 else ('YES — avx2 missing!' if have_avx2 is False else 'check section 1 on server')}")
    print(f"    (b) un-tuned BLAS       : see section 2 verdict")
    print(f"    (c) throttle / steal    : idle={idle_steal if idle_steal is not None else '?'}  load={load_steal if load_steal is not None else '?'}")
    print(f"    (d) co-located contention: see section 5 process list")
    print(f"    (e) genuine silicon     : the single-thread ratio above")
    hr()


if __name__ == "__main__":
    main()
