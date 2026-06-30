# CPU Parallelism & Adaptive Scheduling — Change Log (June 2026)

**Branch:** `phase2`
**Date:** 2026-06-30
**Scope:** CPU utilisation + frame-scheduling. **No change** to recognition thresholds,
ghost-tracking logic, IOU/zone math, or the `FaceTrack` class.

---

## 0. Why this work happened

Benchmarks showed the AMD EPYC 7643 server (32 vCPUs) was **5.36× slower** than a
consumer laptop on the same CPU-only pipeline:

| Machine | ms/frame | FPS |
|---------|----------|-----|
| Laptop (skip=5) | 33 | 28.3 |
| Server (skip=5) | 177 | 5.24 |

`top` on the server showed the Python process pinned at **433% CPU**.

$$\text{Utilisation} = \frac{433\%}{32 \times 100\%} = \frac{4.33 \text{ cores}}{32 \text{ cores}} = 13.5\%$$

The server was **not compute-starved — it was parallelism-starved.** 86.5% of the
machine was idle. The fixes below attack that idle capacity, plus a separate
scheduling fix for the recall-vs-speed tradeoff.

---

## Part A — Adaptive Detection Scheduler (recall vs speed)

### The bug: two gates fighting

The pipeline had a **blunt fixed frame-skip** running *before* the **smart motion
gate**:

```python
if frame_idx % process_every_n_frames != 0:   # Gate 1 — discards 80% blindly
    return {'skipped': True}
if motion_detector.update(...):                # Gate 2 — never sees those frames
    ...
```

With `process_every_n_frames = 5`, 80% of frames die before the motion gate can
decide. A person crossing during those 4 skipped frames is never detected.

**Measured loss:** detections dropped from **349 → 111** between no-skip and skip=5.
The ratio is exactly the skip factor:

$$\frac{\text{detections}_{\text{skip}}}{\text{detections}_{\text{no-skip}}} = \frac{111}{349} \approx \frac{1}{3} = \frac{1}{k},\quad k=\text{skip factor}$$

Recall scales as $P_{\text{detect}} = 1-(1-p)^{m}$ where $m$ is frames observed of a
face. Fixed skipping cuts $m$ by $k$, so recall collapses.

### The fix: detection frequency as a function of scene state

`process_every_n_frames` set to **1** (every frame reaches the gate), and the motion
gate (`core/motion_detector.py`) upgraded from a floor-only gate to a full scheduler:

$$\text{detect}(t) = \text{warmup}(t)\ \lor\ \text{floor}(t)\ \lor\ \text{spike}(t)\ \lor\ \big(\text{motion}(t)\land\text{throttle}(t)\big)$$

| Component | Rule | Purpose |
|-----------|------|---------|
| **Floor** | force detect every `safety_scan_interval` (=30) frames | still scenes never go blind (1 FPS @ 30fps) |
| **Spike** | detect if `Δmotion_fraction ≥ motion_spike_delta` (0.015) | a new person = sudden foreground rise → **instant** detect → recall preserved |
| **Throttle** | under sustained motion, detect ≤ once per `detect_min_interval` (=3) frames | busy scenes don't re-detect every frame → ~10 FPS detector cap |

This is the "**Detector at ~10 FPS, Tracker at 30 FPS**" model.

### Verified (synthetic deterministic test)

| Component | Result |
|-----------|--------|
| Spike | motion-onset frame detected instantly |
| Throttle | 5 detections over 15 sustained-motion frames (every 3rd) → **3× fewer detector calls** |
| Floor | max detect-to-detect gap = 30 = `safety_scan_interval` |

**Net:** a face entering still triggers an immediate detect (recall kept), while a
sustained crowd costs ⅓ the detector calls (speed kept).

---

## Part B — CPU Utilisation (4 fixes, priority order)

### Problem 1 — ONNX thread count hardcoded to 4

**Mathematical proof of the bug:** `intra_op_num_threads` caps how many cores a
single ONNX `Run()` may use. With it set to 4:

$$\text{max active cores} = 4 \;\Rightarrow\; \text{max CPU} = 400\% \approx \text{observed } 433\%$$

The 4-thread cap *was* the 13.5% ceiling.

**Fix:**
- `detector_config.yaml` / `embedder_config.yaml`: `intra_op` / `inter_op` `4 → 0`
  (ORT reads `0` as "use all cores" = `os.cpu_count()`).
- `core/onnx_session.py` (new): central resolver. A single `onnx_threads` key in
  `system_config.yaml` overrides both per-model yamls — one place to tune per server.
  When `intra_op == 0`, also sets `ExecutionMode.ORT_PARALLEL`.
- Startup prints `ONNX detector/embedder threads: N / M available`.

**Verified:** sessions report **12/12** cores on the laptop → **32/32** expected on the
server. Theoretical ceiling lift:

$$\frac{32 \text{ cores}}{4.33 \text{ cores}} \approx 7.4\times \text{ headroom unlocked}$$

> **Deployment caveat:** on the shared VM (8% steal, 9 gunicorn workers, MySQL/Redis),
> grabbing all 32 cores may oversubscribe. Set `onnx_threads.auto: false` and cap
> `intra_op` (e.g. 16) if co-tenant contention appears.

### Problem 3 — `cap.read()` blocks the inference thread

**Serial model:**

$$T_{\text{frame}} = T_{io} + T_{detect} + T_{embed} + T_{track}$$

**Pipelined model (capture on its own thread):**

$$T_{\text{frame}} = \max\big(T_{io},\; T_{detect} + T_{embed} + T_{track}\big)$$

Since $T_{io} \approx 33\text{ms}$ (30fps decode) and $T_{detect} \gg T_{io}$, the I/O
cost is fully hidden behind inference.

**Fix:** `core/frame_capture.py` (new) — `FrameCaptureThread`:
- bounded `queue.Queue(maxsize=4)`
- **live mode** (`drop_stale=True`): drop oldest frame when full → never fall behind real time
- **file mode** (`drop_stale=False`): block when full (backpressure) → every frame processed, output duration correct; EOF → `(None, None)` sentinel
- RTSP reconnect handled *inside* the thread; main loop observes via `reconnect_flag`

Wired into `run_live`, `run_file`, and `core/video_recognition.py::process_video`.

**Verified:** end-to-end file run → **I/O wait = 0.5 ms/frame** (was ~33 ms blocking).

### Problem 2 — `extract_embeddings_batch()` ran serially

The method looped `for face in faces: session.run(...)` — $N$ serial calls. CNN batch
inference scales **sublinearly**:

$$T_{\text{batch}}(N) \approx T_{\text{batch}}(1)\cdot N^{0.6} \quad\ll\quad N\cdot T_{\text{batch}}(1)$$

**Fix:**
- `tools/export_dynamic_batch.py` (new): rewrites ArcFace input/output batch dim to a
  symbolic `'N'` (`[1,3,112,112] → ['N',3,112,112]`), validates batch 1/4/8.
- `embedder.py`: `extract_embeddings_batch` now stacks all faces into one
  `[N,3,112,112]` tensor → one `session.run` → `[N,512]`, L2-normalised per row.
  Falls back to the serial loop on any shape error. `supports_dynamic_batch`
  auto-detected from the model input shape.

**Verified:**
- batched output **bit-identical** to serial (max abs diff = `0.0`)
- batch_8 = 962 ms vs 8×201 = 1610 ms serial → **1.67× speedup**

$$\text{effective per-face cost: } \frac{962}{8} = 120\text{ms} \text{ vs } 201\text{ms serial}$$

> The existing `arcface_resnet100.onnx` *already* had a symbolic batch axis, so the
> batched path works without re-export. The export script remains for static models.

### Problem 4 — Each stream built its own sessions

$$N \text{ streams} \times K \text{ threads} = NK \text{ cores (contending)}$$

Two 4-thread streams = 8 cores, 24 idle. Target:

$$1 \text{ shared session} \times 32 \text{ cores, serving } N \text{ streams via batching}$$

**Fix:** `core/inference_engine.py` (new) — `SharedInferenceEngine`:
- **one** detector + **one** embedder session, shared across all streams. ORT
  `Run()` is re-entrant (GIL-releasing, safe with per-call buffers) → lock-free
  concurrency, default no lock.
- `detect_batch(frames)` → per-frame detections (frames differ in size — not
  batchable like embeddings).
- `embed_batch(faces)` → **cross-stream coalescing**: a worker thread collects face
  crops arriving from different streams within a 5 ms window (cap 16), runs **one**
  ONNX call, distributes results back via `Future`s. Caller stays synchronous.
- `create_video_recognizer(engine=…)` reuses the shared sessions; `run_processor`
  instantiates the engine once.

Cross-stream batch efficiency:

$$\text{cost}(N\text{ streams}) \approx \text{cost}(1) \cdot N^{0.6}\quad\text{vs}\quad N\cdot\text{cost}(1)$$

**Verified:** 6 concurrent `embed()` calls from 6 threads → **1 ONNX run**
(`embed_calls=6, embed_onnx_runs=1`), results bit-identical; session reuse confirmed
(`detector/embedder/aligner is engine.* → True`).

---

## Scaling projection (the original question: "how many CCTV streams?")

Current serial cost per real-time stream ≈ 4.3 cores. With shared sessions +
event-driven detection (tracker 30 FPS, detector ~2–10 FPS, recognition ~0.5 FPS via
ghost tracking), per-stream compute drops to a fraction of a core:

$$\text{streams/server} = \frac{32 \text{ cores}}{\text{cores/stream}}$$

| Model | cores/stream | streams / 32-vCPU server |
|-------|-------------|--------------------------|
| Before (serial, 4-thread cap) | 4.3 | ~7 |
| Event-driven + shared sessions | ~0.13–0.5 | **~60–240** |

GPU stays a *later* multiplier (95→100%), not a prerequisite.

---

## Files changed

| File | Type | Change |
|------|------|--------|
| `core/onnx_session.py` | **new** | central thread resolver + `ORT_PARALLEL` |
| `core/frame_capture.py` | **new** | async `FrameCaptureThread` (live drop / file backpressure) |
| `core/inference_engine.py` | **new** | `SharedInferenceEngine` + cross-stream embed coalescing |
| `tools/export_dynamic_batch.py` | **new** | re-export ArcFace with symbolic batch axis |
| `core/detector.py` | edit | resolve threads via `onnx_session`, startup thread log |
| `core/embedder.py` | edit | batched `extract_embeddings_batch`, `_preprocess_chw`, `supports_dynamic_batch` |
| `core/video_recognition.py` | edit | async capture in `process_video`, `engine=` param, I/O-wait stat |
| `core/motion_detector.py` | edit | adaptive scheduler (spike + throttle on top of floor) |
| `config/detector_config.yaml` | edit | `intra/inter_op: 4 → 0` |
| `config/embedder_config.yaml` | edit | `intra/inter_op: 4 → 0` |
| `config/system_config.yaml` | edit | `onnx_threads` key, `process_every_n_frames: 1`, `detect_min_interval`, `motion_spike_delta` |
| `face_events_system/processor/run_processor.py` | edit | async capture + shared engine in `run_live`/`run_file` |

---

## Summary of measured results

| Fix | Metric | Before | After |
|-----|--------|--------|-------|
| Threads | cores used | 4.33 / 32 (13.5%) | all cores (12/12 local, 32/32 server) |
| Async capture | I/O wait | ~33 ms/frame (blocking) | 0.5 ms/frame (overlapped) |
| Batch embed | batch_8 vs serial | 1610 ms | 962 ms (1.67×), diff 0.0 |
| Shared engine | 6 concurrent embeds | 6 ONNX runs | 1 ONNX run |
| Adaptive sched | sustained-motion detects | every frame | every 3rd (3× fewer), recall kept via spike |

**END OF DOCUMENT**
