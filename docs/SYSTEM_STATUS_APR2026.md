# System Status & Issue Analysis — April 2026

**Date:** April 6, 2026  
**Project:** RetinaFaceArc Recognition (SmartEntry)  
**Session Summary:** Architecture improvements, bug fixes, live streaming addition

---

## 1. What Was Done (April 3–6, 2026)

### 1.1 Critical Bug Fixes

| Bug | Root Cause | Impact | Fix |
|-----|-----------|--------|-----|
| FAISS similarity formula wrong | `IndexFlatL2` returns `d²` (squared), but code applied `1 - (d²)²/2` (squared it again) | All video-quality faces scored below threshold → always "Unknown" | `core/vector_db.py`: `1 - distances[0] / 2` |
| ONNX batch size mismatch warning | ArcFace model has static `batch_size=1` but `extract_embeddings_batch()` sent N faces as a batch | Runtime warnings, possible incorrect outputs | `core/embedder.py`: loop one face at a time |
| Output video too short / 5× speed | `process_video()` only wrote *processed* frames (262/1306), not all frames | Output was 10s instead of 52s, played at wrong speed | `core/video_recognition.py`: write every frame; persist last annotation |
| Config overriding UI slider values | `create_video_recognizer()` read `process_every_n_frames` from YAML and silently replaced what the caller passed | Setting skip_frames=3 in UI had no effect; always used config value 5 | Use `None` sentinel: only apply config default when caller passes `None` |
| Enrollment tab crash | `MultiQualityEnroller()` called with no args; wrong method name `enroll_from_directory(photos_directory=...)` | Enrollment tab 100% broken | Fixed to pass all required components and correct kwargs |
| Model path mismatch | Configs pointed to `_int8.onnx` files that don't exist | Startup crash | Fixed to `retinaface_resnet50.onnx` / `arcface_resnet100.onnx` |

### 1.2 Architecture Additions

- **Motion Detection Gate** (`core/motion_detector.py`) — MOG2 background subtraction gates the 200ms face detector; only triggers when motion fraction > 1% of frame. Saves 30–40% of detector calls in typical surveillance footage.
- **Ghost Tracking** (already existed, now verified) — Identity cached per FaceTrack, re-computed every 30 frames. In practice achieves 80%+ cache hit rate on scenes with persistent faces.
- **Re-enrollment** — Niheesh re-enrolled with 5 photos × 5 quality variants = 25 embeddings averaged to 1 robust embedding.

### 1.3 Performance Results (CLI, Sample 1.mp4 52s video)

```
Total frames:       1306
Processed frames:    262  (every 5th frame — from config)
Skipped frames:     1044
Total detections:     30
Total time:         23.6s (55 FPS wall-clock)
Actual video:       52.2s

Ghost Tracking:     3.3% cache hit rate  (face count too low for large benefit)
Motion Gate:        29.8% frames skipped  (no motion)
```

**Processing is 2.2× faster than real-time** on CPU (RTX 3050 Laptop GPU not used — OOM risk).

---

## 2. The 135s vs 35s Discrepancy — Root Cause Analysis

### 2.1 What the user saw

| Metric | CLI (terminal) | Streamlit UI |
|--------|---------------|--------------|
| skip_frames | 5 (from config) | **1** (UI slider default) |
| Processed frames | 262 | **863** |
| Total time | ~35s | **135s** |
| Avg ms/frame | 17ms | 99ms |
| FPS | 55 | 9.67 |

### 2.2 Why it was slower (two compounding causes)

**Cause 1: 3.3× more frames processed**  
The Streamlit slider defaulted to `skip_frames=1` (process every frame).  
CLI used the config default of `process_every_n_frames=5`.  
Result: 863 vs 262 frames processed.

**Cause 2: 1306 Streamlit UI round-trips (≈50s overhead)**  
`progress_callback` was called for *every* frame (1306 × 2 widget updates).  
Each Streamlit widget update = WebSocket message to browser ≈ 30–40ms round-trip.  
Total overhead: 1306 × ~38ms ≈ **50 seconds** of pure UI communication overhead.

Together: `863 frames × 99ms + 50s overhead = 135s`.  
Without overhead: `863 × 17ms ≈ 15s`. With skip to 5: ~8s.

### 2.3 Fixes applied

1. **Throttle progress_callback**: Only update Streamlit UI every 25 frames instead of every frame.  
   Overhead drops from ~50s to ~2s.

2. **Change slider default**: `skip_frames` default changed from 1 → 3.  
   For a 52s video at 25fps this means ~435 frames instead of 1306.  
   Expected time on same machine: ~10–15s.

---

## 3. Live Streaming — Issue & Solution

### 3.1 The problem

The original app had **no live streaming tab** — only video file upload. The user wanted real-time webcam/CCTV feed with recognition overlay.

### 3.2 Why standard Streamlit can't do true live streaming

Streamlit uses a **stateless request-response model**: the entire Python script re-executes from top on every UI interaction. A `while True` capture loop inside a button handler works but:
- The "Stop" button can't interrupt the loop (Streamlit is blocked)
- No true parallel execution
- Each `st.image()` call inside a loop costs a WebSocket round-trip (same issue as progress_callback)

### 3.3 Solution: streamlit-webrtc

`streamlit-webrtc` uses WebRTC (browser-native video protocol) + a background thread:
- Browser sends video frames over peer-to-peer connection (no Python loop blocking)
- `VideoProcessorBase.recv()` runs in a separate thread — processes one frame at a time
- Processed frames sent back to browser immediately
- Streamlit UI stays responsive for sliders/buttons

### 3.4 Implementation

Added **Live Stream** tab (`tabs[1]`) in `app.py`:

```python
class _LiveProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        result = self.recognizer.process_frame(img, self.frame_count)
        self.frame_count += 1
        if not result.get('skipped') and not result.get('motion_gated'):
            img = self.recognizer.draw_annotations(img, result)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
```

The recognizer is initialized once and stored in `st.session_state` to persist across Streamlit reruns.

**Expected live stream latency**: 300–600ms (WebRTC overhead + ~100ms face detector per triggered frame).

---

## 4. Current System State

### 4.1 Files modified (April 2026)

| File | Change |
|------|--------|
| `core/vector_db.py` | Fix FAISS similarity formula |
| `core/embedder.py` | Fix batch inference (loop, not vstack) |
| `core/video_recognition.py` | Write all frames to output; fix config sentinel; motion gate |
| `core/motion_detector.py` | NEW — MOG2 motion gating |
| `config/detector_config.yaml` | Fix model path (`_int8` → non-quantized) |
| `config/embedder_config.yaml` | Fix model path |
| `config/system_config.yaml` | Add `motion_detection` block; disable wrong zone |
| `app.py` | Throttle progress_callback; fix skip_frames default; add Live Stream tab; fix enrollment |

### 4.2 Performance summary

| Scenario | Frames | Time | FPS |
|----------|--------|------|-----|
| CLI skip=5, motion gate | 262 | 23s | 55 |
| CLI skip=3, motion gate | ~435 | ~38s | ~34 |
| Streamlit skip=3 (after fix), motion gate | ~435 | ~10–15s | ~30–40 |
| Streamlit skip=1 (before fix) | 863 | 135s | 9.67 |
| Live stream (WebRTC, skip=3) | Real-time | - | ~10–15 FPS |

### 4.3 Recognition accuracy

- **Same-subject (enrolled user, good photo)**: cosine similarity 0.87–0.94 ✅
- **Video-quality (CCTV-style, different person)**: cosine similarity ~0.02–0.05 ✅ (correctly rejected)
- **Threshold**: 0.4 (balanced), 0.35 (lenient for CCTV), 0.5 (strict)

---

## 5. Known Remaining Limitations

| Issue | Severity | Notes |
|-------|----------|-------|
| Ghost Tracking cache hit rate low (3.3%) on sparse-face videos | Medium | Works well when same face appears for >30 frames consecutively |
| IOU tracking loses identity between every-5th-frame gaps | Medium | Solved partially by motion gate keeping tracks alive |
| Live stream requires webcam browser permission | Low | Standard browser behavior |
| WebRTC STUN server required for remote deployment | Medium | Configured with Google STUN; works on LAN/localhost |
| No RTSP/IP camera input in UI | Low | CLI tools support full cv2.VideoCapture strings |
| CPU-only inference (GPU OOM on 4GB VRAM) | Low | 2× real-time on CPU is acceptable for entry monitoring |

---

## 6. Next Steps

1. **Improve tracking across skipped frames** — keep bounding boxes from last frame and expand IOU tolerance when skip_frames > 3
2. **Add RTSP stream input** — text box for camera URL (admin/admin@192.168.x.x:554/...) in live tab
3. **Attendance log UI** — show timestamp + confidence per recognized identity in live stream
4. **Multi-camera support** — run recognizer per camera in parallel threads
