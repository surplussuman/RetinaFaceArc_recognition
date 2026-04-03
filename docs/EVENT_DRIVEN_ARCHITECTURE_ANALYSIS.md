# Event-Driven Face Recognition Architecture — Analysis

> Analysis of the proposed CPU-first, event-driven video recognition system.
> Covers what is already implemented, what is partially implemented, what is missing, and known gaps.

---

## 1. The Core Insight — Why This Works

The bottleneck is not *accuracy*, it is *where computation is applied*. The research makes this distinction:

```
OLD MENTAL MODEL:  "Make the heavy model faster"
CORRECT MODEL:     "Use the heavy model less often"
```

Formally, complexity changes from:

```
O(N_frames) → O(N_events)
```

Since in a real CCTV environment N_events << N_frames (1–3 new faces/sec vs 30 frames/sec), the system becomes tractable on CPU.

Three axes of reduction, all multiplicative:

```
T_final = T_base × (1/temporal_skip) × (ROI_area/frame_area) × (motion_area/ROI_area)
```

Each factor is independent. Combined they reduce computation by **10–50×** depending on scene.

---

## 2. What Is Already Implemented (Current Codebase)

### 2.1 Ghost Tracking — FULLY IMPLEMENTED
**File:** `core/video_recognition.py` → `FaceTrack.should_recognize()`

Identity is cached per tracked face. Embedding extraction (the expensive step, ~70ms) is skipped for existing tracks. Re-verification happens only every `recognition_interval` frames (default: 30).

```
Speedup: ~30× on embedding computation
Config:  system_config.yaml → ghost_tracking.recognition_interval
Status:  ✅ Working, production-ready
```

**Gap:** The interval is fixed at 30 frames. The research proposes adaptive interval based on scene motion density. This is not yet implemented.

---

### 2.2 Zone-Based Detection (ROI) — FULLY IMPLEMENTED
**File:** `core/video_recognition.py` → `process_frame()`, zone crop section

When zones are enabled, only the defined ROI rectangles are passed to the detector. Detection complexity becomes proportional to zone area, not full frame area.

```
Speedup:  up to 16× on detection (documented in code comments)
Config:   system_config.yaml → camera_zone
UI:       app.py has an interactive zone editor built-in
Status:   ✅ Working, needs enabling per deployment
```

**Gap:** Zones are static (set once at config time). The research proposes user-defined ROI at upload time in the UI — this IS present in `app.py` but the experience could be improved.

---

### 2.3 Frame Skipping — FULLY IMPLEMENTED
**File:** `core/video_recognition.py` → `process_frame()` early return on `frame_idx % process_every_n_frames`

```
Config:   system_config.yaml → frame_processing.process_every_n_frames
UI:       app.py exposes slider "Process Every N Frames"
Status:   ✅ Working
```

**Gap:** Fixed skip rate. Research proposes dynamic skip based on scene activity — not present.

---

### 2.4 Detection Resolution Cap — FULLY IMPLEMENTED
**File:** `core/video_recognition.py` → `resize_for_detection()`

Frames are capped at 640×360 before being passed to the full-frame detector path. This reduces detector input pixels and therefore detection time.

```
Config:   hardcoded in __init__ → self.max_detection_resolution = (640, 360)
Status:   ✅ Working
Gap:      Should be in config/system_config.yaml, not hardcoded
```

---

### 2.5 Temporal Identity Smoothing — FULLY IMPLEMENTED
**File:** `core/video_recognition.py` → `FaceTrack.vote_identity()`

Exponential moving average of identity votes across frames:
```
identity_votes[id] = α × confidence_new + (1 - α) × votes_history
```
Default α = 0.3, meaning identity is stable across 3–4 frames of uncertainty.

```
Status:  ✅ Working
```

---

### 2.6 Multi-Quality Enrollment — FULLY IMPLEMENTED
**File:** `core/multi_quality_enrollment.py`

5 synthetic quality variants per source photo. Ensures recognition works on degraded CCTV footage (the enrollment gallery already "knows" what the person looks like at low quality).

```
Status:  ✅ Working
```

---

## 3. What Is Proposed But NOT Yet Implemented

### 3.1 Motion Detection Layer (Most Important Missing Piece)

**What the research proposes:**
A cheap pre-filter (frame differencing or MOG2 background subtraction) that runs on every frame and costs ~1–2 ms. Heavy detection only fires inside motion regions.

```python
# Concept
M_t = |Frame_t - Frame_(t-1)|
if M_t > threshold:
    run_detection(motion_regions)
else:
    skip_detection()
```

**Current state:** The codebase has NO motion detection at all. The detector runs on either the full frame or the full zone every `process_every_n_frames` frames. There is no motion gating.

**Mathematical impact:**
```
If motion = 20% of ROI area:
T_detect_new = 0.2 × T_detect_old
→ 5× additional speedup on detection
```

**What needs to be built:**
- `core/motion_detector.py` — wraps `cv2.createBackgroundSubtractorMOG2()` or frame diff
- Integration into `VideoRecognitionSystem.process_frame()` before zone detection
- Config in `system_config.yaml` (motion threshold, min motion area)

**Risk:** MOG2 has a warm-up period (~20–30 frames). During this time, it classifies everything as foreground. Needs a "warm-up frame count" guard before enabling the gate.

---

### 3.2 Adaptive Detection Frequency

**What the research proposes:**
```
f_detect = f(motion_density, crowd_density)
busy_scene → increase detection
static_scene → almost zero detection
```

**Current state:** Frequency is fixed (`process_every_n_frames` slider, 1–5). There is no feedback loop from scene content into the skip rate.

**What needs to be built:**
- A scene activity score computed from motion detector output
- Dynamic update to `process_every_n_frames` based on score
- Smooth transitions to avoid oscillation

**Difficulty:** Medium. Needs the motion detector (3.1) as a dependency.

---

### 3.3 1 FPS Full-Frame Safety Scan

**What the research proposes and correctly models:**
```
Periodic scan: every 1 second regardless of motion
Purpose: catch any face that motion detection missed
```

The research correctly identifies this should NOT be the primary loop — it is a correction signal.

```
Miss probability at 1 FPS for 0.5 sec visible face: P(miss) = 0.5
Mission: catch the 5% of new entrants that motion detection misses
```

**Current state:** Not implemented. The system has no periodic safety scan independent of frame skipping.

**What needs to be built:**
- A timer-based trigger inside `process_frame()` that overrides the skip logic
- At each trigger, run full-frame detection inside the configured zone
- This should run at `1 / fps` interval in frame counts, e.g. every 30 frames at 30 FPS

---

### 3.4 Decoupled Processing and Display Pipeline

**What the research proposes:**
```
Pipeline A (Processing):   runs at computation speed (5–10 FPS)
Pipeline B (Visualization): runs at video FPS (30 FPS), overlays cached results
```

**Current state — the core reason the UI is slow:**

`app.py` processes frames **synchronously** inside the Streamlit callback. Processing speed = playback speed. A 5-second video at 200 ms/frame takes 5s × 30fps × 0.2s = 30 seconds. The 40–50 second real figure accounts for loading + overhead.

There is no decoupling. The user sees the video rendering at processing speed, not real video speed.

**What needs to be built:**
- Offline mode (existing, but slow): process video → save output → user downloads. Already partially works.
- Event feed mode (new): instead of annotated video, output a sequence of detected face crops + identity + timestamp. This is cheap to display and does not require rendering a full video.
- The research's "event feed" idea is technically correct: show detected faces as a card feed rather than a video. This eliminates the display bottleneck entirely.

**Correct current recommendation:** For video uploads, keep the existing offline processing approach (process → show summary + CSV). Do not attempt real-time synchronous display.

---

### 3.5 Multi-Resolution Detection Strategy

**What the research proposes:**
```
Low-resolution scan: whole frame, cheap, detects regions of interest
High-resolution confirmation: only on ROI, precise detection
```

**Current state:** Single-resolution detection only. The detector runs at one resolution (capped at 640×360 for full-frame, or zone size for zone mode).

**What needs to be built:**
- Two-pass detection: fast low-res pass to find candidate regions, precise pass on candidates
- Likely requires SCRFD (the lighter RetinaFace variant) for the fast pass and current ResNet50 for confirmation

**Difficulty:** High. Requires a second model and a two-stage inference engine. Worth adding after motion detection (3.1) is working.

---

### 3.6 Early Rejection Layer (Cheap Pre-Filters)

**What the research proposes:**
```
Before deep model:
- face aspect ratio filter
- skin color mask
- head-like contours
Reject 70–80% of motion regions before DL
```

**Current state:** No pre-filters. Motion regions go directly to the full detector.

**What needs to be built:**
- Contour analysis on motion mask to filter small/non-person-shaped blobs
- Minimum bounding box size check (if motion blob < min_face_size, skip)
- These are pure OpenCV operations, ~0.2 ms cost, can reject most background noise

---

## 4. Mathematical Summary — Expected Gains by Layer

Baseline: 200 ms/frame (RetinaFace on CPU, full frame, 1920×1080)

| Layer | What it does | Reduction | Status |
|-------|-------------|-----------|--------|
| Ghost Tracking | Skip embedding for existing tracks | ~30× on embedding | ✅ Done |
| Frame Skipping | Process 1 in N frames | N× on all stages | ✅ Done |
| Zone Detection | Crop to ROI before detect | 5–16× on detection | ✅ Done |
| Resolution Cap | Resize frame before detect | 2–4× on detection | ✅ Done |
| Motion Gate | Skip detection when no motion | 3–10× on detection | ❌ Not built |
| 1 FPS Safety Scan | Catch missed faces | Accuracy improvement | ❌ Not built |
| Adaptive Frequency | Match compute to activity | 1.5–3× average | ❌ Not built |
| Early Rejection | Filter motion blobs before detect | 2–5× on detection | ❌ Not built |

Layers combine multiplicatively for detection:

```
T_detect_current (with zones + resolution cap) ≈ 200ms × (zone_area/frame_area) × (640/1920)²
Example (30% zone, 640×360 cap): 200 × 0.3 × 0.11 ≈ 6.6 ms/frame

T_detect_future (+ motion gate + early rejection) ≈ 6.6 × 0.2 × 0.3 ≈ 0.4 ms/frame
```

At 0.4 ms/frame for detection + 2 ms/frame for cheap ops + occasional embedding:
```
T_average = E × T_embed × (1/recognition_interval) + N_frames × 0.4ms
           = 2 × 70ms × (1/30) + 30 × 0.4ms
           = 4.7ms + 12ms
           ≈ 17ms/frame → well under 30ms real-time target
```

This math holds for a scene with 2 new people entering per second, which is typical CCTV.

---

## 5. The Display Latency Problem — Root Cause and Fix

**Why the 5-second video takes 40–50 seconds in the UI:**

The current Streamlit app processes frames synchronously while updating a preview placeholder. Every frame write to `st.image()` triggers a UI update. At 200 ms/frame × 150 frames (5s × 30fps / skip_factor_1) = 30 seconds minimum, plus overhead.

The fix is NOT to make processing faster but to decouple display from processing:

```
Current: process frame → update UI → process frame → update UI → ...
Fixed:   process all frames → show summary/faces → done
```

The "event feed" idea from the research is the cleanest fix:
- Instead of rendering an annotated video, extract detected face images + identity + timestamp
- Show them as a card feed in the UI
- Processing becomes fully offline, display is instant

This also enables a cleaner product experience: instead of watching a slow re-render of security footage, users see a clean identity log.

---

## 6. Known Correctness Gaps in the Research

The research is mathematically sound. Two points need clarification for implementation:

### 6.1 The 1 FPS Safety Net — Missed Person Model

The research states:
```
P(miss) = visible_duration / sampling_interval
= 0.5 sec / 1 sec = 0.5 (50% miss rate)
```

This is pessimistic (worst case: person visible for exactly 0.5s). In practice:
- Most CCTV walk-throughs are 2–5 seconds
- P(miss at 1 FPS) = max(0, 1 - visible_duration) ≈ 0 for anyone visible > 1 sec

The 1 FPS scan is a safety net for fast walk-bys, not the primary detection layer. Motion detection (1–2 ms, every frame) handles everything else.

### 6.2 Worst-Case Crowded Scene

The research acknowledges:
```
If 30 new people/sec → O(N_events) ≈ O(N_frames)
```

At a busy shopping mall entrance during peak hour, arrival rates can be 5–10 people/minute, not per second. Even at 1 person per 5 seconds:
```
T = (1/5) × 200ms + 30 × 0.4ms = 40ms + 12ms = 52ms → ~20 FPS
```

Still viable on CPU for a busy scene with motion gating.

---

## 7. Implementation Priority Order

Based on impact vs difficulty:

| Priority | Feature | Impact | Difficulty | Dependency |
|----------|---------|--------|------------|------------|
| 1 | Motion detection gate | Very High | Low | None |
| 2 | 1 FPS safety scan | High | Low | None |
| 3 | Event feed UI mode | High | Medium | None |
| 4 | Move resolution cap to config | Low | Very Low | None |
| 5 | Adaptive detection frequency | Medium | Medium | Motion detector |
| 6 | Early rejection filters | Medium | Low | Motion detector |
| 7 | Multi-resolution two-pass | High | High | Second model |

**Immediate wins (1–2 days):** Items 1, 2, 4 — these unlock the remaining items and deliver measurable speedup.

---

## 8. What This System Does That Traditional Systems Do Not

In simple terms, compared to a conventional frame-by-frame recognition pipeline:

- **Ghost Tracking** — does not re-compute identity every frame once a face is recognized; caches and reuses identity for N frames
- **Zone-Based Detection** — does not process the full image; focuses the detector on user-defined entry-point regions only
- **Frame Skipping** — does not treat every frame equally; processes a fraction at configurable intervals
- **Resolution Capping** — does not pass full-resolution frames to the detector; rescales to the minimum needed for accurate detection
- **Temporal Voting** — does not commit to identity on a single frame; accumulates votes across frames and uses EMA for stable output
- **Multi-Quality Enrollment** — does not enroll one embedding per person; generates 5 quality variants per photo so the model is prepared for CCTV degradation
- **Event Complexity** — computation scales with number of new identities appearing, not with video FPS; idle scenes cost almost nothing
- **Decoupled Display** (proposed) — does not tie processing speed to playback speed; separates recognition pipeline from UI rendering
- **Motion Gating** (proposed) — does not run the detector when nothing moves in the scene; triggers inference only on activity
- **Adaptive Frequency** (proposed) — does not use a fixed processing rate; adjusts detection frequency based on scene activity in real time
