# Video Recognition Parameters Guide
## Complete Reference for All Settings

---

## Sidebar Parameters (Streamlit UI)

### 🎯 Recognition Settings

#### **Recognition Threshold** (0.2 - 0.6)
**Default**: 0.4

**What it does**: Minimum similarity score required to recognize a face as a known person.

**How it works**:
- The system compares face embeddings using cosine similarity (0.0 to 1.0)
- If similarity ≥ threshold → Person is recognized
- If similarity < threshold → Marked as "Unknown"

**Effect on output**:

| Threshold | Effect | Best For | Trade-off |
|-----------|--------|----------|-----------|
| **0.2 - 0.3** | Very lenient | Low-quality CCTV, old photos | ⚠️ High false positives (wrong recognitions) |
| **0.35 - 0.4** | Balanced (Recommended) | Normal CCTV footage | ✅ Good balance |
| **0.45 - 0.5** | Strict | High-quality video, security applications | ⚠️ May miss correct recognitions |
| **0.5+** | Very strict | Critical security (no errors allowed) | ⚠️ Many missed recognitions |

**Example**:
```
Threshold = 0.4:
  John: 0.65 ✅ Recognized (0.65 > 0.4)
  Jane: 0.42 ✅ Recognized (0.42 > 0.4)
  Unknown: 0.38 ❌ Not recognized (0.38 < 0.4)

Threshold = 0.3:
  John: 0.65 ✅ Recognized
  Jane: 0.42 ✅ Recognized
  Unknown: 0.38 ✅ Recognized (but might be wrong!)
```

**When to adjust**:
- ⬇️ **Lower (0.30-0.35)**: Poor quality CCTV, old footage, many unknown faces being missed
- ⬆️ **Higher (0.45-0.50)**: High security needs, good quality video, getting false recognitions

---

### ⚡ Performance Settings

#### **Process Every N Frames** (1 - 5)
**Default**: 1 (process all frames)

**What it does**: Skip frames to speed up processing.

**How it works**:
- 1 = Process every frame (100%)
- 2 = Process every 2nd frame (50% of frames, 2x faster)
- 3 = Process every 3rd frame (33% of frames, 3x faster)
- 5 = Process every 5th frame (20% of frames, 5x faster)

**Effect on output**:

| Setting | Speed | Accuracy Impact | Best For |
|---------|-------|-----------------|----------|
| **1** | Baseline | ✅ Best accuracy, smooth tracking | Final processing, high accuracy needs |
| **2** | 2x faster | ⚠️ Slight: Tracks may be less smooth | Good balance for most videos |
| **3** | 3x faster | ⚠️ Moderate: Some tracking gaps | Quick preview, long videos |
| **5** | 5x faster | ⚠️ Significant: Tracking may fail on fast motion | Very quick preview only |

**Why temporal smoothing helps**:
- System keeps track history (last 5 embeddings)
- Averages embeddings across frames
- Even with skipped frames, recognition remains stable

**Example**:
```
30 FPS video, 5 minutes (9000 frames):
- N=1: Process all 9000 frames (~12 min @ 12 FPS)
- N=2: Process 4500 frames (~6 min @ 12 FPS)
- N=3: Process 3000 frames (~4 min @ 12 FPS)
```

**When to adjust**:
- ⬆️ **Higher (2-3)**: Long videos, preview/testing, slow computer
- ⬇️ **Keep 1**: Short videos, fast motion, critical accuracy needs

---

#### **Show Live Preview**
**Default**: Unchecked (off)

**What it does**: Displays real-time annotated frames during processing.

**Effect on output**:
- ✅ **On**: See recognition happening live (slightly slower)
- ⭕ **Off**: Faster processing, see results only at end

**Performance impact**:
- Adds ~10-20% processing time due to image conversion and display
- Useful for monitoring progress on long videos
- Great for demonstrations

---

#### **Max Frames** (0 = all)
**Default**: 0 (process entire video)

**What it does**: Limit processing to first N frames.

**Use cases**:
- **Testing**: Process first 300 frames (10 sec @ 30fps) to verify settings
- **Preview**: Check if enrolled faces are being detected
- **Debugging**: Quickly test parameter changes

**Example**:
```
Max Frames = 300:
  - 30 FPS video: Processes first 10 seconds
  - 60 FPS video: Processes first 5 seconds
  
Max Frames = 0:
  - Processes entire video regardless of length
```

---

## Advanced Parameters (Code-Level)

These are in `core/video_recognition.py` - modify for advanced tuning:

### **Track IOU Threshold** (0.1 - 0.5)
**Default**: 0.3 (30% overlap)
**Location**: Line ~27 in `VideoRecognitionSystem.__init__()`

```python
self.track_iou_threshold = 0.3
```

**What it does**: Minimum overlap required to match detection to existing track.

**IOU (Intersection over Union)**:
```
IOU = Area of Overlap / Area of Union
```

**Effect**:
- **Lower (0.2)**: More lenient matching, tracks persist through more motion
  - ⚠️ Risk: May incorrectly merge different people
- **Higher (0.4-0.5)**: Stricter matching, more accurate
  - ⚠️ Risk: May create duplicate tracks for same person

**When to adjust**:
- Fast-moving subjects: Lower to 0.2
- Crowded scenes: Keep at 0.3 or raise to 0.35

---

### **Max Frames Missing** (10 - 60)
**Default**: 30 frames (~1 second @ 30fps)
**Location**: Line ~28

```python
self.max_frames_missing = 30
```

**What it does**: How long to keep track alive during occlusions.

**Effect**:
- **Lower (10-15)**: Tracks die quickly during occlusions
  - ✅ Less memory usage
  - ⚠️ New track created when person reappears
- **Higher (45-60)**: Tracks persist through longer occlusions
  - ✅ Better re-identification
  - ⚠️ More memory usage

**Example**:
```
Person walks behind pillar for 0.5 seconds (15 frames @ 30fps):
  Max = 30: ✅ Track maintained, same ID when reappears
  Max = 10: ❌ Track deleted, new ID assigned
```

---

### **Embedding History Size** (3 - 10)
**Default**: 5 embeddings
**Location**: Line ~56 in `FaceTrack.__init__()`

```python
self.embeddings = deque([embedding], maxlen=5)
```

**What it does**: Number of recent embeddings to average for recognition.

**Effect**:
- **Smaller (3)**: Faster adaptation to changes
  - ⚠️ Less stable, more sensitive to single bad frames
- **Larger (7-10)**: More stable recognition
  - ⚠️ Slower to adapt if person changes (puts on glasses, etc.)

---

### **Identity Voting Alpha** (0.1 - 0.5)
**Default**: 0.3 (30% new evidence, 70% history)
**Location**: Line ~96 in `FaceTrack.vote_identity()`

```python
alpha = 0.3
```

**What it does**: Weight for new recognition vs. historical votes.

**Formula**:
```
confidence_new = α × confidence_current + (1-α) × confidence_history
```

**Effect**:
- **Lower (0.1-0.2)**: Very stable, resists change
  - ✅ Good for noisy recognition
  - ⚠️ Slow to correct mistakes
- **Higher (0.4-0.5)**: Adapts quickly
  - ✅ Responsive to changes
  - ⚠️ May flicker between identities

**Example**:
```
Current: John (0.7), History: John (0.8)
Alpha = 0.3: New = 0.3×0.7 + 0.7×0.8 = 0.77 ✅ Stable

Current: Jane (0.6), History: John (0.8)
Alpha = 0.3: Needs several frames to switch
Alpha = 0.5: Switches faster
```

---

## Quality Variants (Enrollment)

Set in `core/multi_quality_enrollment.py`:

### **Slight Blur**
```python
'blur_sigma': 1.0
```
- Simulates slightly out-of-focus cameras
- Helps match faces that aren't perfectly sharp

### **Low Resolution**
```python
'scale_factor': 0.6
```
- 60% of original size
- Simulates distant faces or low-res cameras

### **Poor Lighting**
```python
'brightness': -30
'noise_sigma': 5
```
- Darker by 30 brightness units
- Adds Gaussian noise (σ=5)
- Simulates night-time or poorly lit areas

### **Severe Degradation**
```python
'blur_sigma': 2.0
'scale_factor': 0.5
'brightness': -40
'noise_sigma': 10
```
- Combined worst-case scenario
- Old photos, very low quality CCTV

---

## Recommended Settings by Use Case

### 🏢 **Office/Indoor Security**
```
Recognition Threshold: 0.40
Process Every N Frames: 1-2
Track IOU Threshold: 0.3
Max Frames Missing: 30
```
**Reasoning**: Good lighting, controlled environment, high accuracy needed

---

### 🚗 **Outdoor/Parking CCTV**
```
Recognition Threshold: 0.35
Process Every N Frames: 2-3
Track IOU Threshold: 0.25
Max Frames Missing: 45
```
**Reasoning**: Variable lighting, fast motion, occlusions common

---

### 🏪 **Retail/Crowded Spaces**
```
Recognition Threshold: 0.38
Process Every N Frames: 1-2
Track IOU Threshold: 0.35
Max Frames Missing: 20
```
**Reasoning**: Many people, close distances, need to prevent merging tracks

---

### 📹 **Low-Quality CCTV (480p, compressed)**
```
Recognition Threshold: 0.30-0.35
Process Every N Frames: 1
Track IOU Threshold: 0.3
Embedding History: 7
```
**Reasoning**: Poor quality needs more temporal smoothing, lower threshold

---

### ⚡ **Quick Preview/Testing**
```
Recognition Threshold: 0.40
Process Every N Frames: 3-5
Max Frames: 300
Show Live Preview: On
```
**Reasoning**: Fast feedback, verify system working

---

## Troubleshooting Guide

### Issue: Many false recognitions (wrong names)

**Solution**: ⬆️ Increase recognition threshold
```
Current: 0.35 → Try: 0.40 or 0.45
```

---

### Issue: Missing recognitions (should recognize but marked Unknown)

**Solution**: ⬇️ Lower recognition threshold
```
Current: 0.40 → Try: 0.35 or 0.30
```

---

### Issue: Processing too slow

**Solutions**:
1. ⬆️ Increase frame skip (2 or 3)
2. ⬇️ Reduce video resolution (pre-process with ffmpeg)
3. ⬇️ Turn off live preview

---

### Issue: Track IDs jumping (same person gets multiple IDs)

**Solutions**:
1. ⬇️ Lower IOU threshold (0.25)
2. ⬆️ Increase max frames missing (45)
3. ⬇️ Reduce frame skip (1 = all frames)

---

### Issue: Different people getting same ID

**Solution**: ⬆️ Increase IOU threshold (0.35)

---

### Issue: Flickering identities (name keeps changing)

**Solutions**:
1. ⬇️ Lower alpha (0.2) for more stability
2. ⬆️ Increase embedding history (7)
3. ⬆️ Increase recognition threshold (0.42)

---

## Parameter Priority Guide

When tuning for your specific case, adjust in this order:

1. **Recognition Threshold** - Most impact on accuracy
2. **Process Every N Frames** - Most impact on speed
3. **Show Live Preview** - For monitoring
4. **Track IOU Threshold** - For tracking quality
5. **Max Frames Missing** - For occlusion handling
6. **Advanced parameters** - Only if needed

---

## Testing Workflow

1. **Start with defaults**:
   - Threshold: 0.40
   - Frame skip: 1
   - Max frames: 300 (first 10 seconds)

2. **Run quick test**:
   ```bash
   # Process first 300 frames with preview
   ```

3. **Evaluate results**:
   - Too many "Unknown"? → Lower threshold
   - Wrong recognitions? → Raise threshold
   - Too slow? → Increase frame skip

4. **Full processing**:
   - Once parameters look good, process full video
   - Remove max frames limit
   - Turn off preview (faster)

---

## Quick Reference Table

| Parameter | Impact | Adjust When |
|-----------|--------|-------------|
| Recognition Threshold | Accuracy | Too many wrong/missed recognitions |
| Process Every N Frames | Speed | Video too long or processing too slow |
| Show Live Preview | Monitoring | Want to see progress |
| Track IOU | Tracking | IDs jumping or merging incorrectly |
| Max Frames Missing | Occlusions | People keep getting new IDs |
| Embedding History | Stability | Recognition flickering |
| Voting Alpha | Responsiveness | Want more/less stable identities |

---

## Further Reading

- **Technical Details**: See `docs/TECHNICAL_DOCUMENTATION.md`
- **Usage Examples**: See `docs/PHASE3_GUIDE.md`
- **Quick Start**: See `README.md`
