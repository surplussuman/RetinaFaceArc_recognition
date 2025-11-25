# Phase 3: Video Recognition System
## Complete User Guide

---

## Overview

Phase 3 adds **video face recognition with temporal tracking** for CCTV footage. The system handles:

✅ Low-quality surveillance camera footage  
✅ Multiple people in frame simultaneously  
✅ Temporal smoothing for stable identification  
✅ Real-time processing with configurable frame skip  
✅ Annotated video output + CSV recognition log

---

## Installation

### 1. Install Dependencies

```bash
pip install streamlit pandas
```

All other dependencies should already be installed from Phase 1 & 2.

---

## Usage Options

### Option A: Streamlit Web Interface (Recommended)

**Launch the web interface:**

```bash
streamlit run app.py
```

**Your browser will open automatically at http://localhost:8501**

#### Features:
- 📤 **Upload Video**: Drag & drop MP4, AVI, MOV, or MKV files
- 🎯 **Adjust Settings**: Recognition threshold, frame skip, preview options
- 📊 **View Statistics**: Total detections, unique tracks, recognized identities
- 💾 **Download Results**: Annotated video + CSV log with timestamps
- ➕ **Enroll Users**: Upload photos directly in the web interface

#### Workflow:
1. **Enroll Users** (if not already done):
   - Go to "Enroll Users" tab
   - Enter User ID/Name
   - Upload 3-5 clear photos
   - Click "Enroll User"

2. **Process Video**:
   - Go to "Video Recognition" tab
   - Upload your video file
   - Adjust settings if needed:
     - **Recognition Threshold**: 0.4 (default), lower = more lenient
     - **Process Every N Frames**: 1 (all frames), 2-5 for faster processing
   - Click "Start Recognition"
   - Wait for processing (progress bar shown)

3. **View & Download**:
   - See statistics (detections, tracks, recognized people)
   - Preview annotated video
   - Download annotated video (MP4)
   - Download recognition log (CSV)

---

### Option B: Command Line Interface

**Basic usage:**

```bash
python tools/process_video.py --video test.mp4 --output results/output.mp4
```

**With CSV log:**

```bash
python tools/process_video.py --video test.mp4 --output results/output.mp4 --log results/log.csv
```

**With live preview:**

```bash
python tools/process_video.py --video test.mp4 --output results/output.mp4 --preview
```

**Fast processing (skip frames):**

```bash
python tools/process_video.py --video test.mp4 --output results/output.mp4 --skip-frames 2
```

**Process first 300 frames only (testing):**

```bash
python tools/process_video.py --video test.mp4 --output results/output.mp4 --max-frames 300
```

**Custom recognition threshold:**

```bash
python tools/process_video.py --video test.mp4 --output results/output.mp4 --threshold 0.35
```

#### Command Line Options:

| Option | Description | Default |
|--------|-------------|---------|
| `--video` | Input video file path | *Required* |
| `--output` | Output video file path | None (no output saved) |
| `--log` | CSV log file path | None (no log saved) |
| `--threshold` | Recognition threshold (0.2-0.6) | 0.4 |
| `--skip-frames` | Process every N frames (1-5) | 1 (all frames) |
| `--max-frames` | Maximum frames to process | None (all) |
| `--preview` | Show live preview window | False |

---

## Output Files

### 1. Annotated Video

**Format**: MP4 (H.264)  
**Content**: Original video with bounding boxes and labels

**Annotations**:
- 🟢 **Green Box + Name**: Recognized person
- 🔴 **Red Box + "Unknown"**: Unrecognized face
- Track ID shown for debugging

### 2. Recognition Log (CSV)

**Columns**:
- `frame`: Frame number (0-indexed)
- `timestamp`: Video timestamp (seconds)
- `track_id`: Unique track identifier
- `identity`: Person's name or "Unknown"
- `confidence`: Recognition confidence (0.0-1.0)

**Example**:
```csv
frame,timestamp,track_id,identity,confidence
45,1.50,1,John_Doe,0.6523
46,1.53,1,John_Doe,0.6841
47,1.57,1,John_Doe,0.7012
```

**Use cases**:
- Generate attendance reports
- Track when specific people appear
- Analyze recognition confidence over time
- Audit system performance

### 3. Statistics (JSON)

**Auto-saved** as `<output_name>_stats.json`

**Contains**:
```json
{
  "total_frames": 1800,
  "total_detections": 156,
  "total_tracks": 12,
  "total_recognized": 89,
  "unique_identities": ["John_Doe", "Jane_Smith"],
  "avg_fps": 12.3,
  "processing_time": 146.2
}
```

---

## Performance Tuning

### For Faster Processing:

**1. Skip Frames** (Recommended):
```bash
--skip-frames 2  # Process every 2nd frame (2x faster)
--skip-frames 3  # Process every 3rd frame (3x faster)
```

**Trade-off**: Slight reduction in tracking smoothness, but recognition quality maintained via temporal smoothing.

**2. Lower Resolution** (if needed):
- Pre-process video to 720p or 480p using ffmpeg:
```bash
ffmpeg -i input.mp4 -vf scale=1280:720 output_720p.mp4
```

**3. Test on Subset**:
```bash
--max-frames 300  # Process first 10 seconds (@ 30fps)
```

### For Better Accuracy:

**1. Lower Threshold**:
```bash
--threshold 0.35  # More lenient (higher false positives)
```

**2. Process All Frames**:
```bash
--skip-frames 1  # Default - best accuracy
```

**3. Enroll More Photos**:
- Use 5 photos per person instead of 3
- Include varied lighting and angles

---

## System Architecture

### Video Processing Pipeline:

```
Video Input
    ↓
Frame Extraction (cv2.VideoCapture)
    ↓
For each frame:
    1. Face Detection (RetinaFace)
    2. Face Alignment (5-point)
    3. Embedding Generation (ArcFace)
    4. Track Matching (IOU-based)
    5. Identity Voting (Exponential Moving Average)
    6. Recognition (FAISS search)
    ↓
Annotation (bounding boxes + labels)
    ↓
Video Output (cv2.VideoWriter)
```

### Temporal Smoothing:

**Problem**: Per-frame quality varies in CCTV footage → Unstable recognition

**Solution**: Aggregate evidence across frames

1. **Embedding History**: Keep last 5 embeddings per track
2. **Average Embedding**: Use mean of recent embeddings for recognition
3. **Identity Voting**: Exponential moving average (α=0.3):
   ```
   confidence_new = 0.3 × confidence_current + 0.7 × confidence_history
   ```

**Result**: Stable identification even with frame-to-frame quality drops

### Track-by-Detection:

**IOU (Intersection over Union)** matching:
```
IOU = Area(box1 ∩ box2) / Area(box1 ∪ box2)
```

- **Threshold**: 0.3 (30% overlap required)
- **Max Missing**: 30 frames (1 second @ 30fps)
- **Matching**: Greedy assignment (can upgrade to Hungarian algorithm)

---

## Troubleshooting

### Issue: Low FPS (< 5 FPS)

**Solutions**:
1. Increase `--skip-frames` to 2 or 3
2. Reduce video resolution to 720p
3. Process without preview (`--preview` off)
4. Check CPU usage (close other applications)

### Issue: Poor Recognition Accuracy

**Solutions**:
1. Lower `--threshold` to 0.35 or 0.30
2. Re-enroll users with more photos (5 instead of 3)
3. Check enrollment photo quality (clear, well-lit)
4. Verify database loaded correctly (check console output)

### Issue: Tracking Failures (IDs jumping)

**Solutions**:
1. Process more frames (`--skip-frames 1`)
2. Check video quality (severe motion blur causes issues)
3. Adjust IOU threshold in `core/video_recognition.py` (line ~350)

### Issue: Database Not Found

**Error**: "Database not found. Please enroll users first."

**Solution**:
```bash
python tools/enroll_multi_quality.py --user-id "John_Doe" --directory "path/to/photos/"
```

---

## Best Practices

### 1. Enrollment:
- ✅ Use 3-5 photos per person
- ✅ Vary angles (front, slight left, slight right)
- ✅ Include different expressions (neutral, smiling)
- ✅ Ensure good lighting and focus
- ❌ Avoid sunglasses, masks, or heavy occlusions

### 2. Video Processing:
- ✅ Test on small subset first (`--max-frames 300`)
- ✅ Use `--skip-frames 2` for faster initial testing
- ✅ Save CSV log for analysis
- ✅ Verify enrolled users before processing

### 3. CCTV Optimization:
- ✅ Multi-quality enrollment handles degradation automatically
- ✅ Temporal smoothing compensates for per-frame quality drops
- ✅ Lower threshold for very low quality (0.30-0.35)
- ✅ Process at native resolution (don't upscale)

---

## Example Workflow

### Complete Recognition Pipeline:

```bash
# 1. Enroll users
python tools/enroll_multi_quality.py --user-id "Employee_001" --directory "photos/employee1/"
python tools/enroll_multi_quality.py --user-id "Employee_002" --directory "photos/employee2/"

# 2. Test on first 300 frames
python      tools/process_video.py \
    --video cctv_footage.mp4 \
    --output results/test_output.mp4 \
    --log results/test_log.csv \
    --max-frames 300 \
    --preview

# 3. Review test results
# - Check annotated video
# - Verify recognition accuracy
# - Adjust threshold if needed

# 4. Process full video
python tools/process_video.py \
    --video cctv_footage.mp4 \
    --output results/full_output.mp4 \
    --log results/full_log.csv \
    --threshold 0.35 \
    --skip-frames 2

# 5. Analyze results
# - Open CSV in Excel/pandas
# - Generate attendance report
# - Check when specific people appeared
```

---

## Web Interface Screenshots

### Video Upload
![Upload Interface](docs/images/upload.png)

### Processing
![Processing](docs/images/processing.png)

### Results
![Results](docs/images/results.png)

---

## Advanced Configuration

### Modify Tracking Parameters:

Edit `core/video_recognition.py`:

```python
# Line ~25-30
self.recognition_threshold = 0.4  # Recognition threshold
self.track_iou_threshold = 0.3    # IOU matching threshold
self.max_frames_missing = 30      # Track persistence
self.process_every_n_frames = 1   # Frame skip

# Line ~40-45 (FaceTrack class)
self.bboxes = deque(maxlen=10)       # Keep last 10 bboxes
self.embeddings = deque(maxlen=5)    # Average last 5 embeddings
```

### Modify Exponential Moving Average:

```python
# Line ~120-125
alpha = 0.3  # Weight for new evidence (0.0-1.0)
             # Higher = faster adaptation
             # Lower = more stable
```

---

## Next Steps

1. ✅ **Test with your CCTV footage**
2. ✅ **Tune parameters** for your specific use case
3. ✅ **Generate attendance reports** from CSV logs
4. ✅ **Deploy Streamlit app** for easy access
5. ✅ **Monitor performance** and adjust as needed

---

## Support

For technical details, see:
- `TECHNICAL_DOCUMENTATION.md` - Mathematical foundations
- `README.md` - Quick start guide
- Code comments in `core/video_recognition.py`

For issues:
- Check troubleshooting section above
- Review console output for errors
- Test with subset of frames first
