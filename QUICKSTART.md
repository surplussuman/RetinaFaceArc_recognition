# Phase 1: Quick Start Guide

## 🚀 Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Install required packages
pip install torch torchvision opencv-python numpy pyyaml onnxruntime
```

### Step 2: Download Models (3 minutes)

```powershell
# Automatic download (recommended)
python scripts\download_models.py

# Or check what models you need
python scripts\download_models.py --manual
```

### Step 3: Test the Pipeline

```powershell
# Run comprehensive tests
python tests\test_phase1.py --image path\to\your\face_image.jpg --save-viz

# Output will be in: data\test_output\
```

---

## 📸 What You Need for Testing

**A face image with:**
- Clear, well-lit face
- Resolution ≥ 640×480
- Frontal or near-frontal angle
- Minimal occlusion

**Example test images you can use:**
- Your webcam photo
- Passport photo
- Any clear face photo from your device

---

## ✅ Expected Results

After running the test, you should see:

```
==============================================================
TEST SUMMARY
==============================================================
  Detection: ✓ PASSED
  Alignment: ✓ PASSED
  Embedding: ✓ PASSED
  Recognition: ✓ PASSED
  Benchmark: ✓ PASSED

==============================================================
✓ ALL TESTS PASSED

Phase 1 is complete and working!
Next: Phase 2 - FAISS Vector Database + User Enrollment
==============================================================
```

**Generated files in `data/test_output/`:**
- `detection_result.jpg` - Face detection with landmarks
- `aligned_face_1.jpg` - Aligned face crop
- `alignment_viz_1.jpg` - Before/after alignment
- `recognition_result.jpg` - Recognition with labels

---

## 🎯 What Phase 1 Gives You

✅ **Working Components:**
1. Face detection (RetinaFace)
2. 5-point alignment
3. 512-D face embeddings (ArcFace)
4. Basic gallery matching

✅ **Can Do:**
- Detect faces in images
- Extract face embeddings
- Compare face similarity
- Recognize faces from a small gallery (1-10 people)

❌ **Cannot Do Yet (coming in Phase 2-7):**
- Large-scale recognition (100+ people) → Phase 2 (FAISS)
- Real-time video processing → Phase 3
- Multi-person tracking → Phase 4
- Attendance marking → Phase 5
- Crowd analytics → Phase 6
- Production deployment → Phase 7

---

## 🔧 If Something Goes Wrong

### ❌ "No module named 'onnxruntime'"
```powershell
pip install onnxruntime-gpu  # For GPU
# OR
pip install onnxruntime  # For CPU
```

### ❌ "Failed to load ONNX model"
```powershell
# Check models exist
dir models\

# Should show:
#   retinaface_resnet50.onnx
#   arcface_resnet100.onnx

# If missing, download:
python scripts\download_models.py
```

### ❌ "No face detected"
Your image might have:
- Very small faces (< 50px)
- Poor lighting
- Extreme angles

**Try:**
- Use a clearer image
- Lower confidence threshold in `config/detector_config.yaml`

---

## 📖 Next Steps

Once Phase 1 tests pass:

1. **Read:** `Phase1_CoreSetup.md` for detailed documentation
2. **Experiment:** Try with different images
3. **Ready?** Move to Phase 2 when satisfied with results

---

## 💡 Pro Tips

1. **Start simple:** Test with 1 clear face first
2. **Check outputs:** Always use `--save-viz` to see visualizations
3. **Benchmark:** Note your FPS for performance tracking
4. **Multiple images:** For enrollment, use 5-10 images per person

---

## 📊 Performance Check

**Good performance indicators:**
- Detection: < 50 ms per image
- Alignment: < 2 ms per face
- Embedding: < 10 ms per face
- **Total: > 20 FPS** ✅

If slower:
- Check GPU is being used: `nvidia-smi`
- Reduce input image size
- Use lighter models (MobileNet)

---

## 🎓 Understanding the Output

### Detection Result
```
Detected 1 faces in 42.35 ms

Face 1:
  BBox: [120.5, 80.3, 245.7, 210.6]  ← Face location
  Confidence: 0.997                    ← How sure (0-1)
  Landmarks: 5 points                  ← Eyes, nose, mouth
```

### Alignment Result
```
Face 1:
  Alignment quality: GOOD              ← Quality check
  RMSE error: 1.85 pixels             ← Lower is better (< 3.0 = good)
```

### Embedding Result
```
Face 1:
  Embedding shape: (512,)              ← 512-dimensional vector
  Embedding norm: 1.000000            ← Should be exactly 1.0
  Extraction time: 5.23 ms            ← Inference speed
```

### Recognition Result
```
Face 1:
  Identity: Test_Person               ← Matched person
  Similarity: 0.7234                  ← How similar (0-1)
  Detection confidence: 0.997         ← Detection score
```

**Similarity interpretation:**
- \> 0.6: Very likely same person ✅
- 0.4-0.6: Uncertain (may need adjustment)
- < 0.4: Different person ❌

---

## ✨ Ready to Continue?

If all tests passed, you're ready for **Phase 2: FAISS + Enrollment**!

Phase 2 will add:
- Fast similarity search for 100+ people
- Multi-image enrollment
- Database persistence
- Quality control

**Stay tuned!** 🚀
