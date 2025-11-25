# Phase 1: Core Setup - Detector + Aligner + Embedder + Basic Recognition

## 📋 Overview

Phase 1 implements the foundational pipeline for face recognition:

```
Frame → RetinaFace Detection → 5-Point Alignment → ArcFace Embedding → Basic Recognition
```

This phase focuses on **getting the core components working correctly** with proper mathematical implementation.

---

## ✅ Completed Components

### 1. **RetinaFace Detector** (`core/detector.py`)
- ✅ FPN multi-scale detection (P3, P4, P5, P6, P7)
- ✅ 5-point facial landmark extraction
- ✅ NMS (Non-Maximum Suppression)
- ✅ Support for faces as small as 16×16 pixels
- ✅ ONNX Runtime inference with GPU support

**Key Features:**
- **Mathematical Foundation:** L = L_cls + λ₁·p*·L_box + λ₂·p*·L_pts
- **Anchor Generation:** Multi-scale anchors across FPN levels
- **Box Decoding:** Standard R-CNN style decoding
- **Landmark Regression:** Precise 5-point localization

### 2. **Face Aligner** (`core/aligner.py`)
- ✅ 5-point affine transformation using least-squares: **M = (AᵀA)⁻¹AᵀT**
- ✅ Template-based alignment (112×112 or 160×160)
- ✅ Quality check with RMSE error computation
- ✅ Batch alignment support

**Key Features:**
- **Mathematical Foundation:** Solves least-squares problem for affine matrix
- **Error Propagation:** δu_s ≈ J_u·δm (minimizes geometric variance)
- **Robust Estimation:** RANSAC-based outlier rejection
- **Quality Metrics:** Reprojection error in pixels

### 3. **ArcFace Embedder** (`core/embedder.py`)
- ✅ 512-D L2-normalized embeddings
- ✅ Angular margin loss understanding (m=0.5, s=64)
- ✅ Quality control (blur detection, brightness check)
- ✅ Batch inference support
- ✅ Embedding aggregation (mean/median)

**Key Features:**
- **Mathematical Foundation:** L = -log(exp(s·cos(θ+m)) / Σ exp(s·cos(θ_j)))
- **Geometry:** Embeddings on unit hypersphere (||f|| = 1)
- **Cosine Similarity:** Direct dot product for normalized vectors
- **Quality Checks:** Laplacian variance for blur, brightness range

### 4. **Basic Recognition Pipeline** (`core/basic_recognition.py`)
- ✅ End-to-end integration
- ✅ Gallery management (add/remove/search)
- ✅ Multi-image enrollment with averaging
- ✅ Visualization utilities
- ✅ Performance benchmarking

---

## 📁 Project Structure (Phase 1)

```
project/
├── config/
│   ├── detector_config.yaml      ✅ RetinaFace configuration
│   ├── embedder_config.yaml      ✅ ArcFace configuration
│   ├── faiss_config.yaml         ✅ (for Phase 2)
│   └── system_config.yaml        ✅ Global settings
│
├── core/
│   ├── detector.py               ✅ RetinaFace detector
│   ├── aligner.py                ✅ 5-point affine alignment
│   ├── embedder.py               ✅ ArcFace embedder
│   └── basic_recognition.py      ✅ Integrated pipeline
│
├── models/
│   ├── retinaface_resnet50.onnx  ⚠️  Need to download
│   └── arcface_resnet100.onnx    ⚠️  Need to download
│
├── tests/
│   └── test_phase1.py            ✅ Comprehensive test suite
│
├── requirements.txt              ✅ Dependencies
├── README.md                     ✅ Main documentation
└── Phase1_CoreSetup.md          ✅ This file
```

---

## 🔧 Setup Instructions

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

**Required packages:**
- `torch>=2.0.0`
- `onnxruntime-gpu>=1.16.0` (or `onnxruntime` for CPU)
- `opencv-python>=4.8.0`
- `numpy>=1.24.0`
- `pyyaml>=6.0`

### Step 2: Download Pre-trained Models

You need to download ONNX models and place them in `models/` directory:

#### Option A: Download from official sources

**RetinaFace:**
- Source: [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- Convert PyTorch → ONNX or download pre-converted
- Place as: `models/retinaface_resnet50.onnx`

**ArcFace:**
- Source: [deepinsight/insightface](https://github.com/deepinsight/insightface)
- Model: ResNet100 backbone trained on MS1MV2/MS1MV3
- Place as: `models/arcface_resnet100.onnx`

#### Option B: Use InsightFace model zoo

```python
# Install insightface
pip install insightface

# Download models
import insightface
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Models will be downloaded to ~/.insightface/models/
# Copy the ONNX files to project models/ directory
```

#### Option C: Create placeholder for testing

For development/testing without real models:

```python
# Create dummy ONNX models (won't work for real inference)
# This is just to test code structure
```

---

## 🧪 Testing Phase 1

### Test 1: Quick Module Tests

Test individual components:

```bash
# Test detector
python core/detector.py

# Test aligner
python core/aligner.py

# Test embedder
python core/embedder.py

# Test basic recognition
python core/basic_recognition.py
```

### Test 2: Comprehensive Pipeline Test

Use the test script with a real face image:

```bash
python tests/test_phase1.py --image path/to/face_image.jpg --save-viz
```

**This will test:**
1. ✅ Face detection with bounding boxes and landmarks
2. ✅ Affine alignment with quality metrics
3. ✅ Embedding extraction with timing
4. ✅ Recognition pipeline with gallery
5. ✅ Performance benchmarking (FPS)

**Expected Output:**
```
==============================================================
PHASE 1: CORE PIPELINE TESTING
==============================================================
Image: path/to/face_image.jpg
Output directory: data/test_output
==============================================================

✓ RetinaFace detector initialized
  Model: retinaface_resnet50
  Input size: (640, 640)
  Min face size: 16px

Detected 1 faces in 42.35 ms

Face 1:
  BBox: [120.5, 80.3, 245.7, 210.6]
  Confidence: 0.997
  Landmarks: 5 points

✓ Saved visualization to: data/test_output/detection_result.jpg

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

---

## 📊 Performance Expectations

Based on RTX 3060 or similar mid-range GPU:

| Component | Expected Time | Notes |
|-----------|--------------|-------|
| Detection | 20-40 ms | Depends on image size and number of faces |
| Alignment | 0.5-2 ms | Per face |
| Embedding | 3-8 ms | Per face (ResNet100) |
| **Total Pipeline** | **25-50 ms** | **20-40 FPS** |

**Optimizations (for later phases):**
- Batch inference: Process multiple faces together
- TensorRT: 2-3× speedup
- MobileFaceNet: Lighter embedding model

---

## 🔬 Mathematical Verification

### Verify Detection

```python
from core.detector import RetinaFaceDetector
import cv2

detector = RetinaFaceDetector()
image = cv2.imread("test.jpg")
detections = detector.detect(image)

# Check landmarks
for det in detections:
    landmarks = det['landmarks']
    print(f"Landmarks: {len(landmarks)} points")  # Should be 5
    print(f"Order: left_eye, right_eye, nose, mouth_left, mouth_right")
```

### Verify Alignment

```python
from core.aligner import FaceAligner
import numpy as np

aligner = FaceAligner(output_size=112)

# Check template
print(f"Template shape: {aligner.template.shape}")  # (5, 2)
print(f"Template:\n{aligner.template}")

# Check alignment quality
is_good, error = aligner.check_alignment_quality(landmarks)
print(f"RMSE error: {error:.2f} pixels")  # Should be < 3.0 for good quality
```

### Verify Embedding

```python
from core.embedder import ArcFaceEmbedder
import numpy as np

embedder = ArcFaceEmbedder()
embedding = embedder.extract_embedding(aligned_face)

# Check properties
print(f"Dimension: {embedding.shape[0]}")  # Should be 512
print(f"Norm: {np.linalg.norm(embedding):.6f}")  # Should be ≈ 1.0

# Check similarity
emb1 = embedder.extract_embedding(face1)
emb2 = embedder.extract_embedding(face2)
similarity = embedder.compute_similarity(emb1, emb2)

print(f"Cosine similarity: {similarity:.4f}")
# Same person: > 0.6 (typically 0.65-0.85)
# Different person: < 0.4 (typically 0.0-0.35)
```

---

## 🐛 Troubleshooting

### Issue 1: "Failed to initialize ONNX session"

**Cause:** Missing or incorrect ONNX model files

**Solution:**
```bash
# Check model files exist
ls models/
# Should show:
#   retinaface_resnet50.onnx
#   arcface_resnet100.onnx

# Check file sizes (not 0 bytes)
# RetinaFace: ~30-50 MB
# ArcFace: ~200-250 MB
```

### Issue 2: "CUDA provider not available"

**Cause:** GPU not available or wrong ONNX Runtime version

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Install correct ONNX Runtime
pip uninstall onnxruntime onnxruntime-gpu
pip install onnxruntime-gpu  # For GPU
# OR
pip install onnxruntime  # For CPU only
```

### Issue 3: Low detection accuracy

**Possible causes:**
- Input image too small/large
- Poor lighting conditions
- Extreme face angles

**Solutions:**
```yaml
# Adjust in detector_config.yaml
detection:
  confidence_threshold: 0.5  # Lower for more detections (default: 0.7)
  min_face_size: 10  # Lower for smaller faces (default: 16)
```

### Issue 4: Poor alignment quality (RMSE > 5.0)

**Possible causes:**
- Inaccurate landmark detection
- Extreme pose/occlusion

**Solutions:**
- Use stricter detection threshold (fewer but better detections)
- Filter out low-quality detections
- Use multiple images for enrollment

### Issue 5: Low embedding similarity for same person

**Possible causes:**
- Poor alignment
- Different lighting/pose/expression
- Wrong model

**Solutions:**
- Check alignment quality first
- Use multiple images and average embeddings
- Verify model is trained for face recognition (not age/gender etc.)

---

## 📈 Quality Metrics

### Detection Quality
- **Target:** Detect 95%+ of faces with confidence > 0.7
- **Measure:** Run on test dataset and count detections
- **Acceptable:** ≥ 90% recall at 0.7 threshold

### Alignment Quality
- **Target:** RMSE < 3.0 pixels for 90%+ of faces
- **Measure:** Check reprojection error
- **Acceptable:** RMSE < 5.0 for most faces

### Embedding Quality
- **Target:**
  - Same person: cosine > 0.6 (mean ≈ 0.65)
  - Different person: cosine < 0.4 (mean ≈ 0.15)
- **Measure:** Compute pairwise similarities on labeled dataset
- **Acceptable:** Clear separation between same/different distributions

---

## ✅ Phase 1 Checklist

Before moving to Phase 2, verify:

- [ ] All dependencies installed
- [ ] ONNX models downloaded and placed correctly
- [ ] Detector works (detects faces with landmarks)
- [ ] Aligner works (produces 112×112 aligned crops)
- [ ] Embedder works (produces 512-D normalized vectors)
- [ ] Basic recognition pipeline works (can add to gallery and recognize)
- [ ] All tests pass: `python tests/test_phase1.py --image test.jpg`
- [ ] Performance is acceptable (≥ 20 FPS)

---

## 🎯 What's Next: Phase 2

**Phase 2: FAISS Vector Database + User Enrollment**

Will implement:
- ✨ FAISS index (Flat/HNSW) for fast similarity search
- ✨ User enrollment system (CLI/GUI)
- ✨ Multiple images per user with aggregation
- ✨ Database persistence (save/load)
- ✨ User management (add/delete/list)
- ✨ Quality control during enrollment

**Key improvements over Phase 1:**
- **Scalability:** Handle 100+ users efficiently
- **Speed:** Sub-millisecond search with FAISS
- **Robustness:** Multiple embeddings per user
- **Persistence:** Save/load gallery

---

## 📚 References

### Papers
1. **RetinaFace:** Deng et al. "RetinaFace: Single-stage Dense Face Localisation in the Wild" (CVPR 2020)
2. **ArcFace:** Deng et al. "ArcFace: Additive Angular Margin Loss for Deep Face Recognition" (CVPR 2019)

### Code References
- [biubug6/Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)
- [deepinsight/insightface](https://github.com/deepinsight/insightface)

### Mathematical Details
- See `Idea behind ArcFace.md` for complete mathematical derivations
- Angular margin loss gradient computation
- von Mises-Fisher embedding distribution
- Error propagation in alignment

---

## 💡 Tips for Testing

1. **Start with good quality images:**
   - Well-lit frontal faces
   - Resolution ≥ 640×480
   - Minimal occlusion

2. **Test with variations:**
   - Different angles (±30°)
   - Different lighting
   - With/without glasses
   - Different expressions

3. **Collect statistics:**
   - Detection rate
   - Alignment RMSE distribution
   - Embedding similarity distributions

4. **Visualize results:**
   - Always use `--save-viz` flag
   - Inspect aligned faces manually
   - Check landmark placement

---

## 📞 Support

If you encounter issues:

1. Check this document's troubleshooting section
2. Verify all files are in correct locations
3. Check configuration files for typos
4. Run tests with verbose logging
5. Check main `README.md` for general setup

---

**Phase 1 Status:** ✅ **COMPLETE AND READY FOR TESTING**

Test it thoroughly before moving to Phase 2! 🚀
