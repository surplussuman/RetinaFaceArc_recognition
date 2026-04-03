# RetinaFaceArc Recognition — Agent Guidelines

## Project Overview

Production-grade face recognition pipeline targeting crowded/CCTV environments. Stack: RetinaFace (detection) → ArcFace (embedding) → FAISS (search) → ByteTrack-style temporal tracking.

## Build & Run

```powershell
# Activate env (already created at ./env/)
.\env\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Download ONNX models (required before running anything)
python scripts\download_models.py

# Run main UI
streamlit run app.py                 # Main video recognition  (port 8501)
streamlit run zone_config_app.py     # Zone/ROI configuration UI
```

## Testing

```powershell
python tests\test_phase1.py --image path/to/image.jpg --save-viz   # Core pipeline
python tests\test_phase2.py                                          # DB + enrollment
```

## Enrollment (Users must be enrolled before recognition works)

```powershell
# Recommended — generates 5 quality variants per photo (25 embeddings/user)
python tools\enroll_multi_quality.py --user-id "Name" --directory "path/to/photos/" --replace

# Basic enrollment
python tools\enroll_user.py --user-id "Name" --images img1.jpg img2.jpg
```

## Architecture

```
Frame → RetinaFaceDetector (ONNX) → FaceAligner (5-pt affine → 112×112) →
ArcFaceEmbedder (ONNX → 512-D L2-norm) → VectorDatabase (FAISS FlatL2) →
FaceRecognitionPipeline → VideoRecognizer (Ghost Tracking + temporal smoothing)
```

| Module | File | Key Class |
|--------|------|-----------|
| Detection | `core/detector.py` | `RetinaFaceDetector` |
| Alignment | `core/aligner.py` | `FaceAligner` |
| Embedding | `core/embedder.py` | `ArcFaceEmbedder` |
| Vector DB | `core/vector_db.py` | `VectorDatabase` |
| Pipeline | `core/recognition.py` | `FaceRecognitionPipeline` |
| Video | `core/video_recognition.py` | `VideoRecognizer` |
| Enrollment | `core/enrollment.py` | `EnrollmentSystem` |
| Multi-quality | `core/multi_quality_enrollment.py` | `MultiQualityEnroller` |

## Configuration

All configs live in `config/`. Components load their config at `__init__` via `yaml.safe_load`:

| File | Controls |
|------|---------|
| `config/system_config.yaml` | Thresholds, ghost tracking, ByteTrack, zones, attendance |
| `config/detector_config.yaml` | RetinaFace model path, NMS, min face size |
| `config/embedder_config.yaml` | ArcFace model path, quality checks, head angle limits |
| `config/faiss_config.yaml` | Index type, search k, DB paths |

**Key thresholds** (cosine similarity, range 0–1):
- `threshold: 0.4` — standard balanced recognition
- `strict_threshold: 0.5` — high-security
- `lenient_threshold: 0.35` — low-quality CCTV footage

**Expected model paths** (downloaded by `scripts/download_models.py`):
- `models/retinaface_resnet50_int8.onnx` — primary detector
- `models/arcface_resnet100_int8.onnx` — primary embedder
- `models/retinaface_mobilenet.onnx` / `models/arcface_mobilefacenet.onnx` — lightweight alternatives

## Critical Conventions

### FAISS + L2 → Cosine Similarity
All embeddings are L2-normalized (`||f|| = 1`). FAISS uses `IndexFlatL2`, but similarity is converted:
```python
# cosine_similarity = 1 - (L2_distance² / 2)
similarities = 1 - (distances[0] ** 2) / 2
```
Never change the index type without updating this formula everywhere.

### ONNX Inference — CPU Only
Both models run on `CPUExecutionProvider`. The GPU (RTX 3050 Laptop, 4 GB VRAM) causes OOM when all models load simultaneously. Do not switch to `CUDAExecutionProvider` without testing memory:
```python
providers=['CPUExecutionProvider']   # intentional — GPU OOM on 4GB VRAM
```

### FAISS CPU Only (pip)
`faiss-gpu` has no PyPI wheels for Python 3.13. Install `faiss-cpu` only via pip. For GPU FAISS, use `conda install -c conda-forge faiss-gpu`. The code in `core/vector_db.py` works with both.

### Ghost Tracking (Performance-critical)
`VideoRecognizer` caches identity per tracked face and only re-runs the embedder every 30 frames (`recognition_interval: 30`). This provides ~30× speedup. Do not remove the cache without a performance justification — raw embedding inference is 70–80 ms/frame.

### Multi-Quality Enrollment
`MultiQualityEnroller` generates 5 synthetic quality variants per source photo (blur, low-res, poor lighting, severe, original). This is deliberate — it ensures recognition works on degraded CCTV footage. Prefer `tools/enroll_multi_quality.py` over `tools/enroll_user.py`.

### Deleting Users from FAISS
`IndexFlatL2` does not support deletion. The only way to remove a user is to rebuild the entire index. `VectorDatabase.delete_user()` already does this — never attempt direct index manipulation.

### Quality Levels
```python
# Used consistently across all core modules
quality_levels = {"EXCELLENT": 3, "GOOD": 2, "FAIR": 1, "POOR": 0}
```
Minimum quality for enrollment (default) is FAIR. POOR faces are always skipped.

## Database Persistence

Face database is stored as two paired files — always keep them in sync:
- `data/face_database.index` — FAISS binary index
- `data/face_database.pkl` — user IDs + metadata

Losing either file without the other = unusable database.

## Agent Behavior Rules

- **Never create a new file** when an existing file can be edited to satisfy the need.
- **Never create markdown documentation files** (`.md`) unless explicitly asked by the user. This includes summaries, changelogs, guides, and README updates.
- **Do not add docstrings, comments, or type annotations** to code you didn't change.
- **Do not refactor, reformat, or rename** symbols unless that is the explicit task.
- **Edit config YAML files** (`config/*.yaml`) for threshold/parameter changes — do not hardcode values in Python.
- **One occurrence rule**: When using `replace_string_in_file`, include 3–5 lines of unchanged context before and after so the match is unambiguous.

## Docs Reference

Detailed documentation is in `docs/`. Do not duplicate it here.
- `docs/PARAMETERS_GUIDE.md` — full config parameter reference
- `docs/TECHNICAL_DOCUMENTATION.md` — math + architecture deep-dive
- `docs/PROJECT_STATUS.md` — known limitations, performance numbers
- `QUICKSTART.md` — end-to-end setup walkthrough
