# Phase 3 Installation & Setup Guide

## Prerequisites

You should already have Phase 1 & 2 dependencies installed. If not:

```bash
pip install torch torchvision onnxruntime opencv-python numpy Pillow faiss-cpu tqdm
```

## Phase 3 New Dependencies

Install Streamlit and pandas for the web interface:

```bash
pip install streamlit pandas
```

That's it! Only 2 new packages needed.

## Verification

Check installation:

```bash
python -c "import streamlit; print(f'Streamlit {streamlit.__version__} installed successfully')"
python -c "import pandas; print(f'Pandas {pandas.__version__} installed successfully')"
```

Expected output:
```
Streamlit 1.x.x installed successfully
Pandas 2.x.x installed successfully
```

## Quick Test

Run the test script to verify everything works:

```bash
python tools/test_video_system.py
```

This will:
1. Check your database is loaded
2. Create a test video (if needed)
3. Process first 100 frames
4. Display statistics
5. Verify system is working

## Launch Streamlit

```bash
streamlit run app.py
```

Your browser should open automatically to http://localhost:8501

If not, manually navigate to:
- Local: http://localhost:8501
- Network: http://<your-ip>:8501

## Troubleshooting

### Issue: ModuleNotFoundError: No module named 'streamlit'

**Solution**:
```bash
pip install streamlit
```

### Issue: Port 8501 already in use

**Solution**:
```bash
streamlit run app.py --server.port 8502
```

### Issue: Streamlit shows "No module named 'core'"

**Solution**: Make sure you're running from the project directory:
```bash
cd "d:\Project\Face Recog\project"
streamlit run app.py
```

### Issue: Database not found in Streamlit

**Solution**: Enroll users first:
```bash
python tools/enroll_multi_quality.py --user-id "YourName" --directory "path/to/photos/"
```

## What's Next?

1. **Enroll users**: Use the web interface or CLI
2. **Upload video**: Test with your CCTV footage
3. **Review results**: Download annotated video + CSV log
4. **Tune parameters**: Adjust threshold/frame skip as needed

See `PHASE3_COMPLETE.md` for complete usage guide!
