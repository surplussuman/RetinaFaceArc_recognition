"""
Face Events Processor
======================
Two modes:

  --mode live   (default)
      Read from RTSP stream continuously.
      Send per-frame metadata JSON to backend /metadata.
      No image encoding in the hot loop.  CPU = AI only.

  --mode file   <path>
      Read a video file, batch-process, save face-event images,
      then POST all events to /events at the end.

Usage
-----
# Live mode (RTSP)
python face_events_system/processor/run_processor.py --mode live

# File mode
python face_events_system/processor/run_processor.py --mode file --source "video/Sample 1.mp4"

# Tune thresholds
python face_events_system/processor/run_processor.py --mode live --threshold 0.38
"""
import sys
import os
import cv2
import uuid
import time
import json
import argparse
import requests
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.live_debug import crop_face
from core.vector_db import VectorDatabase
from core.video_recognition import create_video_recognizer
from core.frame_capture import FrameCaptureThread
from core.inference_engine import SharedInferenceEngine

BASE_DIR = Path(__file__).resolve().parents[1]   # face_events_system/
STORAGE_DIR = str(BASE_DIR / "storage" / "images")
EVENT_FILE = str(BASE_DIR / "storage" / "events.json")
os.makedirs(STORAGE_DIR, exist_ok=True)

BACKEND = os.environ.get("BACKEND", "http://127.0.0.1:9002")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Helpers
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def save_event_image(crop, identity, timestamp):
    filename = f"{identity}_{int(timestamp*1000)}_{uuid.uuid4().hex[:6]}.jpg"
    path = os.path.join(STORAGE_DIR, filename)
    # CCTV faces are often 40-80px wide; upscale to 200px so sidebar thumbnails are readable.
    h, w = crop.shape[:2]
    if w < 200:
        scale = 200 / w
        crop = cv2.resize(crop, (200, max(1, int(h * scale))),
                          interpolation=cv2.INTER_LANCZOS4)
    cv2.imwrite(path, crop, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return filename


def save_event_json(event):
    with open(EVENT_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")


def post_event(event):
    try:
        requests.post(f"{BACKEND}/events", json=event, timeout=1.0)
    except Exception:
        pass


def post_metadata(payload):
    """Send per-frame bbox+identity metadata to backend (tiny JSON, not frames)."""
    try:
        requests.post(f"{BACKEND}/metadata", json=payload, timeout=0.5)
    except Exception:
        pass


def load_db():
    vdb = VectorDatabase(embedding_dim=512)
    vdb.load("data/face_database")
    return vdb


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Live mode â€” reads RTSP, sends ONLY metadata per-frame
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_live(source: str, threshold: float, skip_frames: int):
    """
    Live RTSP processing loop.

    Architecture:
        RTSP â†’ process_frame() â†’ POST /metadata (bbox+identity JSON)
        NO image encoding, NO file I/O in the hot loop.
        CPU budget = AI only.
    """
    vdb = load_db()
    # One shared engine (all-core ONNX sessions). Ready to serve multiple streams;
    # here it backs the single live stream.
    engine = SharedInferenceEngine()
    recognizer = create_video_recognizer(
        vdb,
        recognition_threshold=threshold,
        process_every_n_frames=skip_frames,
        engine=engine,
    )

    print(f"\nðŸ”´ Live mode â€” connecting to {source}")
    # Async capture: a dedicated thread reads RTSP into a bounded queue (drop stale
    # frames to stay real-time). The inference loop no longer blocks on cap.read().
    capture = FrameCaptureThread(source, queue_size=4, drop_stale=True)
    if not capture.isOpened():
        print(f"ERROR: Cannot open source: {source}")
        sys.exit(1)
    capture.start()

    fps = capture.fps or 25
    frame_idx = 0
    # {track_id: wall_clock_time_of_last_event}
    # Allow re-posting the same track after EVENT_COOLDOWN seconds so that
    # repeated appearances (e.g. looping RTSP test video) each get a new card.
    EVENT_COOLDOWN = 30.0
    seen_tracks: dict = {}
    print(f"\u2713 Connected \u2014 FPS={fps:.1f}  threshold={threshold}  skip={skip_frames}")
    print("  Press Ctrl+C to stop\n")
    
    # Debug logging every N frames (set DEBUG_FRAMES env to enable)
    debug_every = int(os.environ.get("DEBUG_FRAMES", "0"))
    if debug_every > 0:
        print(f"  DEBUG: logging every {debug_every} frames\n")

    try:
        while True:
            frame_idx, frame = capture.read(timeout=2.0)
            if frame is None:
                # No frame within timeout (reconnecting or stalled) \u2014 keep waiting.
                continue
            if capture.consume_reconnect():
                print("\u26a0  Stream reconnected \u2014 resetting track state")
                seen_tracks.clear()  # let faces fire again after reconnect

            # Debug: log frame info
            if debug_every > 0 and frame_idx % debug_every == 0:
                h, w = frame.shape[:2]
                print(f"[frame {frame_idx}] input frame size: {w}×{h}", flush=True)

            result = recognizer.process_frame(frame, frame_idx)
            
            # Debug: log recognition result
            if debug_every > 0 and frame_idx % debug_every == 0:
                print(f"  result: skipped={result.get('skipped')}, "
                      f"motion_gated={result.get('motion_gated')}, "
                      f"tracks={len(result.get('active_tracks', []))}", flush=True)

            if not result.get("skipped") and not result.get("motion_gated"):
                active = result.get("active_tracks", [])

                if active:
                    # â”€â”€ Per-frame metadata (live bbox overlay) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    tracks_meta = []
                    for track in active:
                        bbox = track.get("bbox")
                        if bbox is None:
                            continue
                        bbox_list = [int(v) for v in bbox[:4]]
                        tracks_meta.append({
                            "track_id":   int(track["track_id"]),
                            "bbox":       bbox_list,
                            "identity":   track.get("identity") or "Unknown",
                            "confidence": float(track.get("confidence") or 0.0),
                        })

                    if tracks_meta:
                        post_metadata({
                            "frame_idx": frame_idx,
                            "timestamp": frame_idx / fps,
                            "tracks":    tracks_meta,
                        })

                    # â”€â”€ One-time event per new track â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
                    now_wall = time.time()
                    for track in active:
                        tid = track["track_id"]
                        last = seen_tracks.get(tid, 0.0)
                        if now_wall - last < EVENT_COOLDOWN:
                            continue
                        seen_tracks[tid] = now_wall

                        bbox = track.get("bbox")
                        if bbox is None:
                            continue
                        crop = crop_face(frame, bbox)
                        if crop.size == 0:
                            continue

                        identity = track.get("identity") or "Unknown"
                        confidence = float(track.get("confidence") or 0.0)
                        timestamp = frame_idx / fps
                        filename = save_event_image(crop, identity, timestamp)

                        event = {
                            "image":      filename,
                            "identity":   identity,
                            "confidence": confidence,
                            "time":       timestamp,
                            "frame":      frame_idx,
                        }
                        # POST to backend (non-blocking best-effort)
                        post_event(event)
                        print(f"EVENT: {identity}  conf={confidence:.2f}  t={timestamp:.2f}s")

    except KeyboardInterrupt:
        print("\nðŸ›‘ Stopped.")
    finally:
        capture.stop()


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# File mode â€” batch-process video, save images, POST events at end
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def run_file(source: str, threshold: float, skip_frames: int):
    vdb = load_db()
    engine = SharedInferenceEngine()
    recognizer = create_video_recognizer(
        vdb,
        recognition_threshold=threshold,
        process_every_n_frames=skip_frames,
        engine=engine,
    )

    # Async capture (file mode): never drop frames; block when buffer full so every
    # frame is processed and output duration stays correct. EOF -> (None, None).
    capture = FrameCaptureThread(source, queue_size=4, drop_stale=False)
    if not capture.isOpened():
        print(f"ERROR: Cannot open video: {source}")
        sys.exit(1)
    capture.start()

    fps = capture.fps or 25
    total = capture.total_frames
    print(f"\nðŸ“‚ File mode â€” {Path(source).name}  ({total} frames @ {fps:.1f} FPS)")

    seen_tracks = set()
    start_time = time.time()
    frame_idx = -1  # last successfully-read index (stays -1 if video is empty)

    while True:
        idx, frame = capture.read(timeout=5.0)
        if frame is None:
            break
        frame_idx = idx

        result = recognizer.process_frame(frame, frame_idx)

        if not result.get("skipped") and not result.get("motion_gated"):
            for track in result.get("active_tracks", []):
                tid = track["track_id"]
                if tid in seen_tracks:
                    continue
                seen_tracks.add(tid)

                bbox = track.get("bbox")
                if bbox is None:
                    continue
                crop = crop_face(frame, bbox)
                if crop.size == 0:
                    continue

                identity = track.get("identity") or "Unknown"
                confidence = float(track.get("confidence") or 0.0)
                timestamp = frame_idx / fps
                filename = save_event_image(crop, identity, timestamp)

                event = {
                    "image":      filename,
                    "identity":   identity,
                    "confidence": confidence,
                    "time":       timestamp,
                    "frame":      frame_idx,
                }
                # Post immediately â€” events fire at most once per new track,
                # so the synchronous POST cost is negligible here.
                save_event_json(event)
                post_event(event)
                print(f"EVENT: {identity}  conf={confidence:.2f}  t={timestamp:.2f}s")

    capture.stop()
    elapsed = time.time() - start_time
    frames_done = frame_idx + 1
    fps_out = frames_done / elapsed if elapsed > 0 else 0
    print(f"\n\u2713 Processing complete in {elapsed:.2f}s  ({fps_out:.1f} FPS)")
    events_sent = len(seen_tracks)
    if events_sent:
        print(f"\u2713 {events_sent} events posted in real-time.")
    else:
        print("\u26a0 No events detected.")


# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
# Entry point
# â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

def main():
    parser = argparse.ArgumentParser(
        description="Face events processor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode", choices=["live", "file"], default="file",
        help="live = RTSP stream (metadata only); file = video file (batch)",
    )
    parser.add_argument(
        "--source",
        default=None,
        help=(
            "For --mode live: RTSP URL (default rtsp://localhost:8554/live). "
            "For --mode file: path to video file (default video/Sample 2.mp4)."
        ),
    )
    parser.add_argument("--threshold", type=float, 
                      default=float(os.environ.get("DETECTION_THRESHOLD", "0.5")),
                      help="Face detection confidence threshold (0.0-1.0)")
    parser.add_argument("--skip-frames", type=int, default=3)
    args = parser.parse_args()

    if args.mode == "live":
        source = args.source or os.environ.get("RTSP_URL", "rtsp://localhost:8554/live")
        run_live(source, args.threshold, args.skip_frames)
    else:
        source = args.source or "video/Sample 2.mp4"
        run_file(source, args.threshold, args.skip_frames)


if __name__ == "__main__":
    main()

