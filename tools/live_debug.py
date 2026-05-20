"""
Live Debug CLI — two modes for diagnosing display vs compute performance
=======================================================================

MODE A  --mode preview
    Processes video and shows annotated frames in a native OpenCV window.
    Proves the pipeline speed without Streamlit overhead.
    Controls: press 'q' to quit, Space to pause.

MODE B  --mode events
    Processes video silently. When a face is detected OR recognised, pops
    an OpenCV window showing only the cropped face + identity label for
    ~2–3 seconds, then closes it automatically.
    Models O(N_events) instead of O(N_frames) display cost.

Usage examples
--------------
# Mode A — live annotated preview
python tools/live_debug.py --video "video/Sample 1.mp4" --mode preview

# Mode B — event popup for detected faces only
python tools/live_debug.py --video "video/Sample 1.mp4" --mode events

# Mode B — popups for RECOGNISED faces only (skip Unknowns)
python tools/live_debug.py --video "video/Sample 1.mp4" --mode events --known-only

# Tune recognition + frame skip
python tools/live_debug.py --video "video/Sample 1.mp4" --mode preview --threshold 0.35 --skip-frames 3
"""

import sys
import time
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import cv2
import numpy as np
from datetime import datetime

from core.vector_db import VectorDatabase
from core.video_recognition import create_video_recognizer


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def load_db(project_root: Path) -> VectorDatabase:
    db_path = project_root / 'data' / 'face_database'
    if not Path(str(db_path) + '.index').exists():
        print("ERROR: No face database found.  Enroll users first:")
        print("  python tools/enroll_multi_quality.py --user-id <Name> --directory photos/")
        sys.exit(1)
    vdb = VectorDatabase(embedding_dim=512)
    vdb.load(db_path)
    users = [u['user_id'] for u in vdb.get_all_users()]
    print(f"✓ Database loaded  |  users: {', '.join(users)}")
    return vdb


def crop_face(frame: np.ndarray, bbox) -> np.ndarray:
    """Return a tight face crop (with small padding) from frame."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in bbox[:4]]
    pad = max(10, int((x2 - x1) * 0.15))
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w, x2 + pad)
    y2 = min(h, y2 + pad)
    return frame[y1:y2, x1:x2].copy()


def make_event_card(crop: np.ndarray, identity: str, confidence: float,
                    timestamp: float, frame_idx: int) -> np.ndarray:
    """Build a tidy event card image: face crop + info bar."""
    target_size = 300
    h, w = crop.shape[:2]
    scale = target_size / max(h, w)
    new_w, new_h = max(1, int(w * scale)), max(1, int(h * scale))
    face_img = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # Info bar below face
    bar_h = 70
    card = np.zeros((new_h + bar_h, max(new_w, 300), 3), dtype=np.uint8)

    # Place face (centred horizontally)
    x_off = (card.shape[1] - new_w) // 2
    card[:new_h, x_off:x_off + new_w] = face_img

    # Info bar
    bar_y = new_h
    if identity and identity != "Unknown":
        color = (0, 200, 80)
        id_text = f"{identity}"
        conf_text = f"similarity={confidence:.3f}"
    else:
        color = (40, 40, 220)
        id_text = "Unknown"
        conf_text = f"frame={frame_idx}"

    cv2.rectangle(card, (0, bar_y), (card.shape[1], card.shape[0]), color, -1)
    cv2.putText(card, id_text,   (8, bar_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    cv2.putText(card, conf_text, (8, bar_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220,220,220), 1)
    ts_text = f"t={timestamp:.2f}s"
    cv2.putText(card, ts_text,   (8, bar_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4,  (190,190,190), 1)

    return card


# ──────────────────────────────────────────────────────────────────────────────
# Mode A — annotated preview
# ──────────────────────────────────────────────────────────────────────────────

def run_preview(video_path: Path, recognizer, max_frames):
    """Process video, render annotated frames in a native OpenCV window."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Video: {width}×{height} @ {fps_src:.1f} FPS  |  {total_frames} frames")

    win_name = "Live Debug — Mode A (press q=quit  space=pause)"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, min(width, 960), min(height, 540))

    frame_idx = 0
    paused = False
    frame_times = []
    start_wall = time.perf_counter()
    last_annotated = None

    print("\n[ Mode A  |  OpenCV Preview  |  press 'q' to quit ]\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret or (max_frames and frame_idx >= max_frames):
                break

            t0 = time.perf_counter()
            result = recognizer.process_frame(frame, frame_idx)
            dt_ms = (time.perf_counter() - t0) * 1000
            frame_times.append(dt_ms)

            if result.get('skipped') or result.get('motion_gated'):
                annotated = last_annotated if last_annotated is not None else frame
            else:
                annotated = recognizer.draw_annotations(frame, result)
                last_annotated = annotated

            # HUD overlay: FPS + frame counter
            elapsed = time.perf_counter() - start_wall
            live_fps = frame_idx / elapsed if elapsed > 0 else 0
            hud = f"Frame {frame_idx}/{total_frames}  |  {dt_ms:.0f}ms  |  ~{live_fps:.1f} FPS"
            cv2.putText(annotated, hud, (10, annotated.shape[0] - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(win_name, annotated)
            frame_idx += 1

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("User quit.")
            break
        elif key == ord(' '):
            paused = not paused
            print("PAUSED" if paused else "RESUMED")

    cap.release()
    cv2.destroyAllWindows()

    total_wall = time.perf_counter() - start_wall
    avg_ms = np.mean(frame_times) if frame_times else 0
    print(f"\n{'='*60}")
    print(f"MODE A COMPLETE")
    print(f"  Frames processed : {frame_idx}")
    print(f"  Wall time        : {total_wall:.2f}s")
    print(f"  Actual FPS       : {frame_idx/total_wall:.1f}")
    print(f"  Avg process time : {avg_ms:.1f}ms/frame")
    print(f"{'='*60}\n")
    print("KEY INSIGHT: compare this FPS vs Streamlit. If roughly equal → bottleneck is compute.")
    print("If MUCH faster → Streamlit overhead is the bottleneck (expected).")


# ──────────────────────────────────────────────────────────────────────────────
# Mode B — event popup
# ──────────────────────────────────────────────────────────────────────────────

def run_events(video_path: Path, recognizer, max_frames, known_only: bool,
               popup_duration: float):
    """
    Process video; show a face-crop popup each time a NEW identity is detected.
    The popup stays open for `popup_duration` seconds then auto-closes.
    O(N_events) display cost — proves event-driven design works.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"ERROR: Cannot open video: {video_path}")
        return

    fps_src = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  Video: {int(cap.get(3))}×{int(cap.get(4))} @ {fps_src:.1f} FPS  |  {total_frames} frames")
    print(f"  known_only={known_only}  |  popup_duration={popup_duration}s")
    print("\n[ Mode B  |  Event Popup  |  running silently — popups appear on detection ]\n")

    frame_idx = 0
    last_annotated = None
    events = []                      # list of event dicts for summary
    seen_track_ids: set = set()      # avoid repeated popup for same track
    open_windows: list = []          # (window_name, close_at_time)

    start_wall = time.perf_counter()

    # Silent console progress
    def _progress(fi, total):
        if fi % 50 == 0:
            pct = fi / total * 100
            print(f"  ... {fi}/{total} ({pct:.0f}%)", end='\r', flush=True)

    while True:
        # ── Close expired popups ──────────────────────────────────────────
        now = time.perf_counter()
        still_open = []
        for wname, close_at in open_windows:
            if now >= close_at:
                cv2.destroyWindow(wname)
            else:
                still_open.append((wname, close_at))
        open_windows = still_open
        # pump events so windows actually render
        cv2.waitKey(1)

        # ── Read next frame ───────────────────────────────────────────────
        ret, frame = cap.read()
        if not ret or (max_frames and frame_idx >= max_frames):
            break

        result = recognizer.process_frame(frame, frame_idx)
        _progress(frame_idx, total_frames)

        if not result.get('skipped') and not result.get('motion_gated'):
            for track in result.get('active_tracks', []):
                tid = track['track_id']
                identity = track.get('identity') or "Unknown"
                confidence = track.get('confidence', 0.0)
                timestamp = frame_idx / fps_src

                if known_only and identity == "Unknown":
                    continue

                # Show popup only once per track (first detection)
                if tid in seen_track_ids:
                    continue
                seen_track_ids.add(tid)

                bbox = track.get('bbox')
                if bbox is None:
                    continue

                crop = crop_face(frame, bbox)
                if crop.size == 0:
                    continue

                card = make_event_card(crop, identity, confidence, timestamp, frame_idx)

                # Unique window name per event
                wname = f"EVENT  {identity}  t={timestamp:.1f}s  track={tid}"
                cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
                cv2.imshow(wname, card)
                open_windows.append((wname, time.perf_counter() + popup_duration))

                events.append({
                    'frame': frame_idx,
                    'timestamp': timestamp,
                    'track_id': tid,
                    'identity': identity,
                    'confidence': confidence,
                })

                print(f"\n  ★ EVENT  [{identity}]  conf={confidence:.3f}  t={timestamp:.2f}s  frame={frame_idx}")

        frame_idx += 1

    # ── Close remaining windows ───────────────────────────────────────────
    cap.release()
    time.sleep(0.5)   # let last popup render briefly
    cv2.destroyAllWindows()

    total_wall = time.perf_counter() - start_wall

    print(f"\n\n{'='*60}")
    print(f"MODE B COMPLETE")
    print(f"  Total frames  : {frame_idx}")
    print(f"  Wall time     : {total_wall:.2f}s")
    print(f"  Actual FPS    : {frame_idx / total_wall:.1f}")
    print(f"  Events fired  : {len(events)}")
    if total_frames > 0:
        event_rate = len(events) / total_frames * 100
        print(f"  Event rate    : {event_rate:.1f}% of frames  ({100-event_rate:.1f}% UI work saved)")
    print(f"\n  Event log:")
    for ev in events:
        print(f"    frame={ev['frame']:5d}  t={ev['timestamp']:6.2f}s  "
              f"track={ev['track_id']}  identity={ev['identity']}  conf={ev['confidence']:.3f}")
    print(f"{'='*60}\n")
    print("KEY INSIGHT: UI only worked for these", len(events), "events, not", frame_idx, "frames.")
    print(f"  Reduction ratio: {len(events)}/{frame_idx} = {len(events)/max(frame_idx,1)*100:.1f}%")


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Live debug: prove compute speed and event-driven display',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('--video',   required=True,          help='Input video file')
    parser.add_argument('--mode',    required=True,          choices=['preview', 'events'],
                        help='preview = annotated OpenCV window; events = face popup on detection')
    parser.add_argument('--threshold', type=float, default=0.4, help='Recognition threshold (default 0.4)')
    parser.add_argument('--skip-frames', type=int, default=3,   help='Process every N frames (default 3)')
    parser.add_argument('--max-frames',  type=int, default=None, help='Cap total frames (default all)')
    parser.add_argument('--known-only',  action='store_true',    help='[events] only popup for recognised faces')
    parser.add_argument('--popup-secs',  type=float, default=2.5,
                        help='[events] seconds each popup stays open (default 2.5)')
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"ERROR: video not found: {video_path}")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"LIVE DEBUG  —  mode={args.mode.upper()}")
    print(f"  video      : {video_path.name}")
    print(f"  threshold  : {args.threshold}")
    print(f"  skip_frames: {args.skip_frames}")
    print(f"{'='*60}\n")

    vdb = load_db(project_root)
    recognizer = create_video_recognizer(
        vdb,
        recognition_threshold=args.threshold,
        process_every_n_frames=args.skip_frames,
    )

    if args.mode == 'preview':
        run_preview(video_path, recognizer, args.max_frames)
    else:
        run_events(video_path, recognizer, args.max_frames, args.known_only, args.popup_secs)


if __name__ == '__main__':
    main()
