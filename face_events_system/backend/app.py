"""
Face Events Backend
====================
Architecture: Separated video pipeline from AI pipeline.

  Pipeline 1 — Video:  RTSP → /stream (MJPEG)  → browser <img>
  Pipeline 2 — AI:     RTSP → processor → POST /metadata → /ws/metadata → canvas overlay
  Pipeline 3 — Events: processor → POST /events  → /ws         → events panel

RTSP_URL env-var overrides the default stream source (load from .env if present).
"""
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pathlib import Path
import json
import os
import cv2
import asyncio
import threading
import time
from dotenv import load_dotenv

# Load .env file from project root
load_dotenv()

app = FastAPI()

# ── CORS: allow all origins in local dev ──────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Storage paths ─────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[1]
EVENT_FILE = str(BASE_DIR / "storage" / "events.json")
IMAGE_DIR = str(BASE_DIR / "storage" / "images")
os.makedirs(os.path.dirname(EVENT_FILE), exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

app.mount("/images", StaticFiles(directory=IMAGE_DIR), name="images")

# ── RTSP source ───────────────────────────────────────────────────────────────
RTSP_URL = os.environ.get("RTSP_URL", "rtsp://localhost:8554/live")
MAX_STREAM_WIDTH = int(os.environ.get("MAX_STREAM_WIDTH", "640"))
MAX_STREAM_HEIGHT = int(os.environ.get("MAX_STREAM_HEIGHT", "360"))


# ─────────────────────────────────────────────────────────────────────────────
# RTSPFrameBuffer
# Reads RTSP once in a background thread; all MJPEG consumers share one copy.
# AI processor connects separately (its own cv2.VideoCapture) — independent.
# ─────────────────────────────────────────────────────────────────────────────
class RTSPFrameBuffer:
    def __init__(self, url: str):
        self.url = url
        self.frame = None
        self.lock = threading.Lock()
        self.running = False
        self.connected = False
        self._thread = None

    def start(self):
        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        cap = None
        while self.running:
            if cap is None or not cap.isOpened():
                cap = cv2.VideoCapture(self.url)
                if not cap.isOpened():
                    self.connected = False
                    time.sleep(2)
                    continue
                self.connected = True
            try:
                ret, frame = cap.read()
                if ret:
                    # Resize to prevent memory overflow on high-res RTSP streams
                    h, w = frame.shape[:2]
                    if w > MAX_STREAM_WIDTH or h > MAX_STREAM_HEIGHT:
                        frame = cv2.resize(frame, (MAX_STREAM_WIDTH, MAX_STREAM_HEIGHT),
                                         interpolation=cv2.INTER_LINEAR)
                    with self.lock:
                        self.frame = frame
                else:
                    cap.release()
                    cap = None
                    self.connected = False
                    time.sleep(1)
            except Exception as e:
                # Handle memory or read errors gracefully
                print(f"⚠ Frame read error: {e}", flush=True)
                cap.release()
                cap = None
                self.connected = False
                time.sleep(1)
        if cap:
            cap.release()

    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None


frame_buffer = RTSPFrameBuffer(RTSP_URL)


@app.on_event("startup")
async def startup():
    # Clear stale events from previous sessions so the UI starts fresh.
    open(EVENT_FILE, "w").close()
    frame_buffer.start()


# ─────────────────────────────────────────────────────────────────────────────
# WebSocket client lists
# ─────────────────────────────────────────────────────────────────────────────
event_clients = []   # /ws       — receives saved face-event cards
meta_clients = []    # /ws/metadata — receives per-frame bbox+identity metadata

# Latest overlay from AI processor — drawn directly onto MJPEG frames
_overlay_lock = threading.Lock()
current_overlay: dict = {"tracks": [], "updated_at": 0.0}


async def _safe_send(ws: WebSocket, clients: list, payload: dict):
    """Send JSON to one websocket; remove it from clients list on error."""
    try:
        await ws.send_json(payload)
    except Exception:
        try:
            await ws.close()
        except Exception:
            pass
        if ws in clients:
            clients.remove(ws)


async def broadcast_event(event: dict):
    for ws in event_clients[:]:
        await _safe_send(ws, event_clients, event)


async def broadcast_metadata(meta: dict):
    for ws in meta_clients[:]:
        await _safe_send(ws, meta_clients, meta)


# ─────────────────────────────────────────────────────────────────────────────
# Video stream endpoint — MJPEG from RTSP (NO AI, raw frames only)
# Browser plays this; AI runs separately. Two pipelines, one CPU load each.
# ─────────────────────────────────────────────────────────────────────────────
MJPEG_FPS = int(os.environ.get("MJPEG_FPS", "15"))          # cap stream fps
MJPEG_QUALITY = int(os.environ.get("MJPEG_QUALITY", "60"))  # JPEG quality


def _draw_overlay(frame: cv2.typing.MatLike, overlay: dict) -> cv2.typing.MatLike:
    """Draw bbox+label for each active track onto the frame (in-place copy)."""
    age = time.time() - overlay["updated_at"]
    if age > 2.5 or not overlay["tracks"]:
        return frame
    frame = frame.copy()
    for track in overlay["tracks"]:
        bbox = track.get("bbox")
        if not bbox or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
        identity = track.get("identity") or "Unknown"
        known = identity != "Unknown"
        color = (0, 210, 80) if known else (42, 140, 255)  # BGR
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = identity if known else "Unknown"
        lx, ly = x1, max(0, y1 - 6)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(frame, (lx, ly - th - 2), (lx + tw + 4, ly + 2), color, -1)
        cv2.putText(frame, label, (lx + 2, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
    return frame


async def _mjpeg_generator():
    interval = 1.0 / MJPEG_FPS
    while True:
        t0 = time.time()
        frame = frame_buffer.get_frame()
        if frame is not None:
            with _overlay_lock:
                overlay_snap = dict(current_overlay)
            frame = _draw_overlay(frame, overlay_snap)
            ok, buf = cv2.imencode(
                ".jpg", frame,
                [cv2.IMWRITE_JPEG_QUALITY, MJPEG_QUALITY]
            )
            if ok:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + buf.tobytes()
                    + b"\r\n"
                )
        # honour rate-limit
        elapsed = time.time() - t0
        await asyncio.sleep(max(0, interval - elapsed))


@app.get("/stream")
async def video_stream():
    """MJPEG stream from RTSP source. Point browser <img src='/stream'> here."""
    return StreamingResponse(
        _mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/stream/status")
def stream_status():
    return {"connected": frame_buffer.connected, "rtsp_url": RTSP_URL}


# ─────────────────────────────────────────────────────────────────────────────
# Events API — face-event cards (images + metadata, saved by processor)
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/events")
def get_events():
    events = []
    if os.path.exists(EVENT_FILE):
        try:
            with open(EVENT_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        events.append(json.loads(line))
                    except Exception:
                        continue
        except Exception:
            return []
    return events[-50:]


@app.post("/events")
async def post_event(event: dict):
    """AI processor posts a face-event; backend persists and broadcasts."""
    os.makedirs(os.path.dirname(EVENT_FILE), exist_ok=True)
    with open(EVENT_FILE, "a") as f:
        f.write(json.dumps(event) + "\n")
    await broadcast_event(event)
    return {"status": "ok"}


@app.websocket("/ws")
async def events_ws(ws: WebSocket):
    """WebSocket for face-event cards (one per new identity)."""
    await ws.accept()
    event_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        if ws in event_clients:
            event_clients.remove(ws)


# ─────────────────────────────────────────────────────────────────────────────
# Live Metadata API — per-frame bbox+identity overlay (no images, just JSON)
# AI processor posts here every frame; browser canvas draws the boxes.
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/metadata")
async def post_metadata(payload: dict):
    """
    AI processor posts per-frame metadata:
    {
      "frame_idx": int,
      "timestamp": float,
      "tracks": [
        {"track_id": int, "bbox": [x1,y1,x2,y2],
         "identity": str, "confidence": float}
      ]
    }
    Stores latest overlay for MJPEG drawing + broadcasts to /ws/metadata clients.
    """
    global current_overlay
    with _overlay_lock:
        current_overlay = {
            "tracks": payload.get("tracks", []),
            "updated_at": time.time(),
        }
    await broadcast_metadata(payload)
    return {"status": "ok"}


@app.websocket("/ws/metadata")
async def metadata_ws(ws: WebSocket):
    """WebSocket for live per-frame bbox overlay metadata."""
    await ws.accept()
    meta_clients.append(ws)
    try:
        while True:
            await ws.receive_text()
    except Exception:
        pass
    finally:
        if ws in meta_clients:
            meta_clients.remove(ws)


@app.get("/env-check")
def env_check():
    """Verify that .env file is being loaded correctly."""
    return {
        "rtsp_url": RTSP_URL,
        "mjpeg_fps": MJPEG_FPS,
        "mjpeg_quality": MJPEG_QUALITY,
        "status": "✓ .env loaded successfully"
    }
