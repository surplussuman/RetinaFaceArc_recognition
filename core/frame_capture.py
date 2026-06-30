"""
Async frame capture — pipeline parallelism for the inference loop.

Problem: a serial loop does
    [cap.read() BLOCKS ~33ms] -> [detect] -> [embed] -> [track] -> [output]
so the inference thread idles during every I/O read.

Fix: a dedicated capture thread fills a small bounded queue; the inference loop
pulls decoded frames with (near) zero wait.

    T_frame_serial   = T_io + T_detect + T_embed + T_track
    T_frame_parallel = max(T_io, T_detect + T_embed + T_track)

Two modes:
    drop_stale=True   (live/RTSP)  -> keep only the freshest frames; drop when full
                                       so we never fall behind real time.
    drop_stale=False  (file)       -> never drop; block when full (backpressure) so
                                       every frame is processed and output duration
                                       stays correct. EOF delivers a (None, None)
                                       sentinel.

RTSP reconnect is handled *inside* the thread. The main loop can observe a
reconnect via `reconnect_flag` (a threading.Event) and reset its own per-stream
state (e.g. seen_tracks) when it fires.
"""

import threading
import queue
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class FrameCaptureThread(threading.Thread):
    def __init__(self,
                 source,
                 queue_size: int = 4,
                 drop_stale: bool = True,
                 reconnect_delay: float = 2.0):
        super().__init__(daemon=True)
        self.source = source
        self.drop_stale = drop_stale
        self.reconnect_delay = reconnect_delay

        self._q: "queue.Queue[Tuple[Optional[int], Optional[np.ndarray]]]" = queue.Queue(maxsize=queue_size)
        self._stop = threading.Event()
        self.reconnect_flag = threading.Event()
        self._frame_idx = 0

        self.cap = cv2.VideoCapture(source)

        # Expose source properties so callers don't need their own VideoCapture.
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

    def isOpened(self) -> bool:
        return self.cap.isOpened()

    # ------------------------------------------------------------------ thread
    def run(self):
        while not self._stop.is_set():
            ret, frame = self.cap.read()

            if not ret:
                if not self.drop_stale:
                    # File mode: natural EOF -> sentinel, then stop.
                    self._put_blocking((None, None))
                    break
                # Live mode: stream lost -> reconnect inside the thread.
                self.cap.release()
                time.sleep(self.reconnect_delay)
                self.cap = cv2.VideoCapture(self.source)
                self.reconnect_flag.set()
                continue

            item = (self._frame_idx, frame)
            self._frame_idx += 1

            if self.drop_stale:
                self._put_drop_oldest(item)
            else:
                self._put_blocking(item)

    def _put_drop_oldest(self, item):
        """Live: keep newest frame; if full, evict the oldest and insert."""
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()  # drop one stale frame
            except queue.Empty:
                pass
            try:
                self._q.put_nowait(item)
            except queue.Full:
                pass

    def _put_blocking(self, item):
        """File: never drop; block until space (or stop requested)."""
        while not self._stop.is_set():
            try:
                self._q.put(item, timeout=0.5)
                return
            except queue.Full:
                continue

    # ------------------------------------------------------------------ reader
    def read(self, timeout: float = 2.0) -> Tuple[Optional[int], Optional[np.ndarray]]:
        """
        Return (frame_idx, frame). Returns (None, None) on EOF (file mode) or if
        no frame arrives within `timeout` seconds.
        """
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return (None, None)

    def consume_reconnect(self) -> bool:
        """True once after each reconnect; clears the flag. Lets the caller reset state."""
        if self.reconnect_flag.is_set():
            self.reconnect_flag.clear()
            return True
        return False

    def stop(self):
        self._stop.set()
        # Drain so a blocked _put_blocking can exit promptly.
        try:
            while True:
                self._q.get_nowait()
        except queue.Empty:
            pass
        if self.cap is not None:
            self.cap.release()
