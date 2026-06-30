"""
SharedInferenceEngine — one set of ONNX sessions serving many streams.

Problem
-------
If each of N streams builds its own detector/embedder session capped at K threads,
the box runs N×K threads fighting for cores (with contention). Worse on a 32-vCPU
server: 2 streams × 4 threads = 8 cores, 24 idle.

Fix
---
ONE detector session + ONE embedder session, each free to use all cores, shared
across every stream. ort.InferenceSession.run() is re-entrant (releases the GIL and
is safe to call from multiple threads as long as each call owns its input/output
buffers — which it does), so streams can call concurrently without a lock.

Cross-stream embedding batch
----------------------------
Embedding IS batchable (all faces are 112×112). embed_batch() coalesces face crops
arriving from different streams within a short collect window into a single ONNX
call. Batch cost scales sublinearly (batch_N ≈ batch_1 × N^0.6).

    N stream threads -> embed queue -> BatchWorker -> one session.run -> N futures

Detection is per-frame (frames differ in size, not batchable the same way) so
detect_batch() simply runs each frame through the shared session.
"""

import threading
import queue
import time
from concurrent.futures import Future
from typing import List, Dict, Optional

import numpy as np

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder


class SharedInferenceEngine:
    def __init__(self,
                 detector_config: str = "config/detector_config.yaml",
                 embedder_config: str = "config/embedder_config.yaml",
                 collect_window_ms: float = 5.0,
                 max_batch_size: int = 16,
                 enable_embed_coalescing: bool = True,
                 use_session_lock: bool = False):
        print("Initializing SharedInferenceEngine (shared ONNX sessions)...")
        self.detector = RetinaFaceDetector(detector_config)
        self.aligner = FaceAligner()
        self.embedder = ArcFaceEmbedder(embedder_config)

        self.collect_window_s = collect_window_ms / 1000.0
        self.max_batch_size = max_batch_size
        # Coalescing only helps when the model has a dynamic batch axis.
        self.enable_embed_coalescing = enable_embed_coalescing and self.embedder.supports_dynamic_batch

        # ORT is re-entrant; default lock-free. Flip use_session_lock=True only if a
        # specific backend proves unsafe under concurrency.
        self._use_lock = use_session_lock
        self._det_lock = threading.Lock()
        self._emb_lock = threading.Lock()

        # Cross-stream embedding coalescer
        self._emb_queue: "queue.Queue" = queue.Queue()
        self._shutdown = threading.Event()
        self._worker: Optional[threading.Thread] = None
        if self.enable_embed_coalescing:
            self._worker = threading.Thread(target=self._embed_worker,
                                            name="embed-batch-worker", daemon=True)
            self._worker.start()

        # Stats
        self.stats = {'embed_calls': 0, 'embed_onnx_runs': 0, 'faces_embedded': 0}

        print(f"  Embed coalescing: {'ON' if self.enable_embed_coalescing else 'OFF'} "
              f"(window={collect_window_ms}ms, max_batch={max_batch_size})")

    # ----------------------------------------------------------------- detect
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """Run the shared detector on one frame (thread-safe)."""
        if self._use_lock:
            with self._det_lock:
                return self.detector.detect(frame)
        return self.detector.detect(frame)

    def detect_batch(self, frames: List[np.ndarray]) -> List[List[Dict]]:
        """Detect faces in each frame independently. Returns detections per frame."""
        return [self.detect(f) for f in frames]

    # ------------------------------------------------------------------ embed
    def embed(self, face: np.ndarray) -> np.ndarray:
        """Embed a single face (routes through the batch path for coalescing)."""
        return self.embed_batch([face])[0]

    def embed_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Embed faces, coalescing with concurrent calls from other streams into one
        ONNX run when coalescing is enabled. Returns [N, 512]. Synchronous to caller.
        """
        if len(faces) == 0:
            return np.array([])

        self.stats['embed_calls'] += 1

        if not self.enable_embed_coalescing:
            # Direct path: no worker, no collect-window latency (e.g. single stream).
            with (self._emb_lock if self._use_lock else _NullCtx()):
                self.stats['embed_onnx_runs'] += 1
                self.stats['faces_embedded'] += len(faces)
                return self.embedder.extract_embeddings_batch(faces)

        fut: "Future" = Future()
        self._emb_queue.put((faces, fut))
        return fut.result()

    def _embed_worker(self):
        """Drain the embed queue, coalesce within collect_window, run one ONNX call."""
        while not self._shutdown.is_set():
            try:
                first = self._emb_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [first]
            total_faces = len(first[0])
            deadline = time.perf_counter() + self.collect_window_s

            # Collect more requests until the window closes or the cap is reached.
            while total_faces < self.max_batch_size:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                try:
                    item = self._emb_queue.get(timeout=remaining)
                except queue.Empty:
                    break
                batch.append(item)
                total_faces += len(item[0])

            self._run_embed_batch(batch)

    def _run_embed_batch(self, batch):
        """batch: list of (faces, future). One ONNX call, split results back per request."""
        all_faces: List[np.ndarray] = []
        spans = []  # (start, count, future)
        for faces, fut in batch:
            spans.append((len(all_faces), len(faces), fut))
            all_faces.extend(faces)

        try:
            with (self._emb_lock if self._use_lock else _NullCtx()):
                embs = self.embedder.extract_embeddings_batch(all_faces)  # [M, 512]
                self.stats['embed_onnx_runs'] += 1
                self.stats['faces_embedded'] += len(all_faces)
            for start, n, fut in spans:
                fut.set_result(embs[start:start + n])
        except Exception as e:
            for _, _, fut in spans:
                if not fut.done():
                    fut.set_exception(e)

    def shutdown(self):
        self._shutdown.set()
        if self._worker is not None:
            self._worker.join(timeout=2.0)


class _NullCtx:
    """No-op context manager for the lock-free path."""
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
