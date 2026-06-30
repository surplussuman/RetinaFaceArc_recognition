"""
Motion Detector — Event-Driven Face Recognition

Implements cheap per-frame motion detection that gates expensive face detection.
Mathematical basis:
    M_t = |Frame_t - Frame_(t-1)|  (frame differencing)
    or MOG2 (background subtraction, more robust)

Complexity shift:
    O(N_frames) × T_detect  →  O(N_events) × T_detect  + O(N_frames) × T_motion
    where T_motion ≈ 1–2ms  vs  T_detect ≈ 200ms

Architecture position:
    Frame → MotionDetector (1ms) → [if motion] → RetinaFaceDetector (200ms)
                                 → [no motion]  → skip detect, age tracks only
"""

import cv2
import numpy as np
from typing import Optional, Tuple, List


class MotionDetector:
    """
    Cheap motion pre-filter for face detection gating.

    Two modes:
      mog2       — MOG2 background subtraction (more robust, handles lighting changes)
      frame_diff — Absolute frame difference (simpler, no warm-up needed)

    The detector outputs:
      - has_motion: bool (whether to run face detection this frame)
      - motion_mask: np.ndarray uint8 (white where motion, black elsewhere)
      - motion_regions: list of (x, y, w, h) bounding rects around motion blobs
      - motion_fraction: float fraction of analyzed area that contains motion
    """

    def __init__(self,
                 method: str = 'mog2',
                 mog2_history: int = 200,
                 mog2_var_threshold: float = 40.0,
                 mog2_detect_shadows: bool = False,
                 warmup_frames: int = 30,
                 min_motion_area: int = 500,
                 motion_gate_threshold: float = 0.01,
                 safety_scan_interval: int = 30,
                 detect_min_interval: int = 1,
                 motion_spike_delta: float = 1.0):
        """
        Args:
            method: 'mog2' or 'frame_diff'
            mog2_history: Number of frames for MOG2 background model
            mog2_var_threshold: MOG2 variance threshold (higher = less sensitive)
            mog2_detect_shadows: Enable shadow detection in MOG2 (slower)
            warmup_frames: Frames before gating activates (MOG2 needs warmup)
            min_motion_area: Minimum contour area (px²) to count as real motion
            motion_gate_threshold: Fraction of frame area that must move to trigger detect
            safety_scan_interval: Force detect every N frames regardless of motion (FLOOR)

            Adaptive detection scheduler (replaces fixed frame-skip):
            detect_min_interval: Under *sustained* motion, run detection at most once
                                 per this many frames (CEILING / throttle). The tracker +
                                 ghost-cache carry identity on the in-between frames.
                                 1 = no throttle (detect every motion frame, old behaviour).
                                 e.g. 3 @ 30fps source => detector caps at ~10 FPS.
            motion_spike_delta:  If motion_fraction jumps by >= this between frames, force an
                                 immediate detection (a new person entering the scene shows
                                 up as a sudden rise in foreground). This is what preserves
                                 recall while the throttle keeps speed. 1.0 = disabled.

        Scheduling model:
            detect(t) = warmup(t) OR floor(t) OR spike(t) OR (sustained_motion(t) AND throttle(t))
            i.e. detection frequency is a function of scene state, not a constant.
        """
        self.method = method
        self.warmup_frames = warmup_frames
        self.min_motion_area = min_motion_area
        self.motion_gate_threshold = motion_gate_threshold
        self.safety_scan_interval = safety_scan_interval
        self.detect_min_interval = max(1, detect_min_interval)
        self.motion_spike_delta = motion_spike_delta

        self._frame_count = 0
        self._prev_gray: Optional[np.ndarray] = None
        self._frames_since_forced_scan = 0
        self._frames_since_detect = 0
        self._prev_motion_fraction = 0.0

        if method == 'mog2':
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=mog2_history,
                varThreshold=mog2_var_threshold,
                detectShadows=mog2_detect_shadows
            )
        else:
            self._bg_subtractor = None

        print(f"[OK] MotionDetector initialized (method={method}, "
              f"warmup={warmup_frames}, gate_threshold={motion_gate_threshold:.3f}, "
              f"safety_scan_every={safety_scan_interval}f)")

    def update(self, frame: np.ndarray, roi: Optional[Tuple[int, int, int, int]] = None
               ) -> Tuple[bool, np.ndarray, List[Tuple[int, int, int, int]], float]:
        """
        Process one frame and decide whether face detection should run.

        Args:
            frame: BGR frame (full resolution)
            roi:   Optional (x, y, w, h) rectangle to restrict analysis to zone area.
                   Coordinates in full-frame space. If None, use full frame.

        Returns:
            (has_motion, motion_mask, motion_regions, motion_fraction)
            - has_motion:      True if face detection should run this frame
            - motion_mask:     uint8 binary mask (same size as analyzed region)
            - motion_regions:  list of (x, y, w, h) in full-frame coords
            - motion_fraction: fraction of analyzed area that moved
        """
        self._frame_count += 1
        self._frames_since_forced_scan += 1
        self._frames_since_detect += 1

        # Crop to ROI if specified
        if roi is not None:
            rx, ry, rw, rh = roi
            # Clamp to frame bounds
            h, w = frame.shape[:2]
            rx, ry = max(0, rx), max(0, ry)
            rw = min(rw, w - rx)
            rh = min(rh, h - ry)
            analysis_frame = frame[ry:ry + rh, rx:rx + rw]
            offset = (rx, ry)
        else:
            analysis_frame = frame
            offset = (0, 0)

        # Always run update() to keep background model warm
        gray = cv2.cvtColor(analysis_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self.method == 'mog2':
            fg_mask = self._bg_subtractor.apply(analysis_frame)
            # MOG2 returns 255 for foreground, 127 for shadow, 0 for background
            _, motion_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
        else:
            # Frame differencing
            if self._prev_gray is None or self._prev_gray.shape != gray.shape:
                self._prev_gray = gray.copy()
                # No previous frame: no motion, but still warm up
                empty_mask = np.zeros(gray.shape, dtype=np.uint8)
                self._frames_since_forced_scan = 0  # detecting now resets the floor
                self._frames_since_detect = 0
                return True, empty_mask, [], 0.0

            diff = cv2.absdiff(gray, self._prev_gray)
            _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        self._prev_gray = gray.copy()

        # Morphological cleanup to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)

        # Find motion blobs
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        motion_regions = []
        total_motion_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area >= self.min_motion_area:
                x, y, w, h = cv2.boundingRect(cnt)
                # Convert to full-frame coordinates
                motion_regions.append((x + offset[0], y + offset[1], w, h))
                total_motion_area += area

        analyzed_area = analysis_frame.shape[0] * analysis_frame.shape[1]
        motion_fraction = total_motion_area / analyzed_area if analyzed_area > 0 else 0.0

        # ---- Adaptive detection scheduler --------------------------------
        # detect(t) = warmup OR floor OR spike OR (sustained_motion AND throttle)

        # During warm-up: always detect (background model is learning)
        in_warmup = self._frame_count <= self.warmup_frames

        # FLOOR — force detect every N frames so still scenes are never blind
        force_scan = self._frames_since_forced_scan >= self.safety_scan_interval

        # Is there meaningful motion right now?
        has_significant_motion = motion_fraction >= self.motion_gate_threshold

        # SPIKE — sudden rise in foreground => new event (person entering).
        # This is what preserves recall: every new face triggers an immediate detect,
        # independent of the throttle below.
        motion_spike = (motion_fraction - self._prev_motion_fraction) >= self.motion_spike_delta

        # CEILING / THROTTLE — under sustained motion, don't re-run the detector every
        # frame. Cap it to once per detect_min_interval; tracker + ghost-cache carry the
        # identity on the skipped frames. (detect_min_interval=1 => no throttle.)
        throttle_ok = self._frames_since_detect >= self.detect_min_interval

        has_motion = (
            in_warmup
            or force_scan
            or motion_spike
            or (has_significant_motion and throttle_ok)
        )

        self._prev_motion_fraction = motion_fraction

        # Any time detection actually fires, reset both the floor and the throttle clocks.
        if has_motion:
            self._frames_since_forced_scan = 0
            self._frames_since_detect = 0

        return has_motion, motion_mask, motion_regions, motion_fraction

    def reset(self):
        """Reset detector state (call when switching cameras or videos)."""
        self._frame_count = 0
        self._prev_gray = None
        self._frames_since_forced_scan = 0
        self._frames_since_detect = 0
        self._prev_motion_fraction = 0.0
        if self.method == 'mog2' and self._bg_subtractor is not None:
            # Re-create to clear learned background
            self._bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=200, varThreshold=40.0, detectShadows=False
            )

    @property
    def is_warmed_up(self) -> bool:
        return self._frame_count > self.warmup_frames
