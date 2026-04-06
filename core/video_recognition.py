"""
Video Face Recognition System for CCTV/Surveillance Footage
============================================================

Handles real-time face recognition in video streams with temporal smoothing
and tracking for robust recognition in low-quality CCTV conditions.

Key Features:
- Frame-by-frame face detection and recognition
- Temporal smoothing: Track faces across frames, average predictions
- Track-by-detection: Assign persistent IDs to people
- Quality-aware processing: Skip very poor frames, focus on good frames
- Optimized for CCTV: Low resolution, compression artifacts, motion blur
- Results output: Annotated video + CSV with timestamps

Mathematical Foundation:
-----------------------
Temporal Smoothing:
    identity_t = argmax(Σ confidence_i * 1{identity_i = k})
    
Track Assignment (IOU-based):
    IOU(box1, box2) = Area(intersection) / Area(union)
    Assign if IOU > 0.3 (same person across frames)

Confidence Aggregation:
    confidence_track = (α * confidence_current + (1-α) * confidence_history)
    α = 0.3 (exponential moving average)

Author: Face Recognition System - Phase 3
Date: 2024
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, deque
from datetime import datetime
import json

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase
from core.motion_detector import MotionDetector


class FaceTrack:
    """
    Represents a tracked face across multiple frames.
    """
    
    def __init__(self, track_id: int, bbox: np.ndarray, embedding: np.ndarray, frame_idx: int):
        self.track_id = track_id
        self.bboxes = deque([bbox], maxlen=10)  # Last 10 bboxes
        self.embeddings = deque([embedding], maxlen=5)  # Last 5 embeddings for averaging
        self.first_seen = frame_idx
        self.last_seen = frame_idx
        self.identity = None  # Recognized identity
        self.identity_confidence = 0.0
        self.identity_votes = defaultdict(float)  # Track votes for each identity
        self.frames_since_seen = 0
        
        # Ghost Tracking: Cache identity to avoid recomputing embeddings every frame
        self.cached_identity = None  # Last recognized identity
        self.cached_confidence = 0.0  # Confidence of cached identity
        self.frames_since_recognition = 0  # Frames since last embedding computation
        
    def update(self, bbox: np.ndarray, embedding: np.ndarray, frame_idx: int):
        """Update track with new detection."""
        self.bboxes.append(bbox)
        self.embeddings.append(embedding)
        self.last_seen = frame_idx
        self.frames_since_seen = 0
        
    def get_average_embedding(self) -> np.ndarray:
        """Compute average embedding from recent detections."""
        avg_emb = np.mean(self.embeddings, axis=0)
        return avg_emb / np.linalg.norm(avg_emb)  # L2 normalize
    
    def get_predicted_bbox(self) -> np.ndarray:
        """Predict next bbox position (simple linear extrapolation)."""
        if len(self.bboxes) < 2:
            bbox = self.bboxes[-1]
            return np.array(bbox) if isinstance(bbox, list) else bbox
        
        # Ensure bboxes are numpy arrays
        last_bbox = self.bboxes[-1]
        prev_bbox = self.bboxes[-2]
        
        if isinstance(last_bbox, list):
            last_bbox = np.array(last_bbox)
        if isinstance(prev_bbox, list):
            prev_bbox = np.array(prev_bbox)
        
        # Simple constant velocity model
        velocity = last_bbox - prev_bbox
        predicted = last_bbox + velocity
        return predicted
    
    def vote_identity(self, identity: str, confidence: float, alpha: float = 0.3):
        """
        Vote for identity with temporal smoothing.
        
        Args:
            identity: Recognized identity
            confidence: Recognition confidence
            alpha: Smoothing factor (0.3 = 30% new, 70% history)
        """
        # Exponential moving average of votes
        self.identity_votes[identity] = (
            alpha * confidence + (1 - alpha) * self.identity_votes.get(identity, 0)
        )
        
        # Update identity to most voted
        if self.identity_votes:
            best_identity = max(self.identity_votes, key=self.identity_votes.get)
            self.identity = best_identity
            self.identity_confidence = self.identity_votes[best_identity]
            
        # Update cached identity for Ghost Tracking
        self.cached_identity = self.identity
        self.cached_confidence = self.identity_confidence
    
    def should_recognize(self, recognition_interval: int) -> bool:
        """
        Ghost Tracking: Determine if this track needs recognition.
        
        Mathematical basis:
            p = 1/recognition_interval (fraction of frames that run embedding)
            If recognition_interval = 30: p = 1/30 → 30× speedup on recognition
        
        Recognition is needed if:
        1. Track is new (never recognized before), OR
        2. Cache is stale (>= recognition_interval frames since last recognition)
        
        Args:
            recognition_interval: Frames between re-verifications (e.g. 30 = 1 sec @ 30fps)
            
        Returns:
            True if embedding extraction + recognition should run
        """
        # New track: must recognize
        if self.cached_identity is None:
            return True
        
        # Existing track: check if cache is stale
        return self.frames_since_recognition >= recognition_interval
    
    def reset_recognition_timer(self):
        """Ghost Tracking: Reset timer after computing new embedding."""
        self.frames_since_recognition = 0
    
    def increment_recognition_timer(self):
        """Ghost Tracking: Increment timer (called every frame for cached tracks)."""
        self.frames_since_recognition += 1
    
    def get_cached_identity(self) -> Tuple[Optional[str], float]:
        """
        Ghost Tracking: Get cached identity without re-recognition.
        
        Returns:
            Tuple of (identity, confidence)
        """
        return (self.cached_identity, self.cached_confidence)


class VideoRecognitionSystem:
    """
    Complete video face recognition system with tracking and temporal smoothing.
    """
    
    def __init__(self,
                 detector: RetinaFaceDetector,
                 aligner: FaceAligner,
                 embedder: ArcFaceEmbedder,
                 vector_db: VectorDatabase,
                 recognition_threshold: float = 0.4,
                 track_iou_threshold: float = 0.3,
                 max_frames_missing: int = 30,
                 process_every_n_frames: int = 1,
                 enable_ghost_tracking: bool = True,
                 recognition_interval: int = 30,
                 enable_zone_detection: bool = False,
                 detection_zones: List[Dict] = None,
                 motion_detector: Optional['MotionDetector'] = None):
        """
        Initialize video recognition system.
        
        Args:
            detector: Face detector
            aligner: Face aligner
            embedder: Face embedder
            vector_db: Vector database with enrolled users
            recognition_threshold: Minimum similarity for recognition
            track_iou_threshold: Min IOU for track assignment (0.3 = 30% overlap)
            max_frames_missing: Max frames a track can be missing before deletion
            process_every_n_frames: Process every N frames (1 = all, 2 = every other)
            enable_ghost_tracking: Enable Ghost Tracking optimization (p = 1/recognition_interval)
            recognition_interval: Frames between re-recognition (30 = 1 sec @ 30fps)
            enable_zone_detection: Enable zone-based detection (16× speedup!)
            detection_zones: List of zones [{'name': str, 'roi': [x,y,w,h], 'enabled': bool}]
            motion_detector: Optional MotionDetector for event-driven gating
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.vector_db = vector_db
        self.recognition_threshold = recognition_threshold
        self.track_iou_threshold = track_iou_threshold
        self.max_frames_missing = max_frames_missing
        self.process_every_n_frames = process_every_n_frames
        
        # Ghost Tracking parameters (Mathematical model: p = 1/recognition_interval)
        self.enable_ghost_tracking = enable_ghost_tracking
        self.recognition_interval = recognition_interval
        
        # Zone Detection (SafePro Secret: 16× speedup by processing only active areas!)
        self.enable_zone_detection = enable_zone_detection
        self.detection_zones = detection_zones if detection_zones else []
        
        # Motion Detection Gate (Event-Driven Architecture)
        # Runs ~1ms/frame; gates 200ms detector to only fire on real activity
        self.motion_detector = motion_detector
        
        # Performance optimization: max resolution for detection
        # For real-time CCTV on CPU: 640×360 (half HD)
        self.max_detection_resolution = (640, 360)  # Aggressive for real-time
        
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 0
        self.frame_count = 0
        
        # Statistics for Ghost Tracking
        self.stats = {
            'embeddings_computed': 0,
            'embeddings_cached': 0,
            'total_detections': 0,
            'zone_detections': 0,  # Detections inside zones
            'total_pixels_processed': 0,  # Track pixel reduction
            'motion_gates_triggered': 0,   # Frames where motion caused detection
            'motion_gates_skipped': 0,     # Frames where no motion → detection skipped
        }
        
        print(f"✓ Video recognition system initialized")
        print(f"  Recognition threshold: {recognition_threshold}")
        print(f"  Track IOU threshold: {track_iou_threshold}")
        print(f"  Process every {process_every_n_frames} frames")
        print(f"  Max detection resolution: {self.max_detection_resolution[0]}×{self.max_detection_resolution[1]}")
        if enable_ghost_tracking:
            print(f"  🚀 Ghost Tracking: ENABLED (recognize every {recognition_interval} frames)")
            print(f"     Expected speedup: ~{recognition_interval}× on recognition")
        if enable_zone_detection:
            print(f"  📍 Zone Detection: ENABLED ({len([z for z in self.detection_zones if z.get('enabled', True)])} zones)")
            print(f"     Expected speedup: ~16× on detection (geometry optimization)")
            for zone in self.detection_zones:
                if zone.get('enabled', True):
                    roi = zone['roi']
                    pixels = roi[2] * roi[3]
                    print(f"     - {zone['name']}: {roi[2]}×{roi[3]} = {pixels:,} pixels")
        if motion_detector is not None:
            print(f"  🎯 Motion Gate: ENABLED (method={motion_detector.method}, "
                  f"warmup={motion_detector.warmup_frames}f, "
                  f"safety_scan_every={motion_detector.safety_scan_interval}f)")
    
    def resize_for_detection(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Intelligently resize frame for detection if too large.
        Only resize if frame is larger than max_detection_resolution.
        
        Args:
            frame: Original frame
            
        Returns:
            (resized_frame, scale_factor)
        """
        h, w = frame.shape[:2]
        max_w, max_h = self.max_detection_resolution
        
        # Only resize if frame is larger than threshold
        if w <= max_w and h <= max_h:
            return frame, 1.0
        
        # Calculate scale to fit within max resolution
        scale = min(max_w / w, max_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """
        Compute Intersection over Union (IOU) between two bboxes.
        
        Args:
            bbox1, bbox2: [x1, y1, x2, y2] format
            
        Returns:
            IOU score [0, 1]
        """
        # Intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Union
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def match_detections_to_tracks(self,
                                   detections: List[Dict],
                                   embeddings: List[np.ndarray]) -> List[Tuple[int, int]]:
        """
        Match new detections to existing tracks using IOU.
        
        Args:
            detections: List of detection dicts with 'bbox'
            embeddings: Corresponding embeddings
            
        Returns:
            List of (detection_idx, track_idx) matches
        """
        if len(self.tracks) == 0 or len(detections) == 0:
            return []
        
        # Compute IOU matrix
        iou_matrix = np.zeros((len(detections), len(self.tracks)))
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            for j, track in enumerate(self.tracks):
                predicted_bbox = track.get_predicted_bbox()
                iou_matrix[i, j] = self.compute_iou(bbox, predicted_bbox)
        
        # Greedy matching (can use Hungarian algorithm for optimal)
        matches = []
        matched_detections = set()
        matched_tracks = set()
        
        # Match highest IOU first
        while True:
            i, j = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
            if iou_matrix[i, j] < self.track_iou_threshold:
                break
            
            matches.append((i, j))
            matched_detections.add(i)
            matched_tracks.add(j)
            
            # Zero out matched rows/cols
            iou_matrix[i, :] = 0
            iou_matrix[:, j] = 0
        
        return matches
    
    def recognize_face(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize face from embedding.
        
        Args:
            embedding: Face embedding
            
        Returns:
            (identity, confidence) or (None, 0.0)
        """
        results = self.vector_db.search(embedding, k=1, threshold=self.recognition_threshold)
        
        if len(results) > 0:
            identity, similarity = results[0]
            return identity, similarity
        
        return None, 0.0
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Dict:
        """
        Process single frame with Ghost Tracking optimization.
        
        Mathematical Model:
            T_frame = T_d + F * p * T_e
            where:
                T_d = detection time (always runs)
                F = faces per frame
                p = 1/recognition_interval (embedding frequency)
                T_e = embedding time per face
        
        With p=1/30: Expected 30× speedup on recognition portion!
        
        Args:
            frame: BGR frame
            frame_idx: Frame number
            
        Returns:
            Dictionary with detections and recognitions
        """
        self.frame_count = frame_idx
        
        # Skip frames if configured
        if frame_idx % self.process_every_n_frames != 0:
            # Age out tracks
            for track in self.tracks:
                track.frames_since_seen += 1
            return {'skipped': True, 'frame_idx': frame_idx}
        
        # MOTION GATE — run cheap motion detector before expensive face detection
        # Cost: ~1ms. Benefit: skip 200ms detector when nothing is moving.
        run_detection = True
        motion_fraction = 0.0
        if self.motion_detector is not None:
            # Use first zone as ROI hint for motion analysis, or full frame
            roi_hint = None
            if self.enable_zone_detection and self.detection_zones:
                z = next((z for z in self.detection_zones if z.get('enabled', True)), None)
                if z:
                    roi_hint = tuple(z['roi'])  # (x, y, w, h)
            has_motion, _mask, _regions, motion_fraction = self.motion_detector.update(frame, roi=roi_hint)
            run_detection = has_motion
            if has_motion:
                self.stats['motion_gates_triggered'] += 1
            else:
                self.stats['motion_gates_skipped'] += 1
                # No detection this frame — only age tracks
                for track in self.tracks:
                    track.frames_since_seen += 1
                    track.increment_recognition_timer()
                self.tracks = [t for t in self.tracks if t.frames_since_seen < self.max_frames_missing]
                return {
                    'frame_idx': frame_idx,
                    'num_detections': 0,
                    'num_tracks': len(self.tracks),
                    'active_tracks': [],
                    'skipped': False,
                    'motion_gated': True,
                    'motion_fraction': motion_fraction,
                }
        
        # STEP 1: ZONE-BASED DETECTION (SafePro Secret: 16× speedup!)
        # Instead of processing full 4K frame (8.3M pixels), process only active zones (518K pixels)
        # Mathematical Model: T_detect ∝ Width × Height
        
        all_detections = []
        
        if self.enable_zone_detection and len(self.detection_zones) > 0:
            # Zone-Based Detection: Process only active areas
            for zone in self.detection_zones:
                if not zone.get('enabled', True):
                    continue
                
                # Extract zone coordinates
                x, y, w, h = zone['roi']
                
                # Crop frame to zone (INSTANT - numpy array slicing)
                zone_frame = frame[y:y+h, x:x+w]
                
                # Track pixels processed for statistics
                self.stats['total_pixels_processed'] += w * h
                
                # Detect directly in zone (NO RESIZE - zones are already optimized size!)
                # This is the key: 1000×800 zone → detect directly → 16× faster than 3840×2160!
                zone_detections = self.detector.detect(zone_frame)
                scale = 1.0  # No scaling applied
                
                # Remap coordinates from zone to full frame
                # Critical: detector gives (x', y') relative to crop, convert to global (X, Y)
                for detection in zone_detections:
                    bbox = detection['bbox']
                    landmarks = detection['landmarks']
                    
                    if isinstance(bbox, list):
                        bbox = np.array(bbox)
                    if isinstance(landmarks, list):
                        landmarks = np.array(landmarks)
                    
                    # Apply scale from resize
                    if scale != 1.0:
                        bbox = bbox / scale
                        landmarks = landmarks / scale
                    
                    # Remap to global coordinates: X_global = x' + x_zone_start
                    detection['bbox'] = bbox + np.array([x, y, x, y])
                    # Landmarks are shape (5, 2): 5 points, each with (x, y)
                    detection['landmarks'] = landmarks + np.array([[x, y]])  # Broadcasting adds [x,y] to each point
                    detection['zone'] = zone['name']  # Track which zone detected this
                    
                    all_detections.append(detection)
                
                self.stats['zone_detections'] += len(zone_detections)
        else:
            # Fallback: Full-frame detection (old method)
            detection_frame, scale = self.resize_for_detection(frame)
            detections = self.detector.detect(detection_frame)
            
            # Scale bboxes and landmarks back to original size
            if scale != 1.0:
                for detection in detections:
                    bbox = detection['bbox']
                    landmarks = detection['landmarks']
                    
                    if isinstance(bbox, list):
                        bbox = np.array(bbox)
                    if isinstance(landmarks, list):
                        landmarks = np.array(landmarks)
                    
                    detection['bbox'] = bbox / scale
                    detection['landmarks'] = landmarks / scale
            
            all_detections = detections
        
        # Use zone detections from here on
        detections = all_detections
        
        # STEP 2: PRELIMINARY IOU MATCHING (Cheap - just bbox overlap)
        # Determine which detections match existing tracks BEFORE extracting embeddings
        preliminary_matches = []
        matched_detections_prelim = set()
        matched_tracks_prelim = set()
        
        if len(detections) > 0 and len(self.tracks) > 0:
            # Compute IOU matrix between all detections and all tracks
            iou_matrix = np.zeros((len(detections), len(self.tracks)))
            
            for i, detection in enumerate(detections):
                det_bbox = detection['bbox']
                if isinstance(det_bbox, list):
                    det_bbox = np.array(det_bbox)
                
                for j, track in enumerate(self.tracks):
                    track_bbox = track.get_predicted_bbox()
                    iou = self.compute_iou(det_bbox, track_bbox)
                    iou_matrix[i, j] = iou
            
            # Greedy matching: highest IOU first
            while True:
                if iou_matrix.size == 0:
                    break
                    
                max_iou = np.max(iou_matrix)
                if max_iou < self.track_iou_threshold:
                    break
                
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                
                preliminary_matches.append((i, j))
                matched_detections_prelim.add(i)
                matched_tracks_prelim.add(j)
                
                iou_matrix[i, :] = 0
                iou_matrix[:, j] = 0
        
        # STEP 3: GHOST TRACKING LOGIC - Determine which detections need embeddings
        detections_needing_embedding = []
        detection_to_track_map = {}  # Maps detection index to track
        
        for det_idx, track_idx in preliminary_matches:
            track = self.tracks[track_idx]
            
            if self.enable_ghost_tracking and not track.should_recognize(self.recognition_interval):
                # ✅ CACHE HIT: Reuse cached identity, skip expensive embedding!
                # This is where we save time: p = 0 for this detection
                self.stats['embeddings_cached'] += 1
                self.stats['total_detections'] += 1
                track.increment_recognition_timer()
                # DON'T add to embedding list
            else:
                # ❌ CACHE MISS: Need to compute embedding (new track or stale cache)
                # This costs T_e time
                detections_needing_embedding.append(det_idx)
                detection_to_track_map[det_idx] = track
                self.stats['total_detections'] += 1
        
        # New detections (not matched to any track) always need embedding
        for det_idx in range(len(detections)):
            if det_idx not in matched_detections_prelim:
                detections_needing_embedding.append(det_idx)
                self.stats['total_detections'] += 1
        
        # STEP 4: EXTRACT EMBEDDINGS (Only for needed detections - F * p * T_e)
        embeddings_map = {}  # Maps detection index to embedding
        
        if len(detections_needing_embedding) > 0:
            # Align faces that need embeddings
            aligned_faces = []
            valid_det_indices = []
            
            for det_idx in detections_needing_embedding:
                landmarks = detections[det_idx]['landmarks']
                aligned_face = self.aligner.align_face(frame, landmarks)
                if aligned_face is not None:
                    aligned_faces.append(aligned_face)
                    valid_det_indices.append(det_idx)
            
            # Batch extract embeddings (FAST!)
            if len(aligned_faces) > 0:
                embeddings_array = self.embedder.extract_embeddings_batch(aligned_faces)
                for i, det_idx in enumerate(valid_det_indices):
                    embeddings_map[det_idx] = embeddings_array[i]
                    self.stats['embeddings_computed'] += 1
        
        # STEP 5: UPDATE TRACKS
        for det_idx, track_idx in preliminary_matches:
            track = self.tracks[track_idx]
            detection = detections[det_idx]
            
            if det_idx in embeddings_map:
                # New embedding computed - full update
                track.update(detection['bbox'], embeddings_map[det_idx], frame_idx)
                
                # Recognize with averaged embedding
                avg_embedding = track.get_average_embedding()
                identity, confidence = self.recognize_face(avg_embedding)
                
                # CRITICAL: Cache result even if "Unknown" for Ghost Tracking to work!
                if identity is not None:
                    track.vote_identity(identity, confidence)
                else:
                    track.vote_identity("Unknown", 0.0)
                
                # Reset timer after computing new embedding
                track.reset_recognition_timer()
            else:
                # Using cached identity - only update bbox, keep old embedding
                track.bboxes.append(detection['bbox'])
                track.last_seen = frame_idx
                track.frames_since_seen = 0
                # Timer already incremented in Step 3
                # Identity stays cached - no re-recognition!
        
        # STEP 6: CREATE NEW TRACKS
        for det_idx in range(len(detections)):
            if det_idx not in matched_detections_prelim and det_idx in embeddings_map:
                track = FaceTrack(
                    self.next_track_id,
                    detections[det_idx]['bbox'],
                    embeddings_map[det_idx],
                    frame_idx
                )
                self.next_track_id += 1
                
                # Try immediate recognition
                identity, confidence = self.recognize_face(embeddings_map[det_idx])
                
                # CRITICAL: Cache result even if "Unknown"
                if identity is not None:
                    track.vote_identity(identity, confidence)
                else:
                    track.vote_identity("Unknown", 0.0)
                
                # Reset timer for new track (just computed first embedding)
                track.reset_recognition_timer()
                
                self.tracks.append(track)
        
        # STEP 7: AGE OUT STALE TRACKS
        for track in self.tracks:
            if track.track_id not in [self.tracks[j].track_id for _, j in preliminary_matches]:
                track.frames_since_seen += 1
        
        self.tracks = [
            track for track in self.tracks
            if track.frames_since_seen < self.max_frames_missing
        ]
        
        # Prepare results
        active_tracks = []
        for track in self.tracks:
            if track.frames_since_seen == 0:  # Currently visible
                bbox = track.bboxes[-1]
                if isinstance(bbox, list):
                    bbox = np.array(bbox)
                
                active_tracks.append({
                    'track_id': track.track_id,
                    'bbox': bbox,
                    'identity': track.identity,
                    'confidence': track.identity_confidence,
                    'first_seen': track.first_seen,
                    'last_seen': track.last_seen
                })
        
        return {
            'frame_idx': frame_idx,
            'num_detections': len(detections),
            'num_tracks': len(self.tracks),
            'active_tracks': active_tracks,
            'skipped': False
        }
    
    def process_video(self,
                     video_path: Path,
                     output_path: Optional[Path] = None,
                     show_preview: bool = False,
                     max_frames: Optional[int] = None,
                     progress_callback: Optional[callable] = None,
                     frame_callback: Optional[callable] = None) -> Dict:
        """
        Process entire video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save annotated video (optional)
            show_preview: Show live preview window
            max_frames: Maximum frames to process (None = all)
            progress_callback: Function(frame_idx, total_frames) called for progress updates
            frame_callback: Function(annotated_frame, stats) called for each processed frame
            
        Returns:
            Dictionary with processing statistics
        """
        print(f"\n{'='*80}")
        print(f"PROCESSING VIDEO: {video_path.name}")
        print(f"{'='*80}\n")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties:")
        print(f"  Resolution: {width}×{height}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Total frames: {total_frames}")
        print(f"  Duration: {total_frames/fps:.2f}s\n")
        
        # Setup output video writer
        writer = None
        if output_path is not None:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                str(output_path),
                fourcc,
                fps,
                (width, height)
            )
            print(f"Output video: {output_path}\n")
        
        # Processing statistics
        stats = {
            'total_frames': 0,
            'processed_frames': 0,
            'skipped_frames': 0,
            'total_detections': 0,
            'unique_tracks': 0,
            'recognized_identities': set(),
            'processing_times': []
        }
        
        # Recognition log (for CSV export)
        recognition_log = []
        
        frame_idx = 0
        start_time = datetime.now()
        last_annotated_frame = None  # Persist last annotation across skipped frames
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if max_frames and frame_idx >= max_frames:
                    break
                
                # Process frame
                frame_start = datetime.now()
                result = self.process_frame(frame, frame_idx)
                frame_time = (datetime.now() - frame_start).total_seconds() * 1000
                
                stats['total_frames'] += 1
                stats['processing_times'].append(frame_time)
                
                if result.get('skipped') and not result.get('motion_gated'):
                    stats['skipped_frames'] += 1
                    # Write last annotated frame (or plain frame) to keep correct duration
                    if writer is not None:
                        writer.write(last_annotated_frame if last_annotated_frame is not None else frame)
                    if show_preview and last_annotated_frame is not None:
                        cv2.imshow('Video Recognition', last_annotated_frame)
                else:
                    if not result.get('motion_gated', False):
                        stats['processed_frames'] += 1
                        stats['total_detections'] += result.get('num_detections', 0)
                    
                    # Draw annotations (even for motion-gated frames, reuse last)
                    if result.get('motion_gated', False):
                        annotated_frame = last_annotated_frame if last_annotated_frame is not None else frame
                    else:
                        annotated_frame = self.draw_annotations(frame, result)
                        last_annotated_frame = annotated_frame
                    
                    # Log recognitions
                    timestamp = frame_idx / fps
                    for track in result.get('active_tracks', []):
                        if track['identity'] is not None:
                            stats['recognized_identities'].add(track['identity'])
                            recognition_log.append({
                                'frame': frame_idx,
                                'timestamp': timestamp,
                                'track_id': track['track_id'],
                                'identity': track['identity'],
                                'confidence': track['confidence']
                            })
                    
                    # Write to output video — always write to maintain correct duration
                    if writer is not None:
                        writer.write(annotated_frame)
                    
                    # Callbacks for external monitoring
                    if frame_callback is not None:
                        frame_callback(annotated_frame, {
                            'frame': frame_idx,
                            'detections': result.get('num_detections', 0),
                            'tracks': result.get('num_tracks', 0)
                        })
                    
                    # Show preview
                    if show_preview:
                        cv2.imshow('Video Recognition', annotated_frame)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                
                frame_idx += 1
                
                # Progress callback
                if progress_callback is not None:
                    progress_callback(frame_idx, total_frames)
                
                # Progress update
                if frame_idx % 100 == 0:
                    elapsed = (datetime.now() - start_time).total_seconds()
                    fps_actual = frame_idx / elapsed if elapsed > 0 else 0
                    print(f"  Processed {frame_idx}/{total_frames} frames ({fps_actual:.1f} FPS)")
        
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_preview:
                cv2.destroyAllWindows()
        
        # Final statistics
        stats['unique_tracks'] = self.next_track_id
        stats['recognized_identities'] = list(stats['recognized_identities'])
        stats['avg_processing_time'] = np.mean(stats['processing_times']) if stats['processing_times'] else 0
        stats['recognition_log'] = recognition_log
        
        total_time = (datetime.now() - start_time).total_seconds()
        
        print(f"\n{'='*80}")
        print(f"PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Total frames: {stats['total_frames']}")
        print(f"Processed frames: {stats['processed_frames']}")
        print(f"Skipped frames: {stats['skipped_frames']}")
        print(f"Total detections: {stats['total_detections']}")
        print(f"Unique tracks: {stats['unique_tracks']}")
        print(f"Recognized identities: {', '.join(stats['recognized_identities']) if stats['recognized_identities'] else 'None'}")
        print(f"Avg processing time: {stats['avg_processing_time']:.1f}ms/frame")
        print(f"Total time: {total_time:.2f}s")
        print(f"Actual FPS: {stats['total_frames']/total_time:.2f}")
        
        # Ghost Tracking Statistics
        if self.enable_ghost_tracking:
            embeddings_computed = self.stats['embeddings_computed']
            embeddings_cached = self.stats['embeddings_cached']
            total_detections = self.stats['total_detections']
            
            if total_detections > 0:
                cache_rate = (embeddings_cached / total_detections) * 100
                expected_embeddings_no_ghost = total_detections
                actual_speedup = expected_embeddings_no_ghost / embeddings_computed if embeddings_computed > 0 else 1.0
                
                print(f"\n{'-'*80}")
                print(f"GHOST TRACKING PERFORMANCE (Mathematical Model: p = 1/{self.recognition_interval})")
                print(f"{'-'*80}")
                print(f"Total detections: {total_detections}")
                print(f"Embeddings computed: {embeddings_computed}")
                print(f"Embeddings cached: {embeddings_cached}")
                print(f"Cache hit rate: {cache_rate:.1f}%")
                print(f"Recognition speedup: {actual_speedup:.1f}× (expected ~{self.recognition_interval}×)")
                print(f"{'-'*80}")
        
        # Motion Gate Statistics
        if self.motion_detector is not None:
            triggered = self.stats['motion_gates_triggered']
            skipped = self.stats['motion_gates_skipped']
            total_gated = triggered + skipped
            skip_rate = (skipped / total_gated * 100) if total_gated > 0 else 0
            print(f"\n{'-'*80}")
            print(f"MOTION GATE PERFORMANCE (Event-Driven Architecture)")
            print(f"{'-'*80}")
            print(f"Frames analyzed by motion: {total_gated}")
            print(f"Detection triggered (motion): {triggered}")
            print(f"Detection skipped (no motion): {skipped}")
            print(f"Detection skip rate: {skip_rate:.1f}% fewer detector runs")
            print(f"{'-'*80}")
        
        # Zone Detection Statistics (SafePro Secret - Geometry Optimization!)
        if self.enable_zone_detection:
            zone_detections = self.stats.get('zone_detections', 0)
            total_pixels_processed = self.stats.get('total_pixels_processed', 0)
            
            # Calculate full-frame pixels for comparison
            if stats['processed_frames'] > 0:
                # Assume standard 4K resolution (can be updated based on actual frame size)
                full_frame_pixels = 3840 * 2160  # 8.3M pixels per frame
                total_full_frame_pixels = full_frame_pixels * stats['processed_frames']
                
                if total_pixels_processed > 0 and total_full_frame_pixels > 0:
                    pixel_reduction = (total_full_frame_pixels - total_pixels_processed) / total_full_frame_pixels * 100
                    speedup_factor = total_full_frame_pixels / total_pixels_processed
                    
                    print(f"\n{'-'*80}")
                    print(f"ZONE DETECTION PERFORMANCE (Geometry Optimization: T_detect ∝ W×H)")
                    print(f"{'-'*80}")
                    print(f"Zone detections: {zone_detections}")
                    print(f"Pixels processed (zones): {total_pixels_processed:,}")
                    print(f"Pixels if full-frame: {total_full_frame_pixels:,}")
                    print(f"Pixel reduction: {pixel_reduction:.1f}%")
                    print(f"Expected detection speedup: ~{speedup_factor:.1f}×")
                    print(f"Active zones: {len([z for z in self.detection_zones if z.get('enabled', True)])}")
                    print(f"{'-'*80}")
        
        print(f"{'='*80}\n")
        
        return stats
    
    def draw_annotations(self, frame: np.ndarray, result: Dict) -> np.ndarray:
        """
        Draw bounding boxes and labels on frame.
        
        Args:
            frame: Input frame
            result: Processing result from process_frame()
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Draw detection zones (green boxes showing active areas)
        if self.enable_zone_detection:
            for zone in self.detection_zones:
                if not zone.get('enabled', True):
                    continue
                
                x, y, w, h = zone['roi']
                
                # Draw zone rectangle
                cv2.rectangle(annotated, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Draw zone label
                zone_label = f"ZONE: {zone['name']}"
                (label_w, label_h), _ = cv2.getTextSize(zone_label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(annotated, (x, y-label_h-10), (x+label_w, y), (0, 255, 0), -1)
                cv2.putText(annotated, zone_label, (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Draw face detections and tracks
        for track in result['active_tracks']:
            bbox = track['bbox'].astype(int)
            identity = track['identity']
            confidence = track['confidence']
            
            # Choose color based on recognition
            if identity is not None and identity != "Unknown":
                color = (0, 255, 0)  # Green for recognized
                label = f"{identity} ({confidence:.2f})"
            else:
                color = (0, 0, 255)  # Red for unknown
                label = f"Unknown (Track {track['track_id']})"
            
            # Draw bounding box
            cv2.rectangle(annotated, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
            
            # Draw label background
            (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                annotated,
                (bbox[0], bbox[1] - label_h - 10),
                (bbox[0] + label_w, bbox[1]),
                color,
                -1
            )
            
            # Draw label text
            cv2.putText(
                annotated,
                label,
                (bbox[0], bbox[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
        
        # Draw frame info
        info_text = f"Frame: {result['frame_idx']} | Tracks: {result['num_tracks']} | Detections: {result['num_detections']}"
        if self.enable_zone_detection:
            info_text += f" | Zone Mode: ON"
        cv2.putText(
            annotated,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )
        
        return annotated


def create_video_recognizer(vector_db: VectorDatabase,
                            recognition_threshold: float = 0.4,
                            process_every_n_frames: Optional[int] = None,
                            enable_ghost_tracking: bool = True,
                            recognition_interval: int = 30) -> VideoRecognitionSystem:
    """
    Create video recognition system with all components.
    
    Args:
        vector_db: Vector database with enrolled users
        recognition_threshold: Minimum similarity for recognition
        process_every_n_frames: Process every N frames (None = read from config, default 1)
        enable_ghost_tracking: Enable Ghost Tracking optimization (p=1/recognition_interval)
        recognition_interval: Frames between re-recognition (30 = 1 sec @ 30fps)
        
    Returns:
        Configured VideoRecognitionSystem
    """
    print("Initializing video recognition system...")
    
    # Load config from system_config.yaml if available
    import yaml
    max_detection_resolution = (640, 360)  # Default
    enable_zone_detection = False
    detection_zones = None
    motion_detector = None
    
    try:
        config_path = Path(__file__).parent.parent / 'config' / 'system_config.yaml'
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
                # Ghost Tracking config
                if 'ghost_tracking' in config:
                    ghost_config = config['ghost_tracking']
                    enable_ghost_tracking = ghost_config.get('enable', enable_ghost_tracking)
                    recognition_interval = ghost_config.get('recognition_interval', recognition_interval)
                    print(f"  Loaded Ghost Tracking config: interval={recognition_interval}")
                
                # Frame processing config
                if 'frame_processing' in config:
                    frame_config = config['frame_processing']
                    # Only use config value when caller passed None (sentinel = "use config default")
                    if process_every_n_frames is None:
                        process_every_n_frames = frame_config.get('process_every_n_frames', 1)
                    
                    if 'max_detection_resolution' in frame_config:
                        res = frame_config['max_detection_resolution']
                        max_detection_resolution = (res['width'], res['height'])
                    
                    print(f"  Loaded Frame Processing config: skip_frames={process_every_n_frames}, resolution={max_detection_resolution}")
                
                # Camera Zone config
                if 'camera_zone' in config:
                    zone_config = config['camera_zone']
                    enable_zone_detection = zone_config.get('enabled', False)
                    
                    if enable_zone_detection and 'zones' in zone_config:
                        detection_zones = zone_config['zones']
                        total_zone_pixels = sum(z['roi'][2] * z['roi'][3] for z in detection_zones if z.get('enabled', True))
                        print(f"  Loaded Camera Zone config: {len(detection_zones)} zones, ~{total_zone_pixels:,} pixels total")
                        for zone in detection_zones:
                            if zone.get('enabled', True):
                                w, h = zone['roi'][2], zone['roi'][3]
                                print(f"    - {zone['name']}: {w}×{h} = {w*h:,} pixels")
                
                # Motion Detection config
                if 'motion_detection' in config:
                    md_config = config['motion_detection']
                    if md_config.get('enable', True):
                        motion_detector = MotionDetector(
                            method=md_config.get('method', 'mog2'),
                            mog2_history=md_config.get('mog2_history', 200),
                            mog2_var_threshold=md_config.get('mog2_var_threshold', 40.0),
                            mog2_detect_shadows=md_config.get('mog2_detect_shadows', False),
                            warmup_frames=md_config.get('warmup_frames', 30),
                            min_motion_area=md_config.get('min_motion_area', 500),
                            motion_gate_threshold=md_config.get('motion_gate_threshold', 0.01),
                            safety_scan_interval=md_config.get('safety_scan_interval', 30),
                        )
                    
    except Exception as e:
        print(f"  Warning: Could not load system_config.yaml, using defaults: {e}")
    
    # Final fallback if config wasn't loaded or process_every_n_frames still None
    if process_every_n_frames is None:
        process_every_n_frames = 1
    
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    system = VideoRecognitionSystem(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        vector_db=vector_db,
        recognition_threshold=recognition_threshold,
        process_every_n_frames=process_every_n_frames,
        enable_ghost_tracking=enable_ghost_tracking,
        recognition_interval=recognition_interval,
        enable_zone_detection=enable_zone_detection,
        detection_zones=detection_zones,
        motion_detector=motion_detector,
    )
    
    # Update max_detection_resolution from config
    system.max_detection_resolution = max_detection_resolution
    
    return system
