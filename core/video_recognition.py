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
                 process_every_n_frames: int = 1):
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
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.vector_db = vector_db
        self.recognition_threshold = recognition_threshold
        self.track_iou_threshold = track_iou_threshold
        self.max_frames_missing = max_frames_missing
        self.process_every_n_frames = process_every_n_frames
        
        # Performance optimization: max resolution for detection
        # For real-time CCTV on CPU: 640×360 (half HD)
        self.max_detection_resolution = (640, 360)  # Aggressive for real-time
        
        self.tracks: List[FaceTrack] = []
        self.next_track_id = 0
        self.frame_count = 0
        
        print(f"✓ Video recognition system initialized")
        print(f"  Recognition threshold: {recognition_threshold}")
        print(f"  Track IOU threshold: {track_iou_threshold}")
        print(f"  Process every {process_every_n_frames} frames")
        print(f"  Max detection resolution: {self.max_detection_resolution[0]}×{self.max_detection_resolution[1]}")
    
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
        Process single frame: detect, align, embed, track, recognize.
        
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
        
        # Smart resize for detection (only if frame is large)
        detection_frame, scale = self.resize_for_detection(frame)
        
        # Detect faces on resized frame
        detections = self.detector.detect(detection_frame)
        
        # Scale bboxes and landmarks back to original size if frame was resized
        if scale != 1.0:
            for detection in detections:
                # Ensure arrays before scaling
                bbox = detection['bbox']
                landmarks = detection['landmarks']
                
                if isinstance(bbox, list):
                    bbox = np.array(bbox)
                if isinstance(landmarks, list):
                    landmarks = np.array(landmarks)
                
                detection['bbox'] = bbox / scale
                detection['landmarks'] = landmarks / scale
        
        # Extract embeddings (BATCH PROCESSING)
        embeddings = []
        valid_detections = []
        aligned_faces = []
        
        # First, align all faces
        for detection in detections:
            landmarks = detection['landmarks']
            aligned_face = self.aligner.align_face(frame, landmarks)
            if aligned_face is not None:
                aligned_faces.append(aligned_face)
                valid_detections.append(detection)
        
        # Batch extract ALL embeddings at once (FAST!)
        if len(aligned_faces) > 0:
            embeddings_array = self.embedder.extract_embeddings_batch(aligned_faces)
            embeddings = list(embeddings_array) if len(embeddings_array) > 0 else []
            
            # If batch failed, remove detections
            if len(embeddings) != len(valid_detections):
                valid_detections = valid_detections[:len(embeddings)]
        
        # Match to tracks
        matches = self.match_detections_to_tracks(valid_detections, embeddings)
        matched_detection_indices = set([m[0] for m in matches])
        matched_track_indices = set([m[1] for m in matches])
        
        # Update matched tracks
        for det_idx, track_idx in matches:
            track = self.tracks[track_idx]
            track.update(
                valid_detections[det_idx]['bbox'],
                embeddings[det_idx],
                frame_idx
            )
            
            # Recognize using averaged embedding
            avg_embedding = track.get_average_embedding()
            identity, confidence = self.recognize_face(avg_embedding)
            
            if identity is not None:
                track.vote_identity(identity, confidence)
        
        # Create new tracks for unmatched detections
        for det_idx, detection in enumerate(valid_detections):
            if det_idx not in matched_detection_indices:
                track = FaceTrack(
                    self.next_track_id,
                    detection['bbox'],
                    embeddings[det_idx],
                    frame_idx
                )
                self.next_track_id += 1
                
                # Try immediate recognition
                identity, confidence = self.recognize_face(embeddings[det_idx])
                if identity is not None:
                    track.vote_identity(identity, confidence)
                
                self.tracks.append(track)
        
        # Age out and remove stale tracks
        self.tracks = [
            track for track in self.tracks
            if track.frames_since_seen < self.max_frames_missing
        ]
        
        # Prepare results
        active_tracks = []
        for track in self.tracks:
            if track.frames_since_seen == 0:  # Currently visible
                # Ensure bbox is numpy array
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
            'num_detections': len(valid_detections),
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
                
                if result['skipped']:
                    stats['skipped_frames'] += 1
                else:
                    stats['processed_frames'] += 1
                    stats['total_detections'] += result['num_detections']
                    
                    # Draw annotations
                    annotated_frame = self.draw_annotations(frame, result)
                    
                    # Log recognitions
                    timestamp = frame_idx / fps
                    for track in result['active_tracks']:
                        if track['identity'] is not None:
                            stats['recognized_identities'].add(track['identity'])
                            recognition_log.append({
                                'frame': frame_idx,
                                'timestamp': timestamp,
                                'track_id': track['track_id'],
                                'identity': track['identity'],
                                'confidence': track['confidence']
                            })
                    
                    # Write to output video
                    if writer is not None:
                        writer.write(annotated_frame)
                    
                    # Callbacks for external monitoring
                    if frame_callback is not None:
                        frame_callback(annotated_frame, {
                            'frame': frame_idx,
                            'detections': result['num_detections'],
                            'tracks': result['num_tracks']
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
        
        for track in result['active_tracks']:
            bbox = track['bbox'].astype(int)
            identity = track['identity']
            confidence = track['confidence']
            
            # Choose color based on recognition
            if identity is not None:
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
                            process_every_n_frames: int = 1) -> VideoRecognitionSystem:
    """
    Create video recognition system with all components.
    
    Args:
        vector_db: Vector database with enrolled users
        recognition_threshold: Minimum similarity for recognition
        process_every_n_frames: Process every N frames (1=all, 2=half, etc)
        
    Returns:
        Configured VideoRecognitionSystem
    """
    print("Initializing video recognition system...")
    
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    return VideoRecognitionSystem(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        vector_db=vector_db,
        recognition_threshold=recognition_threshold,
        process_every_n_frames=process_every_n_frames
    )
