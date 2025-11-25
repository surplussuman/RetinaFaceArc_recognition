"""
Adaptive Threshold Recognition System
=====================================

Implements quality-aware adaptive thresholding for robust recognition.

Mathematical Foundation:
-----------------------
Fixed threshold assumes uniform embedding quality, but real-world images
vary dramatically in quality. Solution: Adjust threshold based on quality:

    threshold_eff = threshold_base - α * (1 - quality_confidence)
    
Where:
- quality_confidence ∈ [0, 1]: Estimated embedding reliability
- α: Sensitivity parameter (how much to lower threshold for poor quality)
- threshold_base: Standard threshold for high-quality images (0.4-0.5)

Quality Confidence Model:
-------------------------
Combines multiple quality metrics into single confidence score:

    quality_confidence = w_blur * f_blur(blur_score) +
                        w_brightness * f_brightness(brightness) +
                        w_resolution * f_resolution(face_size)
    
Where f_* are normalized quality functions mapping metrics to [0, 1].

Example:
- High quality (blur=500, bright=120, size=200px): confidence=0.9, threshold=0.40
- Medium quality (blur=100, bright=100, size=80px): confidence=0.6, threshold=0.32
- Low quality (blur=20, bright=60, size=30px): confidence=0.3, threshold=0.19

Author: Face Recognition System
Date: 2024
"""

import numpy as np
import cv2
from typing import Dict, Tuple, Optional
from pathlib import Path

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase


class AdaptiveThresholdRecognizer:
    """
    Face recognition with quality-adaptive thresholding.
    """
    
    def __init__(self,
                 detector: RetinaFaceDetector,
                 aligner: FaceAligner,
                 embedder: ArcFaceEmbedder,
                 vector_db: VectorDatabase,
                 base_threshold: float = 0.4,
                 min_threshold: float = 0.2,
                 alpha: float = 0.3):
        """
        Initialize adaptive threshold recognizer.
        
        Args:
            detector: Face detector
            aligner: Face aligner
            embedder: Face embedder
            vector_db: Vector database
            base_threshold: Threshold for high-quality images (default: 0.4)
            min_threshold: Minimum threshold for poor quality (default: 0.2)
            alpha: Threshold sensitivity to quality (default: 0.3)
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.vector_db = vector_db
        self.base_threshold = base_threshold
        self.min_threshold = min_threshold
        self.alpha = alpha
        
        print(f"✓ Adaptive threshold recognizer initialized")
        print(f"  Base threshold: {base_threshold}")
        print(f"  Min threshold: {min_threshold}")
        print(f"  Alpha: {alpha}")
    
    def compute_quality_confidence(self,
                                   blur_score: float,
                                   brightness: float,
                                   face_size: float) -> float:
        """
        Compute quality confidence score from image metrics.
        
        Args:
            blur_score: Laplacian variance (higher = sharper)
            brightness: Mean brightness [0, 255]
            face_size: Face bounding box area in pixels
            
        Returns:
            Quality confidence ∈ [0, 1] (1 = excellent, 0 = very poor)
        """
        # Blur confidence (sigmoid centered at 100, scaled to [0,1])
        blur_conf = 1 / (1 + np.exp(-(blur_score - 100) / 50))
        
        # Brightness confidence (optimal range: 80-180)
        if 80 <= brightness <= 180:
            bright_conf = 1.0
        elif brightness < 80:
            bright_conf = max(0.0, brightness / 80)
        else:  # brightness > 180
            bright_conf = max(0.0, 1.0 - (brightness - 180) / 75)
        
        # Resolution confidence (sigmoid centered at 5000 pixels)
        resolution_conf = 1 / (1 + np.exp(-(face_size - 5000) / 3000))
        
        # Weighted combination
        w_blur = 0.4
        w_brightness = 0.3
        w_resolution = 0.3
        
        confidence = (w_blur * blur_conf + 
                     w_brightness * bright_conf + 
                     w_resolution * resolution_conf)
        
        return float(confidence)
    
    def compute_adaptive_threshold(self, quality_confidence: float) -> float:
        """
        Compute adaptive threshold based on quality confidence.
        
        Args:
            quality_confidence: Quality confidence score [0, 1]
            
        Returns:
            Adaptive threshold for this image
        """
        # threshold_eff = threshold_base - α * (1 - quality_confidence)
        threshold = self.base_threshold - self.alpha * (1 - quality_confidence)
        
        # Clip to minimum threshold
        threshold = max(self.min_threshold, threshold)
        
        return float(threshold)
    
    def recognize_face(self,
                      image: np.ndarray,
                      detection: Dict,
                      return_details: bool = False) -> Tuple[Optional[str], float, Dict]:
        """
        Recognize single face with adaptive thresholding.
        
        Args:
            image: Input BGR image
            detection: Face detection dictionary
            return_details: Return detailed quality metrics
            
        Returns:
            Tuple of (user_id or None, similarity, details_dict)
        """
        landmarks = detection['landmarks']
        bbox = detection['bbox']
        
        # Align face
        aligned_face = self.aligner.align_face(image, landmarks)
        if aligned_face is None:
            return None, 0.0, {'error': 'Alignment failed'}
        
        # Extract embedding with quality
        result = self.embedder.extract_embedding(aligned_face, return_quality=True)
        if result is None or result['embedding'] is None:
            return None, 0.0, {'error': 'Embedding extraction failed'}
        
        embedding = result['embedding']
        quality_info = result['quality_info']
        
        # Compute face size
        face_size = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        
        # Compute quality confidence
        quality_confidence = self.compute_quality_confidence(
            blur_score=quality_info['blur_score'],
            brightness=quality_info['brightness'],
            face_size=face_size
        )
        
        # Compute adaptive threshold
        adaptive_threshold = self.compute_adaptive_threshold(quality_confidence)
        
        # Search in database
        results = self.vector_db.search(embedding, k=1, threshold=adaptive_threshold)
        
        details = {
            'blur_score': quality_info['blur_score'],
            'brightness': quality_info['brightness'],
            'face_size': face_size,
            'quality_level': result['quality'],
            'quality_confidence': quality_confidence,
            'adaptive_threshold': adaptive_threshold,
            'base_threshold': self.base_threshold
        }
        
        if len(results) > 0:
            user_id, similarity = results[0]
            details['similarity'] = similarity
            return user_id, similarity, details
        else:
            details['similarity'] = 0.0
            return None, 0.0, details
    
    def recognize_image(self,
                       image_path: Path,
                       return_all_faces: bool = False) -> Dict:
        """
        Recognize all faces in image with adaptive thresholding.
        
        Args:
            image_path: Path to image file
            return_all_faces: Return details for all faces (not just recognized)
            
        Returns:
            Dictionary with recognition results and statistics
        """
        # Load image
        image = cv2.imread(str(image_path))
        if image is None:
            return {'error': 'Failed to load image'}
        
        # Detect faces
        detections = self.detector.detect(image)
        
        if len(detections) == 0:
            return {'error': 'No faces detected', 'num_faces': 0}
        
        # Process each face
        recognized_faces = []
        all_faces = []
        
        for i, detection in enumerate(detections):
            user_id, similarity, details = self.recognize_face(image, detection, return_details=True)
            
            face_info = {
                'face_index': i,
                'detection_confidence': detection['confidence'],
                'bbox': detection['bbox'],
                **details
            }
            
            if user_id is not None:
                face_info['user_id'] = user_id
                recognized_faces.append(face_info)
            
            all_faces.append(face_info)
        
        # Compute statistics
        result = {
            'image': str(image_path.name),
            'num_faces': len(detections),
            'num_recognized': len(recognized_faces),
            'recognized_faces': recognized_faces
        }
        
        if return_all_faces:
            result['all_faces'] = all_faces
        
        # Quality statistics
        if all_faces:
            result['avg_quality_confidence'] = np.mean([f['quality_confidence'] for f in all_faces])
            result['avg_adaptive_threshold'] = np.mean([f['adaptive_threshold'] for f in all_faces])
            result['avg_blur'] = np.mean([f['blur_score'] for f in all_faces])
        
        return result


def create_adaptive_recognizer(vector_db: VectorDatabase,
                               base_threshold: float = 0.4,
                               min_threshold: float = 0.2,
                               alpha: float = 0.3) -> AdaptiveThresholdRecognizer:
    """
    Create adaptive threshold recognizer with all components.
    
    Args:
        vector_db: Vector database
        base_threshold: Threshold for high-quality images
        min_threshold: Minimum threshold for poor quality
        alpha: Threshold sensitivity
        
    Returns:
        Configured AdaptiveThresholdRecognizer
    """
    print("Initializing adaptive threshold recognition system...")
    
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    return AdaptiveThresholdRecognizer(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        vector_db=vector_db,
        base_threshold=base_threshold,
        min_threshold=min_threshold,
        alpha=alpha
    )
