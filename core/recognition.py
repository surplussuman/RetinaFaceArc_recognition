"""
Face Recognition Pipeline
=========================

Complete end-to-end face recognition pipeline integrating detection, alignment,
embedding extraction, and vector database search.

Key Features:
- Single-image and batch recognition
- Quality filtering for reliable recognition
- Configurable similarity thresholds
- Result visualization with bounding boxes and labels
- Support for unknown face detection

Pipeline Flow:
--------------
1. Detect faces (RetinaFace)
2. Align faces (5-point landmarks)
3. Extract embeddings (ArcFace)
4. Search database (FAISS)
5. Return recognized identities with confidence

Author: Face Recognition System
Date: 2024
"""

import numpy as np
import cv2
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase


class FaceRecognitionPipeline:
    """
    Complete face recognition pipeline.
    
    Integrates all components for end-to-end recognition from raw images.
    
    Attributes:
        detector: RetinaFace face detector
        aligner: Landmark-based face aligner
        embedder: ArcFace feature extractor
        vector_db: FAISS vector database for identity matching
        similarity_threshold: Minimum similarity for positive identification
        min_quality: Minimum quality level for recognition
    """
    
    def __init__(self,
                 detector: RetinaFaceDetector,
                 aligner: FaceAligner,
                 embedder: ArcFaceEmbedder,
                 vector_db: VectorDatabase,
                 similarity_threshold: float = 0.4,
                 min_quality: str = "FAIR"):
        """
        Initialize recognition pipeline.
        
        Args:
            detector: Face detector instance
            aligner: Face aligner instance
            embedder: Face embedder instance
            vector_db: Vector database with enrolled users
            similarity_threshold: Minimum similarity for recognition (0-1)
            min_quality: Minimum quality level for recognition
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.vector_db = vector_db
        self.similarity_threshold = similarity_threshold
        self.min_quality = min_quality
        
        # Quality ranking
        self.quality_levels = {"EXCELLENT": 3, "GOOD": 2, "FAIR": 1, "POOR": 0}
        
        print("✓ Face Recognition Pipeline initialized")
        print(f"  Similarity threshold: {similarity_threshold}")
        print(f"  Minimum quality: {min_quality}")
        print(f"  Enrolled users: {len(vector_db)}")
    
    def recognize(self, image: np.ndarray, 
                  top_k: int = 1,
                  return_all_faces: bool = True,
                  min_detection_confidence: float = 0.5) -> List[Dict]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image (BGR format)
            top_k: Number of top matches to return per face
            return_all_faces: If True, return all detected faces (even unknown)
            min_detection_confidence: Minimum detection confidence threshold
        
        Returns:
            List of recognition results, each containing:
                - bbox: Face bounding box [x1, y1, x2, y2]
                - landmarks: 5-point facial landmarks
                - detection_confidence: Detection confidence score
                - identity: Recognized user ID or "Unknown"
                - similarity: Similarity score to matched identity
                - top_matches: List of top-k matches [(user_id, similarity), ...]
                - quality: Face quality level
                - quality_info: Detailed quality metrics
                - embedding: Face embedding (optional)
        """
        results = []
        
        # Detect faces
        detections = self.detector.detect(image)
        
        # Filter by detection confidence
        detections = [d for d in detections if d['confidence'] >= min_detection_confidence]
        
        if len(detections) == 0:
            return results
        
        # Process each detected face
        for i, detection in enumerate(detections):
            bbox = detection['bbox']
            landmarks = detection['landmarks']
            confidence = detection['confidence']
            
            # Align face
            aligned_face = self.aligner.align_face(image, landmarks)
            
            if aligned_face is None:
                # Alignment failed
                if return_all_faces:
                    results.append({
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'detection_confidence': confidence,
                        'identity': 'Unknown',
                        'similarity': 0.0,
                        'top_matches': [],
                        'quality': 'POOR',
                        'quality_info': {},
                        'error': 'Alignment failed'
                    })
                continue
            
            # Extract embedding with quality check
            embedding_result = self.embedder.extract_embedding(aligned_face, return_quality=True)
            embedding = embedding_result['embedding']
            quality = embedding_result['quality']
            quality_info = embedding_result['quality_info']
            
            # Check quality threshold
            if self.quality_levels[quality] < self.quality_levels[self.min_quality]:
                if return_all_faces:
                    results.append({
                        'bbox': bbox,
                        'landmarks': landmarks,
                        'detection_confidence': confidence,
                        'identity': 'Unknown',
                        'similarity': 0.0,
                        'top_matches': [],
                        'quality': quality,
                        'quality_info': quality_info,
                        'error': f'Quality too low: {quality}'
                    })
                continue
            
            # Search in database
            matches = self.vector_db.search(embedding, k=top_k, threshold=self.similarity_threshold)
            
            # Determine identity
            if len(matches) > 0:
                # Recognized
                best_match_id, best_similarity = matches[0]
                identity = best_match_id
                similarity = best_similarity
            else:
                # Unknown
                identity = "Unknown"
                similarity = 0.0
            
            # Build result
            result = {
                'bbox': bbox,
                'landmarks': landmarks,
                'detection_confidence': confidence,
                'identity': identity,
                'similarity': similarity,
                'top_matches': matches,
                'quality': quality,
                'quality_info': quality_info,
                'embedding': embedding
            }
            
            results.append(result)
        
        return results
    
    def recognize_batch(self, images: List[np.ndarray], **kwargs) -> List[List[Dict]]:
        """
        Recognize faces in multiple images.
        
        Args:
            images: List of input images
            **kwargs: Additional arguments for recognize()
        
        Returns:
            List of recognition results (one list per image)
        """
        return [self.recognize(img, **kwargs) for img in images]
    
    def visualize_results(self, 
                          image: np.ndarray,
                          results: List[Dict],
                          show_landmarks: bool = False,
                          show_quality: bool = True,
                          font_scale: float = 0.6,
                          thickness: int = 2) -> np.ndarray:
        """
        Draw recognition results on image.
        
        Args:
            image: Original image
            results: Recognition results from recognize()
            show_landmarks: Whether to draw facial landmarks
            show_quality: Whether to show quality information
            font_scale: Font size for text
            thickness: Line thickness for boxes
        
        Returns:
            Annotated image
        """
        vis_image = image.copy()
        
        for result in results:
            bbox = np.array(result['bbox']).astype(int)
            landmarks = result['landmarks']
            identity = result['identity']
            similarity = result['similarity']
            confidence = result['detection_confidence']
            quality = result['quality']
            
            # Choose color based on identity
            if identity == "Unknown":
                color = (0, 0, 255)  # Red for unknown
            else:
                color = (0, 255, 0)  # Green for recognized
            
            # Draw bounding box
            cv2.rectangle(vis_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), 
                         color, thickness)
            
            # Draw landmarks
            if show_landmarks:
                for (x, y) in landmarks:
                    cv2.circle(vis_image, (int(x), int(y)), 2, (255, 255, 0), -1)
            
            # Prepare label
            if identity == "Unknown":
                label = f"Unknown"
            else:
                label = f"{identity} ({similarity:.3f})"
            
            # Add quality info
            if show_quality:
                label += f" [{quality}]"
            
            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 
                                           font_scale, thickness)
            label_w, label_h = label_size
            
            # Position label above bbox
            label_y = max(bbox[1] - 10, label_h + 5)
            cv2.rectangle(vis_image, 
                         (bbox[0], label_y - label_h - 5),
                         (bbox[0] + label_w + 5, label_y),
                         color, -1)
            
            # Draw label text
            cv2.putText(vis_image, label, 
                       (bbox[0] + 2, label_y - 3),
                       cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                       (255, 255, 255), thickness - 1)
        
        return vis_image
    
    def save_results(self, 
                     output_path: str,
                     results: List[Dict],
                     include_embeddings: bool = False):
        """
        Save recognition results to file.
        
        Args:
            output_path: Path to save results (JSON format)
            results: Recognition results
            include_embeddings: Whether to include embeddings in output
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare serializable results
        serializable_results = []
        for result in results:
            result_copy = result.copy()
            
            # Convert numpy arrays to lists
            result_copy['bbox'] = result['bbox'].tolist()
            result_copy['landmarks'] = result['landmarks'].tolist()
            
            if not include_embeddings:
                result_copy.pop('embedding', None)
            elif 'embedding' in result_copy:
                result_copy['embedding'] = result['embedding'].tolist()
            
            serializable_results.append(result_copy)
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'num_faces': len(results),
                'results': serializable_results
            }, f, indent=2)
        
        print(f"✓ Results saved to {output_path}")
    
    def get_statistics(self, results: List[Dict]) -> Dict:
        """
        Compute statistics from recognition results.
        
        Args:
            results: Recognition results
        
        Returns:
            Statistics dictionary
        """
        if len(results) == 0:
            return {
                'total_faces': 0,
                'recognized': 0,
                'unknown': 0,
                'recognition_rate': 0.0
            }
        
        recognized = sum(1 for r in results if r['identity'] != "Unknown")
        unknown = len(results) - recognized
        
        # Quality distribution
        quality_counts = {}
        for r in results:
            q = r['quality']
            quality_counts[q] = quality_counts.get(q, 0) + 1
        
        # Average confidences
        avg_detection_conf = np.mean([r['detection_confidence'] for r in results])
        recognized_results = [r for r in results if r['identity'] != "Unknown"]
        avg_similarity = np.mean([r['similarity'] for r in recognized_results]) if recognized_results else 0.0
        
        return {
            'total_faces': len(results),
            'recognized': recognized,
            'unknown': unknown,
            'recognition_rate': recognized / len(results),
            'quality_distribution': quality_counts,
            'avg_detection_confidence': float(avg_detection_conf),
            'avg_similarity': float(avg_similarity)
        }


# Example usage
if __name__ == "__main__":
    from core.detector import RetinaFaceDetector
    from core.aligner import FaceAligner
    from core.embedder import ArcFaceEmbedder
    from core.vector_db import VectorDatabase
    
    # Initialize components
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    vector_db = VectorDatabase()
    
    # Create pipeline
    pipeline = FaceRecognitionPipeline(
        detector, aligner, embedder, vector_db,
        similarity_threshold=0.4,
        min_quality="FAIR"
    )
    
    print("\nRecognition pipeline ready!")
    print("Use recognize() to process images.")
