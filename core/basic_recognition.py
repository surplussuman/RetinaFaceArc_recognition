"""
Basic Face Recognition Pipeline (Phase 1)

Integrates:
- RetinaFace detection
- 5-point affine alignment  
- ArcFace embedding extraction
- Basic similarity matching

This is the foundational pipeline before FAISS integration.
"""

import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
import time
from pathlib import Path

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder


class BasicFaceRecognizer:
    """
    Basic face recognition pipeline.
    
    Pipeline: Frame → Detection → Alignment → Embedding → Similarity
    """
    
    def __init__(
        self,
        detector_config: str = "config/detector_config.yaml",
        embedder_config: str = "config/embedder_config.yaml",
        alignment_size: int = 112,
        similarity_threshold: float = 0.40
    ):
        """
        Initialize recognition pipeline.
        
        Args:
            detector_config: Path to detector configuration
            embedder_config: Path to embedder configuration
            alignment_size: Alignment output size (112 or 160)
            similarity_threshold: Cosine similarity threshold for matching
        """
        print("="*60)
        print("Initializing Basic Face Recognition Pipeline")
        print("="*60)
        
        # Initialize components
        self.detector = RetinaFaceDetector(detector_config)
        self.aligner = FaceAligner(output_size=alignment_size)
        self.embedder = ArcFaceEmbedder(embedder_config)
        
        self.similarity_threshold = similarity_threshold
        
        # Gallery: store known face embeddings
        # Format: {"person_id": embedding_vector}
        self.gallery = {}
        
        print(f"\n✓ Pipeline initialized")
        print(f"  Similarity threshold: {self.similarity_threshold}")
        print("="*60)
    
    def add_to_gallery(
        self,
        person_id: str,
        image: np.ndarray,
        replace: bool = True
    ) -> bool:
        """
        Add a person to the gallery.
        
        Args:
            person_id: Unique identifier for the person
            image: Image containing the person's face (BGR)
            replace: Replace existing entry if person_id exists
        
        Returns:
            Success status
        """
        if person_id in self.gallery and not replace:
            print(f"Person '{person_id}' already exists in gallery")
            return False
        
        # Detect face
        detections = self.detector.detect(image)
        
        if len(detections) == 0:
            print(f"No face detected for '{person_id}'")
            return False
        
        if len(detections) > 1:
            print(f"Multiple faces detected for '{person_id}', using first one")
        
        # Use first detection
        det = detections[0]
        landmarks = np.array(det['landmarks'])
        
        # Align face
        aligned = self.aligner.align_face(image, landmarks)
        
        # Check quality
        is_good, metrics = self.embedder.check_face_quality(aligned)
        if not is_good:
            print(f"Warning: Low quality face for '{person_id}': {metrics['issues']}")
        
        # Extract embedding
        embedding = self.embedder.extract_embedding(aligned)
        
        # Add to gallery
        self.gallery[person_id] = embedding
        
        print(f"✓ Added '{person_id}' to gallery (quality: {'GOOD' if is_good else 'POOR'})")
        return True
    
    def add_multiple_to_gallery(
        self,
        person_id: str,
        images: List[np.ndarray]
    ) -> bool:
        """
        Add person using multiple images (more robust).
        
        Extracts embeddings from all images and stores the average.
        
        Args:
            person_id: Unique identifier for the person
            images: List of images containing the person's face
        
        Returns:
            Success status
        """
        embeddings = []
        
        for i, image in enumerate(images):
            # Detect face
            detections = self.detector.detect(image)
            
            if len(detections) == 0:
                print(f"  Image {i+1}/{len(images)}: No face detected")
                continue
            
            det = detections[0]
            landmarks = np.array(det['landmarks'])
            
            # Align face
            aligned = self.aligner.align_face(image, landmarks)
            
            # Check quality
            is_good, metrics = self.embedder.check_face_quality(aligned)
            
            # Extract embedding
            embedding = self.embedder.extract_embedding(aligned)
            embeddings.append(embedding)
            
            print(f"  Image {i+1}/{len(images)}: {'✓' if is_good else '⚠'} Quality: {'GOOD' if is_good else 'POOR'}")
        
        if len(embeddings) == 0:
            print(f"Failed to extract any embeddings for '{person_id}'")
            return False
        
        # Aggregate embeddings
        aggregated = self.embedder.aggregate_embeddings(np.array(embeddings), method="mean")
        
        # Add to gallery
        self.gallery[person_id] = aggregated
        
        print(f"✓ Added '{person_id}' to gallery ({len(embeddings)}/{len(images)} images used)")
        return True
    
    def recognize_face(
        self,
        image: np.ndarray,
        return_all: bool = False
    ) -> List[Dict]:
        """
        Recognize faces in an image.
        
        Args:
            image: Input image (BGR)
            return_all: Return all detections (even unknown)
        
        Returns:
            List of recognition results with format:
            {
                'bbox': [x1, y1, x2, y2],
                'landmarks': [[x1,y1], ...],
                'confidence': detection_confidence,
                'identity': person_id or "Unknown",
                'similarity': cosine_similarity,
                'aligned_face': aligned_face_image
            }
        """
        # Detect faces
        detections = self.detector.detect(image)
        
        if len(detections) == 0:
            return []
        
        results = []
        
        for det in detections:
            landmarks = np.array(det['landmarks'])
            
            # Align face
            aligned = self.aligner.align_face(image, landmarks)
            
            # Extract embedding
            embedding = self.embedder.extract_embedding(aligned)
            
            # Find best match in gallery
            best_match = None
            best_similarity = -1.0
            
            for person_id, gallery_embedding in self.gallery.items():
                similarity = self.embedder.compute_similarity(embedding, gallery_embedding)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = person_id
            
            # Determine identity
            if best_similarity >= self.similarity_threshold:
                identity = best_match
            else:
                identity = "Unknown"
            
            result = {
                'bbox': det['bbox'],
                'landmarks': det['landmarks'],
                'confidence': det['confidence'],
                'identity': identity,
                'similarity': best_similarity,
                'aligned_face': aligned
            }
            
            if return_all or identity != "Unknown":
                results.append(result)
        
        return results
    
    def visualize_results(
        self,
        image: np.ndarray,
        results: List[Dict],
        show_landmarks: bool = True,
        show_confidence: bool = True
    ) -> np.ndarray:
        """
        Visualize recognition results on image.
        
        Args:
            image: Original image
            results: Recognition results
            show_landmarks: Draw landmarks
            show_confidence: Show confidence scores
        
        Returns:
            Annotated image
        """
        vis = image.copy()
        
        for result in results:
            bbox = result['bbox']
            landmarks = result['landmarks']
            identity = result['identity']
            similarity = result['similarity']
            det_conf = result['confidence']
            
            # Draw bounding box
            x1, y1, x2, y2 = [int(v) for v in bbox]
            color = (0, 255, 0) if identity != "Unknown" else (128, 128, 128)
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            
            # Draw landmarks
            if show_landmarks:
                for lm in landmarks:
                    cv2.circle(vis, (int(lm[0]), int(lm[1])), 2, (255, 0, 0), -1)
            
            # Draw label
            label = identity
            if show_confidence:
                label += f" ({similarity:.2f})"
            
            # Text background
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(vis, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
            
            # Text
            cv2.putText(
                vis, label, (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2
            )
        
        return vis
    
    def benchmark(self, image: np.ndarray, num_runs: int = 10) -> Dict[str, float]:
        """
        Benchmark pipeline performance.
        
        Args:
            image: Test image
            num_runs: Number of runs for averaging
        
        Returns:
            Timing statistics
        """
        times = {
            'detection': [],
            'alignment': [],
            'embedding': [],
            'total': []
        }
        
        for _ in range(num_runs):
            # Detection
            t0 = time.time()
            detections = self.detector.detect(image)
            t1 = time.time()
            times['detection'].append((t1 - t0) * 1000)
            
            if len(detections) == 0:
                continue
            
            det = detections[0]
            landmarks = np.array(det['landmarks'])
            
            # Alignment
            t0 = time.time()
            aligned = self.aligner.align_face(image, landmarks)
            t1 = time.time()
            times['alignment'].append((t1 - t0) * 1000)
            
            # Embedding
            t0 = time.time()
            embedding = self.embedder.extract_embedding(aligned)
            t1 = time.time()
            times['embedding'].append((t1 - t0) * 1000)
            
            times['total'].append(
                times['detection'][-1] + 
                times['alignment'][-1] + 
                times['embedding'][-1]
            )
        
        # Compute statistics
        stats = {}
        for key, values in times.items():
            if len(values) > 0:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_min'] = np.min(values)
                stats[f'{key}_max'] = np.max(values)
        
        return stats


if __name__ == "__main__":
    # Test basic recognizer
    print("\n" + "="*60)
    print("Testing Basic Face Recognition Pipeline")
    print("="*60 + "\n")
    
    recognizer = BasicFaceRecognizer(
        alignment_size=112,
        similarity_threshold=0.40
    )
    
    # Create synthetic test images
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    print("\n" + "="*60)
    print("Pipeline Test Complete")
    print("="*60)
    print("\nNote: This test uses random images.")
    print("For real testing, use actual face images with the test script.")
