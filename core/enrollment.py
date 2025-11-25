"""
User Enrollment System for Face Recognition
============================================

Handles user registration with multiple face images. Extracts embeddings from
multiple photos, computes an average embedding, and stores in vector database.

Key Features:
- Multi-image enrollment (3-5 photos recommended)
- Quality filtering (blur, brightness, alignment checks)
- Average embedding computation for robustness
- Batch enrollment from directory
- Webcam capture support (future)

Enrollment Best Practices:
--------------------------
1. Use 3-5 different photos per person
2. Vary angles: frontal, slight left, slight right
3. Vary expressions: neutral, smile
4. Ensure good lighting (not too bright/dark)
5. Avoid heavy occlusions (sunglasses, masks)

Mathematical Foundation:
-----------------------
Average embedding: e_avg = Σ(e_i) / n, then L2-normalize
This reduces noise and improves recognition robustness.

Author: Face Recognition System
Date: 2024
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import cv2
from datetime import datetime

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase


class EnrollmentSystem:
    """
    System for enrolling users into face recognition database.
    
    Handles multi-image enrollment with quality checks and embedding averaging.
    
    Attributes:
        detector: RetinaFace detector for face detection
        aligner: Face aligner for landmark-based alignment
        embedder: ArcFace embedder for feature extraction
        vector_db: FAISS vector database for storage
        min_quality: Minimum quality threshold for enrollment
    """
    
    def __init__(self, 
                 detector: RetinaFaceDetector,
                 aligner: FaceAligner,
                 embedder: ArcFaceEmbedder,
                 vector_db: VectorDatabase,
                 min_quality: str = "GOOD"):
        """
        Initialize enrollment system.
        
        Args:
            detector: Face detector instance
            aligner: Face aligner instance
            embedder: Face embedder instance
            vector_db: Vector database instance
            min_quality: Minimum quality level ("EXCELLENT", "GOOD", "FAIR")
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.vector_db = vector_db
        self.min_quality = min_quality
        
        # Quality ranking
        self.quality_levels = {"EXCELLENT": 3, "GOOD": 2, "FAIR": 1, "POOR": 0}
        
        print("✓ Enrollment system initialized")
        print(f"  Minimum quality: {min_quality}")
    
    def _process_single_image(self, image: np.ndarray, 
                             image_path: str = "unknown") -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Process a single image and extract face embedding.
        
        Args:
            image: Input image (BGR format)
            image_path: Path to image (for logging)
        
        Returns:
            (embedding, quality_info) tuple, or None if processing failed
        """
        # Detect faces
        detections = self.detector.detect(image)
        
        if len(detections) == 0:
            print(f"  [WARNING] No faces detected in {image_path}")
            return None
        
        if len(detections) > 1:
            print(f"  [WARNING] Multiple faces detected in {image_path}, using first one")
        
        # Use first detection
        detection = detections[0]
        bbox = detection['bbox']
        landmarks = detection['landmarks']
        confidence = detection['confidence']
        
        # Align face
        aligned_face = self.aligner.align_face(image, landmarks)
        
        if aligned_face is None:
            print(f"  [WARNING] Alignment failed for {image_path}")
            return None
        
        # Extract embedding with quality check
        result = self.embedder.extract_embedding(aligned_face, return_quality=True)
        embedding = result['embedding']
        quality = result['quality']
        quality_info = result['quality_info']
        
        # Check quality threshold
        if self.quality_levels[quality] < self.quality_levels[self.min_quality]:
            print(f"  [WARNING] Quality too low for {image_path}: {quality}")
            print(f"    Blur: {quality_info['blur_score']:.2f}, "
                  f"Brightness: {quality_info['brightness']:.2f}")
            return None
        
        quality_info['detection_confidence'] = confidence
        
        return embedding, quality_info
    
    def enroll_user(self, 
                    user_id: str,
                    image_paths: List[str],
                    metadata: Optional[Dict] = None) -> bool:
        """
        Enroll a user with multiple images.
        
        Args:
            user_id: Unique identifier for the user (e.g., employee ID, name)
            image_paths: List of image file paths for enrollment
            metadata: Optional user metadata (department, date, etc.)
        
        Returns:
            True if enrollment successful, False otherwise
        """
        print(f"\n{'='*60}")
        print(f"Enrolling user: {user_id}")
        print(f"{'='*60}")
        print(f"Processing {len(image_paths)} images...")
        
        embeddings = []
        quality_scores = []
        successful_images = []
        
        # Process each image
        for i, image_path in enumerate(image_paths, 1):
            print(f"\nImage {i}/{len(image_paths)}: {Path(image_path).name}")
            
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"  [ERROR] Could not read image: {image_path}")
                continue
            
            # Process image
            result = self._process_single_image(image, image_path)
            
            if result is not None:
                embedding, quality_info = result
                embeddings.append(embedding)
                quality_scores.append(quality_info)
                successful_images.append(image_path)
                
                print(f"  ✓ Processed successfully")
                print(f"    Quality: {quality_info['quality']}")
                print(f"    Blur: {quality_info['blur_score']:.2f}")
                print(f"    Brightness: {quality_info['brightness']:.2f}")
                print(f"    Detection confidence: {quality_info['detection_confidence']:.3f}")
        
        # Check if we have enough valid embeddings
        if len(embeddings) == 0:
            print(f"\n[ERROR] No valid embeddings extracted for user '{user_id}'")
            print("  Please provide images with better quality (good lighting, clear faces)")
            return False
        
        if len(embeddings) < 2:
            print(f"\n[WARNING] Only {len(embeddings)} valid image(s) found")
            print("  Recommendation: Use 3-5 images for better recognition accuracy")
        
        # Compute average embedding
        embeddings_array = np.array(embeddings)
        avg_embedding = np.mean(embeddings_array, axis=0)
        
        # L2-normalize the average embedding
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        
        # Prepare metadata
        enrollment_metadata = {
            'enrollment_date': datetime.now().isoformat(),
            'num_images': len(embeddings),
            'successful_images': len(successful_images),
            'total_images': len(image_paths),
            'avg_quality': np.mean([q['blur_score'] for q in quality_scores]),
            'avg_brightness': np.mean([q['brightness'] for q in quality_scores]),
            'avg_detection_confidence': np.mean([q['detection_confidence'] for q in quality_scores]),
            'image_paths': [str(Path(p).name) for p in successful_images]
        }
        
        # Merge with user-provided metadata
        if metadata:
            enrollment_metadata.update(metadata)
        
        # Add to vector database
        success = self.vector_db.add_user(user_id, avg_embedding, enrollment_metadata)
        
        if success:
            print(f"\n{'='*60}")
            print(f"✓ Successfully enrolled user: {user_id}")
            print(f"{'='*60}")
            print(f"  Images processed: {len(successful_images)}/{len(image_paths)}")
            print(f"  Average quality score: {enrollment_metadata['avg_quality']:.2f}")
            print(f"  Embedding dimension: {len(avg_embedding)}")
            print(f"  Total users in database: {len(self.vector_db)}")
        else:
            print(f"\n[ERROR] Failed to add user '{user_id}' to database")
            print("  User may already exist. Use update_user() to modify.")
        
        return success
    
    def enroll_from_directory(self, 
                              user_id: str,
                              directory: str,
                              extensions: Tuple[str] = ('.jpg', '.jpeg', '.png'),
                              metadata: Optional[Dict] = None) -> bool:
        """
        Enroll a user using all images from a directory.
        
        Args:
            user_id: Unique user identifier
            directory: Path to directory containing user images
            extensions: Allowed image file extensions
            metadata: Optional user metadata
        
        Returns:
            True if enrollment successful, False otherwise
        """
        dir_path = Path(directory)
        
        if not dir_path.exists():
            print(f"[ERROR] Directory not found: {directory}")
            return False
        
        # Find all images (case-insensitive)
        image_paths = set()  # Use set to avoid duplicates
        for ext in extensions:
            image_paths.update(dir_path.glob(f"*{ext}"))
            image_paths.update(dir_path.glob(f"*{ext.upper()}"))
        
        image_paths = [str(p) for p in sorted(image_paths)]
        
        if len(image_paths) == 0:
            print(f"[ERROR] No images found in {directory}")
            print(f"  Supported extensions: {extensions}")
            return False
        
        print(f"Found {len(image_paths)} images in {directory}")
        
        return self.enroll_user(user_id, image_paths, metadata)
    
    def update_user(self, 
                    user_id: str,
                    image_paths: List[str],
                    metadata: Optional[Dict] = None) -> bool:
        """
        Update an existing user's enrollment with new images.
        
        Args:
            user_id: User ID to update
            image_paths: New image paths
            metadata: Optional metadata to update
        
        Returns:
            True if update successful, False otherwise
        """
        if user_id not in self.vector_db.user_ids:
            print(f"[ERROR] User '{user_id}' not found in database")
            print("  Use enroll_user() to add a new user")
            return False
        
        # Get existing metadata
        old_metadata = self.vector_db.get_user_metadata(user_id)
        
        # Delete old entry
        self.vector_db.delete_user(user_id)
        
        # Re-enroll with new images
        combined_metadata = old_metadata.copy() if old_metadata else {}
        if metadata:
            combined_metadata.update(metadata)
        combined_metadata['last_updated'] = datetime.now().isoformat()
        
        return self.enroll_user(user_id, image_paths, combined_metadata)
    
    def batch_enroll(self, 
                     user_directories: Dict[str, str],
                     metadata_dict: Optional[Dict[str, Dict]] = None) -> Dict[str, bool]:
        """
        Enroll multiple users from their respective directories.
        
        Args:
            user_directories: Dict mapping user_id to directory path
            metadata_dict: Optional dict mapping user_id to metadata
        
        Returns:
            Dict mapping user_id to enrollment success status
        """
        print(f"\n{'='*60}")
        print(f"Batch Enrollment: {len(user_directories)} users")
        print(f"{'='*60}")
        
        results = {}
        metadata_dict = metadata_dict or {}
        
        for i, (user_id, directory) in enumerate(user_directories.items(), 1):
            print(f"\n[{i}/{len(user_directories)}] Processing: {user_id}")
            
            metadata = metadata_dict.get(user_id, {})
            success = self.enroll_from_directory(user_id, directory, metadata=metadata)
            results[user_id] = success
        
        # Summary
        successful = sum(results.values())
        print(f"\n{'='*60}")
        print(f"Batch Enrollment Complete")
        print(f"{'='*60}")
        print(f"  Successful: {successful}/{len(user_directories)}")
        print(f"  Failed: {len(user_directories) - successful}")
        print(f"  Total users in database: {len(self.vector_db)}")
        
        return results
    
    def remove_user(self, user_id: str) -> bool:
        """
        Remove a user from the database.
        
        Args:
            user_id: User ID to remove
        
        Returns:
            True if removed successfully, False otherwise
        """
        return self.vector_db.delete_user(user_id)
    
    def list_users(self) -> List[Dict]:
        """
        List all enrolled users.
        
        Returns:
            List of user information dictionaries
        """
        return self.vector_db.get_all_users()
    
    def get_user_info(self, user_id: str) -> Optional[Dict]:
        """
        Get information about a specific user.
        
        Args:
            user_id: User ID to query
        
        Returns:
            User metadata dictionary, or None if not found
        """
        return self.vector_db.get_user_metadata(user_id)


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
    
    # Create enrollment system
    enrollment = EnrollmentSystem(detector, aligner, embedder, vector_db, min_quality="GOOD")
    
    # Enroll a user (example)
    # enrollment.enroll_user(
    #     user_id="john_doe",
    #     image_paths=["data/users/john/photo1.jpg", "data/users/john/photo2.jpg"],
    #     metadata={"department": "Engineering", "employee_id": "E001"}
    # )
    
    print("\nEnrollment system ready!")
    print("Use enroll_user() or enroll_from_directory() to add users.")
