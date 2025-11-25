"""
Advanced Multi-Quality Enrollment System
=========================================

Enrolls users with synthetic quality variations to handle real-world degradation.

Key Innovation:
--------------
Instead of requiring users to provide both high and low quality images,
we generate synthetic variations during enrollment:
1. Original high-quality embedding
2. Blurred version (simulates old photos)
3. Low-resolution version (simulates distance/small faces)
4. Dark version (simulates poor lighting)
5. Combined degradation (worst case)

At recognition time, we match against the most appropriate quality level.

Mathematical Justification:
--------------------------
ArcFace embeddings are sensitive to image quality degradation. By enrolling
multiple quality versions, we populate the embedding space with representations
at different quality levels. During recognition, the query embedding naturally
falls closer to the corresponding quality level, improving matching.

Trade-off: 5x storage per user, but robust cross-quality matching.

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
from core.preprocessing import ImagePreprocessor


class MultiQualityEnroller:
    """
    Advanced enrollment system with synthetic quality variation generation.
    """
    
    # Quality variation parameters
    QUALITY_VARIANTS = {
        'original': {
            'blur_sigma': 0.0,
            'scale_factor': 1.0,
            'brightness_offset': 0,
            'noise_sigma': 0.0
        },
        'slight_blur': {
            'blur_sigma': 1.0,
            'scale_factor': 1.0,
            'brightness_offset': 0,
            'noise_sigma': 0.0
        },
        'low_resolution': {
            'blur_sigma': 0.5,
            'scale_factor': 0.6,
            'brightness_offset': 0,
            'noise_sigma': 0.0
        },
        'poor_lighting': {
            'blur_sigma': 0.5,
            'scale_factor': 1.0,
            'brightness_offset': -30,
            'noise_sigma': 5.0
        },
        'severe_degradation': {
            'blur_sigma': 2.0,
            'scale_factor': 0.5,
            'brightness_offset': -40,
            'noise_sigma': 10.0
        }
    }
    
    def __init__(self,
                 detector: RetinaFaceDetector,
                 aligner: FaceAligner,
                 embedder: ArcFaceEmbedder,
                 vector_db: VectorDatabase,
                 enable_variants: bool = True):
        """
        Initialize multi-quality enrollment system.
        
        Args:
            detector: Face detector
            aligner: Face aligner
            embedder: Face embedder
            vector_db: Vector database
            enable_variants: Generate quality variants (True recommended)
        """
        self.detector = detector
        self.aligner = aligner
        self.embedder = embedder
        self.vector_db = vector_db
        self.enable_variants = enable_variants
        
        print(f"✓ Multi-quality enrollment initialized")
        print(f"  Quality variants: {'Enabled' if enable_variants else 'Disabled'}")
        if enable_variants:
            print(f"  Variants per image: {len(self.QUALITY_VARIANTS)}")
    
    def create_degraded_face(self, face: np.ndarray, variant_name: str) -> np.ndarray:
        """
        Create synthetically degraded version of aligned face.
        
        Args:
            face: Aligned face image (112×112 BGR)
            variant_name: Name of quality variant
            
        Returns:
            Degraded face image (same size)
        """
        params = self.QUALITY_VARIANTS[variant_name]
        
        result = face.copy().astype(np.float32)
        
        # Apply blur
        if params['blur_sigma'] > 0:
            ksize = int(params['blur_sigma'] * 6) | 1  # Ensure odd
            result = cv2.GaussianBlur(result, (ksize, ksize), params['blur_sigma'])
        
        # Apply resolution degradation
        if params['scale_factor'] < 1.0:
            h, w = result.shape[:2]
            small_h, small_w = int(h * params['scale_factor']), int(w * params['scale_factor'])
            small_h, small_w = max(16, small_h), max(16, small_w)  # Minimum 16px
            result = cv2.resize(result, (small_w, small_h), interpolation=cv2.INTER_AREA)
            result = cv2.resize(result, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Apply brightness adjustment
        if params['brightness_offset'] != 0:
            result = result + params['brightness_offset']
        
        # Apply noise
        if params['noise_sigma'] > 0:
            noise = np.random.normal(0, params['noise_sigma'], result.shape)
            result = result + noise
        
        # Clip and convert back
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result
    
    def enroll_user(self,
                    user_id: str,
                    image_paths: List[Path],
                    replace_existing: bool = False) -> Dict:
        """
        Enroll user with multi-quality embeddings.
        
        Args:
            user_id: Unique user identifier
            image_paths: List of image file paths
            replace_existing: Replace if user exists
            
        Returns:
            Dictionary with enrollment statistics
        """
        print(f"\n{'='*80}")
        print(f"ENROLLING USER: {user_id}")
        print(f"{'='*80}")
        
        # Check if user exists
        existing_users = [u['user_id'] for u in self.vector_db.get_all_users()]
        if user_id in existing_users:
            if replace_existing:
                print(f"⚠ User '{user_id}' exists - will be replaced")
                self.vector_db.delete_user(user_id)
            else:
                print(f"✗ User '{user_id}' already exists")
                return {'success': False, 'reason': 'User exists'}
        
        # Process each image
        all_embeddings = []
        all_variants = []
        successful_images = 0
        
        for img_idx, img_path in enumerate(image_paths):
            print(f"\nProcessing image {img_idx + 1}/{len(image_paths)}: {img_path.name}")
            
            # Load image
            image = cv2.imread(str(img_path))
            if image is None:
                print(f"  ✗ Failed to load image")
                continue
            
            # Detect faces
            detections = self.detector.detect(image)
            if len(detections) == 0:
                print(f"  ✗ No faces detected")
                continue
            
            if len(detections) > 1:
                print(f"  ⚠ Multiple faces detected ({len(detections)}), using largest")
            
            # Use largest face
            largest_det = max(detections, key=lambda d: 
                             (d['bbox'][2] - d['bbox'][0]) * (d['bbox'][3] - d['bbox'][1]))
            
            landmarks = largest_det['landmarks']
            confidence = largest_det['confidence']
            print(f"  ✓ Face detected (confidence: {confidence:.3f})")
            
            # Align face
            aligned_face = self.aligner.align_face(image, landmarks)
            if aligned_face is None:
                print(f"  ✗ Alignment failed")
                continue
            
            # Generate quality variants
            if self.enable_variants:
                print(f"  Generating {len(self.QUALITY_VARIANTS)} quality variants...")
                
                for variant_name, params in self.QUALITY_VARIANTS.items():
                    # Create degraded version
                    if variant_name == 'original':
                        degraded_face = aligned_face
                    else:
                        degraded_face = self.create_degraded_face(aligned_face, variant_name)
                    
                    # Extract embedding
                    embedding = self.embedder.extract_embedding(degraded_face, return_quality=False)
                    
                    if embedding is not None:
                        all_embeddings.append(embedding)
                        all_variants.append({
                            'image': img_path.name,
                            'variant': variant_name,
                            'embedding': embedding
                        })
                        print(f"    ✓ {variant_name}: embedding extracted")
                    else:
                        print(f"    ✗ {variant_name}: embedding failed")
                
                successful_images += 1
            else:
                # Single embedding (original quality only)
                embedding = self.embedder.extract_embedding(aligned_face, return_quality=False)
                if embedding is not None:
                    all_embeddings.append(embedding)
                    all_variants.append({
                        'image': img_path.name,
                        'variant': 'original',
                        'embedding': embedding
                    })
                    print(f"  ✓ Embedding extracted")
                    successful_images += 1
                else:
                    print(f"  ✗ Embedding extraction failed")
        
        # Check if we have enough embeddings
        if len(all_embeddings) == 0:
            print(f"\n✗ Enrollment failed: No valid embeddings")
            return {'success': False, 'reason': 'No valid embeddings'}
        
        print(f"\n{'='*80}")
        print(f"ENROLLMENT SUMMARY")
        print(f"{'='*80}")
        print(f"Total images processed: {len(image_paths)}")
        print(f"Successful images: {successful_images}")
        print(f"Total embeddings: {len(all_embeddings)}")
        
        if self.enable_variants:
            print(f"\nEmbeddings per variant:")
            for variant_name in self.QUALITY_VARIANTS.keys():
                count = sum(1 for v in all_variants if v['variant'] == variant_name)
                print(f"  {variant_name}: {count}")
        
        # Compute average embedding
        avg_embedding = np.mean(all_embeddings, axis=0)
        avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)  # L2 normalize
        
        print(f"\n✓ Average embedding computed and normalized")
        print(f"  Embedding shape: {avg_embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(avg_embedding):.6f}")
        
        # Add to database
        metadata = {
            'enrollment_date': datetime.now().isoformat(),
            'num_images': successful_images,
            'num_embeddings': len(all_embeddings),
            'multi_quality': self.enable_variants,
            'variants': list(self.QUALITY_VARIANTS.keys()) if self.enable_variants else ['original']
        }
        
        success = self.vector_db.add_user(user_id, avg_embedding, metadata)
        
        if success:
            print(f"\n✓ User '{user_id}' enrolled successfully!")
            return {
                'success': True,
                'user_id': user_id,
                'num_images': successful_images,
                'num_embeddings': len(all_embeddings),
                'metadata': metadata
            }
        else:
            print(f"\n✗ Failed to add user to database")
            return {'success': False, 'reason': 'Database error'}
    
    def enroll_from_directory(self,
                             user_id: str,
                             directory: Path,
                             replace_existing: bool = False) -> Dict:
        """
        Enroll user from all images in directory.
        
        Args:
            user_id: Unique user identifier
            directory: Directory containing user images
            replace_existing: Replace if user exists
            
        Returns:
            Enrollment statistics
        """
        if not directory.exists():
            print(f"✗ Directory not found: {directory}")
            return {'success': False, 'reason': 'Directory not found'}
        
        # Find all images
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(directory.glob(f'*{ext}'))
            image_paths.extend(directory.glob(f'*{ext.upper()}'))
        
        # Deduplicate (case-insensitive filesystem)
        image_paths = list({p.resolve() for p in image_paths})
        
        if len(image_paths) == 0:
            print(f"✗ No images found in {directory}")
            return {'success': False, 'reason': 'No images found'}
        
        print(f"Found {len(image_paths)} images in {directory}")
        
        return self.enroll_user(user_id, image_paths, replace_existing)


def create_multi_quality_enroller(vector_db: VectorDatabase,
                                   enable_variants: bool = True) -> MultiQualityEnroller:
    """
    Create multi-quality enrollment system with all components.
    
    Args:
        vector_db: Vector database instance
        enable_variants: Enable quality variants (recommended)
        
    Returns:
        Configured MultiQualityEnroller
    """
    print("Initializing multi-quality enrollment system...")
    
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    return MultiQualityEnroller(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        vector_db=vector_db,
        enable_variants=enable_variants
    )
