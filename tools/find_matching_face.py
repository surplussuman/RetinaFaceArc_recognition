"""
Find which face in an image matches an enrolled user.

Scans all detected faces and reports similarity to enrolled user.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase
import argparse


def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two L2-normalized embeddings."""
    return float(np.dot(emb1, emb2))


def main():
    parser = argparse.ArgumentParser(description='Find matching face in image')
    parser.add_argument('--image', type=str, required=True, help='Path to test image')
    parser.add_argument('--user-id', type=str, required=True, help='Enrolled user ID')
    
    args = parser.parse_args()
    
    # Initialize system
    print("Initializing...")
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    vector_db = VectorDatabase(embedding_dim=512)
    
    # Load database
    db_path = project_root / 'data' / 'face_database'
    index_path = Path(str(db_path) + '.index')
    if not index_path.exists():
        print(f"Error: Database not found at {index_path}")
        return
    
    vector_db.load(db_path)
    
    # Get enrolled user embedding
    all_users = vector_db.get_all_users()
    user_ids = [u['user_id'] for u in all_users]
    
    if args.user_id not in user_ids:
        print(f"Error: User '{args.user_id}' not found")
        return
    
    user_index = user_ids.index(args.user_id)
    enrolled_embedding = vector_db.index.reconstruct(user_index)
    enrolled_embedding = enrolled_embedding / np.linalg.norm(enrolled_embedding)
    
    # Load image
    image_path = Path(args.image)
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        return
    
    image = cv2.imread(str(image_path))
    
    print(f"\nScanning: {image_path.name}")
    print(f"Looking for: {args.user_id}\n")
    
    # Detect faces
    detections = detector.detect(image)
    print(f"Found {len(detections)} faces\n")
    
    results = []
    
    for i, detection in enumerate(detections):
        landmarks = detection['landmarks']
        confidence = detection['confidence']
        
        # Align face
        aligned_face = aligner.align_face(image, landmarks)
        if aligned_face is None:
            print(f"Face {i}: Alignment failed")
            continue
        
        # Extract embedding
        embedding = embedder.extract_embedding(aligned_face, return_quality=False)
        if embedding is None:
            print(f"Face {i}: Embedding failed")
            continue
        
        # Compute similarity
        similarity = cosine_similarity(embedding, enrolled_embedding)
        
        results.append((i, similarity, confidence))
        print(f"Face {i:2d}: Similarity={similarity:+.4f}, Detection={confidence:.3f}")
    
    # Find best match
    if results:
        best_idx, best_sim, best_conf = max(results, key=lambda x: x[1])
        print(f"\nBest match: Face {best_idx} with similarity {best_sim:+.4f}")
        
        if best_sim >= 0.4:
            print(f"✓ RECOGNIZED as {args.user_id}")
        else:
            print(f"✗ Below threshold (0.4)")


if __name__ == '__main__':
    main()
