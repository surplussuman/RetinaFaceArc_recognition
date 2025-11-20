"""
Phase 1 Example: Basic Face Recognition

This example demonstrates the complete Phase 1 pipeline:
1. Detect faces
2. Align faces
3. Extract embeddings
4. Recognize faces

Usage:
    python examples/phase1_example.py
"""

import cv2
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core import RetinaFaceDetector, FaceAligner, ArcFaceEmbedder, BasicFaceRecognizer


def example_1_detection():
    """Example 1: Detect faces in an image."""
    print("\n" + "="*60)
    print("EXAMPLE 1: Face Detection")
    print("="*60)
    
    # Create a test image (you can replace with: cv2.imread('your_image.jpg'))
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize detector
    detector = RetinaFaceDetector()
    
    # Detect faces
    detections = detector.detect(image)
    
    print(f"\nDetected {len(detections)} faces")
    
    for i, det in enumerate(detections):
        print(f"\nFace {i+1}:")
        print(f"  BBox: {det['bbox']}")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Landmarks: {det['landmarks']}")


def example_2_alignment():
    """Example 2: Align detected face."""
    print("\n" + "="*60)
    print("EXAMPLE 2: Face Alignment")
    print("="*60)
    
    # Create test image and detect face
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detector = RetinaFaceDetector()
    detections = detector.detect(image)
    
    if len(detections) == 0:
        print("No faces detected (using random image)")
        return
    
    # Initialize aligner
    aligner = FaceAligner(output_size=112)
    
    # Align first face
    landmarks = np.array(detections[0]['landmarks'])
    aligned = aligner.align_face(image, landmarks)
    
    print(f"\nAligned face shape: {aligned.shape}")
    
    # Check quality
    is_good, error = aligner.check_alignment_quality(landmarks)
    print(f"Alignment quality: {'GOOD' if is_good else 'POOR'}")
    print(f"RMSE error: {error:.2f} pixels")


def example_3_embedding():
    """Example 3: Extract face embedding."""
    print("\n" + "="*60)
    print("EXAMPLE 3: Embedding Extraction")
    print("="*60)
    
    # Create aligned face (112x112 RGB)
    aligned_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Initialize embedder
    embedder = ArcFaceEmbedder()
    
    # Extract embedding
    embedding = embedder.extract_embedding(aligned_face)
    
    print(f"\nEmbedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"Sample values: {embedding[:10]}")


def example_4_similarity():
    """Example 4: Compare face similarity."""
    print("\n" + "="*60)
    print("EXAMPLE 4: Face Similarity")
    print("="*60)
    
    # Create two random faces
    face1 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    # Initialize embedder
    embedder = ArcFaceEmbedder()
    
    # Extract embeddings
    emb1 = embedder.extract_embedding(face1)
    emb2 = embedder.extract_embedding(face2)
    
    # Compute similarity
    similarity = embedder.compute_similarity(emb1, emb2)
    distance = embedder.compute_distance(emb1, emb2)
    
    print(f"\nCosine similarity: {similarity:.4f}")
    print(f"Euclidean distance: {distance:.4f}")
    print(f"\nInterpretation:")
    print(f"  > 0.6: Likely same person")
    print(f"  0.4-0.6: Uncertain")
    print(f"  < 0.4: Different person")


def example_5_recognition():
    """Example 5: Full recognition pipeline."""
    print("\n" + "="*60)
    print("EXAMPLE 5: Full Recognition Pipeline")
    print("="*60)
    
    # Create test image
    image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    
    # Initialize recognizer
    recognizer = BasicFaceRecognizer(
        alignment_size=112,
        similarity_threshold=0.40
    )
    
    print("\n--- Step 1: Add person to gallery ---")
    # Add a person to the gallery
    # recognizer.add_to_gallery("John_Doe", image)
    print("(Skipped - using random image)")
    
    print("\n--- Step 2: Recognize faces ---")
    results = recognizer.recognize_face(image, return_all=True)
    
    print(f"\nDetected {len(results)} faces")
    
    for i, result in enumerate(results):
        print(f"\nFace {i+1}:")
        print(f"  Identity: {result['identity']}")
        print(f"  Similarity: {result['similarity']:.4f}")
        print(f"  BBox: {result['bbox']}")


def example_6_enrollment():
    """Example 6: Multi-image enrollment."""
    print("\n" + "="*60)
    print("EXAMPLE 6: Multi-Image Enrollment")
    print("="*60)
    
    print("\nBest practice: Use 5-20 images per person")
    print("Images should have:")
    print("  - Different angles (±30°)")
    print("  - Different expressions")
    print("  - Different lighting")
    print("  - With/without glasses (if applicable)")
    
    # Simulate multiple images
    images = [
        np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        for _ in range(5)
    ]
    
    recognizer = BasicFaceRecognizer()
    
    print("\nEnrolling 'John_Doe' with 5 images...")
    # recognizer.add_multiple_to_gallery("John_Doe", images)
    print("(Skipped - using random images)")
    
    print("\n✓ Benefits of multi-image enrollment:")
    print("  - More robust to variations")
    print("  - Better generalization")
    print("  - Higher recognition accuracy")


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print(" "*25 + "PHASE 1 EXAMPLES")
    print("="*80)
    
    print("\nThese examples demonstrate the core Phase 1 functionality.")
    print("For real testing, use actual face images instead of random data.")
    print("\n" + "="*80)
    
    try:
        example_1_detection()
        example_2_alignment()
        example_3_embedding()
        example_4_similarity()
        example_5_recognition()
        example_6_enrollment()
        
        print("\n" + "="*80)
        print("✓ ALL EXAMPLES COMPLETED")
        print("="*80)
        
        print("\n📖 Next Steps:")
        print("  1. Test with real images: python tests/test_phase1.py --image your_image.jpg")
        print("  2. Read Phase1_CoreSetup.md for detailed documentation")
        print("  3. Proceed to Phase 2 when ready")
        print("\n" + "="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nPossible causes:")
        print("  - Missing ONNX models")
        print("  - Missing dependencies")
        print("  - Configuration errors")
        print("\nSee QUICKSTART.md for troubleshooting")


if __name__ == "__main__":
    main()
