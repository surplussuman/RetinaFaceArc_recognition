"""
Phase 1 Test Script: Core Pipeline Testing

Tests:
1. Face detection with RetinaFace
2. 5-point affine alignment
3. ArcFace embedding extraction
4. Basic recognition pipeline
5. Performance benchmarking

Usage:
    python tests/test_phase1.py --image path/to/image.jpg
    python tests/test_phase1.py --image path/to/image.jpg --save-viz
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
import sys
import time

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.basic_recognition import BasicFaceRecognizer


def test_detector(image_path: str, save_dir: Path):
    """Test RetinaFace detector."""
    print("\n" + "="*60)
    print("TEST 1: RetinaFace Detection")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"✗ Failed to load image: {image_path}")
        return False
    
    print(f"Image shape: {image.shape}")
    
    # Initialize detector
    try:
        detector = RetinaFaceDetector()
    except Exception as e:
        print(f"✗ Failed to initialize detector: {e}")
        print("Note: Make sure RetinaFace ONNX model is in models/")
        return False
    
    # Detect faces
    start_time = time.time()
    detections = detector.detect(image)
    elapsed = (time.time() - start_time) * 1000
    
    print(f"\nDetected {len(detections)} faces in {elapsed:.2f} ms")
    
    # Visualize detections
    vis = image.copy()
    for i, det in enumerate(detections):
        bbox = det['bbox']
        landmarks = det['landmarks']
        confidence = det['confidence']
        
        print(f"\nFace {i+1}:")
        print(f"  BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]")
        print(f"  Confidence: {confidence:.3f}")
        print(f"  Landmarks: {len(landmarks)} points")
        
        # Draw bbox
        x1, y1, x2, y2 = [int(v) for v in bbox]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Draw landmarks
        for j, (lx, ly) in enumerate(landmarks):
            cv2.circle(vis, (int(lx), int(ly)), 3, (0, 0, 255), -1)
            cv2.putText(vis, str(j+1), (int(lx)+5, int(ly)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Label
        label = f"Face {i+1} ({confidence:.2f})"
        cv2.putText(vis, label, (x1, y1-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Save visualization
    output_path = save_dir / "detection_result.jpg"
    cv2.imwrite(str(output_path), vis)
    print(f"\n✓ Saved visualization to: {output_path}")
    
    return len(detections) > 0


def test_aligner(image_path: str, save_dir: Path):
    """Test face alignment."""
    print("\n" + "="*60)
    print("TEST 2: Face Alignment")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Detect faces
    try:
        detector = RetinaFaceDetector()
        detections = detector.detect(image)
    except Exception as e:
        print(f"✗ Detection failed: {e}")
        return False
    
    if len(detections) == 0:
        print("✗ No faces detected")
        return False
    
    # Initialize aligner
    aligner = FaceAligner(output_size=112)
    
    # Align faces
    aligned_faces = []
    for i, det in enumerate(detections):
        landmarks = np.array(det['landmarks'])
        
        # Check alignment quality
        is_good, error = aligner.check_alignment_quality(landmarks)
        print(f"\nFace {i+1}:")
        print(f"  Alignment quality: {'GOOD' if is_good else 'POOR'}")
        print(f"  RMSE error: {error:.2f} pixels")
        
        # Align face
        aligned = aligner.align_face(image, landmarks)
        aligned_faces.append(aligned)
        
        # Save aligned face
        output_path = save_dir / f"aligned_face_{i+1}.jpg"
        cv2.imwrite(str(output_path), aligned)
        print(f"  ✓ Saved to: {output_path}")
        
        # Visualize alignment
        vis = aligner.visualize_alignment(image, landmarks, aligned)
        vis_path = save_dir / f"alignment_viz_{i+1}.jpg"
        cv2.imwrite(str(vis_path), vis)
    
    return len(aligned_faces) > 0


def test_embedder(image_path: str, save_dir: Path):
    """Test embedding extraction."""
    print("\n" + "="*60)
    print("TEST 3: ArcFace Embedding")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    
    # Detect and align faces
    try:
        detector = RetinaFaceDetector()
        detections = detector.detect(image)
        
        if len(detections) == 0:
            print("✗ No faces detected")
            return False
        
        aligner = FaceAligner(output_size=112)
        embedder = ArcFaceEmbedder()
        
    except Exception as e:
        print(f"✗ Initialization failed: {e}")
        print("Note: Make sure ArcFace ONNX model is in models/")
        return False
    
    embeddings = []
    
    for i, det in enumerate(detections):
        landmarks = np.array(det['landmarks'])
        aligned = aligner.align_face(image, landmarks)
        
        # Check quality
        is_good, metrics = embedder.check_face_quality(aligned)
        
        # Extract embedding
        start_time = time.time()
        embedding = embedder.extract_embedding(aligned)
        elapsed = (time.time() - start_time) * 1000
        
        embeddings.append(embedding)
        
        print(f"\nFace {i+1}:")
        print(f"  Quality: {'GOOD' if is_good else 'POOR'}")
        if not is_good:
            print(f"  Issues: {metrics['issues']}")
        print(f"  Embedding shape: {embedding.shape}")
        print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
        print(f"  Extraction time: {elapsed:.2f} ms")
        print(f"  Sample values: {embedding[:5]}")
    
    # Test pairwise similarities
    if len(embeddings) > 1:
        print("\nPairwise Similarities:")
        for i in range(len(embeddings)):
            for j in range(i+1, len(embeddings)):
                sim = embedder.compute_similarity(embeddings[i], embeddings[j])
                dist = embedder.compute_distance(embeddings[i], embeddings[j])
                print(f"  Face {i+1} vs Face {j+1}: similarity={sim:.4f}, distance={dist:.4f}")
    
    return True


def test_recognition_pipeline(image_path: str, save_dir: Path):
    """Test end-to-end recognition pipeline."""
    print("\n" + "="*60)
    print("TEST 4: Recognition Pipeline")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    
    try:
        # Initialize recognizer
        recognizer = BasicFaceRecognizer(
            alignment_size=112,
            similarity_threshold=0.40
        )
    except Exception as e:
        print(f"✗ Failed to initialize recognizer: {e}")
        return False
    
    # Test recognition (no gallery yet)
    print("\n--- Test 1: Recognition without gallery ---")
    results = recognizer.recognize_face(image, return_all=True)
    print(f"Detected {len(results)} faces (all unknown)")
    
    # Add first face to gallery as "Test Person"
    if len(results) > 0:
        print("\n--- Test 2: Add face to gallery ---")
        success = recognizer.add_to_gallery("Test_Person", image)
        
        if success:
            print("\n--- Test 3: Recognition with gallery ---")
            results = recognizer.recognize_face(image, return_all=True)
            
            for i, result in enumerate(results):
                print(f"\nFace {i+1}:")
                print(f"  Identity: {result['identity']}")
                print(f"  Similarity: {result['similarity']:.4f}")
                print(f"  Detection confidence: {result['confidence']:.3f}")
            
            # Visualize results
            vis = recognizer.visualize_results(image, results)
            output_path = save_dir / "recognition_result.jpg"
            cv2.imwrite(str(output_path), vis)
            print(f"\n✓ Saved visualization to: {output_path}")
    
    return True


def benchmark_pipeline(image_path: str):
    """Benchmark pipeline performance."""
    print("\n" + "="*60)
    print("TEST 5: Performance Benchmark")
    print("="*60)
    
    # Load image
    image = cv2.imread(image_path)
    
    try:
        recognizer = BasicFaceRecognizer()
        stats = recognizer.benchmark(image, num_runs=10)
        
        print("\nTiming Statistics (10 runs):")
        print(f"  Detection: {stats['detection_mean']:.2f} ± {stats['detection_std']:.2f} ms")
        print(f"  Alignment: {stats['alignment_mean']:.2f} ± {stats['alignment_std']:.2f} ms")
        print(f"  Embedding: {stats['embedding_mean']:.2f} ± {stats['embedding_std']:.2f} ms")
        print(f"  Total: {stats['total_mean']:.2f} ± {stats['total_std']:.2f} ms")
        print(f"\n  FPS (approx): {1000/stats['total_mean']:.1f}")
        
    except Exception as e:
        print(f"✗ Benchmark failed: {e}")
        return False
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Phase 1 Pipeline Testing")
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--save-viz", action="store_true", help="Save visualizations")
    parser.add_argument("--output-dir", type=str, default="data/test_output",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Create output directory
    save_dir = Path(args.output_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if image provided
    if not args.image:
        print("="*60)
        print("PHASE 1 TESTING")
        print("="*60)
        print("\nNo image provided. Usage:")
        print("  python tests/test_phase1.py --image path/to/image.jpg")
        print("\nThis will test:")
        print("  1. RetinaFace detection")
        print("  2. 5-point affine alignment")
        print("  3. ArcFace embedding extraction")
        print("  4. Basic recognition pipeline")
        print("  5. Performance benchmarking")
        print("\nNote: Make sure ONNX models are in models/ directory:")
        print("  - models/retinaface_resnet50.onnx")
        print("  - models/arcface_resnet100.onnx")
        return
    
    # Check if image exists
    if not Path(args.image).exists():
        print(f"✗ Image not found: {args.image}")
        return
    
    print("\n" + "="*60)
    print("PHASE 1: CORE PIPELINE TESTING")
    print("="*60)
    print(f"Image: {args.image}")
    print(f"Output directory: {save_dir}")
    print("="*60)
    
    # Run tests
    test_results = {}
    
    test_results['detection'] = test_detector(args.image, save_dir)
    test_results['alignment'] = test_aligner(args.image, save_dir)
    test_results['embedding'] = test_embedder(args.image, save_dir)
    test_results['recognition'] = test_recognition_pipeline(args.image, save_dir)
    test_results['benchmark'] = benchmark_pipeline(args.image)
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in test_results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  {test_name.capitalize()}: {status}")
    
    all_passed = all(test_results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nPhase 1 is complete and working!")
        print("Next: Phase 2 - FAISS Vector Database + User Enrollment")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the issues before proceeding.")
        print("Common issues:")
        print("  - Missing ONNX models in models/")
        print("  - Incorrect configuration files")
        print("  - Missing dependencies")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
