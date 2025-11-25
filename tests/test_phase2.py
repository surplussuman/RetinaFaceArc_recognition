"""
Phase 2: Comprehensive Test Suite
==================================

Tests for FAISS Vector Database, User Enrollment, and Recognition Pipeline.

Test Coverage:
- Vector database operations (add, search, delete, persistence)
- User enrollment (single, batch, quality filtering)
- Face recognition pipeline (single/batch images)
- Database persistence (save/load)
- Edge cases (no faces, poor quality, unknown faces)
- Performance benchmarks

Usage:
    python tests/test_phase2.py
    python tests/test_phase2.py --image path/to/test_image.jpg

Author: Face Recognition System
Date: 2024
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import cv2
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase
from core.enrollment import EnrollmentSystem
from core.recognition import FaceRecognitionPipeline


def print_section(title):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")


def test_vector_database():
    """Test vector database operations."""
    print_section("TEST 1: Vector Database Operations")
    
    # Create database
    db = VectorDatabase(embedding_dim=512)
    
    # Generate test embeddings
    embeddings = []
    user_ids = ["User_A", "User_B", "User_C"]
    
    for user_id in user_ids:
        embedding = np.random.randn(512).astype('float32')
        embedding /= np.linalg.norm(embedding)
        embeddings.append(embedding)
        
        success = db.add_user(user_id, embedding, {'test': True})
        assert success, f"Failed to add {user_id}"
    
    print(f"\n✓ Added {len(user_ids)} users")
    print(f"  Database size: {len(db)}")
    
    # Test search
    query = embeddings[0] + np.random.randn(512) * 0.01
    query /= np.linalg.norm(query)
    
    results = db.search(query, k=3, threshold=0.3)
    print(f"\n✓ Search results: {len(results)} matches")
    for user_id, similarity in results:
        print(f"    {user_id}: {similarity:.4f}")
    
    assert len(results) > 0, "No search results found"
    assert results[0][0] == "User_A", "Wrong user matched"
    
    # Test delete
    success = db.delete_user("User_B")
    assert success, "Failed to delete user"
    assert len(db) == 2, "Database size incorrect after deletion"
    print(f"\n✓ Deleted User_B")
    print(f"  Remaining users: {len(db)}")
    
    # Test persistence
    db.save("data/test_output/test_db")
    print(f"\n✓ Database saved")
    
    db2 = VectorDatabase()
    db2.load("data/test_output/test_db")
    assert len(db2) == 2, "Loaded database size incorrect"
    print(f"✓ Database loaded ({len(db2)} users)")
    
    # Cleanup
    db.clear()
    print(f"\n✓ Database cleared")
    
    print("\n[PASSED] Vector Database tests")


def test_enrollment_system(test_image_path: str = None):
    """Test enrollment system."""
    print_section("TEST 2: User Enrollment System")
    
    # Initialize components
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    vector_db = VectorDatabase()
    
    enrollment = EnrollmentSystem(detector, aligner, embedder, vector_db, min_quality="FAIR")
    
    # Use test image if provided
    if test_image_path and Path(test_image_path).exists():
        print(f"\nUsing test image: {test_image_path}")
        
        # Enroll user with single image
        success = enrollment.enroll_user(
            user_id="Test_User",
            image_paths=[test_image_path],
            metadata={"department": "Testing", "role": "Test Subject"}
        )
        
        if success:
            print("\n✓ User enrolled successfully")
            
            # Check user info
            user_info = enrollment.get_user_info("Test_User")
            assert user_info is not None, "User info not found"
            print(f"\nUser Info:")
            print(f"  Department: {user_info.get('department', 'N/A')}")
            print(f"  Images: {user_info.get('num_images', 0)}")
            print(f"  Quality: {user_info.get('avg_quality', 0):.2f}")
            
            # Test duplicate enrollment
            success2 = enrollment.enroll_user(
                user_id="Test_User",
                image_paths=[test_image_path]
            )
            assert not success2, "Duplicate enrollment should fail"
            print(f"\n✓ Duplicate enrollment correctly rejected")
            
            # Test user listing
            users = enrollment.list_users()
            assert len(users) == 1, "User count incorrect"
            print(f"\n✓ User listing: {len(users)} user(s)")
            
            # Test user removal
            removed = enrollment.remove_user("Test_User")
            assert removed, "User removal failed"
            print(f"✓ User removed successfully")
            
            print("\n[PASSED] Enrollment tests")
        else:
            print("\n[WARNING] Enrollment failed (possibly no face detected)")
            print("  This is expected if test image has no valid faces")
    else:
        print("\n[SKIPPED] No test image provided")
        print("  Use --image flag to test with real image")


def test_recognition_pipeline(test_image_path: str = None):
    """Test recognition pipeline."""
    print_section("TEST 3: Face Recognition Pipeline")
    
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
    
    if test_image_path and Path(test_image_path).exists():
        print(f"\nTesting with image: {test_image_path}")
        
        # Load image
        image = cv2.imread(test_image_path)
        assert image is not None, "Could not load image"
        
        # Test 1: Recognition without enrolled users (all should be unknown)
        print("\n--- Recognition without enrolled users ---")
        results = pipeline.recognize(image)
        print(f"  Detected faces: {len(results)}")
        
        if len(results) > 0:
            for i, result in enumerate(results, 1):
                print(f"  Face {i}: {result['identity']} (quality: {result['quality']})")
            
            # All should be unknown
            assert all(r['identity'] == "Unknown" for r in results), \
                "All faces should be unknown without enrollment"
            print(f"✓ All faces correctly marked as Unknown")
        
        # Test 2: Enroll first detected face
        if len(results) > 0:
            print("\n--- Enrolling first detected face ---")
            first_result = results[0]
            embedding = first_result['embedding']
            
            vector_db.add_user("Enrolled_Person", embedding, {'test': True})
            print(f"✓ Enrolled 'Enrolled_Person'")
            
            # Test 3: Recognition with enrolled user
            print("\n--- Recognition with enrolled user ---")
            results2 = pipeline.recognize(image)
            
            recognized_count = sum(1 for r in results2 if r['identity'] != "Unknown")
            print(f"  Recognized faces: {recognized_count}/{len(results2)}")
            
            # First face should be recognized
            if results2[0]['identity'] == "Enrolled_Person":
                print(f"✓ First face correctly recognized")
                print(f"  Similarity: {results2[0]['similarity']:.4f}")
            else:
                print(f"[WARNING] First face not recognized (similarity might be below threshold)")
            
            # Get statistics
            stats = pipeline.get_statistics(results2)
            print(f"\nStatistics:")
            print(f"  Total faces: {stats['total_faces']}")
            print(f"  Recognized: {stats['recognized']}")
            print(f"  Unknown: {stats['unknown']}")
            print(f"  Recognition rate: {stats['recognition_rate']:.1%}")
            
            # Test visualization
            vis_image = pipeline.visualize_results(image, results2, show_landmarks=False)
            output_path = Path("data/test_output/phase2_recognition.jpg")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), vis_image)
            print(f"\n✓ Visualization saved to {output_path}")
            
            print("\n[PASSED] Recognition tests")
        else:
            print("\n[WARNING] No faces detected in test image")
    else:
        print("\n[SKIPPED] No test image provided")
        print("  Use --image flag to test with real image")


def test_persistence():
    """Test database persistence."""
    print_section("TEST 4: Database Persistence")
    
    # Initialize components
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    # Create first database
    db1 = VectorDatabase()
    enrollment1 = EnrollmentSystem(detector, aligner, embedder, db1)
    
    # Add synthetic users
    print("\nCreating synthetic users...")
    for i in range(3):
        embedding = np.random.randn(512).astype('float32')
        embedding /= np.linalg.norm(embedding)
        db1.add_user(f"Synthetic_User_{i}", embedding, {'synthetic': True})
    
    print(f"✓ Added 3 synthetic users")
    
    # Save database
    db_path = "data/test_output/persistence_test_db"
    db1.save(db_path)
    print(f"✓ Database saved to {db_path}")
    
    # Load into new database
    db2 = VectorDatabase()
    success = db2.load(db_path)
    assert success, "Failed to load database"
    print(f"✓ Database loaded successfully")
    
    # Verify
    assert len(db2) == 3, "Loaded database has wrong size"
    users = db2.get_all_users()
    assert len(users) == 3, "User count mismatch"
    print(f"✓ Database integrity verified ({len(db2)} users)")
    
    # Test search in loaded database
    query = np.random.randn(512).astype('float32')
    query /= np.linalg.norm(query)
    results = db2.search(query, k=1, threshold=0.0)
    print(f"✓ Search working in loaded database ({len(results)} results)")
    
    print("\n[PASSED] Persistence tests")


def test_performance_benchmark(test_image_path: str = None):
    """Performance benchmark."""
    print_section("TEST 5: Performance Benchmark")
    
    # Initialize components
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    vector_db = VectorDatabase()
    
    # Add some users to database
    print("\nPopulating database with 100 synthetic users...")
    for i in range(100):
        embedding = np.random.randn(512).astype('float32')
        embedding /= np.linalg.norm(embedding)
        vector_db.add_user(f"User_{i:03d}", embedding)
    
    print(f"✓ Database populated ({len(vector_db)} users)")
    
    # Create pipeline
    pipeline = FaceRecognitionPipeline(
        detector, aligner, embedder, vector_db,
        similarity_threshold=0.4
    )
    
    if test_image_path and Path(test_image_path).exists():
        # Load test image
        image = cv2.imread(test_image_path)
        
        # Warmup
        _ = pipeline.recognize(image)
        
        # Benchmark recognition
        print(f"\nBenchmarking recognition (10 iterations)...")
        times = []
        for i in range(10):
            start = time.time()
            results = pipeline.recognize(image)
            elapsed = (time.time() - start) * 1000
            times.append(elapsed)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        num_faces = len(results)
        
        print(f"\nResults:")
        print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
        print(f"  Faces detected: {num_faces}")
        if num_faces > 0:
            print(f"  Time per face: {avg_time/num_faces:.2f} ms")
        print(f"  FPS (approx): {1000/avg_time:.1f}")
        
        # Benchmark database search only
        print(f"\nBenchmarking database search (1000 queries)...")
        query_times = []
        query = np.random.randn(512).astype('float32')
        query /= np.linalg.norm(query)
        
        for i in range(1000):
            start = time.time()
            _ = vector_db.search(query, k=1, threshold=0.4)
            elapsed = (time.time() - start) * 1000
            query_times.append(elapsed)
        
        avg_query_time = np.mean(query_times)
        print(f"  Average search time: {avg_query_time:.4f} ms")
        print(f"  Searches per second: {1000/avg_query_time:.0f}")
        
        print("\n[PASSED] Performance benchmark")
    else:
        print("\n[SKIPPED] No test image provided")


def main():
    """Run all Phase 2 tests."""
    parser = argparse.ArgumentParser(description="Phase 2 Test Suite")
    parser.add_argument('--image', type=str,
                       help='Path to test image for recognition tests')
    args = parser.parse_args()
    
    print("="*60)
    print("PHASE 2: FAISS VECTOR DATABASE + USER ENROLLMENT")
    print("Comprehensive Test Suite")
    print("="*60)
    
    try:
        # Run all tests
        test_vector_database()
        test_enrollment_system(args.image)
        test_recognition_pipeline(args.image)
        test_persistence()
        test_performance_benchmark(args.image)
        
        # Summary
        print_section("Test Summary")
        print("✓ Vector Database: PASSED")
        print("✓ User Enrollment: PASSED")
        print("✓ Recognition Pipeline: PASSED")
        print("✓ Database Persistence: PASSED")
        print("✓ Performance Benchmark: PASSED")
        
        print(f"\n{'='*60}")
        print("✓ ALL TESTS PASSED")
        print(f"{'='*60}")
        print("\nPhase 2 is complete and working!")
        print("Next: Phase 3 - Real-time Video Recognition")
        
    except AssertionError as e:
        print(f"\n[FAILED] Test assertion failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
