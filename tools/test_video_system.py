"""
Test Video Recognition System
==============================

Quick test script to verify video recognition is working.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import cv2
import numpy as np
from core.vector_db import VectorDatabase
from core.video_recognition import create_video_recognizer


def create_test_video(output_path: Path, duration_seconds: int = 5, fps: int = 30):
    """
    Create a simple test video with a static frame.
    
    Args:
        output_path: Output video path
        duration_seconds: Video duration in seconds
        fps: Frames per second
    """
    # Video properties
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    
    # Create a simple frame
    total_frames = duration_seconds * fps
    
    for i in range(total_frames):
        # Create gradient background
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (50 + i % 50, 100, 150)
        
        # Add text
        text = f"Test Video - Frame {i+1}/{total_frames}"
        cv2.putText(frame, text, (50, height//2), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        
        writer.write(frame)
    
    writer.release()
    print(f"✓ Test video created: {output_path}")


def test_video_recognition():
    """Test video recognition system."""
    
    print("=" * 60)
    print("Video Recognition System Test")
    print("=" * 60)
    
    # Check database
    db_path = project_root / 'data' / 'face_database'
    index_path = Path(str(db_path) + '.index')
    
    if not index_path.exists():
        print("\n❌ Error: Database not found!")
        print("\nPlease enroll users first:")
        print("  python tools/enroll_multi_quality.py --user-id <name> --directory <photos/>")
        return False
    
    # Load database
    print("\n1. Loading database...")
    vector_db = VectorDatabase(embedding_dim=512)
    vector_db.load(db_path)
    
    user_count = vector_db.get_user_count()
    print(f"   ✓ Database loaded ({user_count} users)")
    
    users = vector_db.get_all_users()
    print(f"   Enrolled users:")
    for user in users:
        # Count embeddings for this user
        count = sum(1 for uid in vector_db.user_ids if uid == user['user_id'])
        print(f"     • {user['user_id']} ({count} embeddings)")
    
    # Create recognizer
    print("\n2. Creating video recognizer...")
    recognizer = create_video_recognizer(
        vector_db,
        recognition_threshold=0.4,
        process_every_n_frames=1
    )
    print("   ✓ Recognizer created")
    
    # Check for test video
    print("\n3. Checking for test video...")
    test_video_path = project_root / 'test_video.mp4'
    
    if not test_video_path.exists():
        print(f"   ⚠ Test video not found: {test_video_path}")
        print("\n   Creating simple test video (5 seconds)...")
        create_test_video(test_video_path, duration_seconds=5)
    else:
        print(f"   ✓ Found test video: {test_video_path}")
    
    # Get video info
    cap = cv2.VideoCapture(str(test_video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    
    print(f"\n   Video Info:")
    print(f"     • Frames: {total_frames}")
    print(f"     • FPS: {fps:.1f}")
    print(f"     • Resolution: {width}×{height}")
    print(f"     • Duration: {total_frames/fps:.1f}s")
    
    # Process first 100 frames as test
    print(f"\n4. Processing first 100 frames...")
    
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)
    
    output_video = results_dir / 'test_output.mp4'
    
    try:
        stats = recognizer.process_video(
            test_video_path,
            output_path=output_video,
            show_preview=False,
            max_frames=100
        )
        
        print("\n   ✓ Processing complete!")
        print(f"\n   Statistics:")
        print(f"     • Total detections: {stats['total_detections']}")
        print(f"     • Unique tracks: {stats['total_tracks']}")
        print(f"     • Recognized: {stats['total_recognized']}")
        print(f"     • Avg FPS: {stats['avg_fps']:.1f}")
        print(f"     • Processing time: {stats['processing_time']:.1f}s")
        
        if stats.get('unique_identities'):
            print(f"     • Identified: {', '.join(stats['unique_identities'])}")
        
        print(f"\n   Output saved to: {output_video}")
        
        if stats['recognition_log']:
            print(f"\n   Recognition events: {len(stats['recognition_log'])}")
            print(f"     First few entries:")
            for entry in stats['recognition_log'][:5]:
                print(f"       Frame {entry['frame']}: Track {entry['track_id']} = {entry['identity']} ({entry['confidence']:.3f})")
        
        return True
        
    except Exception as e:
        print(f"\n   ❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main test function."""
    
    success = test_video_recognition()
    
    print("\n" + "=" * 60)
    if success:
        print("✅ TEST PASSED - Video recognition system is working!")
        print("\nNext steps:")
        print("  1. Process your own video:")
        print("     python tools/process_video.py --video your_video.mp4 --output results/output.mp4")
        print("\n  2. Or use Streamlit web interface:")
        print("     streamlit run app.py")
    else:
        print("❌ TEST FAILED - See errors above")
    print("=" * 60)


if __name__ == '__main__':
    main()
