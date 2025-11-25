"""
Video Face Recognition CLI Tool
================================

Process videos with face recognition and tracking.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import csv
from core.vector_db import VectorDatabase
from core.video_recognition import create_video_recognizer


def save_recognition_log(log: list, output_path: Path):
    """Save recognition log to CSV."""
    if not log:
        return
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['frame', 'timestamp', 'track_id', 'identity', 'confidence'])
        writer.writeheader()
        writer.writerows(log)
    
    print(f"✓ Recognition log saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Process video with face recognition and tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process video with output
  python process_video.py --video test.mp4 --output results/output.mp4
  
  # Process with preview
  python process_video.py --video test.mp4 --output results/output.mp4 --preview
  
  # Process every 2nd frame (faster)
  python process_video.py --video test.mp4 --output results/output.mp4 --skip-frames 2
  
  # Process first 300 frames only
  python process_video.py --video test.mp4 --output results/output.mp4 --max-frames 300
        """
    )
    
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--output', type=str, help='Output video file path (optional)')
    parser.add_argument('--log', type=str, help='Recognition log CSV path (optional)')
    parser.add_argument('--threshold', type=float, default=0.4, help='Recognition threshold (default: 0.4)')
    parser.add_argument('--skip-frames', type=int, default=1, help='Process every N frames (default: 1 = all)')
    parser.add_argument('--max-frames', type=int, help='Maximum frames to process (optional)')
    parser.add_argument('--preview', action='store_true', help='Show live preview window')
    
    args = parser.parse_args()
    
    # Load database
    db_path = project_root / 'data' / 'face_database'
    index_path = Path(str(db_path) + '.index')
    if not index_path.exists():
        print("Error: Database not found. Please enroll users first.")
        print(f"Run: python tools/enroll_multi_quality.py --user-id <name> --directory <photos/>")
        return
    
    vector_db = VectorDatabase(embedding_dim=512)
    vector_db.load(db_path)
    
    print(f"✓ Database loaded ({vector_db.get_user_count()} users)")
    enrolled_users = [u['user_id'] for u in vector_db.get_all_users()]
    print(f"  Enrolled users: {', '.join(enrolled_users)}\n")
    
    # Create video recognizer
    recognizer = create_video_recognizer(
        vector_db,
        recognition_threshold=args.threshold,
        process_every_n_frames=args.skip_frames
    )
    
    # Process video
    video_path = Path(args.video)
    if not video_path.exists():
        print(f"Error: Video not found: {video_path}")
        return
    
    output_path = Path(args.output) if args.output else None
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    stats = recognizer.process_video(
        video_path,
        output_path=output_path,
        show_preview=args.preview,
        max_frames=args.max_frames
    )
    
    # Save recognition log
    if args.log and stats['recognition_log']:
        log_path = Path(args.log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        save_recognition_log(stats['recognition_log'], log_path)
    
    # Save statistics
    if output_path:
        stats_path = output_path.parent / (output_path.stem + '_stats.json')
        with open(stats_path, 'w') as f:
            # Convert sets to lists for JSON serialization
            stats_copy = stats.copy()
            stats_copy.pop('recognition_log', None)  # Remove log (saved separately)
            stats_copy.pop('processing_times', None)  # Remove raw times
            json.dump(stats_copy, f, indent=2)
        print(f"✓ Statistics saved to {stats_path}")


if __name__ == '__main__':
    main()
