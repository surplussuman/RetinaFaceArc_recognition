"""
User Enrollment CLI Tool
=========================

Command-line tool for enrolling users into the face recognition system.

Usage Examples:
--------------
# Enroll from multiple images
python tools/enroll_user.py --user-id "John_Doe" --images img1.jpg img2.jpg img3.jpg

# Enroll from directory
python tools/enroll_user.py --user-id "Jane_Smith" --directory data/users/jane/

# Batch enrollment from parent directory
python tools/enroll_user.py --batch-directory data/users/

# Update existing user
python tools/enroll_user.py --user-id "John_Doe" --images new1.jpg new2.jpg --update

Author: Face Recognition System
Date: 2024
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase
from core.enrollment import EnrollmentSystem


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Enroll users into face recognition system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # User identification
    parser.add_argument('--user-id', type=str,
                       help='Unique user identifier (e.g., employee ID, name)')
    
    # Input options
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('--images', nargs='+',
                            help='List of image paths for enrollment')
    input_group.add_argument('--directory', type=str,
                            help='Directory containing user images')
    input_group.add_argument('--batch-directory', type=str,
                            help='Parent directory with subdirectories for each user')
    
    # Metadata options
    parser.add_argument('--metadata', type=str,
                       help='Additional metadata in JSON format')
    
    # Action options
    parser.add_argument('--update', action='store_true',
                       help='Update existing user (re-enrollment)')
    parser.add_argument('--delete', type=str,
                       help='Delete user from database')
    parser.add_argument('--list', action='store_true',
                       help='List all enrolled users')
    parser.add_argument('--info', type=str,
                       help='Show information about specific user')
    
    # Database options
    parser.add_argument('--db-path', type=str, default='data/face_database',
                       help='Path to vector database (default: data/face_database)')
    
    # Configuration options
    parser.add_argument('--similarity-threshold', type=float, default=0.4,
                       help='Similarity threshold for recognition (default: 0.4)')
    parser.add_argument('--min-quality', type=str, default='GOOD',
                       choices=['EXCELLENT', 'GOOD', 'FAIR'],
                       help='Minimum quality for enrollment (default: GOOD)')
    
    return parser.parse_args()


def main():
    """Main enrollment function."""
    args = parse_args()
    
    # Initialize components
    print("Initializing face recognition system...")
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    # Load or create vector database
    vector_db = VectorDatabase(embedding_dim=512)
    db_path = Path(args.db_path)
    if db_path.with_suffix('.index').exists():
        vector_db.load(args.db_path)
    else:
        print(f"Creating new database: {args.db_path}")
    
    # Create enrollment system
    enrollment = EnrollmentSystem(
        detector, aligner, embedder, vector_db,
        min_quality=args.min_quality
    )
    
    # Parse metadata
    metadata = None
    if args.metadata:
        import json
        metadata = json.loads(args.metadata)
    
    # Handle different actions
    if args.list:
        # List all users
        users = enrollment.list_users()
        if len(users) == 0:
            print("\nNo users enrolled in database")
        else:
            print(f"\n{'='*60}")
            print(f"Enrolled Users ({len(users)})")
            print(f"{'='*60}")
            for i, user in enumerate(users, 1):
                print(f"\n{i}. {user['user_id']}")
                print(f"   Enrollment date: {user.get('enrollment_date', 'N/A')}")
                print(f"   Images: {user.get('num_images', 'N/A')}")
                print(f"   Quality: {user.get('avg_quality', 'N/A'):.2f}")
    
    elif args.info:
        # Show user info
        user_info = enrollment.get_user_info(args.info)
        if user_info is None:
            print(f"\n[ERROR] User '{args.info}' not found")
        else:
            print(f"\n{'='*60}")
            print(f"User Information: {args.info}")
            print(f"{'='*60}")
            for key, value in user_info.items():
                print(f"  {key}: {value}")
    
    elif args.delete:
        # Delete user
        success = enrollment.remove_user(args.delete)
        if success:
            print(f"\n✓ User '{args.delete}' deleted successfully")
            # Save database
            vector_db.save(args.db_path)
        else:
            print(f"\n[ERROR] Failed to delete user '{args.delete}'")
    
    elif args.batch_directory:
        # Batch enrollment
        parent_dir = Path(args.batch_directory)
        if not parent_dir.exists():
            print(f"[ERROR] Directory not found: {args.batch_directory}")
            return
        
        # Find subdirectories
        user_dirs = {d.name: str(d) for d in parent_dir.iterdir() if d.is_dir()}
        
        if len(user_dirs) == 0:
            print(f"[ERROR] No subdirectories found in {args.batch_directory}")
            return
        
        # Batch enroll
        results = enrollment.batch_enroll(user_dirs)
        
        # Save database
        vector_db.save(args.db_path)
        print(f"\n✓ Database saved to {args.db_path}")
    
    else:
        # Single user enrollment
        if not args.user_id:
            print("[ERROR] --user-id is required for enrollment")
            return
        
        # Determine input method
        if args.images:
            # Enroll from image list
            if args.update:
                success = enrollment.update_user(args.user_id, args.images, metadata)
            else:
                success = enrollment.enroll_user(args.user_id, args.images, metadata)
        
        elif args.directory:
            # Enroll from directory
            if args.update:
                # Get images from directory
                dir_path = Path(args.directory)
                extensions = ('.jpg', '.jpeg', '.png')
                image_paths = []
                for ext in extensions:
                    image_paths.extend(dir_path.glob(f"*{ext}"))
                    image_paths.extend(dir_path.glob(f"*{ext.upper()}"))
                image_paths = [str(p) for p in sorted(image_paths)]
                success = enrollment.update_user(args.user_id, image_paths, metadata)
            else:
                success = enrollment.enroll_from_directory(args.user_id, args.directory, metadata=metadata)
        
        else:
            print("[ERROR] Must provide either --images or --directory")
            return
        
        # Save database if successful
        if success:
            vector_db.save(args.db_path)
            print(f"\n✓ Database saved to {args.db_path}")


if __name__ == "__main__":
    main()
