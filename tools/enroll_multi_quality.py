"""
Multi-Quality User Enrollment CLI Tool
======================================

Enrolls users with synthetic quality variations for robust recognition.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
from core.vector_db import VectorDatabase
from core.multi_quality_enrollment import create_multi_quality_enroller


def main():
    parser = argparse.ArgumentParser(
        description='Enroll user with multi-quality embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Enroll from directory with quality variants
  python enroll_multi_quality.py --user-id "John" --directory photos/john/
  
  # Enroll without variants (single quality)
  python enroll_multi_quality.py --user-id "Jane" --directory photos/jane/ --no-variants
  
  # Replace existing user
  python enroll_multi_quality.py --user-id "Bob" --directory photos/bob/ --replace
        """
    )
    
    parser.add_argument(
        '--user-id',
        type=str,
        required=True,
        help='Unique user identifier'
    )
    
    parser.add_argument(
        '--directory',
        type=str,
        required=True,
        help='Directory containing user images'
    )
    
    parser.add_argument(
        '--replace',
        action='store_true',
        help='Replace existing user if exists'
    )
    
    parser.add_argument(
        '--no-variants',
        action='store_true',
        help='Disable quality variants (faster but less robust)'
    )
    
    args = parser.parse_args()
    
    # Initialize database
    db_path = project_root / 'data' / 'face_database'
    vector_db = VectorDatabase(embedding_dim=512)
    
    # Load existing database if exists
    index_path = Path(str(db_path) + '.index')
    if index_path.exists():
        vector_db.load(db_path)
        print(f"✓ Loaded existing database ({vector_db.get_user_count()} users)")
    else:
        print("✓ Created new database")
    
    # Create enroller
    enable_variants = not args.no_variants
    enroller = create_multi_quality_enroller(vector_db, enable_variants=enable_variants)
    
    # Enroll user
    directory = Path(args.directory)
    result = enroller.enroll_from_directory(
        user_id=args.user_id,
        directory=directory,
        replace_existing=args.replace
    )
    
    if result['success']:
        # Save database
        vector_db.save(db_path)
        print(f"\n✓ Database saved to {db_path}")
        print(f"\nTotal users in database: {vector_db.get_user_count()}")
    else:
        print(f"\n✗ Enrollment failed: {result.get('reason', 'Unknown error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()
