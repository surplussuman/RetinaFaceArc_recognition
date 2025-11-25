"""
Face Recognition CLI Tool
=========================

Command-line tool for recognizing faces in images using the enrolled database.

Usage Examples:
--------------
# Recognize faces in single image
python tools/recognize_image.py --image photo.jpg

# Recognize and save visualization
python tools/recognize_image.py --image photo.jpg --save-viz results/output.jpg

# Recognize with custom threshold
python tools/recognize_image.py --image photo.jpg --threshold 0.5

# Batch recognition on directory
python tools/recognize_image.py --directory images/ --save-results results/

# Show statistics only
python tools/recognize_image.py --image photo.jpg --stats-only

Author: Face Recognition System
Date: 2024
"""

import argparse
import sys
from pathlib import Path
import cv2
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.detector import RetinaFaceDetector
from core.aligner import FaceAligner
from core.embedder import ArcFaceEmbedder
from core.vector_db import VectorDatabase
from core.recognition import FaceRecognitionPipeline


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Recognize faces in images using enrolled database",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--image', type=str,
                            help='Path to input image')
    input_group.add_argument('--directory', type=str,
                            help='Directory containing images to process')
    
    # Output options
    parser.add_argument('--save-viz', type=str,
                       help='Save visualization to specified path')
    parser.add_argument('--save-results', type=str,
                       help='Save recognition results (JSON) to specified path')
    parser.add_argument('--output-directory', type=str,
                       help='Output directory for batch processing')
    
    # Display options
    parser.add_argument('--show', action='store_true',
                       help='Display result image')
    parser.add_argument('--stats-only', action='store_true',
                       help='Show only statistics without visualization')
    parser.add_argument('--show-landmarks', action='store_true',
                       help='Draw facial landmarks on visualization')
    parser.add_argument('--show-quality', action='store_true', default=True,
                       help='Show quality information (default: True)')
    
    # Database options
    parser.add_argument('--db-path', type=str, default='data/face_database',
                       help='Path to vector database (default: data/face_database)')
    
    # Recognition parameters
    parser.add_argument('--threshold', type=float, default=0.4,
                       help='Similarity threshold for recognition (default: 0.4)')
    parser.add_argument('--top-k', type=int, default=1,
                       help='Number of top matches to return (default: 1)')
    parser.add_argument('--min-quality', type=str, default='FAIR',
                       choices=['EXCELLENT', 'GOOD', 'FAIR', 'POOR'],
                       help='Minimum quality for recognition (default: FAIR)')
    parser.add_argument('--min-detection-confidence', type=float, default=0.5,
                       help='Minimum detection confidence (default: 0.5)')
    
    return parser.parse_args()


def process_single_image(image_path: str, pipeline: FaceRecognitionPipeline, args):
    """Process a single image and return results."""
    print(f"\nProcessing: {Path(image_path).name}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Could not read image: {image_path}")
        return None
    
    # Recognize faces
    start_time = time.time()
    results = pipeline.recognize(
        image,
        top_k=args.top_k,
        return_all_faces=True,
        min_detection_confidence=args.min_detection_confidence
    )
    elapsed_time = (time.time() - start_time) * 1000
    
    # Print results
    print(f"  Detection time: {elapsed_time:.2f} ms")
    print(f"  Detected faces: {len(results)}")
    
    if len(results) > 0 and not args.stats_only:
        print(f"\n  {'='*56}")
        print(f"  {'Face':<8} {'Identity':<20} {'Similarity':<12} {'Quality':<10}")
        print(f"  {'='*56}")
        for i, result in enumerate(results, 1):
            identity = result['identity']
            similarity = result['similarity']
            quality = result['quality']
            print(f"  {i:<8} {identity:<20} {similarity:.4f}       {quality:<10}")
    
    # Get statistics
    stats = pipeline.get_statistics(results)
    
    return {
        'image_path': image_path,
        'image': image,
        'results': results,
        'stats': stats,
        'elapsed_time': elapsed_time
    }


def main():
    """Main recognition function."""
    args = parse_args()
    
    # Initialize components
    print("Initializing face recognition system...")
    detector = RetinaFaceDetector()
    aligner = FaceAligner()
    embedder = ArcFaceEmbedder()
    
    # Load vector database
    vector_db = VectorDatabase(embedding_dim=512)
    db_path = Path(args.db_path)
    
    if not db_path.with_suffix('.index').exists():
        print(f"[ERROR] Database not found: {args.db_path}")
        print("Please enroll users first using tools/enroll_user.py")
        return
    
    vector_db.load(args.db_path)
    
    # Create recognition pipeline
    pipeline = FaceRecognitionPipeline(
        detector, aligner, embedder, vector_db,
        similarity_threshold=args.threshold,
        min_quality=args.min_quality
    )
    
    # Process input
    if args.image:
        # Single image processing
        result_data = process_single_image(args.image, pipeline, args)
        
        if result_data is None:
            return
        
        # Display statistics
        stats = result_data['stats']
        print(f"\n{'='*60}")
        print("Recognition Statistics")
        print(f"{'='*60}")
        print(f"  Total faces: {stats['total_faces']}")
        print(f"  Recognized: {stats['recognized']}")
        print(f"  Unknown: {stats['unknown']}")
        print(f"  Recognition rate: {stats['recognition_rate']:.1%}")
        print(f"  Avg detection confidence: {stats['avg_detection_confidence']:.3f}")
        if stats['recognized'] > 0:
            print(f"  Avg similarity: {stats['avg_similarity']:.3f}")
        
        # Quality distribution
        print(f"\n  Quality Distribution:")
        for quality, count in stats['quality_distribution'].items():
            print(f"    {quality}: {count}")
        
        # Save or display visualization
        if not args.stats_only:
            vis_image = pipeline.visualize_results(
                result_data['image'],
                result_data['results'],
                show_landmarks=args.show_landmarks,
                show_quality=args.show_quality
            )
            
            if args.save_viz:
                output_path = Path(args.save_viz)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(output_path), vis_image)
                print(f"\n✓ Visualization saved to {output_path}")
            
            if args.show:
                cv2.imshow("Recognition Results", vis_image)
                print("\nPress any key to close...")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        # Save results JSON
        if args.save_results:
            pipeline.save_results(args.save_results, result_data['results'])
    
    else:
        # Batch directory processing
        input_dir = Path(args.directory)
        if not input_dir.exists():
            print(f"[ERROR] Directory not found: {args.directory}")
            return
        
        # Find all images
        extensions = ('.jpg', '.jpeg', '.png')
        image_paths = []
        for ext in extensions:
            image_paths.extend(input_dir.glob(f"*{ext}"))
            image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
        
        if len(image_paths) == 0:
            print(f"[ERROR] No images found in {args.directory}")
            return
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        # Setup output directory
        output_dir = Path(args.output_directory) if args.output_directory else input_dir / 'results'
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process all images
        all_stats = []
        for image_path in image_paths:
            result_data = process_single_image(str(image_path), pipeline, args)
            
            if result_data is None:
                continue
            
            all_stats.append(result_data['stats'])
            
            # Save visualization
            if not args.stats_only:
                vis_image = pipeline.visualize_results(
                    result_data['image'],
                    result_data['results'],
                    show_landmarks=args.show_landmarks,
                    show_quality=args.show_quality
                )
                
                output_path = output_dir / f"recognized_{image_path.name}"
                cv2.imwrite(str(output_path), vis_image)
            
            # Save results JSON
            if args.save_results:
                results_path = output_dir / f"{image_path.stem}_results.json"
                pipeline.save_results(str(results_path), result_data['results'])
        
        # Aggregate statistics
        if len(all_stats) > 0:
            total_faces = sum(s['total_faces'] for s in all_stats)
            total_recognized = sum(s['recognized'] for s in all_stats)
            total_unknown = sum(s['unknown'] for s in all_stats)
            
            print(f"\n{'='*60}")
            print(f"Batch Processing Complete")
            print(f"{'='*60}")
            print(f"  Images processed: {len(all_stats)}")
            print(f"  Total faces: {total_faces}")
            print(f"  Recognized: {total_recognized}")
            print(f"  Unknown: {total_unknown}")
            print(f"  Overall recognition rate: {total_recognized/total_faces:.1%}")
            print(f"\n✓ Results saved to {output_dir}")


if __name__ == "__main__":
    main()
