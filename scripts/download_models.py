"""
Model Download Script

Downloads pre-trained ONNX models for RetinaFace and ArcFace.

Usage:
    python scripts/download_models.py
    python scripts/download_models.py --model retinaface
    python scripts/download_models.py --model arcface
"""

import os
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_from_insightface():
    """
    Download models from InsightFace model zoo.
    
    This will download both detection and recognition models.
    """
    print("="*60)
    print("Downloading models from InsightFace Model Zoo")
    print("="*60)
    
    try:
        import insightface
        from insightface.app import FaceAnalysis
        print("\n✓ InsightFace installed")
    except ImportError:
        print("\n✗ InsightFace not installed")
        print("\nInstalling insightface...")
        os.system(f"{sys.executable} -m pip install insightface")
        import insightface
        from insightface.app import FaceAnalysis
    
    print("\nInitializing FaceAnalysis (this will download models)...")
    print("Models will be cached in: ~/.insightface/models/")
    
    try:
        app = FaceAnalysis(
            name='buffalo_l',  # Model pack name
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        
        print("\n✓ Models downloaded successfully!")
        print("\nModel location: ~/.insightface/models/buffalo_l/")
        
        # Get model paths
        home = Path.home()
        model_dir = home / ".insightface" / "models" / "buffalo_l"
        
        if model_dir.exists():
            print(f"\nModels found in: {model_dir}")
            print("\nAvailable models:")
            for model_file in model_dir.glob("*.onnx"):
                print(f"  - {model_file.name}")
            
            # Copy to project directory
            print("\n" + "="*60)
            print("Copying models to project directory...")
            print("="*60)
            
            project_dir = Path(__file__).parent.parent
            project_model_dir = project_dir / "models"
            project_model_dir.mkdir(exist_ok=True)
            
            # Map InsightFace models to our naming
            model_mapping = {
                "det_10g.onnx": "retinaface_resnet50.onnx",  # Detection model
                "w600k_r50.onnx": "arcface_resnet100.onnx",  # Recognition model
            }
            
            import shutil
            for src_name, dst_name in model_mapping.items():
                src_path = model_dir / src_name
                dst_path = project_model_dir / dst_name
                
                if src_path.exists():
                    shutil.copy2(src_path, dst_path)
                    size_mb = dst_path.stat().st_size / (1024 * 1024)
                    print(f"✓ Copied {dst_name} ({size_mb:.1f} MB)")
                else:
                    print(f"⚠ {src_name} not found")
            
            print(f"\n✓ Models copied to: {project_model_dir}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False
    
    return True


def download_manual_instructions():
    """
    Print manual download instructions.
    """
    print("="*60)
    print("Manual Download Instructions")
    print("="*60)
    
    print("\n📥 RetinaFace Model:")
    print("   Source: https://github.com/biubug6/Pytorch_Retinaface")
    print("   Or: https://github.com/deepinsight/insightface")
    print("   Model: RetinaFace with ResNet50 backbone")
    print("   Save as: models/retinaface_resnet50.onnx")
    
    print("\n📥 ArcFace Model:")
    print("   Source: https://github.com/deepinsight/insightface")
    print("   Model: ArcFace ResNet100 (trained on MS1MV3)")
    print("   Save as: models/arcface_resnet100.onnx")
    
    print("\n📦 Using InsightFace (Recommended):")
    print("   pip install insightface")
    print("   python scripts/download_models.py")
    
    print("\n⚠️  Alternative Models:")
    print("   For faster inference, use:")
    print("   - RetinaFace with MobileNet: retinaface_mobilenet.onnx")
    print("   - ArcFace with MobileFaceNet: arcface_mobilefacenet.onnx")
    
    print("\n" + "="*60)


def check_models():
    """Check if models exist and are valid."""
    print("="*60)
    print("Checking Model Files")
    print("="*60)
    
    project_dir = Path(__file__).parent.parent
    model_dir = project_dir / "models"
    
    required_models = {
        "retinaface_resnet50.onnx": {
            "min_size": 10 * 1024 * 1024,  # 10 MB
            "description": "RetinaFace detection model"
        },
        "arcface_resnet100.onnx": {
            "min_size": 100 * 1024 * 1024,  # 100 MB
            "description": "ArcFace recognition model"
        }
    }
    
    all_present = True
    
    for model_name, info in required_models.items():
        model_path = model_dir / model_name
        
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            min_size_mb = info["min_size"] / (1024 * 1024)
            
            if model_path.stat().st_size >= info["min_size"]:
                print(f"✓ {model_name}: {size_mb:.1f} MB")
                print(f"    {info['description']}")
            else:
                print(f"⚠ {model_name}: {size_mb:.1f} MB (expected > {min_size_mb:.0f} MB)")
                print(f"    File may be corrupted or incomplete")
                all_present = False
        else:
            print(f"✗ {model_name}: NOT FOUND")
            print(f"    {info['description']}")
            all_present = False
    
    print("\n" + "="*60)
    
    if all_present:
        print("✓ All models present and valid!")
        print("\nYou can now run tests:")
        print("  python tests/test_phase1.py --image test.jpg")
    else:
        print("✗ Some models are missing")
        print("\nDownload models using:")
        print("  python scripts/download_models.py")
        print("\nOr see manual instructions:")
        print("  python scripts/download_models.py --help")
    
    print("="*60)
    
    return all_present


def main():
    parser = argparse.ArgumentParser(
        description="Download pre-trained models for face recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download all models automatically
  python scripts/download_models.py
  
  # Check if models exist
  python scripts/download_models.py --check
  
  # Show manual download instructions
  python scripts/download_models.py --manual
        """
    )
    
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Check if models exist and are valid'
    )
    
    parser.add_argument(
        '--manual',
        action='store_true',
        help='Show manual download instructions'
    )
    
    args = parser.parse_args()
    
    if args.check:
        check_models()
    elif args.manual:
        download_manual_instructions()
    else:
        # Try automatic download
        print("\n🚀 Starting automatic model download...\n")
        
        success = download_from_insightface()
        
        if success:
            print("\n" + "="*60)
            print("✓ Setup Complete!")
            print("="*60)
            check_models()
        else:
            print("\n" + "="*60)
            print("⚠ Automatic download failed")
            print("="*60)
            download_manual_instructions()


if __name__ == "__main__":
    main()
