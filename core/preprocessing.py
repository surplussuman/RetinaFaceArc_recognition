"""
Image preprocessing for face recognition quality enhancement.

This module implements preprocessing techniques to improve recognition
on low-quality images (old photos, CCTV footage, poor lighting, blur).

Mathematical Foundations:
-------------------------
1. CLAHE (Contrast Limited Adaptive Histogram Equalization):
   - Operates on tiles (default 8×8), computes local CDF
   - Clip limit prevents noise amplification
   - Transform: I'(x,y) = CDF_tile(I(x,y)) * 255
   
2. Bilateral Filter (Edge-Preserving Denoising):
   - I_filtered(x) = (1/W) Σ I(x_i) · G_σs(||x-x_i||) · G_σr(|I(x)-I(x_i)|)
   - σs: spatial variance (neighbor distance)
   - σr: range variance (intensity difference)
   - Preserves edges while smoothing homogeneous regions
   
3. Unsharp Masking (Sharpness Enhancement):
   - I_sharp = I + α·(I - G_σ*I)
   - G_σ*I: Gaussian-blurred image
   - α: sharpening strength (default 1.5)
   
4. Adaptive Contrast Enhancement:
   - I_contrast = α·I + β
   - α: contrast multiplier, β: brightness offset
   - Expands dynamic range for better feature extraction

Author: Copilot
Date: 2024
"""

import cv2
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path


class ImagePreprocessor:
    """
    Image preprocessing pipeline for face recognition quality enhancement.
    
    Applies sequence of transformations to improve recognition accuracy
    on degraded images while preserving facial features.
    """
    
    def __init__(
        self,
        enable_clahe: bool = True,
        enable_denoise: bool = True,
        enable_sharpen: bool = True,
        enable_contrast: bool = True,
        clahe_clip_limit: float = 2.0,
        clahe_tile_size: int = 8,
        bilateral_d: int = 5,
        bilateral_sigma_color: float = 50.0,
        bilateral_sigma_space: float = 50.0,
        sharpen_sigma: float = 1.0,
        sharpen_alpha: float = 1.5,
        contrast_alpha: float = 1.2,
        contrast_beta: float = 10.0
    ):
        """
        Initialize preprocessor with configurable parameters.
        
        Args:
            enable_clahe: Apply CLAHE for low-light enhancement
            enable_denoise: Apply bilateral filter for noise reduction
            enable_sharpen: Apply unsharp masking for sharpness
            enable_contrast: Apply adaptive contrast enhancement
            clahe_clip_limit: CLAHE clip limit (2.0 = moderate enhancement)
            clahe_tile_size: CLAHE tile grid size (8×8 default)
            bilateral_d: Bilateral filter diameter (5 = small neighborhood)
            bilateral_sigma_color: Bilateral range variance (50 = moderate)
            bilateral_sigma_space: Bilateral spatial variance (50 = moderate)
            sharpen_sigma: Gaussian blur sigma for unsharp mask (1.0)
            sharpen_alpha: Sharpening strength (1.5 = strong)
            contrast_alpha: Contrast multiplier (1.2 = 20% increase)
            contrast_beta: Brightness offset (10 = slight brightening)
        """
        self.enable_clahe = enable_clahe
        self.enable_denoise = enable_denoise
        self.enable_sharpen = enable_sharpen
        self.enable_contrast = enable_contrast
        
        # CLAHE parameters
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_size = (clahe_tile_size, clahe_tile_size)
        
        # Bilateral filter parameters
        self.bilateral_d = bilateral_d
        self.bilateral_sigma_color = bilateral_sigma_color
        self.bilateral_sigma_space = bilateral_sigma_space
        
        # Sharpening parameters
        self.sharpen_sigma = sharpen_sigma
        self.sharpen_alpha = sharpen_alpha
        
        # Contrast parameters
        self.contrast_alpha = contrast_alpha
        self.contrast_beta = contrast_beta
        
        # Initialize CLAHE object
        if self.enable_clahe:
            self.clahe = cv2.createCLAHE(
                clipLimit=self.clahe_clip_limit,
                tileGridSize=self.clahe_tile_size
            )
    
    def preprocess(
        self,
        image: np.ndarray,
        return_steps: bool = False
    ) -> np.ndarray | Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Apply full preprocessing pipeline to input image.
        
        Pipeline order:
        1. CLAHE (if enabled) - normalize lighting
        2. Bilateral filter (if enabled) - reduce noise
        3. Unsharp masking (if enabled) - enhance sharpness
        4. Adaptive contrast (if enabled) - expand dynamic range
        
        Args:
            image: Input BGR image (H×W×3 uint8)
            return_steps: Return intermediate results for visualization
            
        Returns:
            Preprocessed BGR image (H×W×3 uint8)
            If return_steps=True: (final_image, {step_name: step_image})
        """
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        
        if len(image.shape) != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected BGR image (H×W×3), got shape {image.shape}")
        
        # Store intermediate results if requested
        steps = {} if return_steps else None
        
        # Start with input image
        result = image.copy()
        if return_steps:
            steps['input'] = image.copy()
        
        # Step 1: CLAHE (on luminance channel in LAB color space)
        if self.enable_clahe:
            result = self._apply_clahe(result)
            if return_steps:
                steps['clahe'] = result.copy()
        
        # Step 2: Bilateral filtering (denoise while preserving edges)
        if self.enable_denoise:
            result = self._apply_bilateral_filter(result)
            if return_steps:
                steps['denoise'] = result.copy()
        
        # Step 3: Unsharp masking (enhance sharpness)
        if self.enable_sharpen:
            result = self._apply_unsharp_mask(result)
            if return_steps:
                steps['sharpen'] = result.copy()
        
        # Step 4: Adaptive contrast enhancement
        if self.enable_contrast:
            result = self._apply_contrast_enhancement(result)
            if return_steps:
                steps['contrast'] = result.copy()
        
        if return_steps:
            return result, steps
        return result
    
    def _apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to luminance channel in LAB color space.
        
        CLAHE (Contrast Limited Adaptive Histogram Equalization):
        - Operates on local tiles to adapt to varying lighting
        - Clip limit prevents noise amplification in homogeneous regions
        - Applied to L channel (luminance) to avoid color distortion
        
        Args:
            image: Input BGR image (H×W×3 uint8)
            
        Returns:
            CLAHE-enhanced BGR image (H×W×3 uint8)
        """
        # Convert BGR → LAB
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Split into L, A, B channels
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Apply CLAHE to L channel
        l_enhanced = self.clahe.apply(l_channel)
        
        # Merge channels back
        lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
        
        # Convert LAB → BGR
        bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
        
        return bgr_enhanced
    
    def _apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Apply bilateral filter for edge-preserving denoising.
        
        Bilateral Filter:
        I_filtered(x) = (1/W) Σ I(x_i) · G_σs(||x-x_i||) · G_σr(|I(x)-I(x_i)|)
        
        Two Gaussian kernels:
        - Spatial (σs): Weight by neighbor distance
        - Range (σr): Weight by intensity similarity
        
        Result: Smooths homogeneous regions, preserves edges
        
        Args:
            image: Input BGR image (H×W×3 uint8)
            
        Returns:
            Denoised BGR image (H×W×3 uint8)
        """
        return cv2.bilateralFilter(
            image,
            d=self.bilateral_d,
            sigmaColor=self.bilateral_sigma_color,
            sigmaSpace=self.bilateral_sigma_space
        )
    
    def _apply_unsharp_mask(self, image: np.ndarray) -> np.ndarray:
        """
        Apply unsharp masking for sharpness enhancement.
        
        Unsharp Masking:
        I_sharp = I + α·(I - G_σ*I)
        
        Where:
        - I: Original image
        - G_σ*I: Gaussian-blurred version
        - α: Sharpening strength
        - (I - G_σ*I): High-frequency details
        
        Args:
            image: Input BGR image (H×W×3 uint8)
            
        Returns:
            Sharpened BGR image (H×W×3 uint8)
        """
        # Compute Gaussian-blurred version
        blurred = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=self.sharpen_sigma
        )
        
        # Compute detail layer: I - G_σ*I
        # Use float32 to prevent overflow/underflow
        detail = image.astype(np.float32) - blurred.astype(np.float32)
        
        # Add scaled detail back: I + α·detail
        sharpened = image.astype(np.float32) + self.sharpen_alpha * detail
        
        # Clip to valid range and convert back to uint8
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        return sharpened
    
    def _apply_contrast_enhancement(self, image: np.ndarray) -> np.ndarray:
        """
        Apply adaptive contrast enhancement.
        
        Linear Transform:
        I_contrast = α·I + β
        
        Where:
        - α > 1: Increase contrast (expand dynamic range)
        - β > 0: Increase brightness (shift dynamic range)
        
        Args:
            image: Input BGR image (H×W×3 uint8)
            
        Returns:
            Contrast-enhanced BGR image (H×W×3 uint8)
        """
        # Apply linear transform
        enhanced = cv2.convertScaleAbs(
            image,
            alpha=self.contrast_alpha,
            beta=self.contrast_beta
        )
        
        return enhanced
    
    def preprocess_face(
        self,
        face_image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Preprocess face crop for embedding extraction.
        
        Applies full preprocessing pipeline optimized for face recognition.
        Optionally resizes to target size (e.g., 112×112 for ArcFace).
        
        Args:
            face_image: Face crop BGR image (H×W×3 uint8)
            target_size: Optional (width, height) for resizing
            
        Returns:
            Preprocessed face image (uint8)
        """
        # Apply preprocessing pipeline
        preprocessed = self.preprocess(face_image)
        
        # Resize if target size specified
        if target_size is not None:
            preprocessed = cv2.resize(
                preprocessed,
                target_size,
                interpolation=cv2.INTER_LINEAR
            )
        
        return preprocessed
    
    def create_degraded_version(
        self,
        image: np.ndarray,
        blur_sigma: float = 1.5,
        scale_factor: float = 0.7,
        brightness_offset: int = -20
    ) -> np.ndarray:
        """
        Create synthetically degraded version of image for multi-quality enrollment.
        
        Simulates quality degradation from:
        - Old photos (blur, fading)
        - Low resolution (downsampling)
        - Poor lighting (brightness reduction)
        
        Args:
            image: Input BGR image (H×W×3 uint8)
            blur_sigma: Gaussian blur sigma (1.5 = moderate blur)
            scale_factor: Resolution scale (0.7 = 30% reduction)
            brightness_offset: Brightness shift (-20 = darker)
            
        Returns:
            Degraded BGR image (same size as input)
        """
        # Apply Gaussian blur
        degraded = cv2.GaussianBlur(
            image,
            ksize=(0, 0),
            sigmaX=blur_sigma
        )
        
        # Reduce resolution
        h, w = degraded.shape[:2]
        small_h, small_w = int(h * scale_factor), int(w * scale_factor)
        degraded = cv2.resize(degraded, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        degraded = cv2.resize(degraded, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Adjust brightness
        degraded = cv2.convertScaleAbs(
            degraded,
            alpha=1.0,
            beta=brightness_offset
        )
        
        return degraded
    
    def get_config(self) -> Dict:
        """
        Get current preprocessing configuration.
        
        Returns:
            Dictionary of all preprocessing parameters
        """
        return {
            'enable_clahe': self.enable_clahe,
            'enable_denoise': self.enable_denoise,
            'enable_sharpen': self.enable_sharpen,
            'enable_contrast': self.enable_contrast,
            'clahe_clip_limit': self.clahe_clip_limit,
            'clahe_tile_size': self.clahe_tile_size,
            'bilateral_d': self.bilateral_d,
            'bilateral_sigma_color': self.bilateral_sigma_color,
            'bilateral_sigma_space': self.bilateral_sigma_space,
            'sharpen_sigma': self.sharpen_sigma,
            'sharpen_alpha': self.sharpen_alpha,
            'contrast_alpha': self.contrast_alpha,
            'contrast_beta': self.contrast_beta
        }


def create_default_preprocessor() -> ImagePreprocessor:
    """
    Create preprocessor with default parameters optimized for face recognition.
    
    Returns:
        ImagePreprocessor with balanced settings for quality enhancement
    """
    return ImagePreprocessor(
        enable_clahe=True,
        enable_denoise=True,
        enable_sharpen=True,
        enable_contrast=True,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        bilateral_d=5,
        bilateral_sigma_color=50.0,
        bilateral_sigma_space=50.0,
        sharpen_sigma=1.0,
        sharpen_alpha=1.5,
        contrast_alpha=1.2,
        contrast_beta=10.0
    )


def create_aggressive_preprocessor() -> ImagePreprocessor:
    """
    Create preprocessor with aggressive enhancement for very poor quality images.
    
    Use for:
    - Old photos with significant fading/blur
    - CCTV footage with low light
    - Heavily compressed images
    
    Returns:
        ImagePreprocessor with strong enhancement settings
    """
    return ImagePreprocessor(
        enable_clahe=True,
        enable_denoise=True,
        enable_sharpen=True,
        enable_contrast=True,
        clahe_clip_limit=3.0,  # Stronger local contrast
        clahe_tile_size=4,     # Smaller tiles for more local adaptation
        bilateral_d=7,         # Larger neighborhood for stronger denoising
        bilateral_sigma_color=75.0,  # More aggressive color smoothing
        bilateral_sigma_space=75.0,  # Larger spatial influence
        sharpen_sigma=1.5,     # More blur in unsharp mask
        sharpen_alpha=2.0,     # Stronger sharpening
        contrast_alpha=1.5,    # Stronger contrast boost
        contrast_beta=20.0     # Stronger brightness boost
    )


def create_minimal_preprocessor() -> ImagePreprocessor:
    """
    Create preprocessor with minimal enhancement for high-quality images.
    
    Use for:
    - Recent photos with good lighting
    - Professional photography
    - High-resolution sources
    
    Returns:
        ImagePreprocessor with conservative settings
    """
    return ImagePreprocessor(
        enable_clahe=True,
        enable_denoise=False,  # Skip denoising for clean images
        enable_sharpen=False,  # Skip sharpening for already sharp images
        enable_contrast=True,
        clahe_clip_limit=1.5,  # Gentle local contrast
        clahe_tile_size=16,    # Larger tiles for less aggressive adaptation
        contrast_alpha=1.1,    # Gentle contrast boost
        contrast_beta=5.0      # Minimal brightness adjustment
    )


if __name__ == '__main__':
    """
    Demonstration of preprocessing pipeline on sample image.
    """
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python preprocessing.py <image_path>")
        print("Applies preprocessing pipeline and saves intermediate steps")
        sys.exit(1)
    
    image_path = Path(sys.argv[1])
    if not image_path.exists():
        print(f"Error: Image not found: {image_path}")
        sys.exit(1)
    
    # Load image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Failed to load image: {image_path}")
        sys.exit(1)
    
    print(f"Processing: {image_path.name}")
    print(f"Image size: {image.shape[1]}×{image.shape[0]}")
    
    # Create preprocessor
    preprocessor = create_default_preprocessor()
    
    # Process with intermediate steps
    result, steps = preprocessor.preprocess(image, return_steps=True)
    
    # Save intermediate steps
    output_dir = Path('preprocessing_steps')
    output_dir.mkdir(exist_ok=True)
    
    for step_name, step_image in steps.items():
        output_path = output_dir / f"{image_path.stem}_{step_name}.jpg"
        cv2.imwrite(str(output_path), step_image)
        print(f"  Saved: {output_path.name}")
    
    print(f"\nPreprocessing complete. Check {output_dir}/ for results.")
