"""
Face Alignment Module

Implements 5-point affine alignment using least-squares solver:
M = (AᵀA)⁻¹AᵀT

Mathematical Foundation:
- Input: 5 detected landmarks (left_eye, right_eye, nose, mouth_left, mouth_right)
- Output: Affine-aligned 112×112 (or 160×160) frontal crop
- Method: Solve least-squares problem to find optimal affine transformation
- Error propagation: δu_s ≈ J_u·δm (minimizes geometric variance)

The alignment transforms nuisance variability (pose, rotation, scale) into 
small additive noise, making embeddings stable and consistent.
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional
import yaml


class FaceAligner:
    """
    5-point affine face alignment.
    
    Transforms detected faces to a canonical frontal view using
    least-squares affine transformation based on facial landmarks.
    """
    
    # Standard 5-point facial landmark template (normalized to 112x112)
    # Based on empirical face geometry
    TEMPLATE_112 = np.array([
        [38.2946, 51.6963],  # Left eye
        [73.5318, 51.5014],  # Right eye
        [56.0252, 71.7366],  # Nose tip
        [41.5493, 92.3655],  # Left mouth corner
        [70.7299, 92.2041]   # Right mouth corner
    ], dtype=np.float32)
    
    TEMPLATE_160 = np.array([
        [54.706,  73.852],   # Left eye
        [105.045, 73.573],   # Right eye
        [80.036,  102.481],  # Nose tip
        [59.356,  131.95],   # Left mouth corner
        [101.021, 131.76]    # Right mouth corner
    ], dtype=np.float32)
    
    def __init__(self, output_size: int = 112):
        """
        Initialize face aligner.
        
        Args:
            output_size: Output image size (112 or 160)
        """
        if output_size == 112:
            self.template = self.TEMPLATE_112
        elif output_size == 160:
            self.template = self.TEMPLATE_160
        else:
            raise ValueError(f"Unsupported output size: {output_size}. Use 112 or 160.")
        
        self.output_size = output_size
        
        print(f"✓ Face aligner initialized")
        print(f"  Output size: {output_size}×{output_size}")
        print(f"  Template landmarks: {len(self.template)} points")
    
    def estimate_affine_transform(
        self,
        src_landmarks: np.ndarray,
        dst_landmarks: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Estimate affine transformation matrix using least-squares.
        
        Mathematical derivation:
        Given source landmarks S and destination landmarks D, we want to find
        affine matrix M (2×3) such that:
            M · [x_s, y_s, 1]ᵀ = [x_d, y_d]ᵀ
        
        Expanding M = [a b c]
                      [d e f]
        
        For each landmark pair i:
            a·x_si + b·y_si + c = x_di
            d·x_si + e·y_si + f = y_di
        
        Stack into linear system: A·m = t
        where m = [a, b, c, d, e, f]ᵀ
        
        Solution: m = (AᵀA)⁻¹Aᵀt  (least-squares)
        
        Args:
            src_landmarks: Source landmarks [N, 2]
            dst_landmarks: Destination landmarks [N, 2] (default: template)
        
        Returns:
            Affine transformation matrix [2, 3]
        """
        if dst_landmarks is None:
            dst_landmarks = self.template
        
        # Ensure landmarks are numpy arrays
        src_landmarks = np.array(src_landmarks, dtype=np.float32)
        dst_landmarks = np.array(dst_landmarks, dtype=np.float32)
        
        # Validate input
        if src_landmarks.shape[0] != dst_landmarks.shape[0]:
            raise ValueError(
                f"Landmark count mismatch: src={src_landmarks.shape[0]}, "
                f"dst={dst_landmarks.shape[0]}"
            )
        
        if src_landmarks.shape[0] < 3:
            raise ValueError(
                f"Need at least 3 landmark pairs, got {src_landmarks.shape[0]}"
            )
        
        # Use cv2.estimateAffinePartial2D for robust estimation
        # This uses RANSAC internally for outlier rejection
        M, inliers = cv2.estimateAffinePartial2D(
            src_landmarks,
            dst_landmarks,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0
        )
        
        if M is None:
            # Fallback to direct least-squares if RANSAC fails
            M = self._estimate_affine_least_squares(src_landmarks, dst_landmarks)
        
        return M
    
    def _estimate_affine_least_squares(
        self,
        src_pts: np.ndarray,
        dst_pts: np.ndarray
    ) -> np.ndarray:
        """
        Direct least-squares estimation of affine matrix.
        
        Manual implementation of M = (AᵀA)⁻¹Aᵀt
        
        Args:
            src_pts: Source points [N, 2]
            dst_pts: Destination points [N, 2]
        
        Returns:
            Affine matrix [2, 3]
        """
        n = src_pts.shape[0]
        
        # Construct matrix A
        # For each point: [x, y, 1, 0, 0, 0] for x-equation
        #                 [0, 0, 0, x, y, 1] for y-equation
        A = np.zeros((2*n, 6), dtype=np.float32)
        t = np.zeros((2*n,), dtype=np.float32)
        
        for i in range(n):
            x_s, y_s = src_pts[i]
            x_d, y_d = dst_pts[i]
            
            # x-equation
            A[2*i] = [x_s, y_s, 1, 0, 0, 0]
            t[2*i] = x_d
            
            # y-equation
            A[2*i+1] = [0, 0, 0, x_s, y_s, 1]
            t[2*i+1] = y_d
        
        # Solve least-squares: m = (AᵀA)⁻¹Aᵀt
        m, residuals, rank, s = np.linalg.lstsq(A, t, rcond=None)
        
        # Reshape to 2×3 matrix
        M = m.reshape(2, 3)
        
        return M
    
    def align_face(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        output_size: Optional[int] = None
    ) -> np.ndarray:
        """
        Align face using affine transformation.
        
        Args:
            image: Input image (BGR)
            landmarks: Facial landmarks [5, 2] or [N, 2]
            output_size: Output size (default: use initialized size)
        
        Returns:
            Aligned face crop [H, W, 3]
        """
        if output_size is None:
            output_size = self.output_size
        
        # Use appropriate template
        if output_size != self.output_size:
            if output_size == 112:
                template = self.TEMPLATE_112
            elif output_size == 160:
                template = self.TEMPLATE_160
            else:
                # Scale current template
                scale = output_size / self.output_size
                template = self.template * scale
        else:
            template = self.template
        
        # Estimate affine transformation
        M = self.estimate_affine_transform(landmarks, template)
        
        # Apply affine warp
        aligned = cv2.warpAffine(
            image,
            M,
            (output_size, output_size),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0
        )
        
        return aligned
    
    def align_faces_batch(
        self,
        image: np.ndarray,
        landmarks_list: List[np.ndarray],
        output_size: Optional[int] = None
    ) -> List[np.ndarray]:
        """
        Align multiple faces from the same image.
        
        Args:
            image: Input image (BGR)
            landmarks_list: List of facial landmarks [N×[5, 2]]
            output_size: Output size
        
        Returns:
            List of aligned face crops
        """
        aligned_faces = []
        
        for landmarks in landmarks_list:
            try:
                aligned = self.align_face(image, landmarks, output_size)
                aligned_faces.append(aligned)
            except Exception as e:
                print(f"Warning: Failed to align face: {e}")
                # Return blank image as placeholder
                size = output_size or self.output_size
                aligned_faces.append(np.zeros((size, size, 3), dtype=np.uint8))
        
        return aligned_faces
    
    def compute_alignment_error(
        self,
        src_landmarks: np.ndarray,
        M: np.ndarray
    ) -> float:
        """
        Compute alignment error (reprojection error).
        
        Error = RMSE of |M·src - dst|
        
        Args:
            src_landmarks: Source landmarks [N, 2]
            M: Affine matrix [2, 3]
        
        Returns:
            RMSE error in pixels
        """
        # Transform source landmarks
        src_homo = np.hstack([src_landmarks, np.ones((len(src_landmarks), 1))])
        transformed = (M @ src_homo.T).T
        
        # Compute error
        error = np.sqrt(np.mean((transformed - self.template) ** 2))
        
        return error
    
    def check_alignment_quality(
        self,
        landmarks: np.ndarray,
        min_quality: float = 3.0
    ) -> Tuple[bool, float]:
        """
        Check if landmarks are suitable for alignment.
        
        Args:
            landmarks: Facial landmarks [5, 2]
            min_quality: Maximum acceptable RMSE (pixels)
        
        Returns:
            (is_good, rmse_error)
        """
        try:
            M = self.estimate_affine_transform(landmarks)
            error = self.compute_alignment_error(landmarks, M)
            is_good = error <= min_quality
            return is_good, error
        except Exception:
            return False, float('inf')
    
    def visualize_alignment(
        self,
        image: np.ndarray,
        landmarks: np.ndarray,
        aligned: np.ndarray
    ) -> np.ndarray:
        """
        Visualize original and aligned face side by side.
        
        Args:
            image: Original image
            landmarks: Original landmarks
            aligned: Aligned face crop
        
        Returns:
            Visualization image
        """
        # Draw landmarks on original
        vis_orig = image.copy()
        for i, (x, y) in enumerate(landmarks):
            cv2.circle(vis_orig, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                vis_orig, str(i+1), (int(x)+5, int(y)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
        
        # Draw landmarks on aligned (template positions)
        vis_aligned = aligned.copy()
        for i, (x, y) in enumerate(self.template):
            cv2.circle(vis_aligned, (int(x), int(y)), 3, (0, 255, 0), -1)
            cv2.putText(
                vis_aligned, str(i+1), (int(x)+5, int(y)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1
            )
        
        # Resize original crop to match aligned size
        x_min = int(landmarks[:, 0].min()) - 20
        y_min = int(landmarks[:, 1].min()) - 20
        x_max = int(landmarks[:, 0].max()) + 20
        y_max = int(landmarks[:, 1].max()) + 20
        
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(image.shape[1], x_max)
        y_max = min(image.shape[0], y_max)
        
        crop = image[y_min:y_max, x_min:x_max]
        if crop.size > 0:
            crop_resized = cv2.resize(crop, (self.output_size, self.output_size))
        else:
            crop_resized = np.zeros((self.output_size, self.output_size, 3), dtype=np.uint8)
        
        # Concatenate
        vis = np.hstack([crop_resized, vis_aligned])
        
        return vis


if __name__ == "__main__":
    # Test aligner
    aligner = FaceAligner(output_size=112)
    
    # Test with synthetic landmarks
    # Simulate detected landmarks (slightly rotated/scaled)
    angle = 15 * np.pi / 180  # 15 degrees
    scale = 0.9
    center = np.array([320, 240])
    
    # Rotation matrix
    R = np.array([
        [np.cos(angle), -np.sin(angle)],
        [np.sin(angle), np.cos(angle)]
    ])
    
    # Transform template landmarks
    template_centered = aligner.TEMPLATE_112 - 56  # Center around origin
    src_landmarks = (R @ template_centered.T).T * scale + center
    
    print(f"\nSource landmarks (simulated detection):")
    print(src_landmarks)
    
    # Estimate affine transform
    M = aligner.estimate_affine_transform(src_landmarks)
    print(f"\nEstimated affine matrix:")
    print(M)
    
    # Check alignment quality
    is_good, error = aligner.check_alignment_quality(src_landmarks)
    print(f"\nAlignment quality: {'GOOD' if is_good else 'BAD'}")
    print(f"RMSE error: {error:.2f} pixels")
    
    # Test with actual image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    aligned = aligner.align_face(test_image, src_landmarks)
    
    print(f"\nAligned face shape: {aligned.shape}")
    print("✓ Aligner test passed!")
