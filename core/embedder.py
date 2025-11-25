"""
ArcFace Embedding Module

Implements face embedding extraction using ArcFace with angular margin loss.

Mathematical Foundation:
- Loss: L = -log(exp(s·cos(θ+m)) / Σ exp(s·cos(θ_j)))
  where:
    - θ = angle between embedding and class center
    - m = angular margin (default: 0.5 radians)
    - s = scale factor (default: 64)
    
- Embedding space: 512-D unit hypersphere (||f|| = 1)
- Geometry: Angular margin creates explicit separation buffer between classes
- Gradient: Contains rotational term (sin m) that reduces angular error

Performance:
- Same-class cosine similarity: μ_s ≈ 0.6, σ_s ≈ 0.1
- Different-class cosine similarity: μ_b ≈ 0, σ_b ≈ 1/√d ≈ 0.045
"""

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Optional, Tuple
import yaml
import time


class ArcFaceEmbedder:
    """
    ArcFace face embedding extractor.
    
    Extracts 512-dimensional L2-normalized embeddings from aligned face images.
    Trained with additive angular margin loss for maximum inter-class separation.
    """
    
    def __init__(self, config_path: str = "config/embedder_config.yaml"):
        """
        Initialize ArcFace embedder.
        
        Args:
            config_path: Path to embedder configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model parameters
        self.model_path = self.config['model']['path']
        self.embedding_dim = self.config['embedding']['dimension']
        self.normalize = self.config['embedding']['normalize']
        self.flip_test = self.config['embedding']['postprocess']['flip_test']
        
        # Preprocessing parameters
        self.input_size = tuple(self.config['preprocessing']['input_size'])
        self.mean = np.array(self.config['preprocessing']['mean'], dtype=np.float32)
        self.std = np.array(self.config['preprocessing']['std'], dtype=np.float32)
        self.pixel_scale = self.config['preprocessing']['pixel_scale']
        self.pixel_bias = self.config['preprocessing']['pixel_bias']
        
        # Quality control
        self.min_face_size = self.config['quality']['min_face_size']
        self.check_blur = self.config['quality']['check_blur']
        self.blur_threshold = self.config['quality']['blur_threshold']
        
        # ArcFace parameters (for reference)
        self.margin = self.config['arcface']['margin']
        self.scale = self.config['arcface']['scale']
        
        # Initialize ONNX Runtime session
        self._init_session()
        
        print(f"✓ ArcFace embedder initialized")
        print(f"  Model: {self.config['model']['name']}")
        print(f"  Embedding dimension: {self.embedding_dim}")
        print(f"  Input size: {self.input_size}")
        print(f"  ArcFace parameters: m={self.margin}, s={self.scale}")
    
    def _init_session(self):
        """Initialize ONNX Runtime inference session."""
        providers = self.config['optimization']['execution_providers']
        
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config['optimization']['session']['intra_op_num_threads']
        sess_options.inter_op_num_threads = self.config['optimization']['session']['inter_op_num_threads']
        
        # Graph optimization
        opt_level_map = {
            'ORT_DISABLE_ALL': ort.GraphOptimizationLevel.ORT_DISABLE_ALL,
            'ORT_ENABLE_BASIC': ort.GraphOptimizationLevel.ORT_ENABLE_BASIC,
            'ORT_ENABLE_EXTENDED': ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
            'ORT_ENABLE_ALL': ort.GraphOptimizationLevel.ORT_ENABLE_ALL,
        }
        sess_options.graph_optimization_level = opt_level_map[
            self.config['optimization']['session']['graph_optimization_level']
        ]
        
        try:
            self.session = ort.InferenceSession(
                self.model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            print(f"  Execution providers: {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")
    
    def preprocess(self, face: np.ndarray) -> np.ndarray:
        """
        Preprocess aligned face for embedding extraction.
        
        Args:
            face: Aligned face image (BGR format) [H, W, 3]
        
        Returns:
            Preprocessed tensor [1, 3, H, W]
        """
        # Resize if needed
        if face.shape[:2] != self.input_size:
            face = cv2.resize(face, self.input_size, interpolation=cv2.INTER_LINEAR)
        
        # BGR to RGB
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # Normalize: (pixel / scale) + bias
        # Typically: (pixel / 127.5) - 1.0 → range [-1, 1]
        face = face.astype(np.float32)
        face = (face - self.mean) / self.std
        
        # Alternative normalization (if using pixel_scale and pixel_bias)
        # face = (face / self.pixel_scale) + self.pixel_bias
        
        # Transpose to CHW and add batch dimension
        tensor = np.transpose(face, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor
    
    def postprocess(self, embedding: np.ndarray) -> np.ndarray:
        """
        Post-process embedding (L2 normalization).
        
        Args:
            embedding: Raw embedding [1, D] or [D]
        
        Returns:
            Normalized embedding [D]
        """
        # Remove batch dimension if present
        if embedding.ndim == 2:
            embedding = embedding[0]
        
        # L2 normalization: f_norm = f / ||f||
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def extract_embedding(self, face: np.ndarray, return_quality: bool = False):
        """
        Extract embedding from aligned face.
        
        Args:
            face: Aligned face image (BGR) [H, W, 3]
            return_quality: If True, return dict with embedding and quality info
        
        Returns:
            If return_quality=False: L2-normalized embedding [512]
            If return_quality=True: dict with 'embedding', 'quality', 'quality_info'
        """
        # Preprocess
        tensor = self.preprocess(face)
        
        # Inference
        embedding = self.session.run([self.output_name], {self.input_name: tensor})[0]
        
        # Postprocess
        embedding = self.postprocess(embedding)
        
        if return_quality:
            # Check face quality
            is_good, quality_metrics = self.check_face_quality(face)
            
            # Determine quality level
            blur_score = quality_metrics.get('blur_score', 0)
            brightness = quality_metrics.get('brightness', 0)
            
            if len(quality_metrics.get('issues', [])) == 0:
                if blur_score >= self.blur_threshold * 2:
                    quality = "EXCELLENT"
                else:
                    quality = "GOOD"
            elif blur_score >= self.blur_threshold * 0.5:
                quality = "FAIR"
            else:
                quality = "POOR"
            
            return {
                'embedding': embedding,
                'quality': quality,
                'quality_info': {
                    'blur_score': blur_score,
                    'brightness': brightness,
                    'quality': quality,
                    'issues': quality_metrics.get('issues', [])
                }
            }
        
        return embedding
    
    def extract_embedding_with_flip(self, face: np.ndarray) -> np.ndarray:
        """
        Extract embedding with horizontal flip test-time augmentation.
        
        Averages embeddings from original and flipped image for better stability.
        
        Args:
            face: Aligned face image (BGR) [H, W, 3]
        
        Returns:
            L2-normalized averaged embedding [512]
        """
        # Original embedding
        emb1 = self.extract_embedding(face)
        
        # Flipped embedding
        face_flipped = cv2.flip(face, 1)
        emb2 = self.extract_embedding(face_flipped)
        
        # Average and re-normalize
        embedding = (emb1 + emb2) / 2.0
        
        if self.normalize:
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding = embedding / norm
        
        return embedding
    
    def extract_embeddings_batch(self, faces: List[np.ndarray]) -> np.ndarray:
        """
        Extract embeddings for multiple faces (batched inference).
        
        Args:
            faces: List of aligned face images (BGR)
        
        Returns:
            Embeddings array [N, 512]
        """
        if len(faces) == 0:
            return np.array([])
        
        # Preprocess all faces
        tensors = [self.preprocess(face) for face in faces]
        batch_tensor = np.vstack(tensors)
        
        # Batch inference
        embeddings = self.session.run([self.output_name], {self.input_name: batch_tensor})[0]
        
        # Postprocess each embedding
        normalized_embeddings = []
        for emb in embeddings:
            emb_norm = self.postprocess(emb)
            normalized_embeddings.append(emb_norm)
        
        return np.array(normalized_embeddings)
    
    def compute_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        For L2-normalized vectors: cosine_similarity = dot_product
        
        Args:
            emb1: First embedding [512]
            emb2: Second embedding [512]
        
        Returns:
            Cosine similarity [-1, 1] (higher = more similar)
        """
        similarity = np.dot(emb1, emb2)
        return float(similarity)
    
    def compute_distance(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """
        Compute Euclidean distance between embeddings.
        
        For normalized vectors: L2_distance = sqrt(2 - 2*cosine_similarity)
        
        Args:
            emb1: First embedding [512]
            emb2: Second embedding [512]
        
        Returns:
            Euclidean distance [0, 2] (lower = more similar)
        """
        distance = np.linalg.norm(emb1 - emb2)
        return float(distance)
    
    def check_face_quality(self, face: np.ndarray) -> Tuple[bool, dict]:
        """
        Check face image quality before embedding extraction.
        
        Checks:
        - Image size
        - Blur detection (Laplacian variance)
        - Brightness
        
        Args:
            face: Face image (BGR)
        
        Returns:
            (is_good, quality_metrics)
        """
        h, w = face.shape[:2]
        metrics = {}
        issues = []
        
        # Check size
        if min(h, w) < self.min_face_size:
            issues.append(f"Too small: {w}x{h} < {self.min_face_size}")
        metrics['size'] = (w, h)
        
        # Check blur
        if self.check_blur:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            metrics['blur_score'] = laplacian_var
            
            if laplacian_var < self.blur_threshold:
                issues.append(f"Blurry: {laplacian_var:.1f} < {self.blur_threshold}")
        
        # Check brightness
        if self.config['quality']['check_brightness']:
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            mean_brightness = gray.mean()
            metrics['brightness'] = mean_brightness
            
            min_bright, max_bright = self.config['quality']['brightness_range']
            if not (min_bright <= mean_brightness <= max_bright):
                issues.append(f"Brightness: {mean_brightness:.1f} not in [{min_bright}, {max_bright}]")
        
        is_good = len(issues) == 0
        metrics['issues'] = issues
        
        return is_good, metrics
    
    def aggregate_embeddings(
        self,
        embeddings: np.ndarray,
        method: str = "mean"
    ) -> np.ndarray:
        """
        Aggregate multiple embeddings into a single representative embedding.
        
        Args:
            embeddings: Multiple embeddings [N, 512]
            method: Aggregation method ("mean", "median")
        
        Returns:
            Aggregated embedding [512]
        """
        if len(embeddings) == 0:
            raise ValueError("Cannot aggregate empty embedding list")
        
        if len(embeddings) == 1:
            return embeddings[0]
        
        if method == "mean":
            agg = np.mean(embeddings, axis=0)
        elif method == "median":
            agg = np.median(embeddings, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Re-normalize
        if self.normalize:
            norm = np.linalg.norm(agg)
            if norm > 0:
                agg = agg / norm
        
        return agg


if __name__ == "__main__":
    # Test embedder
    embedder = ArcFaceEmbedder()
    
    # Test with synthetic face
    test_face = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    
    print(f"\nExtracting embedding from test face...")
    embedding = embedder.extract_embedding(test_face)
    
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding norm: {np.linalg.norm(embedding):.6f}")
    print(f"Embedding sample: {embedding[:10]}")
    
    # Test similarity
    test_face2 = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
    embedding2 = embedder.extract_embedding(test_face2)
    
    similarity = embedder.compute_similarity(embedding, embedding2)
    distance = embedder.compute_distance(embedding, embedding2)
    
    print(f"\nSimilarity between two random faces:")
    print(f"  Cosine similarity: {similarity:.4f}")
    print(f"  Euclidean distance: {distance:.4f}")
    
    # Test quality check
    is_good, metrics = embedder.check_face_quality(test_face)
    print(f"\nQuality check: {'PASS' if is_good else 'FAIL'}")
    print(f"  Metrics: {metrics}")
    
    # Test batch processing
    test_faces = [test_face, test_face2]
    embeddings = embedder.extract_embeddings_batch(test_faces)
    print(f"\nBatch embeddings shape: {embeddings.shape}")
    
    # Test aggregation
    aggregated = embedder.aggregate_embeddings(embeddings, method="mean")
    print(f"Aggregated embedding shape: {aggregated.shape}")
    print(f"Aggregated embedding norm: {np.linalg.norm(aggregated):.6f}")
    
    print("\n✓ Embedder test passed!")
