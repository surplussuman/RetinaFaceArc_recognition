"""
RetinaFace Detector Module

Implements FPN-based face detection with:
- Multi-scale detection (P3, P4, P5, P6, P7)
- 5-point facial landmark regression
- Smooth L1 loss for bounding boxes and landmarks
- Support for faces as small as 16×16 pixels

Mathematical Foundation:
- FPN: Feature Pyramid Network for multi-scale features
- Loss: L = L_cls + λ₁·p*·L_box + λ₂·p*·L_pts
- NMS: Non-Maximum Suppression with IOU threshold
"""

import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
import yaml
from pathlib import Path
import time

from core.onnx_session import (
    resolve_thread_config, configure_session_options, effective_threads, resolve_model_path,
)


class RetinaFaceDetector:
    """
    RetinaFace face detector with FPN backbone.
    
    Detects faces and 5 facial landmarks:
    1. Left eye
    2. Right eye
    3. Nose tip
    4. Left mouth corner
    5. Right mouth corner
    """
    
    def __init__(self, config_path: str = "config/detector_config.yaml"):
        """
        Initialize RetinaFace detector.
        
        Args:
            config_path: Path to detector configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Model parameters
        self.model_path = self.config['model']['path']
        self.confidence_threshold = self.config['detection']['confidence_threshold']
        self.nms_threshold = self.config['detection']['nms']['iou_threshold']
        self.top_k = self.config['detection']['nms']['top_k']
        self.keep_top_k = self.config['detection']['nms']['keep_top_k']
        
        # Preprocessing parameters
        self.input_size = tuple(self.config['preprocessing']['input_size'])
        self.mean = np.array(self.config['preprocessing']['mean'], dtype=np.float32)
        self.std = np.array(self.config['preprocessing']['std'], dtype=np.float32)
        self.scale = self.config['preprocessing']['scale']
        
        # FPN parameters
        self.feature_strides = self.config['detection']['strides']
        self.anchor_scales = self.config['detection']['scales']

        # Phase 1c — dynamic input (default off => exact legacy 640x640 path).
        dyn = self.config['preprocessing'].get('dynamic_input', {}) or {}
        self.dynamic_input = bool(dyn.get('enabled', False))
        self.dyn_stride_multiple = int(dyn.get('stride_multiple', max(self.feature_strides)))
        self.dyn_max_size = int(dyn.get('max_size', self.input_size[0]))

        # Per-input-size anchor cache (thread-safe; detect() is called concurrently
        # by SharedInferenceEngine across streams).
        import threading as _threading
        self._anchor_cache = {}
        self._anchor_lock = _threading.Lock()

        # Initialize ONNX Runtime session
        self._init_session()

        # Generate anchors
        self._generate_anchors()
        
        print(f"[OK] RetinaFace detector initialized")
        print(f"  Model: {self.config['model']['name']}")
        print(f"  Input size: {self.input_size}")
        print(f"  Min face size: {self.config['detection']['min_face_size']}px")
    
    def _init_session(self):
        """Initialize ONNX Runtime inference session."""
        providers = self.config['optimization']['execution_providers']
        
        sess_options = ort.SessionOptions()

        # Resolve thread counts: system_config.yaml -> onnx_threads overrides the
        # per-model yaml. 0 == use all available cores (fixes 4-core utilisation cap).
        local_intra = self.config['optimization']['session']['intra_op_num_threads']
        local_inter = self.config['optimization']['session']['inter_op_num_threads']
        intra_op, inter_op = resolve_thread_config(local_intra, local_inter)
        configure_session_options(sess_options, intra_op, inter_op)
        self._effective_threads = effective_threads(intra_op)

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
        
        # Phase 1b: swap to INT8 model when quantization is enabled (A/B, default off).
        model_path = resolve_model_path(self.model_path)

        try:
            self.session = ort.InferenceSession(
                model_path,
                sess_options=sess_options,
                providers=providers
            )
            self.input_name = self.session.get_inputs()[0].name
            print(f"  Execution providers: {self.session.get_providers()}")
            print(f"  ONNX detector threads: {self._effective_threads} / {os.cpu_count()} available")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")
    
    def _build_anchors(self, input_w: int, input_h: int):
        """
        Build anchor centers for a given input size (W, H).

        For RetinaFace/InsightFace det_10g model:
        - Each FPN level has 2 anchor scales
        - Anchor centers are grid points at stride intervals
        - At 640x640: stride8(80x80x2=12800) + stride16(40x40x2=3200) + stride32(20x20x2=800) = 16800

        Returns (anchor_centers [N,2], num_anchors_per_level list). Pure function of
        the input size — no instance state mutated, so it is safe to call concurrently.
        """
        centers = []
        num_per_level = []
        for stride, scales in zip(self.feature_strides, self.anchor_scales):
            feat_h = input_h // stride
            feat_w = input_w // stride

            # InsightFace-style meshgrid of grid points scaled by stride.
            ac = np.stack(np.mgrid[:feat_h, :feat_w][::-1], axis=-1).astype(np.float32)
            ac = (ac * stride).reshape((-1, 2))

            # 2 anchors per location (num_anchors=2) -> duplicate the centers.
            if len(scales) > 1:
                ac = np.stack([ac] * len(scales), axis=1).reshape((-1, 2))

            num_per_level.append(len(ac))
            centers.append(ac)

        return np.vstack(centers).astype(np.float32), num_per_level

    def _generate_anchors(self):
        """Build and cache the default-size anchors (legacy behaviour)."""
        w, h = self.input_size
        self.anchor_centers, self.num_anchors_per_level = self._build_anchors(w, h)
        self._anchor_cache[(w, h)] = (self.anchor_centers, self.num_anchors_per_level)
        print(f"  Generated {len(self.anchor_centers)} anchor centers across {len(self.feature_strides)} FPN levels")
        print(f"  Anchors per level: {self.num_anchors_per_level}")

    def _get_anchors(self, input_w: int, input_h: int):
        """Return cached anchors for (W, H), building+caching on first use (thread-safe)."""
        key = (input_w, input_h)
        cached = self._anchor_cache.get(key)
        if cached is not None:
            return cached[0]
        with self._anchor_lock:
            cached = self._anchor_cache.get(key)
            if cached is None:
                cached = self._build_anchors(input_w, input_h)
                self._anchor_cache[key] = cached
        return cached[0]

    def _compute_dynamic_size(self, img_w: int, img_h: int):
        """
        Pick an input (W, H) for dynamic mode: scale so the longer side fits
        max_size (never upscale), then round each dim UP to a stride multiple so
        the FPN feature maps are integer-sized. Matches the frame/ROI aspect ratio
        instead of forcing a square -> fewer wasted-padding FLOPs.
        """
        import math
        m = self.dyn_stride_multiple
        s = min(self.dyn_max_size / float(max(img_w, img_h)), 1.0)
        tw = min(self.dyn_max_size, int(math.ceil((img_w * s) / m) * m))
        th = min(self.dyn_max_size, int(math.ceil((img_h * s) / m) * m))
        # Guard against zero-size for tiny crops.
        return max(tw, m), max(th, m)
    
    def _preprocess(self, image: np.ndarray, target_size: Tuple[int, int]):
        """
        Preprocess image for detection at an explicit (target_w, target_h).

        Matches InsightFace preprocessing:
        - Resize maintaining aspect ratio
        - Pad to target size (top-left alignment)
        - Normalize with mean=127.5, std=128.0

        Returns (tensor [1,3,th,tw], scale, padding (0,0), (target_w, target_h)).
        """
        img_h, img_w = image.shape[:2]
        target_w, target_h = target_size

        # Calculate resize scale (maintain aspect ratio)
        im_ratio = float(img_h) / img_w
        model_ratio = float(target_h) / target_w

        if im_ratio > model_ratio:
            new_height = target_h
            new_width = int(new_height / im_ratio)
        else:
            new_width = target_w
            new_height = int(new_width * im_ratio)

        scale = float(new_height) / img_h

        # Resize image
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

        # Pad to target size (InsightFace style - top-left alignment)
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[:new_height, :new_width] = resized

        # Normalize (note: InsightFace uses RGB order in blobFromImage with swapRB=True)
        # This is equivalent to: (BGR_pixel - 127.5) / 128.0 in BGR order
        normalized = (padded.astype(np.float32) - self.mean) / self.std

        # Transpose to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)

        return tensor, scale, (0, 0), (target_w, target_h)

    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Backward-compatible preprocess (always the fixed configured input_size).
        Returns (tensor, scale, padding). detect() uses _preprocess directly so it
        can pick a dynamic size and the matching anchors.
        """
        tensor, scale, padding, _ = self._preprocess(image, tuple(self.input_size))
        return tensor, scale, padding
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int],
        anchor_centers: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Post-process detection outputs.
        
        Args:
            outputs: Model outputs [scores, boxes, landmarks]
            scale: Resize scale factor
            padding: Padding (pad_w, pad_h)
            orig_shape: Original image shape (height, width)
        
        Returns:
            List of detections, each containing:
                - bbox: [x1, y1, x2, y2]
                - confidence: detection score
                - landmarks: [[x1, y1], [x2, y2], [x3, y3], [x4, y4], [x5, y5]]
        """
        # Parse outputs (model-specific)
        # RetinaFace outputs are separated by FPN level
        # 3 levels: P3, P4, P5 with different numbers of anchors
        
        # The model outputs 9 tensors: 3 for scores, 3 for boxes, 3 for landmarks
        # Each FPN level has its own outputs
        fmc = 3  # Number of FPN levels

        # Anchors matching the input size used for THIS call (dynamic input). Falls
        # back to the default-size anchors for the legacy fixed-size path.
        anchors = self.anchor_centers if anchor_centers is None else anchor_centers
        
        # Concatenate outputs from all FPN levels
        # But first, multiply box/landmark predictions by stride
        scores_list = []
        boxes_list = []
        landmarks_list = []
        
        for idx, stride in enumerate(self.feature_strides):
            # Scores
            scores_list.append(outputs[idx])
            
            # Box predictions (multiply by stride for LTRB distance encoding)
            bbox_preds = outputs[idx + fmc] * stride
            boxes_list.append(bbox_preds)
            
            # Landmark predictions (multiply by stride)
            kps_preds = outputs[idx + fmc * 2] * stride
            landmarks_list.append(kps_preds)
        
        scores = np.concatenate(scores_list, axis=0)[:, 0]
        boxes = np.concatenate(boxes_list, axis=0)
        landmarks = np.concatenate(landmarks_list, axis=0)
        
        # Filter by confidence
        inds = np.where(scores > self.confidence_threshold)[0]
        scores = scores[inds]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        anchor_centers_filtered = anchors[inds]  # Filter anchors too!
        
        # Apply top_k
        if len(scores) > self.top_k:
            order = scores.argsort()[::-1][:self.top_k]
            scores = scores[order]
            boxes = boxes[order]
            landmarks = landmarks[order]
            anchor_centers_filtered = anchor_centers_filtered[order]
        
        # Decode boxes (apply anchor offsets) using filtered anchors
        boxes = self._decode_boxes(boxes, anchor_centers_filtered)
        landmarks = self._decode_landmarks(landmarks, anchor_centers_filtered)
        
        # DEBUG: Print before adjustment
        if self.config['debug'].get('verbose', False):
            print(f"  Before adjustment - boxes[0]: {boxes[0] if len(boxes)>0 else 'none'}")
        
        # Adjust for padding and scale
        pad_w, pad_h = padding
        boxes = (boxes - np.array([pad_w, pad_h, pad_w, pad_h])) / scale
        landmarks = (landmarks - np.array([pad_w, pad_h] * 5)) / scale
        
        # DEBUG: Print after adjustment
        if self.config['debug'].get('verbose', False):
            print(f"  After adjustment - boxes[0]: {boxes[0] if len(boxes)>0 else 'none'}")
        
        # Clip to image boundaries
        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        landmarks[:, [0, 2, 4, 6, 8]] = np.clip(landmarks[:, [0, 2, 4, 6, 8]], 0, orig_w)
        landmarks[:, [1, 3, 5, 7, 9]] = np.clip(landmarks[:, [1, 3, 5, 7, 9]], 0, orig_h)
        
        # DEBUG: Print after clipping
        if self.config['debug'].get('verbose', False):
            print(f"  After clipping - boxes[0]: {boxes[0] if len(boxes)>0 else 'none'}")
        
        # Apply NMS
        keep = self._nms(boxes, scores, self.nms_threshold)
        boxes = boxes[keep]
        scores = scores[keep]
        landmarks = landmarks[keep]
        
        # Keep top_k after NMS
        if len(scores) > self.keep_top_k:
            order = scores.argsort()[::-1][:self.keep_top_k]
            boxes = boxes[order]
            scores = scores[order]
            landmarks = landmarks[order]
        
        # Format output
        detections = []
        for i in range(len(scores)):
            detection = {
                'bbox': boxes[i].tolist(),
                'confidence': float(scores[i]),
                'landmarks': landmarks[i].reshape(5, 2).tolist()
            }
            detections.append(detection)
        
        return detections
    
    def _decode_boxes(self, box_deltas: np.ndarray, anchor_centers: np.ndarray) -> np.ndarray:
        """
        Decode bounding box deltas using anchors.
        
        InsightFace RetinaFace uses LTRB (Left-Top-Right-Bottom) distance encoding:
        - box_deltas are already multiplied by stride
        - Format: [left_distance, top_distance, right_distance, bottom_distance]
        - Decoding: x1 = cx - left, y1 = cy - top, x2 = cx + right, y2 = cy + bottom
        
        Args:
            box_deltas: Predicted box distances [N, 4], already scaled by stride
            anchor_centers: Corresponding anchor centers [N, 2]
        
        Returns:
            Decoded boxes [N, 4] in [x1, y1, x2, y2] format
        """
        # LTRB distance decoding
        x1 = anchor_centers[:, 0] - box_deltas[:, 0]
        y1 = anchor_centers[:, 1] - box_deltas[:, 1]
        x2 = anchor_centers[:, 0] + box_deltas[:, 2]
        y2 = anchor_centers[:, 1] + box_deltas[:, 3]
        
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        
        return boxes
    
    def _decode_landmarks(self, landmark_deltas: np.ndarray, anchor_centers: np.ndarray) -> np.ndarray:
        """
        Decode landmark offsets using anchors.
        
        InsightFace landmark encoding:
        - landmark_deltas are already multiplied by stride
        - Each landmark: px = center_x + delta_x, py = center_y + delta_y
        
        Args:
            landmark_deltas: Predicted landmark offsets [N, 10], already scaled by stride
            anchor_centers: Corresponding anchor centers [N, 2]
        
        Returns:
            Decoded landmarks [N, 10] in [x1, y1, x2, y2, ..., x5, y5] format
        """
        # Decode each landmark point
        landmarks = np.zeros_like(landmark_deltas)
        for i in range(5):
            landmarks[:, i*2] = anchor_centers[:, 0] + landmark_deltas[:, i*2]
            landmarks[:, i*2+1] = anchor_centers[:, 1] + landmark_deltas[:, i*2+1]
        
        return landmarks
    
    def _nms(self, boxes: np.ndarray, scores: np.ndarray, threshold: float) -> List[int]:
        """
        Non-Maximum Suppression.
        
        Args:
            boxes: Bounding boxes [N, 4]
            scores: Confidence scores [N]
            threshold: IOU threshold
        
        Returns:
            Indices of kept boxes
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            
            # Compute IOU with remaining boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            
            # Keep boxes with IOU < threshold
            inds = np.where(iou <= threshold)[0]
            order = order[inds + 1]
        
        return keep
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        Detect faces in image.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            List of detections with bounding boxes, confidence, and landmarks
        """
        start_time = time.time()

        # Choose input size: dynamic (match frame/ROI aspect ratio, fewer padding
        # FLOPs) or the fixed configured size. Default off => exact legacy path.
        if self.dynamic_input:
            img_h, img_w = image.shape[:2]
            target_size = self._compute_dynamic_size(img_w, img_h)
        else:
            target_size = tuple(self.input_size)

        # Preprocess at the chosen size
        tensor, scale, padding, input_wh = self._preprocess(image, target_size)

        # Inference
        outputs = self.session.run(None, {self.input_name: tensor})

        # Postprocess with anchors matching the size actually fed to the model
        orig_shape = image.shape[:2]
        anchors = self._get_anchors(*input_wh)
        detections = self.postprocess(outputs, scale, padding, orig_shape, anchor_centers=anchors)
        
        # Log timing
        if self.config['debug']['log_inference_time']:
            elapsed = (time.time() - start_time) * 1000
            print(f"  Detection time: {elapsed:.2f} ms ({len(detections)} faces)")
        
        return detections
    
    def detect_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Batch detection for multiple images.
        
        Args:
            images: List of input images (BGR format)
        
        Returns:
            List of detection lists
        """
        # TODO: Implement true batch processing
        # For now, process sequentially
        return [self.detect(img) for img in images]


if __name__ == "__main__":
    # Test detector
    detector = RetinaFaceDetector()
    
    # Test on sample image
    test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    detections = detector.detect(test_image)
    
    print(f"\nDetected {len(detections)} faces")
    for i, det in enumerate(detections):
        print(f"Face {i+1}:")
        print(f"  BBox: {det['bbox']}")
        print(f"  Confidence: {det['confidence']:.3f}")
        print(f"  Landmarks: {det['landmarks']}")
