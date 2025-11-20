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

import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Tuple, Dict, Optional
import yaml
from pathlib import Path
import time


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
        
        # Initialize ONNX Runtime session
        self._init_session()
        
        # Generate anchors
        self._generate_anchors()
        
        print(f"✓ RetinaFace detector initialized")
        print(f"  Model: {self.config['model']['name']}")
        print(f"  Input size: {self.input_size}")
        print(f"  Min face size: {self.config['detection']['min_face_size']}px")
    
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
            print(f"  Execution providers: {self.session.get_providers()}")
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ONNX session: {e}")
    
    def _generate_anchors(self):
        """
        Generate anchor boxes for each FPN level.
        
        Anchors are generated based on:
        - Feature map stride
        - Anchor scales (min, max) per level
        - Anchor aspect ratios (typically [1.0] for faces)
        """
        self.anchors = []
        
        for stride, scales in zip(self.feature_strides, self.anchor_scales):
            # Calculate feature map size
            feat_h = self.input_size[1] // stride
            feat_w = self.input_size[0] // stride
            
            # Generate anchor centers
            shift_x = np.arange(0, feat_w) * stride
            shift_y = np.arange(0, feat_h) * stride
            shift_x, shift_y = np.meshgrid(shift_x, shift_y)
            
            # Flatten and stack
            shifts = np.vstack((
                shift_x.ravel(),
                shift_y.ravel(),
                shift_x.ravel(),
                shift_y.ravel()
            )).transpose()
            
            # Generate anchors for this level
            for scale in scales:
                # For aspect ratio = 1.0
                anchor = np.array([
                    -scale / 2, -scale / 2,
                    scale / 2, scale / 2
                ], dtype=np.float32)
                
                # Apply shifts
                anchors = shifts + anchor
                self.anchors.append(anchors)
        
        self.anchors = np.vstack(self.anchors).astype(np.float32)
        print(f"  Generated {len(self.anchors)} anchors across {len(self.feature_strides)} FPN levels")
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for detection.
        
        Args:
            image: Input image (BGR format)
        
        Returns:
            - Preprocessed image tensor
            - Scale factor
            - Padding (pad_w, pad_h)
        """
        img_h, img_w = image.shape[:2]
        target_w, target_h = self.input_size
        
        # Calculate resize scale
        scale = min(target_w / img_w, target_h / img_h)
        
        # Resize image
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Pad to target size
        pad_w = (target_w - new_w) // 2
        pad_h = (target_h - new_h) // 2
        
        padded = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
        
        # Normalize (BGR format, subtract mean, divide by std)
        normalized = (padded.astype(np.float32) - self.mean) / self.std
        
        # Transpose to CHW format and add batch dimension
        tensor = np.transpose(normalized, (2, 0, 1))
        tensor = np.expand_dims(tensor, axis=0)
        
        return tensor, scale, (pad_w, pad_h)
    
    def postprocess(
        self,
        outputs: List[np.ndarray],
        scale: float,
        padding: Tuple[int, int],
        orig_shape: Tuple[int, int]
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
        # Typical RetinaFace output: [scores, boxes, landmarks]
        scores = outputs[0]  # Shape: [N, num_anchors, 2]
        boxes = outputs[1]   # Shape: [N, num_anchors, 4]
        landmarks = outputs[2]  # Shape: [N, num_anchors, 10]
        
        # Remove batch dimension
        scores = scores[0]
        boxes = boxes[0]
        landmarks = landmarks[0]
        
        # Get face class scores (class 1)
        scores = scores[:, 1]
        
        # Filter by confidence
        inds = np.where(scores > self.confidence_threshold)[0]
        scores = scores[inds]
        boxes = boxes[inds]
        landmarks = landmarks[inds]
        
        # Apply top_k
        if len(scores) > self.top_k:
            order = scores.argsort()[::-1][:self.top_k]
            scores = scores[order]
            boxes = boxes[order]
            landmarks = landmarks[order]
        
        # Decode boxes (apply anchor offsets)
        boxes = self._decode_boxes(boxes[inds])
        landmarks = self._decode_landmarks(landmarks[inds])
        
        # Adjust for padding and scale
        pad_w, pad_h = padding
        boxes = (boxes - np.array([pad_w, pad_h, pad_w, pad_h])) / scale
        landmarks = (landmarks - np.array([pad_w, pad_h] * 5)) / scale
        
        # Clip to image boundaries
        orig_h, orig_w = orig_shape
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)
        landmarks[:, [0, 2, 4, 6, 8]] = np.clip(landmarks[:, [0, 2, 4, 6, 8]], 0, orig_w)
        landmarks[:, [1, 3, 5, 7, 9]] = np.clip(landmarks[:, [1, 3, 5, 7, 9]], 0, orig_h)
        
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
    
    def _decode_boxes(self, box_deltas: np.ndarray) -> np.ndarray:
        """
        Decode bounding box deltas using anchors.
        
        Box encoding (standard R-CNN style):
        dx = (pred_x - anchor_x) / anchor_w
        dy = (pred_y - anchor_y) / anchor_h
        dw = log(pred_w / anchor_w)
        dh = log(pred_h / anchor_h)
        
        Args:
            box_deltas: Predicted box offsets [N, 4]
        
        Returns:
            Decoded boxes [N, 4] in [x1, y1, x2, y2] format
        """
        # Anchor boxes [x1, y1, x2, y2]
        anchors = self.anchors[:len(box_deltas)]
        
        # Convert anchors to [cx, cy, w, h]
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + anchor_w * 0.5
        anchor_cy = anchors[:, 1] + anchor_h * 0.5
        
        # Decode
        dx, dy, dw, dh = box_deltas[:, 0], box_deltas[:, 1], box_deltas[:, 2], box_deltas[:, 3]
        
        pred_cx = dx * anchor_w + anchor_cx
        pred_cy = dy * anchor_h + anchor_cy
        pred_w = np.exp(dw) * anchor_w
        pred_h = np.exp(dh) * anchor_h
        
        # Convert to [x1, y1, x2, y2]
        boxes = np.stack([
            pred_cx - pred_w * 0.5,
            pred_cy - pred_h * 0.5,
            pred_cx + pred_w * 0.5,
            pred_cy + pred_h * 0.5
        ], axis=1)
        
        return boxes
    
    def _decode_landmarks(self, landmark_deltas: np.ndarray) -> np.ndarray:
        """
        Decode landmark offsets using anchors.
        
        Args:
            landmark_deltas: Predicted landmark offsets [N, 10]
        
        Returns:
            Decoded landmarks [N, 10] in [x1, y1, x2, y2, ..., x5, y5] format
        """
        anchors = self.anchors[:len(landmark_deltas)]
        
        # Anchor dimensions
        anchor_w = anchors[:, 2] - anchors[:, 0]
        anchor_h = anchors[:, 3] - anchors[:, 1]
        anchor_cx = anchors[:, 0] + anchor_w * 0.5
        anchor_cy = anchors[:, 1] + anchor_h * 0.5
        
        # Decode each landmark point
        landmarks = np.zeros_like(landmark_deltas)
        for i in range(5):
            landmarks[:, i*2] = landmark_deltas[:, i*2] * anchor_w + anchor_cx
            landmarks[:, i*2+1] = landmark_deltas[:, i*2+1] * anchor_h + anchor_cy
        
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
        
        # Preprocess
        tensor, scale, padding = self.preprocess(image)
        
        # Inference
        outputs = self.session.run(None, {self.input_name: tensor})
        
        # Postprocess
        orig_shape = image.shape[:2]
        detections = self.postprocess(outputs, scale, padding, orig_shape)
        
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
