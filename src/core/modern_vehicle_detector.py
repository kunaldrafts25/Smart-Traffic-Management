import cv2
import numpy as np
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass
import logging

# Setup logging
logger = logging.getLogger("VehicleDetector")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


@dataclass
class DetectionResult:
    vehicle_count: int
    detections: List[Dict[str, Any]]
    confidence_scores: List[float]
    processing_time: float
    frame_id: int
    timestamp: float
    model_name: str


class ModernVehicleDetector:
    VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
    VEHICLE_IDS = [2, 3, 5, 7, 1]  # COCO class IDs
    
    def __init__(self, config_path: Optional[str] = None):
        self.model = None
        self.model_name = "yolov8n"
        self.confidence_threshold = 0.25
        self.frame_count = 0
        self.total_time = 0.0        
        try:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"PyTorch {torch.__version__} on {self.device}")
        except ImportError:
            self.device = "cpu"
        
        self._load_model()
    
    def _load_model(self):
        try:
            from ultralytics import YOLO
            model_path = Path("models/yolov8n.pt")
            if model_path.exists():
                self.model = YOLO(str(model_path))
                logger.info(f"YOLO loaded: {model_path} on {self.device}")
            else:
                # Download from ultralytics
                self.model = YOLO("yolov8n.pt")
                logger.info(f"YOLO downloaded: yolov8n.pt on {self.device}")
                
        except Exception as e:
            logger.error(f"YOLO load failed: {e}")
            self.model = None
    
    def detect_vehicles(self, frame: np.ndarray, frame_id: int = 0) -> DetectionResult:
        start_time = time.time()
        
        if self.model is None:
            return DetectionResult(
                vehicle_count=0, detections=[], confidence_scores=[],
                processing_time=0.0, frame_id=frame_id,
                timestamp=time.time(), model_name=self.model_name
            )
        
        try:
            results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            detections = []
            confidences = []
            
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0])
                    if cls_id in self.VEHICLE_IDS:
                        conf = float(box.conf[0])
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        detections.append({
                            'class_id': cls_id,
                            'class_name': self.VEHICLE_CLASSES[self.VEHICLE_IDS.index(cls_id)],
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        confidences.append(conf)
            
            processing_time = time.time() - start_time
            self.frame_count += 1
            self.total_time += processing_time
            
            logger.info("Detection done")
            logger.info("Perf: detect_vehicles done")
            
            return DetectionResult(
                vehicle_count=len(detections),
                detections=detections,
                confidence_scores=confidences,
                processing_time=processing_time,
                frame_id=frame_id,
                timestamp=time.time(),
                model_name=self.model_name
            )
            
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return DetectionResult(
                vehicle_count=0, detections=[], confidence_scores=[],
                processing_time=0.0, frame_id=frame_id,
                timestamp=time.time(), model_name=self.model_name
            )
    
    def draw_detections(self, frame: np.ndarray, result: DetectionResult) -> np.ndarray:
        output = frame.copy()
        
        colors = {
            'car': (0, 255, 0),
            'motorcycle': (255, 165, 0),
            'bus': (255, 0, 0),
            'truck': (0, 0, 255),
            'bicycle': (255, 255, 0)
        }
        
        for det in result.detections:
            x1, y1, x2, y2 = det['bbox']
            cls_name = det['class_name']
            conf = det['confidence']
            color = colors.get(cls_name, (255, 255, 255))
            
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            cv2.putText(output, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        return output
    
    def is_model_loaded(self) -> bool:
        return self.model is not None
