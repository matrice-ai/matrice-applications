from ultralytics import YOLO
import numpy as np
from typing import List, Dict, Tuple
import torch

class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.5):
        """
        Initialize the object detector.
        
        Args:
            model_name: Name of the YOLO model to use
            conf_threshold: Confidence threshold for detections
        """
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold
        self.class_names = self.model.names
        
    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.
        
        Args:
            frame: Input frame as numpy array
            
        Returns:
            List of dictionaries containing detection results
        """
        results = self.model(frame, conf=self.conf_threshold)[0]
        detections = []
        
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.class_names[class_id]
            
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class': class_name,
                'class_id': class_id
            }
            detections.append(detection)
            
        return detections
    
    def filter_classes(self, detections: List[Dict], target_classes: List[str]) -> List[Dict]:
        """
        Filter detections to only include target classes.
        
        Args:
            detections: List of detection dictionaries
            target_classes: List of class names to keep
            
        Returns:
            Filtered list of detections
        """
        return [det for det in detections if det['class'] in target_classes] 