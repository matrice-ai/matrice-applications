import cv2
import json
from typing import Dict, List, Tuple
import os
from datetime import datetime
import numpy as np

class VideoProcessor:
    def __init__(self, video_path: str):
        """Initialize video processor with video path."""
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_frame = 0
        
    def read_frame(self) -> Tuple[bool, np.ndarray]:
        """Read next frame from video."""
        ret, frame = self.cap.read()
        if ret:
            self.current_frame += 1
        return ret, frame
    
    def get_current_time(self) -> float:
        """Get current time in seconds."""
        return self.current_frame / self.fps
    
    def create_video_writer(self, output_path: str) -> cv2.VideoWriter:
        """Create video writer for output."""
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        return cv2.VideoWriter(output_path, fourcc, self.fps, (self.width, self.height))
    
    def draw_results(self, frame: np.ndarray, detections: List[Dict], counts: Dict[str, Dict[str, int]], track_ids: bool = False) -> np.ndarray:
        """
        Draw detection boxes, counts, and track IDs on the frame.
        
        Args:
            frame: Input frame
            detections: List of detection dictionaries
            counts: Dictionary of counts per class
            track_ids: Whether to show track IDs
            
        Returns:
            Frame with visualizations
        """
        # Draw detection boxes and labels
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class']
            conf = det['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Prepare label
            label = f"{class_name}: {conf:.2f}"
            if track_ids and 'track_id' in det:
                label += f" ID: {det['track_id']}"
                
            # Draw label background
            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
            cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
            
            # Draw label text
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        
        # Draw counts with larger text and dark blue color
        y_offset = 40  # Increased starting position
        for class_name, count_info in counts.items():
            present_count = count_info.get('present', 0)
            total_count = count_info.get('total', 0)
            
            # Draw present count in black with larger text
            present_text = f"{class_name}: {present_count}"
            cv2.putText(frame, present_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
            
            # Draw total count in dark blue with larger text
            total_text = f"Total {class_name}: {total_count}"
            cv2.putText(frame, total_text, (20, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (139, 0, 139), 2)  # Dark blue color
            
            y_offset += 80  # Increased spacing between classes
            
        return frame
    
    def release(self):
        """Release video capture."""
        self.cap.release()
    
    def _convert_to_serializable(self, obj):
        """Convert numpy types to Python native types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                            np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    def save_frame_results(self, results: Dict, output_dir: str):
        """Save frame results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"results_{timestamp}.json")
        
        # Convert numpy types to Python native types
        serializable_results = self._convert_to_serializable(results)
        
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f, indent=4) 