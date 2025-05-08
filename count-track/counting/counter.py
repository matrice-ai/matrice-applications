from typing import List, Dict
from collections import defaultdict

class ObjectCounter:
    def __init__(self, target_classes: List[str] = None):
        """
        Initialize the object counter.
        
        Args:
            target_classes: List of classes to count (default: ['person', 'car'])
        """
        self.target_classes = target_classes or ['person', 'car']
        self.present_counts = defaultdict(int)
        
    def update(self, detections: List[Dict]) -> Dict[str, int]:
        """
        Update counts based on new detections.
        
        Args:
            detections: List of detection dictionaries
            
        Returns:
            Dictionary of present counts per class
        """
        # Reset present counts
        self.present_counts.clear()
        
        # Count objects in current frame
        for det in detections:
            class_name = det['class']
            if class_name in self.target_classes:
                self.present_counts[class_name] += 1
        
        return dict(self.present_counts)
    
    def reset(self):
        """Reset the counter."""
        self.present_counts.clear() 