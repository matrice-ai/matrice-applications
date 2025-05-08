import argparse
import os
from typing import Dict, List
import json
from datetime import datetime
import numpy as np

from detection.detector import ObjectDetector
from counting.counter import ObjectCounter
from tracking.tracker import ObjectTracker
from utils.video_utils import VideoProcessor

def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32,
                        np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

def process_video(video_path: str, output_dir: str, mode: str = 'both'):
    """
    Process video with detection, counting, and tracking.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        mode: Processing mode ('counting', 'tracking', or 'both')
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(output_dir, timestamp)
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize components
    detector = ObjectDetector()
    counter = ObjectCounter()
    tracker = ObjectTracker()
    video_processor = VideoProcessor(video_path)
    
    # Create video writer
    output_video = os.path.join(output_dir, f"{mode}_output.mp4")
    video_writer = video_processor.create_video_writer(output_video)
    
    # Process frames
    frame_results = []
    while True:
        ret, frame = video_processor.read_frame()
        if not ret:
            break
            
        # Get current time
        current_time = video_processor.get_current_time()
        
        # Detect objects
        detections = detector.detect(frame)
        
        # Update counts and tracks
        present_counts = counter.update(detections)
        if mode in ['tracking', 'both']:
            detections = tracker.update(detections, frame)
            total_counts = tracker.get_track_counts()
        else:
            total_counts = {class_name: {'total': count} for class_name, count in present_counts.items()}
        
        # Combine counts
        combined_counts = {}
        for class_name in set(present_counts.keys()) | set(total_counts.keys()):
            combined_counts[class_name] = {
                'present': present_counts.get(class_name, 0),
                'total': total_counts.get(class_name, 0)
            }
        
        # Draw results
        frame = video_processor.draw_results(
            frame, 
            detections, 
            combined_counts,
            track_ids=(mode in ['tracking', 'both'])
        )
        
        # Write frame
        video_writer.write(frame)
        
        # Store results
        frame_results.append({
            'timestamp': current_time,
            'detections': convert_to_serializable(detections),
            'counts': combined_counts
        })
    
    # Save results
    results_file = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(results_file, 'w') as f:
        json.dump(frame_results, f, indent=2)
    
    # Cleanup
    video_writer.release()
    video_processor.release()

def main():
    parser = argparse.ArgumentParser(description='Object Detection, Counting, and Tracking')
    parser.add_argument('--video', required=True, help='Path to input video')
    parser.add_argument('--output', default='output', help='Output directory')
    parser.add_argument('--mode', choices=['counting', 'tracking', 'both'], default='both',
                      help='Processing mode')
    
    args = parser.parse_args()
    process_video(args.video, args.output, args.mode)

if __name__ == '__main__':
    main() 