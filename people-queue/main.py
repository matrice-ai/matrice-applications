#!/usr/bin/env python
# People Queue Counter - Main Entry Point

import os
import sys
import cv2
import matplotlib.pyplot as plt
from queue_counter import QueueCounter

def main():
    """Main entry point for the People Queue Counter application"""
    
    # Get the absolute path to the video file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    video_path = os.path.join(script_dir, "cr.mp4")
    
    # Check if video file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        print("Please place 'cr.mp4' in the same directory as this script.")
        return
    
    # Set output path
    output_path = os.path.join(script_dir, "queue_output.mp4")
    
    # Initialize the queue counter with 10 seconds buffer time
    print(f"Initializing QueueCounter with video: {video_path}")
    counter = QueueCounter(
        model_path="yolov8n.pt", 
        confidence=0.4,
        buffer_time_seconds=10  # Only count people in queue if they've been there for 10+ seconds
    )
    
    # Get first frame for area definitions
    first_frame = counter.get_first_frame(video_path)
    if first_frame is None:
        print("Could not read the first frame of the video.")
        return
    
    # Define queue areas
    print("\nDefining queue areas...")
    queue_indices = counter.define_multiple_queue_areas(first_frame)
    
    # Define cash counter areas
    print("\nDefining cash counter areas...")
    counter_indices = counter.define_cash_counter_areas(first_frame)
    
    if not queue_indices:
        print("No queue areas defined. Exiting.")
        return
    
    # Process the video - don't display frames during processing (display_every=0)
    print("\nProcessing video. This may take some time...")
    print("Queue and counter statistics will be printed every 100 frames")
    counter.process_video(video_path, output_path, display_every=0)
    
    print(f"Processing complete. Output saved to: {output_path}")
    print(f"Queue trend graph saved as: queue_trends.png")
    
    # Display unique counts for each queue
    print("\nFinal Queue Statistics:")
    for i, queue_area in enumerate(counter.queue_areas):
        unique_count = len(counter.unique_queue_counts[i])
        print(f"{queue_area['name']}: {unique_count} unique people")
    
    # Display final counter statistics
    if counter.cash_counters:
        print("\nFinal Counter Statistics:")
        for i, counter_area in enumerate(counter.cash_counters):
            serviced_count = counter.people_serviced[i]
            staff_time = counter.counter_staff_times[i]['total_staffed']
            empty_time = counter.counter_staff_times[i]['total_empty']
            
            # Format times in minutes:seconds
            staff_mins, staff_secs = divmod(int(staff_time), 60)
            empty_mins, empty_secs = divmod(int(empty_time), 60)
            
            print(f"{counter_area['name']}:")
            print(f"  People serviced: {serviced_count}")
            print(f"  Staffed time: {staff_mins}m {staff_secs}s")
            print(f"  Empty time: {empty_mins}m {empty_secs}s")
    
    print("\nCheck the output video to see tracking results and queue counts.")

if __name__ == "__main__":
    main() 