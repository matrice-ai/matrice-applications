#!/usr/bin/env python
# Queue Counter Implementation

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from IPython.display import clear_output
from collections import defaultdict
import time
import random

class QueueCounter:
    def __init__(self, model_path="yolov8n.pt", confidence=0.5, device=None, buffer_time_seconds=10):
        """
        Initialize the QueueCounter object
        
        Args:
            model_path (str): Path to the YOLOv8 model
            confidence (float): Detection confidence threshold
            device (str): Device to run model on ('cpu', 'cuda', or None for auto)
            buffer_time_seconds (int): Time in seconds a person must be in queue to count
        """
        self.conf = confidence
        self.device = device if device else "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        
        # Load YOLOv8 model - will download if not present
        self.model = YOLO(model_path)
        
        # Only track people (class id 0 in COCO dataset)
        self.target_class = 0  # person class
        
        # Queue areas - will be set by the user (polygon, color, name)
        self.queue_areas = []
        
        # Tracking - simple ID assignment and track management
        self.next_id = 0
        self.tracks = {}
        self.track_history = defaultdict(lambda: [])
        self.max_track_length = 20
        self.max_disappeared = 15
        self.disappeared = {}
        
        # Buffer time tracking - how long each person has been in each queue
        self.buffer_time_seconds = buffer_time_seconds
        self.person_queue_times = {}  # Format: {track_id: {queue_idx: first_frame_time}}
        
        # Unique counts per queue
        self.unique_queue_counts = defaultdict(set)  # Format: {queue_idx: set(track_ids)}
        
        # Fixed set of distinctive colors for queues
        self.color_palette = [
            (255, 0, 0),     # Red
            (0, 255, 0),     # Green
            (0, 0, 255),     # Blue
            (255, 255, 0),   # Yellow
            (255, 0, 255),   # Magenta
            (0, 255, 255),   # Cyan
            (255, 165, 0),   # Orange
            (128, 0, 128),   # Purple
            (0, 128, 0),     # Dark Green
            (128, 0, 0),     # Maroon
        ]

    def add_queue_area(self, points, name=None):
        """
        Add a queue area to track
        
        Args:
            points (list): List of (x, y) coordinates defining the queue area
            name (str): Optional name for this queue area
        
        Returns:
            int: Index of the added queue area
        """
        queue_idx = len(self.queue_areas)
        color = self.color_palette[queue_idx % len(self.color_palette)]
        
        if name is None:
            name = f"Queue {queue_idx+1}"
        
        self.queue_areas.append({
            'polygon': np.array(points, dtype=np.int32),
            'color': color,
            'name': name
        })
        
        return queue_idx
    
    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using OpenCV's pointPolygonTest"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def update_tracks(self, detections, frame_time):
        """
        Update tracking information
        
        Args:
            detections: List of [x1, y1, x2, y2, conf, cls]
            frame_time: Timestamp or frame number for time tracking
            
        Returns:
            dict: Updated tracks with IDs
        """
        # If no detections, mark all tracks as disappeared
        if len(detections) == 0:
            for track_id in list(self.tracks.keys()):
                self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                if self.disappeared[track_id] > self.max_disappeared:
                    del self.tracks[track_id]
                    del self.disappeared[track_id]
                    if track_id in self.track_history:
                        del self.track_history[track_id]
                    if track_id in self.person_queue_times:
                        del self.person_queue_times[track_id]
            return self.tracks
        
        # If no existing tracks, create new ones for all detections
        if len(self.tracks) == 0:
            for i, bbox in enumerate(detections):
                self.tracks[self.next_id] = bbox
                self.disappeared[self.next_id] = 0
                self.next_id += 1
        else:
            # Associate detections with existing tracks (simple IoU-based approach)
            track_ids = list(self.tracks.keys())
            track_boxes = np.array(list(self.tracks.values()))
            
            # Compute IoU between all tracks and detections
            matched_tracks = set()
            matched_detections = set()
            
            for i, detection in enumerate(detections):
                best_iou = 0.3  # IoU threshold
                best_id = None
                
                for track_id in track_ids:
                    if track_id in matched_tracks:
                        continue
                        
                    track_box = self.tracks[track_id]
                    iou = self.bbox_iou(detection, track_box)
                    
                    if iou > best_iou:
                        best_iou = iou
                        best_id = track_id
                
                if best_id is not None:
                    # Update existing track
                    self.tracks[best_id] = detection
                    self.disappeared[best_id] = 0
                    matched_tracks.add(best_id)
                    matched_detections.add(i)
                    
                    # Add to track history
                    centerX = (detection[0] + detection[2]) // 2
                    centerY = (detection[1] + detection[3]) // 2
                    self.track_history[best_id].append((centerX, centerY))
                    # Limit history length
                    if len(self.track_history[best_id]) > self.max_track_length:
                        self.track_history[best_id] = self.track_history[best_id][-self.max_track_length:]
            
            # Mark unmatched tracks as disappeared
            for track_id in track_ids:
                if track_id not in matched_tracks:
                    self.disappeared[track_id] = self.disappeared.get(track_id, 0) + 1
                    if self.disappeared[track_id] > self.max_disappeared:
                        del self.tracks[track_id]
                        del self.disappeared[track_id]
                        if track_id in self.track_history:
                            del self.track_history[track_id]
                        if track_id in self.person_queue_times:
                            del self.person_queue_times[track_id]
            
            # Create new tracks for unmatched detections
            for i, detection in enumerate(detections):
                if i not in matched_detections:
                    self.tracks[self.next_id] = detection
                    self.disappeared[self.next_id] = 0
                    
                    # Initialize track history
                    centerX = (detection[0] + detection[2]) // 2
                    centerY = (detection[1] + detection[3]) // 2
                    self.track_history[self.next_id].append((centerX, centerY))
                    
                    self.next_id += 1
                    
        return self.tracks
    
    def bbox_iou(self, box1, box2):
        """Calculate IoU between two bounding boxes"""
        # Determine coordinates of intersection
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        # Calculate area of intersection
        width_inter = max(0, x2_inter - x1_inter + 1)
        height_inter = max(0, y2_inter - y1_inter + 1)
        area_inter = width_inter * height_inter
        
        # Calculate area of both boxes
        width_box1 = box1[2] - box1[0] + 1
        height_box1 = box1[3] - box1[1] + 1
        width_box2 = box2[2] - box2[0] + 1
        height_box2 = box2[3] - box2[1] + 1
        area_box1 = width_box1 * height_box1
        area_box2 = width_box2 * height_box2
        
        # Calculate Union area
        area_union = area_box1 + area_box2 - area_inter
        
        # Calculate IoU
        iou = area_inter / area_union
        
        return iou
    
    def process_frame(self, frame, frame_time):
        """
        Process a single frame
        
        Args:
            frame (numpy.ndarray): Input frame
            frame_time: Timestamp or frame number for buffer time tracking
        
        Returns:
            tuple: (Annotated frame, dict of counts per queue)
        """
        # Detect objects using YOLOv8
        results = self.model(frame, conf=self.conf, device=self.device)[0]
        
        # Extract person detections
        person_detections = []
        for detection in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = detection
            if int(cls) == self.target_class:
                person_detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
        
        # Update tracks
        tracks = self.update_tracks(person_detections, frame_time)
        
        # Count people in queue areas
        queue_counts = {}
        in_queue_tracks = defaultdict(list)  # Format: {queue_idx: [track_ids]}
        
        for track_id, bbox in tracks.items():
            # Use center point to determine if in queue
            center_x = (bbox[0] + bbox[2]) // 2
            center_y = (bbox[1] + bbox[3]) // 2
            center_point = (center_x, center_y)
            
            # Check each queue area
            for queue_idx, queue_area in enumerate(self.queue_areas):
                polygon = queue_area['polygon']
                
                if self.is_point_in_polygon(center_point, polygon):
                    # Initialize queue time tracking for this person if not already tracked
                    if track_id not in self.person_queue_times:
                        self.person_queue_times[track_id] = {}
                    
                    # Store first time seen in this queue if not already stored
                    if queue_idx not in self.person_queue_times[track_id]:
                        self.person_queue_times[track_id][queue_idx] = frame_time
                    
                    # Calculate time spent in queue
                    time_in_queue = frame_time - self.person_queue_times[track_id][queue_idx]
                    
                    # If person has been in queue longer than buffer time, count them
                    if time_in_queue >= self.buffer_time_seconds:
                        in_queue_tracks[queue_idx].append(track_id)
                        
                        # Add to unique counts for this queue
                        self.unique_queue_counts[queue_idx].add(track_id)
                else:
                    # If person is not in this queue anymore, reset their time
                    if track_id in self.person_queue_times and queue_idx in self.person_queue_times[track_id]:
                        del self.person_queue_times[track_id][queue_idx]
        
        # Calculate current counts for each queue
        for queue_idx, tracks_in_queue in in_queue_tracks.items():
            queue_counts[queue_idx] = len(tracks_in_queue)
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw queue areas
        for queue_idx, queue_area in enumerate(self.queue_areas):
            polygon = queue_area['polygon']
            color = queue_area['color']
            name = queue_area['name']
            
            # Draw polygon
            cv2.polylines(annotated_frame, [polygon], True, color, 2)
            
            # Add queue name and count
            count = queue_counts.get(queue_idx, 0)
            unique_count = len(self.unique_queue_counts[queue_idx])
            
            # Calculate position for queue text - find the top-left point of the polygon
            if len(polygon) > 0:
                min_x = min(pt[0] for pt in polygon)
                min_y = min(pt[1] for pt in polygon)
                text_pos = (min_x, min_y - 10)
            else:
                text_pos = (50, 50 + queue_idx * 30)
                
            cv2.putText(
                annotated_frame,
                f"{name}: {count} (Total: {unique_count})",
                text_pos,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2
            )
        
        # Draw bounding boxes and track history
        for track_id, bbox in tracks.items():
            x1, y1, x2, y2 = bbox[:4]
            
            # Determine color based on which queue(s) the person is in
            in_any_queue = False
            box_color = (128, 128, 128)  # Default gray
            
            for queue_idx, tracks_in_queue in in_queue_tracks.items():
                if track_id in tracks_in_queue:
                    box_color = self.queue_areas[queue_idx]['color']
                    in_any_queue = True
                    break
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 2)
            
            # Draw ID
            cv2.putText(
                annotated_frame, 
                f"ID: {track_id}", 
                (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                box_color, 
                2
            )
            
            # Draw trace path if available
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                points = np.array(self.track_history[track_id], dtype=np.int32)
                cv2.polylines(annotated_frame, [points], False, box_color, 2)
        
        # Add total count info
        total_people_in_queues = sum(queue_counts.values())
        cv2.putText(
            annotated_frame, 
            f"Total people in queues: {total_people_in_queues}", 
            (20, 40), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
        
        return annotated_frame, queue_counts
    
    def process_video(self, video_path, output_path=None, display_every=0):
        """
        Process a video file
        
        Args:
            video_path (str): Path to the input video
            output_path (str): Path to save the output video (None to not save)
            display_every (int): Display every nth frame (0 for no display)
        
        Returns:
            dict: Queue counts over time {queue_idx: [counts]}
        """
        if not self.queue_areas:
            print("No queue areas defined. Please define at least one queue area.")
            # Define queue areas using grid method before processing
            self.define_multiple_queue_areas(self.get_first_frame(video_path))
        
        cap = cv2.VideoCapture(video_path)
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate buffer time in frames
        self.buffer_frames = int(self.buffer_time_seconds * fps)
        print(f"Buffer time set to {self.buffer_time_seconds} seconds ({self.buffer_frames} frames)")
        
        # Set up output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Dictionary to track counts per queue over time
        queue_counts_over_time = defaultdict(list)
        frame_count = 0
        
        # Process each frame
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            # Process the frame (use frame count as time measure)
            annotated_frame, queue_counts = self.process_frame(frame, frame_count)
            
            # Save counts for each queue
            for queue_idx in range(len(self.queue_areas)):
                queue_counts_over_time[queue_idx].append(queue_counts.get(queue_idx, 0))
            
            # Save frame to output video if needed
            if out:
                out.write(annotated_frame)
            
            # Display progress
            if frame_count % 100 == 0:
                counts_str = ", ".join([f"{q['name']}: {c}" for q, c in zip(self.queue_areas, 
                                                                         [queue_counts.get(i, 0) for i in range(len(self.queue_areas))])])
                print(f"Processing frame {frame_count}/{total_frames} - Counts: {counts_str}")
            
            # Optionally display frames during processing (if display_every > 0)
            if display_every > 0 and frame_count % display_every == 0:
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
                plt.axis('off')
                plt.show()
            
            frame_count += 1
        
        # Release resources
        cap.release()
        if out:
            out.release()
            print(f"Output video saved to: {output_path}")
        
        # Plot queue count over time
        self.plot_queue_trends(queue_counts_over_time, fps)
        
        # Return all counts
        return queue_counts_over_time
    
    def plot_queue_trends(self, queue_counts_over_time, fps):
        """Plot trends of queue counts over time"""
        plt.figure(figsize=(12, 6))
        
        for queue_idx, counts in queue_counts_over_time.items():
            if queue_idx < len(self.queue_areas):
                queue_name = self.queue_areas[queue_idx]['name']
                color = [c/255 for c in self.queue_areas[queue_idx]['color']]  # Convert to 0-1 range for matplotlib
                
                time_points = [i/fps for i in range(len(counts))]
                plt.plot(time_points, counts, label=queue_name, color=color)
        
        plt.title("Queue Counts Over Time")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Number of People in Queue")
        plt.grid(True)
        plt.legend()
        plt.savefig("queue_trends.png")
        plt.close()

    def get_first_frame(self, video_path):
        """Get the first frame of a video for queue area definition"""
        cap = cv2.VideoCapture(video_path)
        ret, first_frame = cap.read()
        cap.release()
        
        if not ret:
            raise ValueError("Failed to read video")
            
        return first_frame

    def define_multiple_queue_areas(self, frame):
        """
        Define multiple queue areas using a grid overlay
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            int: Number of defined queue areas
        """
        h, w = frame.shape[:2]
        
        # Create a grid overlay
        grid_frame = frame.copy()
        grid_step = 50  # Grid size
        
        # Draw vertical lines
        for x in range(0, w, grid_step):
            cv2.line(grid_frame, (x, 0), (x, h), (255, 255, 255), 1)
            # Add x-coordinate labels
            if x % 100 == 0:  # Label every 100 pixels
                cv2.putText(grid_frame, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # Draw horizontal lines
        for y in range(0, h, grid_step):
            cv2.line(grid_frame, (0, y), (w, y), (255, 255, 255), 1)
            # Add y-coordinate labels
            if y % 100 == 0:  # Label every 100 pixels
                cv2.putText(grid_frame, str(y), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # Display the grid
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB))
        plt.title("Grid Overlay - Use coordinates to define queue areas")
        plt.axis('off')
        plt.show()
        
        print("Define multiple queue areas using the grid reference.")
        
        # Ask how many queue areas to define
        num_queues = 0
        while num_queues <= 0:
            try:
                num_queues = int(input("How many queue areas do you want to define? "))
                if num_queues <= 0:
                    print("Please enter a positive number.")
            except ValueError:
                print("Please enter a valid number.")
        
        for i in range(num_queues):
            print(f"\nDefining Queue Area {i+1}")
            print("You need to specify 3-4 points (x,y) for the queue area polygon.")
            print("Format: x1,y1 x2,y2 x3,y3 [x4,y4]")
            print("Example: 100,300 300,300 300,100 100,100")
            
            # Ask for queue name
            queue_name = input(f"Enter name for Queue {i+1} (or press Enter for default): ")
            if not queue_name:
                queue_name = f"Queue {i+1}"
            
            valid_input = False
            while not valid_input:
                try:
                    points_input = input(f"Enter the points for {queue_name}: ")
                    points_pairs = points_input.split()
                    
                    if len(points_pairs) < 3:
                        print("Need at least 3 points. Please try again.")
                        continue
                    
                    queue_points = []
                    for pair in points_pairs:
                        x, y = map(int, pair.split(','))
                        queue_points.append((x, y))
                    
                    # Visualize the input area
                    display_frame = frame.copy()
                    
                    # Draw existing queue areas
                    for q_idx, q_area in enumerate(self.queue_areas):
                        cv2.polylines(display_frame, [q_area['polygon']], True, q_area['color'], 2)
                        # Add queue name
                        min_x = min(pt[0] for pt in q_area['polygon'])
                        min_y = min(pt[1] for pt in q_area['polygon'])
                        cv2.putText(display_frame, q_area['name'], (min_x, min_y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_area['color'], 2)
                    
                    # Draw new queue area
                    points_array = np.array(queue_points, np.int32)
                    preview_color = self.color_palette[len(self.queue_areas) % len(self.color_palette)]
                    cv2.polylines(display_frame, [points_array], True, preview_color, 2)
                    
                    plt.figure(figsize=(12, 8))
                    plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                    plt.title(f"Queue Area: {queue_name}")
                    plt.axis('off')
                    plt.show()
                    
                    confirmation = input("Is this area correct? (y/n): ")
                    if confirmation.lower() == 'y':
                        valid_input = True
                        self.add_queue_area(queue_points, queue_name)
                except Exception as e:
                    print(f"Invalid input format: {e}. Please try again.")
        
        # Display all queue areas
        if self.queue_areas:
            display_frame = frame.copy()
            for q_idx, q_area in enumerate(self.queue_areas):
                cv2.polylines(display_frame, [q_area['polygon']], True, q_area['color'], 2)
                # Add queue name
                min_x = min(pt[0] for pt in q_area['polygon'])
                min_y = min(pt[1] for pt in q_area['polygon'])
                cv2.putText(display_frame, q_area['name'], (min_x, min_y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, q_area['color'], 2)
            
            plt.figure(figsize=(12, 8))
            plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
            plt.title("All Defined Queue Areas")
            plt.axis('off')
            plt.show()
        
        print(f"Defined {len(self.queue_areas)} queue areas.")
        return len(self.queue_areas)
        
    def define_queue_area_with_grid(self, frame):
        """
        Define a single queue area using a grid overlay (for backward compatibility)
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            list: Points defining queue area
        """
        h, w = frame.shape[:2]
        
        # Create a grid overlay
        grid_frame = frame.copy()
        grid_step = 50  # Grid size
        
        # Draw vertical lines
        for x in range(0, w, grid_step):
            cv2.line(grid_frame, (x, 0), (x, h), (255, 255, 255), 1)
            # Add x-coordinate labels
            if x % 100 == 0:  # Label every 100 pixels
                cv2.putText(grid_frame, str(x), (x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # Draw horizontal lines
        for y in range(0, h, grid_step):
            cv2.line(grid_frame, (0, y), (w, y), (255, 255, 255), 1)
            # Add y-coordinate labels
            if y % 100 == 0:  # Label every 100 pixels
                cv2.putText(grid_frame, str(y), (5, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
        # Display the grid
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(grid_frame, cv2.COLOR_BGR2RGB))
        plt.title("Grid Overlay - Use coordinates to define queue area")
        plt.axis('off')
        plt.show()
        
        print("Using the grid as reference, define your queue area.")
        print("You need to specify 4 points (x,y) for the queue area polygon.")
        print("Format: x1,y1 x2,y2 x3,y3 x4,y4")
        print("Example: 100,300 300,300 300,100 100,100")
        
        valid_input = False
        queue_points = None
        
        while not valid_input:
            try:
                points_input = input("Enter the corner points of your queue area: ")
                points_pairs = points_input.split()
                
                if len(points_pairs) < 3:
                    print("Need at least 3 points. Please try again.")
                    continue
                
                queue_points = []
                for pair in points_pairs:
                    x, y = map(int, pair.split(','))
                    queue_points.append((x, y))
                
                # Visualize the input area
                display_frame = frame.copy()
                points_array = np.array(queue_points, np.int32)
                cv2.polylines(display_frame, [points_array], True, (0, 255, 0), 2)
                
                plt.figure(figsize=(12, 8))
                plt.imshow(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
                plt.title("Defined Queue Area")
                plt.axis('off')
                plt.show()
                
                confirmation = input("Is this area correct? (y/n): ")
                if confirmation.lower() == 'y':
                    valid_input = True
                    self.add_queue_area(queue_points, "Queue 1")
            except Exception as e:
                print(f"Invalid input format: {e}. Please try again.")
        
        return queue_points 