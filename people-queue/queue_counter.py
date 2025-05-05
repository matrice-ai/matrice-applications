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
        
        # Cash counter areas - will be set by the user
        self.cash_counters = []
        
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
        
        # Cash counter tracking
        self.counter_staff_times = {}  # Format: {counter_idx: {'staff_present': bool, 'last_change': time, 'total_empty': seconds, 'total_staffed': seconds}}
        self.people_serviced = defaultdict(int)  # Format: {counter_idx: count}
        self.being_serviced = defaultdict(set)  # Format: {counter_idx: set(track_ids)}
        
        # Store when people entered and exited each cash counter for service tracking
        self.service_entry_times = {}  # Format: {counter_idx: {track_id: entry_time}}
        
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
        
        # Fixed set of distinctive colors for cash counters (different from queue colors)
        self.counter_color_palette = [
            (0, 165, 255),   # Orange (BGR)
            (255, 191, 0),   # Deep Sky Blue (BGR)
            (238, 130, 238), # Violet (BGR)
            (255, 105, 180), # Hot Pink (BGR)
            (173, 255, 47),  # Green Yellow (BGR)
            (64, 224, 208),  # Turquoise (BGR)
            (255, 165, 0),   # Orange (BGR)
            (218, 112, 214), # Orchid (BGR)
            (152, 251, 152), # Pale Green (BGR)
            (147, 112, 219), # Medium Purple (BGR)
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
    
    def add_cash_counter(self, points, name=None):
        """
        Add a cash counter area to track
        
        Args:
            points (list): List of (x, y) coordinates defining the cash counter area
            name (str): Optional name for this cash counter
        
        Returns:
            int: Index of the added cash counter
        """
        counter_idx = len(self.cash_counters)
        color = self.counter_color_palette[counter_idx % len(self.counter_color_palette)]
        
        if name is None:
            name = f"Counter {counter_idx+1}"
        
        self.cash_counters.append({
            'polygon': np.array(points, dtype=np.int32),
            'color': color,
            'name': name
        })
        
        # Initialize counter statistics
        self.counter_staff_times[counter_idx] = {
            'staff_present': False,
            'last_change': time.time(),
            'total_empty': 0,
            'total_staffed': 0
        }
        
        self.service_entry_times[counter_idx] = {}
        
        return counter_idx
    
    def is_point_in_polygon(self, point, polygon):
        """Check if a point is inside a polygon using OpenCV's pointPolygonTest"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def update_tracks(self, detections, frame_time):
        """
        Update tracking information with improved ID consistency
        
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
            # Associate detections with existing tracks using multiple metrics
            track_ids = list(self.tracks.keys())
            track_boxes = np.array(list(self.tracks.values()))
            
            # Compute centers for all tracks and detections
            track_centers = []
            for track_id in track_ids:
                bbox = self.tracks[track_id]
                center_x = (bbox[0] + bbox[2]) // 2
                center_y = (bbox[1] + bbox[3]) // 2
                track_centers.append((center_x, center_y))
            
            detection_centers = []
            for det in detections:
                center_x = (det[0] + det[2]) // 2
                center_y = (det[1] + det[3]) // 2
                detection_centers.append((center_x, center_y))
            
            # Compute IoU between all tracks and detections
            matched_tracks = set()
            matched_detections = set()
            
            # Dictionary to store distance metrics for all possible matches
            match_metrics = []
            
            # Calculate metrics for all track-detection pairs
            for t_idx, track_id in enumerate(track_ids):
                track_box = self.tracks[track_id]
                track_center = track_centers[t_idx]
                
                # Get track's recent trajectory (if available)
                recent_trajectory = None
                if track_id in self.track_history and len(self.track_history[track_id]) >= 2:
                    recent_positions = self.track_history[track_id][-5:]  # Last 5 positions
                    if len(recent_positions) >= 2:
                        # Calculate average movement vector
                        dx_sum = 0
                        dy_sum = 0
                        for i in range(1, len(recent_positions)):
                            dx_sum += recent_positions[i][0] - recent_positions[i-1][0]
                            dy_sum += recent_positions[i][1] - recent_positions[i-1][1]
                        
                        avg_dx = dx_sum / (len(recent_positions) - 1)
                        avg_dy = dy_sum / (len(recent_positions) - 1)
                        recent_trajectory = (avg_dx, avg_dy)
                
                for d_idx, detection in enumerate(detections):
                    det_center = detection_centers[d_idx]
                    
                    # Calculate IoU score
                    iou = self.bbox_iou(track_box, detection)
                    
                    # Calculate center distance
                    center_distance = np.sqrt((track_center[0] - det_center[0])**2 + 
                                              (track_center[1] - det_center[1])**2)
                    
                    # Calculate size similarity (area ratio)
                    track_area = (track_box[2] - track_box[0]) * (track_box[3] - track_box[1])
                    det_area = (detection[2] - detection[0]) * (detection[3] - detection[1])
                    area_ratio = min(track_area, det_area) / max(track_area, det_area)
                    
                    # Calculate trajectory consistency if available
                    trajectory_score = 1.0  # Default score if no trajectory
                    if recent_trajectory is not None:
                        # Calculate expected position based on trajectory
                        expected_x = track_center[0] + recent_trajectory[0]
                        expected_y = track_center[1] + recent_trajectory[1]
                        
                        # Calculate how well the detection matches the expected position
                        expected_distance = np.sqrt((expected_x - det_center[0])**2 + 
                                                   (expected_y - det_center[1])**2)
                        
                        # Convert to a score (closer is better)
                        max_expected_distance = 50  # Pixels
                        trajectory_score = max(0, 1 - (expected_distance / max_expected_distance))
                    
                    # Calculate combined metric score
                    # Weight different metrics based on importance
                    iou_weight = 0.45
                    distance_weight = 0.35
                    size_weight = 0.1
                    trajectory_weight = 0.1
                    
                    # For distance, convert to a score (closer is better)
                    max_distance = 100  # Pixels
                    distance_score = max(0, 1 - (center_distance / max_distance))
                    
                    # Calculate final score
                    combined_score = (iou_weight * iou + 
                                     distance_weight * distance_score + 
                                     size_weight * area_ratio +
                                     trajectory_weight * trajectory_score)
                    
                    # Store this potential match
                    match_metrics.append((combined_score, t_idx, d_idx, track_id))
            
            # Sort matches by score (highest first)
            match_metrics.sort(reverse=True)
            
            # Assign tracks to detections by greedy algorithm
            for score, _, d_idx, track_id in match_metrics:
                # Skip if either track or detection is already matched
                if track_id in matched_tracks or d_idx in matched_detections:
                    continue
                
                # Skip if score is too low
                if score < 0.2:  # Threshold for matching
                    continue
                
                # Update the track with this detection
                detection = detections[d_idx]
                self.tracks[track_id] = detection
                self.disappeared[track_id] = 0
                matched_tracks.add(track_id)
                matched_detections.add(d_idx)
                    
                    # Add to track history
                    centerX = (detection[0] + detection[2]) // 2
                    centerY = (detection[1] + detection[3]) // 2
                self.track_history[track_id].append((centerX, centerY))
                    # Limit history length
                if len(self.track_history[track_id]) > self.max_track_length:
                    self.track_history[track_id] = self.track_history[track_id][-self.max_track_length:]
            
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
        
        # Count people in cash counter areas
        counter_counts = {}
        in_counter_tracks = defaultdict(list)  # Format: {counter_idx: [track_ids]}
        
        # Track people being serviced
        previously_serviced = {idx: set(self.being_serviced[idx]) for idx in range(len(self.cash_counters))}
        for idx in range(len(self.cash_counters)):
            self.being_serviced[idx] = set()
        
        for track_id, bbox in tracks.items():
            # Get person center point
            centerX = (bbox[0] + bbox[2]) // 2
            centerY = (bbox[1] + bbox[3]) // 2
            center_point = (centerX, centerY)
            
            # Check if person is in any queue
            for queue_idx, queue_area in enumerate(self.queue_areas):
                if self.is_point_in_polygon(center_point, queue_area['polygon']):
                    in_queue_tracks[queue_idx].append(track_id)
                
                    # Track how long the person has been in this queue
                    if track_id not in self.person_queue_times:
                        self.person_queue_times[track_id] = {}
                    
                    # Initialize the first time we see this person in this queue
                    if queue_idx not in self.person_queue_times[track_id]:
                        self.person_queue_times[track_id][queue_idx] = frame_time
                    
            # Check if person is in any cash counter
            for counter_idx, counter_area in enumerate(self.cash_counters):
                if self.is_point_in_polygon(center_point, counter_area['polygon']):
                    in_counter_tracks[counter_idx].append(track_id)
                    self.being_serviced[counter_idx].add(track_id)
                    
                    # Initialize entry time for this person at this counter
                    if track_id not in self.service_entry_times[counter_idx]:
                        self.service_entry_times[counter_idx][track_id] = frame_time
        
        # Process completed services (people who were at a counter but aren't anymore)
        for counter_idx in range(len(self.cash_counters)):
            # Find people who were being serviced but are no longer at the counter
            serviced_people = previously_serviced[counter_idx] - self.being_serviced[counter_idx]
            
            # Increment the count of people serviced for each counter
            self.people_serviced[counter_idx] += len(serviced_people)
            
            # Clean up entry times for people who have left
            for track_id in serviced_people:
                if track_id in self.service_entry_times[counter_idx]:
                    del self.service_entry_times[counter_idx][track_id]
        
        # Update counter staff status
        for counter_idx, counter_area in enumerate(self.cash_counters):
            # Get current time for duration tracking
            current_time = time.time()
            
            # Determine if staff is present (first person in the counter area is considered staff)
            staff_present = len(in_counter_tracks[counter_idx]) > 0
            
            # If staff status changed, update timings
            if staff_present != self.counter_staff_times[counter_idx]['staff_present']:
                # Calculate duration since last change
                duration = current_time - self.counter_staff_times[counter_idx]['last_change']
                
                # Update the appropriate counter
                if self.counter_staff_times[counter_idx]['staff_present']:
                    self.counter_staff_times[counter_idx]['total_staffed'] += duration
                else:
                    self.counter_staff_times[counter_idx]['total_empty'] += duration
                
                # Record the change
                self.counter_staff_times[counter_idx]['staff_present'] = staff_present
                self.counter_staff_times[counter_idx]['last_change'] = current_time
        
        # Apply buffer time to count people in queue
        for queue_idx, track_ids in in_queue_tracks.items():
            # Count people who've been in queue longer than buffer time
            buffered_count = 0
            
            for track_id in track_ids:
                queue_entry_time = self.person_queue_times[track_id].get(queue_idx, frame_time)
                time_in_queue = frame_time - queue_entry_time
                
                # Only count if they've been in queue longer than buffer time
                if time_in_queue >= self.buffer_time_seconds:
                    buffered_count += 1
                    # Add to unique counts set
                    self.unique_queue_counts[queue_idx].add(track_id)
            
            queue_counts[queue_idx] = buffered_count
        
        # Store the count for each counter
        for counter_idx in range(len(self.cash_counters)):
            counter_counts[counter_idx] = len(in_counter_tracks[counter_idx])
        
        # Annotate frame
        annotated_frame = frame.copy()
        
        # Draw queue areas
        for queue_idx, queue_area in enumerate(self.queue_areas):
            cv2.polylines(annotated_frame, [queue_area['polygon']], True, queue_area['color'], 2)
            
            # Add queue name and count
            queue_count = queue_counts.get(queue_idx, 0)
            text = f"{queue_area['name']}: {queue_count}"
            
            # Get a good position for the text (above the top-left corner of polygon)
            min_x = min(point[0] for point in queue_area['polygon'])
            min_y = min(point[1] for point in queue_area['polygon'])
                text_pos = (min_x, min_y - 10)
            
            cv2.putText(annotated_frame, text, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, queue_area['color'], 2)
        
        # Draw cash counter areas
        for counter_idx, counter_area in enumerate(self.cash_counters):
            cv2.polylines(annotated_frame, [counter_area['polygon']], True, counter_area['color'], 2)
            
            # Add counter name and staff status
            staff_present = self.counter_staff_times[counter_idx]['staff_present']
            staff_status = "Staff Present" if staff_present else "Empty"
            
            # Calculate timing information
            staff_time = self.counter_staff_times[counter_idx]['total_staffed']
            empty_time = self.counter_staff_times[counter_idx]['total_empty']
            
            # Get people serviced count
            serviced_count = self.people_serviced[counter_idx]
            
            # Position for the counter name (above the bottom-left corner of polygon)
            min_x = min(point[0] for point in counter_area['polygon'])
            max_y = max(point[1] for point in counter_area['polygon'])
            
            name_pos = (min_x, max_y + 20)
            status_pos = (min_x, max_y + 40)
            serviced_pos = (min_x, max_y + 60)
            time_pos = (min_x, max_y + 80)
            
            # Draw counter information
            cv2.putText(annotated_frame, f"{counter_area['name']}", name_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_area['color'], 2)
            cv2.putText(annotated_frame, f"Status: {staff_status}", status_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_area['color'], 2)
            cv2.putText(annotated_frame, f"Serviced: {serviced_count}", serviced_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.7, counter_area['color'], 2)
            
            # Format times in minutes:seconds
            staff_mins, staff_secs = divmod(int(staff_time), 60)
            empty_mins, empty_secs = divmod(int(empty_time), 60)
            
            time_text = f"Staffed: {staff_mins}m{staff_secs}s | Empty: {empty_mins}m{empty_secs}s"
            cv2.putText(annotated_frame, time_text, time_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.6, counter_area['color'], 2)
        
        # Draw tracks
        for track_id, bbox in tracks.items():
            x1, y1, x2, y2 = bbox[:4]
            
            # Determine color based on whether they're in a queue or at a counter
            track_color = (0, 0, 0)  # Default black
            
            # Check if person is in a queue
            for queue_idx, queue_area in enumerate(self.queue_areas):
                if track_id in in_queue_tracks[queue_idx]:
                    track_color = queue_area['color']
                    break
            
            # Check if person is at a counter (counter takes precedence for coloring)
            for counter_idx, counter_area in enumerate(self.cash_counters):
                if track_id in in_counter_tracks[counter_idx]:
                    track_color = counter_area['color']
                    break
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), track_color, 2)
            
            # Add ID
            cv2.putText(annotated_frame, f"ID: {track_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, track_color, 2)
            
            # Draw track history (trails)
            if track_id in self.track_history and len(self.track_history[track_id]) > 1:
                # Draw lines connecting previous positions
                for i in range(1, len(self.track_history[track_id])):
                    pt1 = self.track_history[track_id][i - 1]
                    pt2 = self.track_history[track_id][i]
                    cv2.line(annotated_frame, pt1, pt2, track_color, 2)
        
        return annotated_frame, queue_counts, counter_counts
    
    def process_video(self, video_path, output_path=None, display_every=0):
        """
        Process a video file and count people in queue areas
        
        Args:
            video_path (str): Path to input video
            output_path (str): Path to save output video (optional)
            display_every (int): How often to display/update during processing (0 = never)
        
        Returns:
            dict: Queue counts over time
        """
        # Open the video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return None
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Initialize output video writer if specified
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Initialize queue tracking
        frame_idx = 0
        queue_counts_over_time = []
        counter_counts_over_time = []
        
        # Process video frames
        while cap.isOpened():
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Use frame index as timestamp (for buffer time tracking)
            frame_time = frame_idx / fps
            
            # Process the frame
            annotated_frame, queue_counts, counter_counts = self.process_frame(frame, frame_time)
            
            # Store counts for plotting
            queue_counts_over_time.append(queue_counts)
            counter_counts_over_time.append(counter_counts)
            
            # Write to output video if specified
            if output_path:
                out.write(annotated_frame)
            
            # Display progress
            if display_every > 0 and frame_idx % display_every == 0:
                cv2.imshow('Queue Detection', annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_idx % 100 == 0:
                print(f"Processing frame {frame_idx}/{total_frames} ({frame_idx/total_frames*100:.1f}%)")
                
                # Print queue information
                for i, queue_area in enumerate(self.queue_areas):
                    count = queue_counts.get(i, 0)
                    print(f"  {queue_area['name']}: {count} people")
                
                # Print counter information
                for i, counter_area in enumerate(self.cash_counters):
                    staff_present = self.counter_staff_times[i]['staff_present']
                    status = "Staff Present" if staff_present else "Empty"
                    serviced_count = self.people_serviced[i]
                    print(f"  {counter_area['name']}: {status}, Serviced: {serviced_count}")
            
            # Increment frame counter
            frame_idx += 1
        
        # Clean up
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        # Plot queue trends
        self.plot_queue_trends(queue_counts_over_time, fps)
        
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

    def define_cash_counter_areas(self, frame):
        """
        Interactive definition of cash counter areas in a frame
        
        Args:
            frame: Image frame to display for selection
        
        Returns:
            list: Indices of defined cash counter areas
        """
        # Clone frame for drawing
        draw_frame = frame.copy()
        
        # Instructions
        print("Define Cash Counter Areas:")
        print("- Left-click to add points")
        print("- Right-click to complete the current cash counter")
        print("- Press 'q' to finish defining cash counters")
        print("- Press 'c' to cancel the current cash counter")
        
        # Parameters for drawing
        counter_indices = []
        current_points = []
        
        # Mouse callback function
        def mouse_callback(event, x, y, flags, param):
            nonlocal current_points, draw_frame
            
            if event == cv2.EVENT_LBUTTONDOWN:
                # Add point to current shape
                current_points.append((x, y))
                
                # Redraw the frame
                draw_frame = frame.copy()
                
                # Draw existing cash counters
                for area in self.cash_counters:
                    cv2.polylines(draw_frame, [area['polygon']], True, area['color'], 2)
                    min_x = min(point[0] for point in area['polygon'])
                    min_y = min(point[1] for point in area['polygon'])
                    cv2.putText(draw_frame, area['name'], (min_x, min_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, area['color'], 2)
                
                # Draw current shape
                if len(current_points) > 1:
                    points_array = np.array(current_points, dtype=np.int32)
                    cv2.polylines(draw_frame, [points_array], False, (0, 255, 255), 2)
                    
                # Draw all points
                for pt in current_points:
                    cv2.circle(draw_frame, pt, 5, (0, 255, 255), -1)
            
            elif event == cv2.EVENT_RBUTTONDOWN and len(current_points) > 2:
                # Complete the current counter
                counter_idx = self.add_cash_counter(current_points)
                counter_indices.append(counter_idx)
                
                print(f"Added {self.cash_counters[counter_idx]['name']}")
                
                # Redraw the frame
                draw_frame = frame.copy()
                
                # Draw all counters, including the new one
                for area in self.cash_counters:
                    cv2.polylines(draw_frame, [area['polygon']], True, area['color'], 2)
                    min_x = min(point[0] for point in area['polygon'])
                    min_y = min(point[1] for point in area['polygon'])
                    cv2.putText(draw_frame, area['name'], (min_x, min_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, area['color'], 2)
                
                # Reset current points for the next counter
                current_points = []
        
        # Create window and set mouse callback
        cv2.namedWindow('Define Cash Counter Areas')
        cv2.setMouseCallback('Define Cash Counter Areas', mouse_callback)
        
        # Main loop for cash counter definition
        while True:
            cv2.imshow('Define Cash Counter Areas', draw_frame)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Finish defining cash counters
                break
            
            elif key == ord('c'):
                # Cancel current cash counter
                current_points = []
                
                # Redraw the frame
                draw_frame = frame.copy()
                
                # Draw existing cash counters
                for area in self.cash_counters:
                    cv2.polylines(draw_frame, [area['polygon']], True, area['color'], 2)
                    min_x = min(point[0] for point in area['polygon'])
                    min_y = min(point[1] for point in area['polygon'])
                    cv2.putText(draw_frame, area['name'], (min_x, min_y - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, area['color'], 2)
        
        # Clean up
        cv2.destroyAllWindows()
        
        return counter_indices 