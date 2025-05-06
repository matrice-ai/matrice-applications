"""
utils.py - Utilities for Face Recognition System

This module provides utility functions for:
- Visualization
- Analytics
- Metrics
- Helper functions
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import os
import json
from collections import Counter

# Initialize logger
logger = logging.getLogger('face_recognition.utils')

# Colors for visualization
COLORS = {
    'blue': (255, 0, 0),      # BGR format
    'green': (0, 255, 0),
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'orange': (0, 140, 255),
    'purple': (255, 0, 255)
}

# Visualization functions
def draw_face_box(frame, face_loc, text="", color=COLORS['green'], confidence=None):
    """
    Draw a box around a face with optional text label
    
    Args:
        frame: Frame to draw on
        face_loc: Face location (top, right, bottom, left)
        text: Optional text label
        color: Box color (BGR)
        confidence: Optional confidence score
        
    Returns:
        Frame with annotations
    """
    top, right, bottom, left = face_loc
    
    # Draw box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Add text background
    if text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (left, top - text_size[1] - 10), (left + text_size[0] + 10, top), color, -1)
        cv2.putText(frame, text, (left + 5, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, COLORS['white'], 1)
    
    # Add confidence if provided
    if confidence is not None:
        conf_text = f"{confidence:.2f}"
        cv2.putText(frame, conf_text, (left + 5, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame


def draw_landmarks(frame, landmarks):
    """
    Draw facial landmarks on the frame for visualization
    
    Args:
        frame: Frame to draw on
        landmarks: Dictionary of facial landmarks
        
    Returns:
        Frame with landmark annotations
    """
    for feature, points in landmarks.items():
        color = COLORS['blue']
        if feature == "left_eye" or feature == "right_eye":
            color = COLORS['green']
        elif feature == "nose_bridge" or feature == "nose_tip":
            color = COLORS['red']
        elif feature == "top_lip" or feature == "bottom_lip":
            color = COLORS['yellow']
            
        # Draw points
        for point in points:
            cv2.circle(frame, point, 2, color, -1)
            
        # Connect points with lines for better visualization
        if len(points) > 1:
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts], False, color, 1)
    
    return frame


def add_status_panel(frame, title, fps, stats):
    """
    Add a modern status panel with information to the frame
    
    Args:
        frame: Input image frame
        title: Title for the panel
        fps: FPS value to display
        stats: Dictionary of statistics to display
        
    Returns:
        Frame with status panel
    """
    h, w = frame.shape[:2]
    
    # Copy the frame to avoid modifying original
    display_frame = frame.copy()
    
    # Define panel parameters
    panel_height = 30 + 20 * (len(stats) + 1)  # Title + stats
    panel_width = 180
    panel_x = 10
    panel_y = 10
    
    # Panel colors
    BACKGROUND = (20, 20, 20)  # Dark gray
    TITLE_COLOR = (0, 140, 255)  # Orange
    TEXT_COLOR = (255, 255, 255)  # White
    VALUE_COLOR = (200, 200, 200)  # Light gray
    BORDER_COLOR = (60, 60, 60)  # Darker gray for border
    
    # Create panel with alpha blending
    alpha = 0.7
    overlay = display_frame.copy()
    
    # Draw panel background with rounded corners
    cv2.rectangle(overlay, 
                 (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 BACKGROUND, -1)
    cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0, display_frame)
    
    # Add a thin border
    cv2.rectangle(display_frame, 
                 (panel_x, panel_y), 
                 (panel_x + panel_width, panel_y + panel_height), 
                 BORDER_COLOR, 1, cv2.LINE_AA)
    
    # Add title with separator line
    cv2.putText(display_frame, title, 
               (panel_x + 10, panel_y + 20), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, TITLE_COLOR, 1, cv2.LINE_AA)
    
    # Add horizontal separator
    cv2.line(display_frame, 
            (panel_x + 5, panel_y + 30), 
            (panel_x + panel_width - 5, panel_y + 30), 
            BORDER_COLOR, 1, cv2.LINE_AA)
    
    # Add FPS if available
    y_offset = panel_y + 50
    if fps > 0:
        fps_color = (0, 255, 0) if fps > 20 else (0, 255, 255) if fps > 10 else (0, 0, 255)
        fps_text = f"FPS: {fps:.1f}"
        cv2.putText(display_frame, fps_text, 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, fps_color, 1, cv2.LINE_AA)
        y_offset += 20
    
    # Add stats
    for i, (key, value) in enumerate(stats.items()):
        # Key
        cv2.putText(display_frame, f"{key}:", 
                   (panel_x + 10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1, cv2.LINE_AA)
        
        # Value
        cv2.putText(display_frame, str(value), 
                   (panel_x + 100, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, VALUE_COLOR, 1, cv2.LINE_AA)
        
        y_offset += 20
    
    return display_frame


# Time utilities
def format_time_delta(seconds):
    """
    Format time delta in a human-readable format
    
    Args:
        seconds: Number of seconds
        
    Returns:
        Formatted string (e.g. "2h 30m" or "45s")
    """
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds / 60)
        return f"{minutes}m"
    else:
        hours = int(seconds / 3600)
        minutes = int((seconds % 3600) / 60)
        return f"{hours}h {minutes}m"


def is_time_between(start_time, end_time, check_time=None):
    """
    Check if the current time is between the start and end times
    
    Args:
        start_time: Start time (HH:MM format)
        end_time: End time (HH:MM format)
        check_time: Time to check (datetime object, or None for current time)
        
    Returns:
        True if current time is between start and end times
    """
    if check_time is None:
        check_time = datetime.now().time()
    else:
        if isinstance(check_time, datetime):
            check_time = check_time.time()
    
    start = datetime.strptime(start_time, "%H:%M").time()
    end = datetime.strptime(end_time, "%H:%M").time()
    
    if start <= end:
        return start <= check_time <= end
    else:  # Handle case where time range spans midnight
        return check_time >= start or check_time <= end


# Analytics functions
def calculate_dwell_time(visits, location=None):
    """
    Calculate average dwell time from visit logs
    
    Args:
        visits: List of visit dictionaries
        location: Optional location filter
        
    Returns:
        Average dwell time in seconds
    """
    # Group visits by person ID and calculate time differences
    person_visits = {}
    for visit in visits:
        person_id = visit.get("personId")
        visit_time = visit.get("timestamp")
        visit_loc = visit.get("location")
        
        # Filter by location if specified
        if location and visit_loc != location:
            continue
            
        try:
            timestamp = datetime.fromisoformat(visit_time)
            
            if person_id not in person_visits:
                person_visits[person_id] = []
                
            person_visits[person_id].append(timestamp)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Calculate dwell times
    dwell_times = []
    for person_id, timestamps in person_visits.items():
        if len(timestamps) >= 2:
            # Sort timestamps
            timestamps.sort()
            
            # Calculate time differences between consecutive visits
            for i in range(len(timestamps) - 1):
                time_diff = (timestamps[i+1] - timestamps[i]).total_seconds()
                
                # Only count as same session if less than 1 hour apart
                if time_diff < 3600:
                    dwell_times.append(time_diff)
    
    # Calculate average
    if dwell_times:
        return sum(dwell_times) / len(dwell_times)
    else:
        return 0


def get_customer_visit_frequency(visits, days=30, customer_id=None):
    """
    Calculate visit frequency for customers
    
    Args:
        visits: List of visit dictionaries
        days: Number of days to look back
        customer_id: Optional customer ID to filter by
        
    Returns:
        Dictionary with visit frequency statistics
    """
    # Set time threshold
    now = datetime.now()
    threshold = now - timedelta(days=days)
    
    # Filter visits
    filtered_visits = []
    customer_visits = {}
    
    for visit in visits:
        if visit.get("type") != "customer":
            continue
            
        person_id = visit.get("personId")
        
        # Filter by customer ID if specified
        if customer_id and person_id != customer_id:
            continue
            
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            
            # Only include visits within the time threshold
            if timestamp >= threshold:
                filtered_visits.append(visit)
                
                # Count visits per customer
                if person_id not in customer_visits:
                    customer_visits[person_id] = []
                    
                customer_visits[person_id].append(timestamp)
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    # Calculate visit frequencies
    visit_days = {}  # Days when customers visited
    for person_id, timestamps in customer_visits.items():
        visit_days[person_id] = set()
        for ts in timestamps:
            visit_days[person_id].add(ts.date())
    
    # Calculate statistics
    total_customers = len(customer_visits)
    
    if total_customers == 0:
        return {
            "total_customers": 0,
            "avg_visits_per_customer": 0,
            "avg_visit_days": 0,
            "return_rate": 0
        }
    
    total_visits = sum(len(timestamps) for timestamps in customer_visits.values())
    total_days = sum(len(days) for days in visit_days.values())
    
    # Count customers with multiple visits
    customers_with_returns = sum(1 for person_id, days in visit_days.items() if len(days) > 1)
    
    return {
        "total_customers": total_customers,
        "avg_visits_per_customer": total_visits / total_customers,
        "avg_visit_days": total_days / total_customers,
        "return_rate": customers_with_returns / total_customers if total_customers > 0 else 0
    }


def generate_hourly_traffic_report(visits, days=7):
    """
    Generate hourly traffic report
    
    Args:
        visits: List of visit dictionaries
        days: Number of days to include
        
    Returns:
        Dictionary with hourly visit counts
    """
    # Set time threshold
    now = datetime.now()
    threshold = now - timedelta(days=days)
    
    # Initialize counters
    hourly_counts = {hour: 0 for hour in range(24)}
    
    # Count visits by hour
    for visit in visits:
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            
            # Only include visits within the time threshold
            if timestamp >= threshold:
                hour = timestamp.hour
                hourly_counts[hour] += 1
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    return hourly_counts


def generate_visit_heatmap(visits, days=7):
    """
    Generate visit heatmap by day and hour
    
    Args:
        visits: List of visit dictionaries
        days: Number of days to include
        
    Returns:
        2D numpy array with visit counts by day and hour
    """
    # Set time threshold
    now = datetime.now()
    threshold = now - timedelta(days=days)
    
    # Initialize heatmap (days x hours)
    heatmap = np.zeros((7, 24))
    
    # Count visits by day and hour
    for visit in visits:
        try:
            timestamp = datetime.fromisoformat(visit.get("timestamp", ""))
            
            # Only include visits within the time threshold
            if timestamp >= threshold:
                day = timestamp.weekday()  # 0 = Monday, 6 = Sunday
                hour = timestamp.hour
                heatmap[day, hour] += 1
        except Exception as e:
            logger.error(f"Error parsing timestamp: {e}")
    
    return heatmap


def plot_hourly_traffic(hourly_counts):
    """
    Plot hourly traffic chart
    
    Args:
        hourly_counts: Dictionary with hourly visit counts
        
    Returns:
        Matplotlib figure
    """
    hours = list(range(24))
    counts = [hourly_counts[hour] for hour in hours]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(hours, counts, width=0.8, color='blue', alpha=0.7)
    
    ax.set_xlabel('Hour of Day')
    ax.set_ylabel('Visit Count')
    ax.set_title('Hourly Traffic')
    ax.set_xticks(hours)
    ax.set_xticklabels([f"{h:02d}:00" for h in hours], rotation=45)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return fig


def plot_visit_heatmap(heatmap):
    """
    Plot visit heatmap
    
    Args:
        heatmap: 2D numpy array with visit counts by day and hour
        
    Returns:
        Matplotlib figure
    """
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hours = [f"{h:02d}:00" for h in range(24)]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(heatmap, cmap='viridis')
    
    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Visit Count", rotation=-90, va="bottom")
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(hours)))
    ax.set_yticks(np.arange(len(days)))
    ax.set_xticklabels(hours)
    ax.set_yticklabels(days)
    
    # Rotate the x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    ax.set_title("Visit Heatmap by Day and Hour")
    fig.tight_layout()
    
    return fig


# Image saving functions
def save_detected_face(frame, face_location, save_dir, prefix="face", quality_threshold=0.5):
    """
    Save a detected face to disk if it meets quality threshold
    
    Args:
        frame: Image frame
        face_location: Face location (top, right, bottom, left)
        save_dir: Directory to save to
        prefix: Filename prefix
        quality_threshold: Minimum quality score to save
        
    Returns:
        Path to saved image or None if face wasn't saved
    """
    try:
        # Ensure directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Crop face
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        
        # Calculate quality
        from face_processor import FaceProcessor
        processor = FaceProcessor()
        quality = processor.get_face_quality_score(face_img)
        
        # Save if quality is good enough
        if quality >= quality_threshold:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{prefix}_{timestamp}_{quality:.2f}.jpg"
            filepath = os.path.join(save_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, face_img)
            logger.info(f"Saved face to {filepath}")
            
            return filepath
    except Exception as e:
        logger.error(f"Error saving face: {e}")
    
    return None


# Helper functions
def generate_unique_id(prefix=""):
    """Generate a unique ID with optional prefix"""
    import uuid
    unique_id = str(uuid.uuid4().hex[:8])
    if prefix:
        return f"{prefix}_{unique_id}"
    return unique_id


def read_json_file(filepath):
    """Read JSON file, with error handling"""
    try:
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                return json.load(f)
    except Exception as e:
        logger.error(f"Error reading {filepath}: {e}")
    
    return None


def write_json_file(data, filepath):
    """Write data to JSON file, with error handling"""
    try:
        directory = os.path.dirname(filepath)
        os.makedirs(directory, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=4)
        return True
    except Exception as e:
        logger.error(f"Error writing to {filepath}: {e}")
    
    return False


def draw_stylish_box(frame, face_location, label, color, confidence=None, secondary_text=None):
    """
    Draw a stylish bounding box and text label on a face
    
    Args:
        frame: Image frame to draw on
        face_location: (top, right, bottom, left) coordinates
        label: Text label to display
        color: (B,G,R) color tuple for the box
        confidence: Optional confidence score to display
        secondary_text: Optional additional text to display below the box
        
    Returns:
        Modified frame with stylish box and text
    """
    top, right, bottom, left = face_location
    
    # Make a copy to avoid modifying original
    draw_frame = frame.copy()
    
    # Calculate face dimensions
    face_width = right - left
    face_height = bottom - top
    
    # Draw thinner, stylish box with anti-aliased edges
    cv2.rectangle(draw_frame, (left, top), (right, bottom), color, 1, cv2.LINE_AA)
    
    # Add corner markers for a modern look
    corner_length = min(30, face_width // 4, face_height // 4)
    line_thickness = 2
    
    # Top-left corner
    cv2.line(draw_frame, (left, top), (left + corner_length, top), color, line_thickness, cv2.LINE_AA)
    cv2.line(draw_frame, (left, top), (left, top + corner_length), color, line_thickness, cv2.LINE_AA)
    
    # Top-right corner
    cv2.line(draw_frame, (right, top), (right - corner_length, top), color, line_thickness, cv2.LINE_AA)
    cv2.line(draw_frame, (right, top), (right, top + corner_length), color, line_thickness, cv2.LINE_AA)
    
    # Bottom-left corner
    cv2.line(draw_frame, (left, bottom), (left + corner_length, bottom), color, line_thickness, cv2.LINE_AA)
    cv2.line(draw_frame, (left, bottom), (left, bottom - corner_length), color, line_thickness, cv2.LINE_AA)
    
    # Bottom-right corner
    cv2.line(draw_frame, (right, bottom), (right - corner_length, bottom), color, line_thickness, cv2.LINE_AA)
    cv2.line(draw_frame, (right, bottom), (right, bottom - corner_length), color, line_thickness, cv2.LINE_AA)
    
    # Define font settings - use a cleaner font
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    
    # Calculate text size
    (text_width, text_height), _ = cv2.getTextSize(label, font, font_scale, font_thickness)
    
    # Draw cleaner text label with gradient background
    # Background with gradient alpha
    alpha = 0.7
    overlay = draw_frame.copy()
    cv2.rectangle(overlay, 
                 (left - 1, top - text_height - 12),
                 (left + text_width + 10, top),
                 color, -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, draw_frame, 1 - alpha, 0, draw_frame)
    
    # Draw text with shadow for better readability
    shadow_offset = 1
    cv2.putText(draw_frame, label, 
               (left + 5 + shadow_offset, top - 5 + shadow_offset), 
               font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
    cv2.putText(draw_frame, label, 
               (left + 5, top - 5), 
               font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # Add confidence if provided
    if confidence is not None:
        conf_text = f"{confidence:.2f}"
        # Draw confidence as a small pill-shaped badge
        (conf_width, conf_height), _ = cv2.getTextSize(conf_text, font, font_scale, font_thickness)
        
        # Draw confidence badge
        badge_left = right - conf_width - 10
        badge_top = bottom + 5
        badge_right = right
        badge_bottom = badge_top + conf_height + 10
        
        # Draw rounded rectangle badge
        overlay = draw_frame.copy()
        cv2.rectangle(overlay, 
                     (badge_left - 5, badge_top),
                     (badge_right + 5, badge_bottom),
                     color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.7, draw_frame, 0.3, 0, draw_frame)
        
        # Add confidence text
        cv2.putText(draw_frame, conf_text, 
                   (badge_left, badge_bottom - 5), 
                   font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
    
    # Add secondary text if provided
    if secondary_text is not None:
        # Draw below the face
        cv2.putText(draw_frame, secondary_text, 
                   (left, bottom + 20), 
                   font, font_scale, color, font_thickness, cv2.LINE_AA)
    
    return draw_frame 