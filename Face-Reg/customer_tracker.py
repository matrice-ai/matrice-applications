"""
customer_tracker.py - Customer Recognition and Tracking Module

This module handles detection, recognition, and tracking of customers.
It differentiates between returning customers and new visitors, maintaining 
tracking information to determine if they're frequent visitors.
"""

import cv2
import numpy as np
import logging
import threading
import time
from datetime import datetime
from collections import defaultdict, deque

from models import get_face_detector, get_embedding_model
from face_processor import FaceProcessor
from db_manager import db_manager
from utils import draw_stylish_box

# Initialize logger
logger = logging.getLogger('face_recognition.customer')

# Configuration
CUSTOMER_FACE_RESET_SECONDS = 300  # 5 minutes before resetting tracking if face disappears

class TrackedFace:
    """Class to hold tracking information for a detected face"""
    
    def __init__(self, face_id, face_location, embedding, quality_score=0.0):
        """Initialize tracked face"""
        self.face_id = face_id
        self.bbox = face_location  # (top, right, bottom, left)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.frames_tracked = 1
        self.embedding = embedding
        self.quality_score = quality_score
        self.identified = False
        self.identified_as = None
        self.center_points = []  # For motion tracking
        
        # Calculate initial center point
        top, right, bottom, left = face_location
        self.center_points.append((
            (left + right) // 2,
            (top + bottom) // 2
        ))
    
    def update(self, face_location, embedding=None, quality_score=None):
        """Update tracking info"""
        self.bbox = face_location
        self.last_seen = time.time()
        self.frames_tracked += 1
        
        # Update embedding if provided and better quality
        if embedding is not None and quality_score is not None:
            if quality_score > self.quality_score:
                self.embedding = embedding
                self.quality_score = quality_score
        
        # Update center point
        top, right, bottom, left = face_location
        self.center_points.append((
            (left + right) // 2,
            (top + bottom) // 2
        ))
        # Keep only the last 30 points
        if len(self.center_points) > 30:
            self.center_points = self.center_points[-30:]
    
    @property
    def total_time(self):
        """Get total tracking time in seconds"""
        return self.last_seen - self.first_seen
    
    @property
    def is_expired(self):
        """Check if tracking is expired"""
        return time.time() - self.last_seen > CUSTOMER_FACE_RESET_SECONDS
    
    @property
    def last_center(self):
        """Get the most recent center point"""
        if not self.center_points:
            return None
        return self.center_points[-1]
    
    def get_motion_vector(self):
        """Calculate motion vector from center points history"""
        if len(self.center_points) < 2:
            return (0, 0)
        
        # Get the last two center points
        last = self.center_points[-1]
        prev = self.center_points[-2]
        
        # Calculate vector
        return (last[0] - prev[0], last[1] - prev[1])
    
    def predict_next_location(self):
        """Predict next face location based on motion history"""
        if len(self.center_points) < 2:
            return self.bbox
        
        # Get motion vector
        dx, dy = self.get_motion_vector()
        
        # Apply motion to bbox
        top, right, bottom, left = self.bbox
        return (top + dy, right + dx, bottom + dy, left + dx)


class CustomerTracker:
    """Customer tracking manager that handles recognition of customers"""
    
    def __init__(self, detector_type="mtcnn", embedding_model="facenet", match_threshold=0.6):
        """
        Initialize customer tracker
        
        Args:
            detector_type: Face detection model ("mtcnn" or "hog")
            embedding_model: Face embedding model ("facenet" or "mobilefacenet")
            match_threshold: Matching threshold (lower is more strict)
        """
        self.detector = get_face_detector(detector_type)
        self.embedding_model = get_embedding_model(embedding_model)
        self.face_processor = FaceProcessor()
        self.match_threshold = match_threshold
        self.detector_type = detector_type
        
        logger.info(f"Customer tracker initialized with {detector_type} detector and {embedding_model}")
        
        # Face tracking state
        self.tracked_faces = {}  # Dictionary of tracked faces by ID
        self.next_customer_id = 0  # Counter for assigning temporary IDs
        
        # Recognition stats
        self.recognition_count = 0
        self.recognized_customers = set()
        self.new_customers = 0
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Thread safety
        self.track_lock = threading.Lock()
        self.stats_lock = threading.Lock()
    
    def track_face(self, face_location, embedding, quality_score):
        """
        Track a face - determine if it's a new tracking entry or update existing
        
        Args:
            face_location: Face bounding box (top, right, bottom, left)
            embedding: Face embedding vector
            quality_score: Face quality score
            
        Returns:
            ID of the tracked face
        """
        # Calculate face center point
        top, right, bottom, left = face_location
        face_center = ((left + right) // 2, (top + bottom) // 2)
        
        # Search for existing tracked faces with similar position
        best_match_id = None
        best_distance = float('inf')
        face_width = right - left
        
        with self.track_lock:
            # First, clean up expired tracks
            self._cleanup_expired_tracks()
            
            # Find best match based on location
            for face_id, tracked in self.tracked_faces.items():
                # Skip already identified faces (they're matched by embedding)
                if tracked.identified:
                    continue
                
                # Get last position
                if tracked.last_center is None:
                    continue
                    
                # Calculate center point distance
                tx, ty = tracked.last_center
                distance = np.sqrt((tx - face_center[0])**2 + (ty - face_center[1])**2)
                
                # Consider it a match if within reasonable distance (half face width)
                if distance < face_width / 2 and distance < best_distance:
                    best_match_id = face_id
                    best_distance = distance
            
            # If found, update the tracking
            if best_match_id:
                self.tracked_faces[best_match_id].update(face_location, embedding, quality_score)
                return best_match_id
            
            # If no match, create a new tracking ID
            new_id = f"customer_{self.next_customer_id}"
            self.next_customer_id += 1
            self.tracked_faces[new_id] = TrackedFace(new_id, face_location, embedding, quality_score)
            return new_id
    
    def _cleanup_expired_tracks(self):
        """Remove expired face tracks"""
        expired_ids = []
        for face_id, tracked in self.tracked_faces.items():
            if tracked.is_expired:
                expired_ids.append(face_id)
        
        # Remove expired tracks
        for face_id in expired_ids:
            logger.info(f"Removing expired tracking for {face_id}")
            del self.tracked_faces[face_id]
    
    def detect_and_track_customers(self, frame, location="Unknown"):
        """
        Detect and track customers in a frame
        
        Args:
            frame: Input image frame
            location: Location identifier for logging
            
        Returns:
            Tuple of (tracked_customers, face_locations)
            - tracked_customers: List of dictionaries with customer and tracking info
            - face_locations: List of face locations (top, right, bottom, left)
        """
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed > 0:
            fps = 1 / elapsed
            self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        # Detect faces
        face_locations, probs, landmarks = self.detector.detect_faces(frame)
        
        # No faces found
        if not face_locations:
            return [], []
        
        # Extract face embeddings
        face_embeddings = self.embedding_model.get_embeddings(frame, face_locations)
        
        tracked_customers = []
        
        # Process each detected face
        for i, (face_loc, embedding) in enumerate(zip(face_locations, face_embeddings)):
            # Skip if no embedding could be extracted
            if embedding is None or len(embedding) == 0:
                continue
            
            # Process face for quality assessment
            face_img = self.face_processor.crop_face(frame, face_loc)
            quality_score = self.face_processor.get_face_quality_score(face_img) if face_img is not None else 0.0
            
            # Get landmarks if available
            face_landmarks = None
            
            # Handle landmarks differently based on detector type
            if self.detector_type == "mtcnn" and landmarks is not None:
                # For MTCNN, landmarks is a numpy array with shape [n_faces, 5, 2]
                # where 5 points are: left eye, right eye, nose, left mouth, right mouth
                if i < len(landmarks):
                    # Convert MTCNN landmarks to face_recognition format
                    mtcnn_landmarks = landmarks[i]
                    
                    # Only proceed if we have valid landmarks
                    if mtcnn_landmarks is not None and isinstance(mtcnn_landmarks, np.ndarray):
                        # Create a dictionary format similar to face_recognition's landmarks
                        face_landmarks = {
                            "left_eye": mtcnn_landmarks[0].tolist(),  # Left eye
                            "right_eye": mtcnn_landmarks[1].tolist(),  # Right eye
                            "nose_tip": mtcnn_landmarks[2].tolist(),  # Nose
                            "left_mouth": mtcnn_landmarks[3].tolist(),  # Left mouth
                            "right_mouth": mtcnn_landmarks[4].tolist(),  # Right mouth
                            "full_landmarks": {
                                "left_eye": [mtcnn_landmarks[0].tolist()],
                                "right_eye": [mtcnn_landmarks[1].tolist()],
                                "nose_bridge": [mtcnn_landmarks[2].tolist()],
                                "top_lip": [mtcnn_landmarks[3].tolist(), mtcnn_landmarks[4].tolist()],
                                "bottom_lip": [mtcnn_landmarks[3].tolist(), mtcnn_landmarks[4].tolist()]
                            }
                        }
            
            # Try to match with customer database
            customer_match = db_manager.find_matching_customer(embedding, self.match_threshold)
            
            if customer_match:
                # Found a customer match
                customer_id = customer_match["customerId"]
                confidence = customer_match["confidence"]
                
                # Update database and logs
                db_manager.update_customer(
                    customer_match,
                    {"embedding": embedding, "embedding_list": embedding.tolist(), "quality_score": quality_score},
                    location
                )
                
                # Log recognition event
                db_manager.log_recognition_event(
                    "customer",
                    customer_id,
                    confidence,
                    location
                )
                
                # Update stats
                with self.stats_lock:
                    self.recognition_count += 1
                    self.recognized_customers.add(customer_id)
                
                # Create result entry
                result = {
                    "type": "customer",
                    "customer_id": customer_id,
                    "visit_count": customer_match.get("visitCount", 1),
                    "first_seen": customer_match.get("firstSeen", ""),
                    "confidence": confidence,
                    "face_location": face_loc,
                    "quality_score": quality_score,
                    "detection_time": datetime.now().isoformat()
                }
                
                # Update or create tracking info
                with self.track_lock:
                    # Check if we have this face already tracked
                    tracked_id = None
                    for face_id, tracked in self.tracked_faces.items():
                        if tracked.identified and tracked.identified_as and tracked.identified_as.get("id") == customer_id:
                            tracked_id = face_id
                            # Update tracking
                            tracked.update(face_loc, embedding, quality_score)
                            break
                    
                    # Create new tracking if not found
                    if not tracked_id:
                        new_id = f"customer_{customer_id}"
                        self.tracked_faces[new_id] = TrackedFace(new_id, face_loc, embedding, quality_score)
                        self.tracked_faces[new_id].identified = True
                        self.tracked_faces[new_id].identified_as = {
                            "type": "customer",
                            "id": customer_id
                        }
                        tracked_id = new_id
                    
                    # Add tracking ID to result
                    result["tracking_id"] = tracked_id
                
                tracked_customers.append(result)
                logger.info(f"Recognized customer: {customer_id} ({confidence:.2f})")
                
            else:
                # This is a new customer, register immediately
                face_data = {
                    "embedding": embedding,
                    "embedding_list": embedding.tolist(),
                    "quality_score": quality_score
                }
                
                # Add as a new customer
                customer = db_manager.add_new_customer(face_data, location)
                
                # Update tracking
                tracking_id = self.track_face(face_loc, embedding, quality_score)
                
                # Mark as identified
                with self.track_lock:
                    tracked_face = self.tracked_faces.get(tracking_id)
                    tracked_face.identified = True
                    tracked_face.identified_as = {
                        "type": "customer",
                        "id": customer["customerId"]
                    }
                
                # Update stats
                with self.stats_lock:
                    self.new_customers += 1
                
                # Create result entry
                result = {
                    "type": "customer",
                    "customer_id": customer["customerId"],
                    "visit_count": 1,
                    "confidence": 1.0,  # High confidence since we just created it
                    "face_location": face_loc,
                    "quality_score": quality_score,
                    "tracking_id": tracking_id,
                    "detection_time": datetime.now().isoformat()
                }
                
                tracked_customers.append(result)
                logger.info(f"Registered new customer: {customer['customerId']}")
        
        return tracked_customers, face_locations
    
    def get_tracking_stats(self):
        """Get tracking statistics"""
        with self.track_lock:
            return {
                "total_tracked_faces": len(self.tracked_faces),
                "identified_faces": sum(1 for f in self.tracked_faces.values() if f.identified),
                "customer_faces": len(self.tracked_faces)
            }
    
    def get_recognition_stats(self):
        """Get recognition statistics"""
        with self.stats_lock:
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            return {
                "total_recognitions": self.recognition_count,
                "unique_customers_recognized": len(self.recognized_customers),
                "new_customers_enrolled": self.new_customers,
                "average_fps": avg_fps
            }
    
    def get_customer_details(self, customer_id):
        """Get detailed information about a customer"""
        return db_manager.get_customer_info(customer_id)
    
    def get_customer_visit_stats(self, customer_id=None):
        """Get visit statistics for a customer or all customers"""
        return db_manager.get_customer_visit_stats(customer_id)
    
    def draw_customer_tracking(self, frame, tracked_customers):
        """
        Draw customer tracking results on frame
        
        Args:
            frame: Input image frame
            tracked_customers: List of tracked customers from detect_and_track_customers
            
        Returns:
            Frame with tracking visualization
        """
        # Colors for different customer types
        RETURNING_COLOR = (255, 128, 0)  # Vibrant orange-blue (BGR)
        NEW_COLOR = (128, 0, 255)       # Purple
        
        # Make a copy to avoid modifying original
        draw_frame = frame.copy()
        
        # Draw each tracked face
        for customer in tracked_customers:
            # Get the face location
            face_loc = customer["face_location"]
            
            # Determine color based on visit count
            visit_count = customer.get("visit_count", 1)
            color = RETURNING_COLOR if visit_count > 1 else NEW_COLOR
            
            # Create labels
            customer_id = customer.get("customer_id", "N/A")
            label = f"Customer {customer_id}"
            
            # Create secondary text
            secondary_text = None
            if visit_count > 1:
                secondary_text = f"Visit: {visit_count}"
            
            # Use the stylish box utility
            draw_frame = draw_stylish_box(
                draw_frame, 
                face_loc, 
                label, 
                color, 
                confidence=customer.get("confidence", None),
                secondary_text=secondary_text
            )
            
            # Draw tracking line of movement (optional)
            tracking_id = customer.get("tracking_id")
            if tracking_id and tracking_id in self.tracked_faces:
                tracked = self.tracked_faces[tracking_id]
                if len(tracked.center_points) >= 2:
                    # Draw path
                    points = tracked.center_points
                    for i in range(1, len(points)):
                        cv2.line(draw_frame, points[i-1], points[i], color, 1, cv2.LINE_AA)
        
        return draw_frame 