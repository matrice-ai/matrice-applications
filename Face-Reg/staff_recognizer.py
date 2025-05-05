"""
staff_recognizer.py - Staff Recognition Module

This module handles detection and recognition of staff members from the enrollment database.
It uses state-of-the-art face detection and recognition models.
"""

import cv2
import numpy as np
import logging
import threading
from datetime import datetime

from models import get_face_detector, get_embedding_model
from face_processor import FaceProcessor
from db_manager import db_manager

# Initialize logger
logger = logging.getLogger('face_recognition.staff')

class StaffRecognizer:
    """Staff recognition manager that handles recognition of enrolled staff"""
    
    def __init__(self, detector_type="mtcnn", embedding_model="facenet", match_threshold=0.6):
        """
        Initialize staff recognizer
        
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
        
        logger.info(f"Staff recognizer initialized with {detector_type} detector and {embedding_model}")
        
        # Recognition stats
        self.recognition_count = 0
        self.recognized_staff = set()
        
        # Thread safety
        self.stats_lock = threading.Lock()
    
    def detect_and_recognize_staff(self, frame, location="Unknown"):
        """
        Detect and recognize staff in a single frame
        
        Args:
            frame: Input image frame
            location: Location identifier for logging
            
        Returns:
            Tuple of (recognized_staff, face_locations)
            - recognized_staff: List of dictionaries with staff info
            - face_locations: List of face locations (top, right, bottom, left)
        """
        # Detect faces
        face_locations, probs, landmarks = self.detector.detect_faces(frame)
        
        # No faces found
        if not face_locations:
            return [], []
        
        # Extract face embeddings
        face_embeddings = self.embedding_model.get_embeddings(frame, face_locations)
        
        recognized_staff = []
        
        # Process each detected face
        for i, (face_loc, embedding) in enumerate(zip(face_locations, face_embeddings)):
            # Skip if no embedding could be extracted
            if embedding is None or len(embedding) == 0:
                continue
                
            # Try to match with staff database
            staff_match = db_manager.find_matching_staff(embedding, self.match_threshold)
            
            if staff_match:
                # Get facial landmarks if available
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
                
                # Process face for quality assessment and additional features
                face_img = self.face_processor.crop_face(frame, face_loc)
                quality_score = self.face_processor.get_face_quality_score(face_img) if face_img is not None else 0.0
                
                # Create result entry
                result = {
                    "staff_id": staff_match["staffId"],
                    "name": f"{staff_match['firstName']} {staff_match['lastName']}",
                    "position": staff_match.get("position", ""),
                    "department": staff_match.get("department", ""),
                    "confidence": staff_match["confidence"],
                    "face_location": face_loc,
                    "quality_score": quality_score,
                    "detection_time": datetime.now().isoformat()
                }
                
                # Log the recognition
                self._log_staff_recognition(staff_match["staffId"], staff_match["confidence"], location)
                
                recognized_staff.append(result)
        
        return recognized_staff, face_locations
    
    def _log_staff_recognition(self, staff_id, confidence, location):
        """Log staff recognition event"""
        # Log to database
        db_manager.log_recognition_event("staff", staff_id, confidence, location)
        
        # Update stats
        with self.stats_lock:
            self.recognition_count += 1
            self.recognized_staff.add(staff_id)
    
    def get_recognition_stats(self):
        """Get recognition statistics"""
        with self.stats_lock:
            return {
                "total_recognitions": self.recognition_count,
                "unique_staff_recognized": len(self.recognized_staff),
                "recognized_staff_ids": list(self.recognized_staff)
            }
    
    def get_staff_details(self, staff_id):
        """Get detailed information about a staff member"""
        return db_manager.get_staff_info(staff_id)
    
    def draw_staff_recognition(self, frame, recognized_staff):
        """
        Draw staff recognition results on frame
        
        Args:
            frame: Input image frame
            recognized_staff: List of recognized staff from detect_and_recognize_staff
            
        Returns:
            Frame with recognition visualization
        """
        # Draw each recognized staff member
        for staff in recognized_staff:
            face_loc = staff["face_location"]
            top, right, bottom, left = face_loc
            
            # Draw box around face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            
            # Display name and position
            label = f"{staff['name']} - {staff['position']}"
            confidence = f"{staff['confidence']:.2f}"
            
            # Add text background
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
            cv2.rectangle(frame, (left, top - text_size[1] - 10), (left + text_size[0] + 10, top), (0, 255, 0), -1)
            cv2.putText(frame, label, (left + 5, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
            
            # Add confidence
            cv2.putText(frame, confidence, (left + 5, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame 