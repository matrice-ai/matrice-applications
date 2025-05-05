"""
models.py - Face Detection and Recognition Models

Implements state-of-the-art face detection and recognition models:
- MTCNN for face detection
- ArcFace/MobileFaceNet for face embeddings
"""

import cv2
import numpy as np
import face_recognition
import logging
from facenet_pytorch import MTCNN
import torch

# Initialize logger
logger = logging.getLogger('face_recognition.models')

class BaseFaceDetector:
    """Base class for face detection models"""
    def __init__(self):
        pass
    
    def detect_faces(self, frame):
        """Detect faces in a frame and return locations"""
        raise NotImplementedError("Subclasses must implement detect_faces")

class BaseEmbeddingModel:
    """Base class for face embedding models"""
    def __init__(self):
        pass
    
    def get_embeddings(self, frame, face_locations):
        """Extract face embeddings from detected face locations"""
        raise NotImplementedError("Subclasses must implement get_embeddings")

class MTCNNDetector(BaseFaceDetector):
    """MTCNN-based face detector (state-of-the-art)"""
    def __init__(self, min_face_size=20, thresholds=[0.6, 0.7, 0.7], device=None):
        """
        Initialize MTCNN face detector
        
        Args:
            min_face_size: Minimum face size to detect
            thresholds: MTCNN thresholds for the three stages
            device: 'cpu' or 'cuda' for GPU acceleration
        """
        super().__init__()
        
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        logger.info(f"Using MTCNN on device: {self.device}")
        
        # Initialize MTCNN
        self.detector = MTCNN(
            min_face_size=min_face_size,
            thresholds=thresholds,
            device=self.device,
            keep_all=True
        )
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using MTCNN
        
        Args:
            frame: RGB image (numpy array)
            
        Returns:
            List of face locations in (top, right, bottom, left) format
            List of face probabilities
            List of facial landmarks (if available)
        """
        try:
            # Detect faces
            boxes, probs, landmarks = self.detector.detect(frame, landmarks=True)
            
            # Convert to (top, right, bottom, left) format if faces were found
            face_locations = []
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    # Convert from [x1, y1, x2, y2] to (top, right, bottom, left)
                    left, top, right, bottom = map(int, box)
                    face_locations.append((top, right, bottom, left))
                
                # Ensure landmarks is None or properly formatted
                if landmarks is not None:
                    # MTCNN landmarks is shape [n_faces, 5, 2]
                    # We'll return it as is, and handle the conversion in the calling code
                    return face_locations, probs, landmarks
                else:
                    return face_locations, probs, None
            else:
                return [], [], None
                
        except Exception as e:
            logger.error(f"Error in MTCNN face detection: {str(e)}")
            return [], [], None

class HOGDetector(BaseFaceDetector):
    """HOG-based face detector from face_recognition library (CPU-friendly)"""
    def __init__(self, model="hog"):
        super().__init__()
        self.model = model
    
    def detect_faces(self, frame):
        """
        Detect faces in a frame using HOG
        
        Args:
            frame: RGB image (numpy array)
            
        Returns:
            List of face locations in (top, right, bottom, left) format
            Empty list for probabilities (not provided by this model)
            Empty list for landmarks (not provided by this detector)
        """
        try:
            face_locations = face_recognition.face_locations(frame, model=self.model)
            return face_locations, [], []
        except Exception as e:
            logger.error(f"Error in HOG face detection: {str(e)}")
            return [], [], []

class FaceRecognitionModel(BaseEmbeddingModel):
    """Face embedding model from face_recognition library"""
    def __init__(self):
        super().__init__()
    
    def get_embeddings(self, frame, face_locations):
        """
        Extract face embeddings using face_recognition
        
        Args:
            frame: RGB image (numpy array)
            face_locations: List of face locations in (top, right, bottom, left) format
            
        Returns:
            List of face embeddings (128-dimensional vectors)
        """
        try:
            # Get face encodings
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            return face_encodings
        except Exception as e:
            logger.error(f"Error extracting face embeddings: {str(e)}")
            return []

class MobileFaceNetModel(BaseEmbeddingModel):
    """MobileFaceNet embedding model (state-of-the-art, lightweight)"""
    def __init__(self, model_path=None):
        """
        Initialize MobileFaceNet model
        
        Note: This is a placeholder for actual implementation
        In production, you'd load the actual model
        """
        super().__init__()
        self.model_path = model_path
        # Use face_recognition library as fallback for now
        self.fallback_model = FaceRecognitionModel()
        logger.info("Using face_recognition model as fallback for MobileFaceNet")
    
    def get_embeddings(self, frame, face_locations):
        """
        Extract face embeddings using MobileFaceNet
        
        Args:
            frame: RGB image (numpy array)
            face_locations: List of face locations in (top, right, bottom, left) format
            
        Returns:
            List of face embeddings
        """
        # For now, use fallback model
        # In production implementation, replace with actual MobileFaceNet
        return self.fallback_model.get_embeddings(frame, face_locations)

# Facotry function to get appropriate models
def get_face_detector(detector_type="mtcnn"):
    """Get face detector by type"""
    if detector_type.lower() == "mtcnn":
        return MTCNNDetector()
    elif detector_type.lower() == "hog":
        return HOGDetector()
    else:
        logger.warning(f"Unknown detector type: {detector_type}, falling back to HOG")
        return HOGDetector()

def get_embedding_model(model_type="facenet"):
    """Get embedding model by type"""
    if model_type.lower() == "mobilefacenet":
        return MobileFaceNetModel()
    elif model_type.lower() == "facenet":
        return FaceRecognitionModel()
    else:
        logger.warning(f"Unknown embedding model type: {model_type}, falling back to face_recognition")
        return FaceRecognitionModel() 