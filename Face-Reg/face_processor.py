"""
face_processor.py - Face Preprocessing and Quality Assessment

Handles:
- Face alignment
- Image preprocessing
- Quality assessment of faces
- Image utilities
"""

import cv2
import numpy as np
import logging
from scipy.spatial import distance

# Initialize logger
logger = logging.getLogger('face_recognition.processor')

class FaceProcessor:
    """Face image processing and quality assessment"""
    
    def __init__(self, target_face_size=(112, 112)):
        """
        Initialize face processor
        
        Args:
            target_face_size: Target size for face alignment (width, height)
        """
        self.target_size = target_face_size
    
    def align_face(self, frame, landmarks):
        """
        Align face based on detected landmarks
        
        Args:
            frame: Original image containing the face
            landmarks: Facial landmarks (eyes, nose, etc.)
            
        Returns:
            Aligned face image
        """
        try:
            # This is a simplified alignment - in production use more reference points
            if not landmarks or 'left_eye' not in landmarks or 'right_eye' not in landmarks:
                logger.warning("Cannot align face: landmarks not available")
                return None
                
            # Get eye centers
            left_eye = landmarks['left_eye']
            right_eye = landmarks['right_eye']
            
            if isinstance(left_eye, list) and len(left_eye) > 0:
                left_eye_center = np.mean(left_eye, axis=0).astype(int)
            else:
                left_eye_center = left_eye
                
            if isinstance(right_eye, list) and len(right_eye) > 0:
                right_eye_center = np.mean(right_eye, axis=0).astype(int)
            else:
                right_eye_center = right_eye
            
            # Calculate angle
            dy = right_eye_center[1] - left_eye_center[1]
            dx = right_eye_center[0] - left_eye_center[0]
            angle = np.degrees(np.arctan2(dy, dx))
            
            # Get center of eyes
            eye_center = ((left_eye_center[0] + right_eye_center[0]) // 2,
                          (left_eye_center[1] + right_eye_center[1]) // 2)
            
            # Rotation matrix
            M = cv2.getRotationMatrix2D(eye_center, angle, 1)
            
            # Warp image
            height, width = frame.shape[:2]
            warped = cv2.warpAffine(frame, M, (width, height), flags=cv2.INTER_CUBIC)
            
            return warped
            
        except Exception as e:
            logger.error(f"Error aligning face: {str(e)}")
            return frame  # Return original frame if alignment fails
    
    def crop_face(self, frame, face_location, margin=0.2):
        """
        Crop face from frame with margin
        
        Args:
            frame: Original image
            face_location: (top, right, bottom, left) coordinates
            margin: Percentage margin to add around the face
            
        Returns:
            Cropped face image
        """
        try:
            if not face_location or len(face_location) != 4:
                return None
                
            top, right, bottom, left = face_location
            
            # Calculate margin in pixels
            height = bottom - top
            width = right - left
            margin_pixels_h = int(height * margin)
            margin_pixels_w = int(width * margin)
            
            # Add margin with bounds checking
            frame_height, frame_width = frame.shape[:2]
            top = max(0, top - margin_pixels_h)
            bottom = min(frame_height, bottom + margin_pixels_h)
            left = max(0, left - margin_pixels_w)
            right = min(frame_width, right + margin_pixels_w)
            
            # Crop face
            face_img = frame[top:bottom, left:right]
            
            return face_img
            
        except Exception as e:
            logger.error(f"Error cropping face: {str(e)}")
            return None
    
    def resize_face(self, face_img):
        """
        Resize face to target size
        
        Args:
            face_img: Input face image
            
        Returns:
            Resized face image
        """
        if face_img is None:
            return None
            
        try:
            return cv2.resize(face_img, self.target_size, interpolation=cv2.INTER_AREA)
        except Exception as e:
            logger.error(f"Error resizing face: {str(e)}")
            return face_img
    
    def normalize_face(self, face_img):
        """
        Normalize face image (adjust lighting, contrast)
        
        Args:
            face_img: Input face image
            
        Returns:
            Normalized face image
        """
        if face_img is None:
            return None
            
        try:
            # Convert to float and normalize
            face_float = face_img.astype(np.float32) / 255.0
            
            # Apply histogram equalization in LAB color space
            face_lab = cv2.cvtColor(face_float, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(face_lab)
            l_eq = cv2.equalizeHist((l * 255).astype(np.uint8)) / 255.0
            face_lab_eq = cv2.merge([l_eq, a, b])
            face_rgb_eq = cv2.cvtColor(face_lab_eq, cv2.COLOR_LAB2RGB)
            
            # Convert back to uint8
            face_rgb_eq = (face_rgb_eq * 255).astype(np.uint8)
            
            return face_rgb_eq
        except Exception as e:
            logger.error(f"Error normalizing face: {str(e)}")
            return face_img
    
    def get_face_quality_score(self, face_img):
        """
        Calculate quality score for a face image
        Higher score = better quality
        
        Args:
            face_img: Face image to assess
            
        Returns:
            Quality score between 0 and 1
        """
        if face_img is None:
            return 0.0
            
        try:
            # Check image size (faces should be reasonably large)
            height, width = face_img.shape[:2]
            size_score = min(1.0, (height * width) / (100 * 100))
            
            # Blur detection using Laplacian variance
            try:
                gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY)
                blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                blur_score = min(1.0, blur_variance / 500)
            except:
                blur_score = 0.3  # Default if conversion fails
            
            # Brightness assessment
            try:
                hsv = cv2.cvtColor(face_img, cv2.COLOR_RGB2HSV)
                brightness = hsv[..., 2].mean() / 255
                brightness_score = 1.0 - 2.0 * abs(brightness - 0.5)  # Penalize too dark or too bright
            except:
                brightness_score = 0.5  # Default if conversion fails
            
            # Combine scores with weights
            quality_score = 0.4 * size_score + 0.4 * blur_score + 0.2 * brightness_score
            
            return quality_score
            
        except Exception as e:
            logger.error(f"Error calculating face quality: {str(e)}")
            return 0.3  # Return moderate default score
    
    def extract_facial_features(self, face_img, landmarks):
        """
        Extract geometric features from facial landmarks
        
        Args:
            face_img: Face image
            landmarks: Facial landmarks dictionary
            
        Returns:
            Dictionary of facial metrics
        """
        if face_img is None or not landmarks:
            return {}
            
        try:
            features = {}
            
            # Extract eye positions
            if 'left_eye' in landmarks and 'right_eye' in landmarks:
                left_eye = landmarks['left_eye']
                right_eye = landmarks['right_eye']
                
                if isinstance(left_eye, list) and isinstance(right_eye, list):
                    # Calculate eye centers if landmarks are lists of points
                    left_eye_center = np.mean(left_eye, axis=0)
                    right_eye_center = np.mean(right_eye, axis=0)
                else:
                    # Use directly if already centers
                    left_eye_center = np.array(left_eye)
                    right_eye_center = np.array(right_eye)
                
                # Calculate eye distance
                eye_distance = distance.euclidean(left_eye_center, right_eye_center)
                features['eye_distance'] = eye_distance
                
                # Calculate eye aspect ratio if available
                if isinstance(left_eye, list) and len(left_eye) >= 6:
                    # Simplified Eye Aspect Ratio (EAR) calculation
                    ear_left = self._calculate_ear(left_eye)
                    ear_right = self._calculate_ear(right_eye)
                    features['eye_aspect_ratio'] = (ear_left + ear_right) / 2
            
            # Add more facial metrics as needed
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {str(e)}")
            return {}
    
    def _calculate_ear(self, eye_points):
        """
        Calculate Eye Aspect Ratio (EAR) from eye landmarks
        
        Args:
            eye_points: List of eye landmark points
            
        Returns:
            EAR value
        """
        if len(eye_points) < 6:
            return 0.0
            
        try:
            # Compute the euclidean distance between eye landmarks
            # Simplified version - in production use specific indices for each model
            vertical_1 = distance.euclidean(eye_points[1], eye_points[5])
            vertical_2 = distance.euclidean(eye_points[2], eye_points[4])
            horizontal = distance.euclidean(eye_points[0], eye_points[3])
            
            # Calculate EAR
            ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
            return ear
        except:
            return 0.0
    
    def process_face(self, frame, face_location, landmarks=None):
        """
        Full face processing pipeline
        
        Args:
            frame: Original image
            face_location: Face bounding box (top, right, bottom, left)
            landmarks: Optional facial landmarks for alignment
            
        Returns:
            Dictionary with processed face and metadata
        """
        try:
            # Crop face with margin
            face_img = self.crop_face(frame, face_location)
            if face_img is None:
                return None
                
            # Align face if landmarks are available
            aligned_face = None
            if landmarks:
                aligned_face = self.align_face(face_img, landmarks)
            
            # Use aligned face if available, otherwise use cropped face
            processed_face = aligned_face if aligned_face is not None else face_img
            
            # Resize face
            resized_face = self.resize_face(processed_face)
            
            # Normalize face
            normalized_face = self.normalize_face(resized_face)
            
            # Calculate quality score
            quality_score = self.get_face_quality_score(normalized_face)
            
            # Extract facial features
            facial_features = {}
            if landmarks:
                facial_features = self.extract_facial_features(normalized_face, landmarks)
            
            # Create result dictionary
            result = {
                "original_face": face_img,
                "processed_face": normalized_face,
                "quality_score": quality_score,
                "facial_features": facial_features,
                "face_location": face_location
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in face processing pipeline: {str(e)}")
            return None 