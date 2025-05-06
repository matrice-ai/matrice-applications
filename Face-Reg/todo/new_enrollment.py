"""
Staff Enrollment Module - Proof of Concept with SOTA Models

This script captures video from a laptop camera, extracts facial features using
state-of-the-art deep learning models, and stores them in a local JSON file
(simulating MongoDB storage).

Dependencies:
- OpenCV
- Streamlit
- NumPy
- InsightFace (SOTA face detection and recognition)
- tqdm
- json

Install with: 
pip install opencv-python streamlit numpy tqdm onnxruntime
pip install -U insightface==0.7.3
pip install onnx
"""

import cv2
import numpy as np
import streamlit as st
import os
import json
import time
import uuid
from datetime import datetime
from tqdm import tqdm
import shutil
import insightface
from insightface.app import FaceAnalysis
from insightface.utils import face_align

# Configuration
MIN_FACE_FRAMES = 10  # Minimum number of good quality face frames required
FACE_SIMILARITY_THRESHOLD = 0.6  # Threshold for face similarity (lower is more strict)
STORAGE_DIR = "face_recognition_data"
FRAMES_DIR = f"{STORAGE_DIR}/staff_frames"
DB_FILE = f"{STORAGE_DIR}/staff_database.json"
FACE_EMBEDDING_SIZE = 512  # ArcFace embedding size

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)


def load_database():
    """Load the staff database from local storage"""
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    else:
        # Initialize empty database structure
        db = {
            "staff": [],
            "staff_embeddings": []
        }
        save_database(db)
        return db


def save_database(db):
    """Save the staff database to local storage"""
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)


def get_face_quality_score(face_obj):
    """
    Calculate a quality score for a face based on multiple factors
    Returns a score between 0 and 1 (higher is better)
    """
    # Check detection confidence
    det_score = float(face_obj.det_score)
    
    # Check face size
    bbox = face_obj.bbox.astype(int)
    face_width = bbox[2] - bbox[0]
    face_height = bbox[3] - bbox[1]
    size_score = min(1.0, (face_width * face_height) / (100 * 100))
    
    # Get face landmarks and check for pose
    landmarks = face_obj.landmark_2d_106
    if landmarks is not None:
        # Calculate face angle from landmarks
        pose_score = get_pose_quality(landmarks)
    else:
        pose_score = 0.5  # Default if landmarks not available
    
    # Combine scores (weighted average)
    quality_score = 0.4 * det_score + 0.3 * size_score + 0.3 * pose_score
    
    return min(1.0, quality_score)


def get_pose_quality(landmarks):
    """
    Estimate how frontal the face is based on landmarks
    Returns a score between 0 and 1 (higher is more frontal)
    """
    # Get the nose tip and chin (center points)
    nose_tip = landmarks[51]  # Approximate nose tip
    chin = landmarks[93]  # Approximate chin point
    
    # Get left and right side points
    left_face = landmarks[0]  # Left face edge
    right_face = landmarks[32]  # Right face edge
    
    # Calculate face width
    face_width = np.linalg.norm(right_face - left_face)
    
    # Calculate deviation from center
    # If perfectly frontal, nose tip and chin should be aligned with face center
    face_center_x = (left_face[0] + right_face[0]) / 2
    nose_deviation = abs(nose_tip[0] - face_center_x) / (face_width / 2)
    chin_deviation = abs(chin[0] - face_center_x) / (face_width / 2)
    
    # Convert to a quality score (lower deviation = higher quality)
    pose_quality = 1.0 - (nose_deviation * 0.7 + chin_deviation * 0.3)
    return max(0.0, min(1.0, pose_quality))


def estimate_face_angles(landmarks):
    """
    Estimate face yaw, pitch, and roll angles from landmarks
    Returns estimated angles in degrees
    """
    # For a precise implementation, we'd use 3D face landmark detection
    # This is a simplified 2D approximation
    
    # Get key facial landmarks
    nose_tip = landmarks[51]  # Nose tip
    left_eye = np.mean(landmarks[60:68], axis=0)  # Left eye center
    right_eye = np.mean(landmarks[68:76], axis=0)  # Right eye center
    left_mouth = landmarks[76]  # Left mouth corner
    right_mouth = landmarks[82]  # Right mouth corner
    
    # Face midpoint
    face_midpoint_x = (landmarks[0][0] + landmarks[32][0]) / 2  # Left to right face edge
    
    # Calculate eye midpoint
    eye_midpoint = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
    
    # Calculate mouth midpoint
    mouth_midpoint = [(left_mouth[0] + right_mouth[0]) / 2, (left_mouth[1] + right_mouth[1]) / 2]
    
    # Calculate face width and height for normalization
    face_width = np.linalg.norm(landmarks[0] - landmarks[32])  # Width between face edges
    face_height = np.linalg.norm(landmarks[51] - landmarks[93])  # Height from nose to chin
    
    # Estimate yaw (left/right head rotation)
    # Compare nose position to face center
    yaw = (nose_tip[0] - face_midpoint_x) / (face_width / 2) * 45
    
    # Estimate pitch (up/down head tilt)
    # Compare vertical relationship between eyes and mouth
    vertical_ratio = (eye_midpoint[1] - nose_tip[1]) / (mouth_midpoint[1] - nose_tip[1])
    pitch = (vertical_ratio - 0.67) * 30  # 0.67 is approximate "neutral" ratio
    
    # Estimate roll (head tilt left/right)
    eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
    roll = np.degrees(eye_angle)
    
    return {
        "yaw": float(yaw),
        "pitch": float(pitch),
        "roll": float(roll)
    }


class StaffEnrollmentApp:
    def __init__(self):
        self.db = load_database()
        
        # Initialize InsightFace models
        self.face_app = FaceAnalysis(
            name="buffalo_l",  # Using the large buffalo model (SCRFD + ArcFace)
            root=".",  # Model download directory
            providers=['CPUExecutionProvider']  # Change to CUDAExecutionProvider for GPU
        )
        
        # Initialize with all required models
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))
        
    def run(self):
        st.set_page_config(page_title="Staff Enrollment System", layout="wide")
        
        st.title("Staff Enrollment System")
        st.write("Capture staff faces from different angles to enroll them in the face recognition system")
        
        # Model info
        st.write("Using **InsightFace** with **SCRFD** face detector and **ArcFace** recognition model")
        
        # Sidebar for staff information
        with st.sidebar:
            st.header("Staff Information")
            staff_id = st.text_input("Staff ID", value=f"ST{int(time.time())}")
            first_name = st.text_input("First Name")
            last_name = st.text_input("Last Name")
            email = st.text_input("Email")
            position = st.text_input("Position")
            department = st.text_input("Department")
            
            # Staff list
            st.header("Enrolled Staff")
            if self.db["staff"]:
                for staff in self.db["staff"]:
                    st.write(f"• {staff['firstName']} {staff['lastName']} ({staff['staffId']})")
            else:
                st.write("No staff enrolled yet")
        
        # Main content - camera capture
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("Camera Feed")
            start_capture = st.button("Start Video Capture")
            stop_capture = st.button("Stop Capture")
            
            # Placeholder for the camera feed
            camera_placeholder = st.empty()
            
            # Status indicators
            status_placeholder = st.empty()
            progress_placeholder = st.empty()
            
            # Instructions
            st.write("""
            **Instructions:**
            1. Enter staff information in the sidebar
            2. Click "Start Video Capture"
            3. Slowly move your face: front → left → right → up → down
            4. System will automatically capture face images from different angles
            5. Click "Stop Capture" when done
            """)
        
        with col2:
            st.header("Captured Faces")
            faces_placeholder = st.empty()
            
            # Enrollment action
            enroll_button = st.button("Enroll Staff Member")
            enrollment_status = st.empty()
        
        # Video capture logic
        if start_capture:
            # Validate staff information
            if not first_name or not last_name or not staff_id:
                st.error("Please enter required staff information")
                return
            
            # Check if staff ID already exists
            if any(staff["staffId"] == staff_id for staff in self.db["staff"]):
                st.error(f"Staff ID '{staff_id}' already exists. Please use a different ID.")
                return
            
            # Create staff directory
            staff_dir = f"{FRAMES_DIR}/{staff_id}"
            if os.path.exists(staff_dir):
                shutil.rmtree(staff_dir)
            os.makedirs(staff_dir)
            
            # Start capturing
            self.capture_video(
                staff_id=staff_id,
                camera_placeholder=camera_placeholder,
                status_placeholder=status_placeholder,
                progress_placeholder=progress_placeholder,
                faces_placeholder=faces_placeholder,
                stop_button=stop_capture
            )
        
        # Enrollment logic
        if enroll_button:
            if not first_name or not last_name or not staff_id:
                enrollment_status.error("Please enter required staff information")
                return
            
            # Check if we have enough good quality face images
            staff_dir = f"{FRAMES_DIR}/{staff_id}"
            if not os.path.exists(staff_dir) or len(os.listdir(staff_dir)) < MIN_FACE_FRAMES:
                enrollment_status.error(f"Not enough face images captured. Need at least {MIN_FACE_FRAMES}.")
                return
            
            # Process enrollment
            enrollment_status.info("Processing enrollment...")
            self.enroll_staff(
                staff_id=staff_id,
                first_name=first_name,
                last_name=last_name,
                email=email,
                position=position,
                department=department,
                progress_placeholder=progress_placeholder
            )
            enrollment_status.success("Staff enrolled successfully!")

    def capture_video(self, staff_id, camera_placeholder, status_placeholder, progress_placeholder, faces_placeholder, stop_button):
        """Capture video from the camera and extract faces using SOTA models"""
        # Initialize the webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            status_placeholder.error("Error: Could not open camera")
            return
        
        # Set parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Variables to track capture
        captured_faces = []
        last_capture_time = time.time() - 10  # Allow immediate first capture
        capture_interval = 1.0  # seconds between captures
        frame_count = 0
        
        # To track face angles for diversity
        captured_angles = []
        
        while True:
            # Check if user clicked stop
            if stop_button:
                break
                
            # Read a frame
            ret, frame = cap.read()
            if not ret:
                status_placeholder.error("Error: Failed to capture frame")
                break
            
            # Convert to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces using InsightFace
            faces = self.face_app.get(rgb_frame)
            
            # Draw rectangles around faces
            frame_with_faces = rgb_frame.copy()
            for face in faces:
                bbox = face.bbox.astype(int)
                # Draw rectangle
                cv2.rectangle(frame_with_faces, 
                              (bbox[0], bbox[1]), 
                              (bbox[2], bbox[3]), 
                              (0, 255, 0), 2)
                
                # Draw quality score
                quality = get_face_quality_score(face)
                cv2.putText(frame_with_faces, 
                            f"Q: {quality:.2f}", 
                            (bbox[0], bbox[1] - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)
            
            # Display the frame
            camera_placeholder.image(frame_with_faces, channels="RGB", use_column_width=True)
            
            # If we found exactly one face, consider capturing it
            current_time = time.time()
            if len(faces) == 1 and (current_time - last_capture_time) >= capture_interval:
                face = faces[0]
                
                # Calculate quality score
                quality_score = get_face_quality_score(face)
                
                # Only accept reasonably good quality
                if quality_score >= 0.5:
                    # Get face landmarks and estimate angles
                    if face.landmark_2d_106 is not None:
                        face_angles = estimate_face_angles(face.landmark_2d_106)
                        
                        # Check if we already have a similar face angle
                        is_unique_angle = True
                        for existing_angle in captured_angles:
                            yaw_diff = abs(existing_angle["yaw"] - face_angles["yaw"])
                            pitch_diff = abs(existing_angle["pitch"] - face_angles["pitch"])
                            
                            # If this angle is too similar to an existing one, skip it
                            if yaw_diff < 15 and pitch_diff < 15:
                                is_unique_angle = False
                                break
                        
                        if is_unique_angle or len(captured_faces) < 3:  # Always capture first few frames
                            # Extract face image
                            bbox = face.bbox.astype(int)
                            face_img = rgb_frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                            
                            # Save face image to file
                            frame_filename = f"{FRAMES_DIR}/{staff_id}/frame_{len(captured_faces):03d}.jpg"
                            cv2.imwrite(frame_filename, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                            
                            # Add to captured faces
                            captured_face = {
                                "embeddingId": f"emb_{staff_id}_{len(captured_faces):03d}",
                                "vector": face.embedding.tolist(),  # ArcFace embedding
                                "faceAngle": face_angles,
                                "qualityScore": float(quality_score),
                                "detectScore": float(face.det_score),
                                "captureDate": datetime.now().isoformat(),
                                "sourceFrame": os.path.basename(frame_filename)
                            }
                            
                            captured_faces.append(captured_face)
                            captured_angles.append(face_angles)
                            
                            # Update last capture time
                            last_capture_time = current_time
            
            # Update status
            face_count = len(faces)
            if face_count == 0:
                status_text = "No faces detected. Please position yourself in front of the camera."
            elif face_count > 1:
                status_text = "Multiple faces detected. Please ensure only one person is in frame."
            else:
                status_text = f"Face detected. Captured {len(captured_faces)} unique angles."
            
            status_placeholder.text(status_text)
            
            # Show progress
            if MIN_FACE_FRAMES > 0:
                progress = min(1.0, len(captured_faces) / MIN_FACE_FRAMES)
                progress_placeholder.progress(progress)
            
            # Display thumbnails of captured faces
            if captured_faces:
                # Read the saved images to display
                face_images = []
                for i, face_data in enumerate(captured_faces[-6:]):  # Show last 6 faces
                    frame_path = f"{FRAMES_DIR}/{staff_id}/{face_data['sourceFrame']}"
                    if os.path.exists(frame_path):
                        img = cv2.imread(frame_path)
                        if img is not None:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            # Add angle info
                            angle_text = f"Yaw: {face_data['faceAngle']['yaw']:.1f}°"
                            cv2.putText(img, angle_text, (5, 15), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                            face_images.append(img)
                
                # Combine images into a grid
                if face_images:
                    # Create a grid (2x3 or fewer)
                    cols = min(3, len(face_images))
                    rows = (len(face_images) + cols - 1) // cols
                    
                    # Find max dimensions
                    max_h = max(img.shape[0] for img in face_images)
                    max_w = max(img.shape[1] for img in face_images)
                    
                    # Create a blank canvas
                    grid = np.zeros((rows * max_h, cols * max_w, 3), dtype=np.uint8)
                    
                    # Place images
                    for i, img in enumerate(face_images):
                        row = i // cols
                        col = i % cols
                        h, w = img.shape[:2]
                        
                        y_offset = row * max_h
                        x_offset = col * max_w
                        
                        # Center the image in its cell
                        y_start = y_offset + (max_h - h) // 2
                        x_start = x_offset + (max_w - w) // 2
                        
                        grid[y_start:y_start+h, x_start:x_start+w] = img
                    
                    # Display the grid
                    faces_placeholder.image(grid, caption=f"Captured Faces ({len(captured_faces)} total)", use_column_width=True)
            
            # Frame rate control
            time.sleep(0.03)  # ~30 FPS
            frame_count += 1
            
            # Auto-stop if we have enough faces
            if len(captured_faces) >= MIN_FACE_FRAMES * 1.5:  # Collect a few extra for good measure
                status_placeholder.success(f"Successfully captured {len(captured_faces)} face angles!")
                break
        
        # Release the camera
        cap.release()
        
        return captured_faces

    def enroll_staff(self, staff_id, first_name, last_name, email, position, department, progress_placeholder):
        """Enroll a staff member using the captured face data"""
        # Create staff record
        staff_record = {
            "_id": str(uuid.uuid4()),
            "staffId": staff_id,
            "firstName": first_name,
            "lastName": last_name,
            "email": email,
            "position": position,
            "department": department,
            "status": "active",
            "enrollmentDate": datetime.now().isoformat(),
            "lastUpdated": datetime.now().isoformat()
        }
        
        # Process face embeddings
        embeddings_record = {
            "_id": str(uuid.uuid4()),
            "staffId": staff_id,
            "embeddings": [],
            "modelVersion": "arcface_r100",  # ArcFace ResNet100
            "vectorDimension": FACE_EMBEDDING_SIZE,  # ArcFace embedding size
            "lastUpdated": datetime.now().isoformat()
        }
        
        # Read all face images and process them
        staff_dir = f"{FRAMES_DIR}/{staff_id}"
        all_frames = [f for f in os.listdir(staff_dir) if f.endswith('.jpg')]
        
        progress_placeholder.text("Processing face embeddings...")
        
        for i, frame_file in enumerate(tqdm(all_frames)):
            frame_path = os.path.join(staff_dir, frame_file)
            
            # Read image and convert to RGB
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process with InsightFace
            faces = self.face_app.get(rgb_frame)
            if not faces:
                continue
                
            face = faces[0]  # Should be only one face per saved image
            
            # Get quality score
            quality_score = get_face_quality_score(face)
            
            # Get face angles
            if face.landmark_2d_106 is not None:
                face_angles = estimate_face_angles(face.landmark_2d_106)
            else:
                face_angles = {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
            
            # Add to embeddings
            embeddings_record["embeddings"].append({
                "embeddingId": f"emb_{staff_id}_{i:03d}",
                "vector": face.embedding.tolist(),
                "faceAngle": face_angles,
                "qualityScore": float(quality_score),
                "detectScore": float(face.det_score),
                "captureDate": datetime.fromtimestamp(os.path.getmtime(frame_path)).isoformat(),
                "sourceFrame": frame_file
            })
        
        # Update database
        self.db["staff"].append(staff_record)
        self.db["staff_embeddings"].append(embeddings_record)
        save_database(self.db)
        
        progress_placeholder.text(f"Enrollment complete! {len(embeddings_record['embeddings'])} face embeddings stored.")


if __name__ == "__main__":
    app = StaffEnrollmentApp()
    app.run()