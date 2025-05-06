"""
Staff Enrollment Module - CLI Version

This script captures video from a laptop camera, extracts facial features,
and stores them in a local JSON file (simulating MongoDB storage).
"""

import cv2
import face_recognition
import numpy as np
import os
import json
import time
import uuid
from datetime import datetime
from tqdm import tqdm
import shutil
import random
import string
from facenet_pytorch import MTCNN
import torch

# Configuration
FACE_DETECTION_MODEL = "mtcnn"  # Changed from "hog" to "mtcnn" for consistency
MIN_FACE_FRAMES = 10  # Minimum number of good quality face frames required
STORAGE_DIR = "face_recognition_data"
FRAMES_DIR = f"{STORAGE_DIR}/staff_frames"
DB_FILE = f"{STORAGE_DIR}/staff_database.json"
TARGET_FRAMES = 15  # Total frames to capture across different angles

# Colors for visualization
BLUE = (255, 0, 0)        # BGR format
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)
os.makedirs(FRAMES_DIR, exist_ok=True)

# Initialize MTCNN detector
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn_detector = MTCNN(
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    device=device,
    keep_all=True
)

# Import utilities
from utils import draw_stylish_box

def load_database():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    else:
        db = {"staff": [], "staff_embeddings": []}
        save_database(db)
        return db


def save_database(db):
    with open(DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)


def get_face_quality_score(face_image):
    height, width = face_image.shape[:2]
    size_score = min(1.0, (height * width) / (100 * 100))
    gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
    blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    blur_score = min(1.0, blur_variance / 500)
    return 0.5 * size_score + 0.5 * blur_score


def detect_faces_mtcnn(frame):
    """Detect faces using MTCNN"""
    boxes, probs, landmarks = mtcnn_detector.detect(frame, landmarks=True)
    
    # Convert to face_recognition format (top, right, bottom, left)
    face_locations = []
    if boxes is not None:
        for box in boxes:
            left, top, right, bottom = map(int, box)
            face_locations.append((top, right, bottom, left))
    
    return face_locations, probs, landmarks


def extract_face_embeddings(frame, face_locations):
    face_data = []
    if not face_locations:
        return face_data
    encodings = face_recognition.face_encodings(frame, face_locations)
    for encoding, loc in zip(encodings, face_locations):
        top, right, bottom, left = loc
        face_image = frame[top:bottom, left:right]
        quality = get_face_quality_score(face_image)
        face_data.append({
            "embedding": encoding.tolist(),  # Ensure we're storing as list for consistency
            "quality_score": quality,
            "location": {"top": top, "right": right, "bottom": bottom, "left": left},
            "size": {"width": right - left, "height": bottom - top}
        })
    return face_data


def get_facial_landmarks(frame, face_location):
    """
    Get facial landmarks for a detected face.
    This function supports both face_recognition and MTCNN formats.
    
    Args:
        frame: Frame containing the face
        face_location: Location of face (top, right, bottom, left)
        
    Returns:
        Dictionary with landmark information
    """
    top, right, bottom, left = face_location
    
    try:
        # Try to get landmarks using face_recognition library
        landmarks = face_recognition.face_landmarks(frame, [(top, right, bottom, left)])
        
        if landmarks and len(landmarks) > 0:
            # Successfully got landmarks from face_recognition
            landmarks = landmarks[0]  # Get the first face's landmarks
            
            nose_bridge = landmarks.get('nose_bridge', [])
            nose_tip = nose_bridge[-1] if nose_bridge else [(left + right)//2, (top + bottom)//2]
            left_eye_points = landmarks.get('left_eye', [])
            right_eye_points = landmarks.get('right_eye', [])
            
            # For easier access and visualization
            left_eye = [
                sum(x for x, _ in left_eye_points)/len(left_eye_points) if left_eye_points else left + (right-left)//4,
                sum(y for _, y in left_eye_points)/len(left_eye_points) if left_eye_points else top + (bottom-top)//3
            ]
            right_eye = [
                sum(x for x, _ in right_eye_points)/len(right_eye_points) if right_eye_points else right - (right-left)//4,
                sum(y for _, y in right_eye_points)/len(right_eye_points) if right_eye_points else top + (bottom-top)//3
            ]
            
            return {"full_landmarks": landmarks, "left_eye": left_eye, "right_eye": right_eye, "nose_tip": nose_tip}
        
        else:
            # If face_recognition failed, create basic landmark structure
            center_x, center_y = (left + right) // 2, (top + bottom) // 2
            eye_y = top + (bottom - top) // 3
            mouth_y = bottom - (bottom - top) // 3
            
            # Rough landmark estimates based on face geometry
            basic_landmarks = {
                "left_eye": [left + (right - left) // 4, eye_y],
                "right_eye": [right - (right - left) // 4, eye_y],
                "nose_tip": [center_x, center_y],
                "full_landmarks": {
                    "left_eye": [[left + (right - left) // 4, eye_y]],
                    "right_eye": [[right - (right - left) // 4, eye_y]],
                    "nose_bridge": [[center_x, center_y]],
                    "top_lip": [[center_x - (right - left) // 4, mouth_y], [center_x + (right - left) // 4, mouth_y]],
                    "bottom_lip": [[center_x - (right - left) // 4, mouth_y + 10], [center_x + (right - left) // 4, mouth_y + 10]]
                }
            }
            return basic_landmarks
    
    except Exception as e:
        print(f"Warning: Failed to get facial landmarks: {e}")
        
        # Fallback to basic landmark structure
        center_x, center_y = (left + right) // 2, (top + bottom) // 2
        eye_y = top + (bottom - top) // 3
        
        basic_landmarks = {
            "left_eye": [left + (right - left) // 4, eye_y],
            "right_eye": [right - (right - left) // 4, eye_y],
            "nose_tip": [center_x, center_y],
            "full_landmarks": {
                "left_eye": [[left + (right - left) // 4, eye_y]],
                "right_eye": [[right - (right - left) // 4, eye_y]],
                "nose_bridge": [[center_x, center_y]],
                "top_lip": [[center_x - 20, bottom - 20], [center_x + 20, bottom - 20]],
                "bottom_lip": [[center_x - 20, bottom - 10], [center_x + 20, bottom - 10]]
            }
        }
        return basic_landmarks


def estimate_face_angle(landmarks):
    left_eye = landmarks['left_eye']
    right_eye = landmarks['right_eye']
    nose_tip = landmarks['nose_tip']
    eye_mid = [(left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2]
    yaw = (nose_tip[0] - eye_mid[0]) * 45
    pitch = (nose_tip[1] - eye_mid[1]) * 30
    if right_eye[1] != left_eye[1]:
        roll = np.arctan((right_eye[1]-left_eye[1])/(right_eye[0]-left_eye[0])) * 180/np.pi
    else:
        roll = 0
    return {"yaw": yaw, "pitch": pitch, "roll": roll}


def draw_landmarks(frame, landmarks):
    """Draw facial landmarks on the frame for visualization"""
    for feature, points in landmarks["full_landmarks"].items():
        color = BLUE
        if feature == "left_eye" or feature == "right_eye":
            color = GREEN
        elif feature == "nose_bridge" or feature == "nose_tip":
            color = RED
        elif feature == "top_lip" or feature == "bottom_lip":
            color = YELLOW
            
        # Draw points
        for point in points:
            cv2.circle(frame, point, 2, color, -1)
            
        # Connect points with lines for better visualization
        if len(points) > 1:
            pts = np.array(points, np.int32)
            cv2.polylines(frame, [pts], False, color, 1)
    
    return frame


def generate_staff_id():
    """Generate a unique staff ID with timestamp and random characters"""
    timestamp = int(time.time()) % 10000  # last 4 digits of timestamp
    letters = ''.join(random.choice(string.ascii_uppercase) for _ in range(3))
    return f"ST{letters}{timestamp}"


def get_instruction_text(captured_count, face_angle=None):
    """Generate instruction text based on current capture progress"""
    if captured_count == 0:
        return "Look straight at the camera"
    elif captured_count < 3:
        return "Turn your head slightly left"
    elif captured_count < 6:
        return "Turn your head slightly right"
    elif captured_count < 9:
        return "Look slightly up"
    elif captured_count < 12:
        return "Look slightly down"
    else:
        return "Almost done! Vary your expression slightly"


def create_info_panel(captured_count, target_count):
    """
    Create a stylish info panel for enrollment progress
    
    Args:
        captured_count: Number of faces captured so far
        target_count: Target number of faces to capture
        
    Returns:
        Styled info panel image
    """
    # Define panel size and colors
    panel_width = 640
    panel_height = 150
    
    # Colors
    BACKGROUND = (30, 30, 30)
    TEXT_COLOR = (255, 255, 255)
    PROGRESS_BG = (50, 50, 50)
    PROGRESS_FG = (0, 200, 100)
    ACCENT = (0, 120, 255)
    
    # Create panel with dark background
    panel = np.ones((panel_height, panel_width, 3), dtype=np.uint8) * 30
    
    # Add title with modern font
    title = "Staff Enrollment Progress"
    cv2.putText(panel, title, (20, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.8, ACCENT, 2, cv2.LINE_AA)
    
    # Add progress text with clean font
    progress_text = f"Captured: {captured_count}/{target_count} frames"
    cv2.putText(panel, progress_text, (20, 60), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)
    
    # Calculate progress
    progress = min(1.0, captured_count / target_count)
    
    # Draw modern progress bar
    bar_width = int(panel_width * 0.8)
    bar_height = 15
    bar_x = int(panel_width * 0.1)
    bar_y = 80
    
    # Draw rounded background bar
    cv2.rectangle(panel, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                 PROGRESS_BG, cv2.FILLED, cv2.LINE_AA)
    
    # Draw filled progress
    filled_width = int(bar_width * progress)
    if filled_width > 0:
        cv2.rectangle(panel, (bar_x, bar_y), (bar_x + filled_width, bar_y + bar_height), 
                     PROGRESS_FG, cv2.FILLED, cv2.LINE_AA)
    
    # Add percentage
    percent_text = f"{int(progress * 100)}%"
    cv2.putText(panel, percent_text, (bar_x + bar_width + 10, bar_y + 12), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    
    # Add instructions
    instructions = "Position your face in the guide box and follow the angle instructions"
    cv2.putText(panel, instructions, (20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1, cv2.LINE_AA)
    
    # Add quit instruction
    quit_text = "Press 'q' to quit"
    text_size = cv2.getTextSize(quit_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
    cv2.putText(panel, quit_text, (panel_width - text_size[0] - 20, 120), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, ACCENT, 1, cv2.LINE_AA)
    
    return panel


def capture_video(staff_id):
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't open camera.")
        return []
    
    # Set lower resolution for faster processing
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create directory for staff face images
    staff_dir = os.path.join(FRAMES_DIR, staff_id)
    os.makedirs(staff_dir, exist_ok=True)
    
    # Set up tracking variables
    last_time = time.time()
    interval = 0.5  # seconds between captures
    captured = []
    target_angles = [
        {"yaw": 0, "pitch": 0},    # Front
        {"yaw": -30, "pitch": 0},  # Left
        {"yaw": 30, "pitch": 0},   # Right
        {"yaw": 0, "pitch": 15},   # Up
        {"yaw": 0, "pitch": -15}   # Down
    ]
    
    # Color definitions
    RED = (0, 0, 255)
    GREEN = (0, 255, 0)
    BLUE = (255, 0, 0)
    YELLOW = (0, 255, 255)
    
    print("Starting face capture. Press 'q' to stop.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to grab frame.")
            break
        
        # Get frame dimensions
        h, w = frame.shape[:2]
        
        # Create info panel
        info_panel = create_info_panel(len(captured), TARGET_FRAMES)
        
        # Convert to RGB (face_recognition uses RGB)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces using MTCNN (faster than HOG for this purpose)
        locs, probs, landmarks_list = detect_faces_mtcnn(rgb)
        
        # Make a copy for display with instructions
        display_frame = frame.copy()
        
        # Draw guide box to help position face
        guide_padding = 50
        guide_left = w // 2 - 100
        guide_top = h // 2 - 125
        guide_right = w // 2 + 100
        guide_bottom = h // 2 + 125
        
        # Draw stylish guide box with anti-aliased edges
        cv2.rectangle(display_frame, (guide_left, guide_top), (guide_right, guide_bottom), BLUE, 1, cv2.LINE_AA)
        
        # Add corner markers for the guide box
        corner_length = 20
        line_thickness = 1
        
        # Top-left corner
        cv2.line(display_frame, (guide_left, guide_top), (guide_left + corner_length, guide_top), BLUE, line_thickness, cv2.LINE_AA)
        cv2.line(display_frame, (guide_left, guide_top), (guide_left, guide_top + corner_length), BLUE, line_thickness, cv2.LINE_AA)
        
        # Top-right corner
        cv2.line(display_frame, (guide_right, guide_top), (guide_right - corner_length, guide_top), BLUE, line_thickness, cv2.LINE_AA)
        cv2.line(display_frame, (guide_right, guide_top), (guide_right, guide_top + corner_length), BLUE, line_thickness, cv2.LINE_AA)
        
        # Bottom-left corner
        cv2.line(display_frame, (guide_left, guide_bottom), (guide_left + corner_length, guide_bottom), BLUE, line_thickness, cv2.LINE_AA)
        cv2.line(display_frame, (guide_left, guide_bottom), (guide_left, guide_bottom - corner_length), BLUE, line_thickness, cv2.LINE_AA)
        
        # Bottom-right corner
        cv2.line(display_frame, (guide_right, guide_bottom), (guide_right - corner_length, guide_bottom), BLUE, line_thickness, cv2.LINE_AA)
        cv2.line(display_frame, (guide_right, guide_bottom), (guide_right, guide_bottom - corner_length), BLUE, line_thickness, cv2.LINE_AA)
        
        # Display instruction text
        instruction_text = get_instruction_text(len(captured))
        
        # Add instruction with styled text
        cv2.putText(display_frame, instruction_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(display_frame, instruction_text, (20, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Process exactly one face
        if len(locs) == 1:
            face_location = locs[0]
            top, right, bottom, left = face_location
            
            # Get landmarks for the detected face
            landmarks = get_facial_landmarks(rgb, face_location)
            
            # Create label for the current angle
            angle = estimate_face_angle(landmarks)
            
            # Determine which target angle we're working on
            angle_index = min(len(captured) // 3, len(target_angles) - 1)
            target = target_angles[angle_index]
            
            # Determine if the face is at a good angle for capture
            yaw_diff = abs(angle["yaw"] - target["yaw"])
            pitch_diff = abs(angle["pitch"] - target["pitch"])
            good_angle = yaw_diff < 10 and pitch_diff < 10
            
            # Set color based on face position quality
            box_color = GREEN if good_angle else YELLOW
            
            # Draw face box with the utility
            label = "Face Detected"
            angle_text = f"Yaw: {angle['yaw']:.1f}, Pitch: {angle['pitch']:.1f}"
            
            display_frame = draw_stylish_box(
                display_frame, 
                face_location, 
                label, 
                box_color, 
                secondary_text=angle_text
            )
            
            # Capture face if it's time
            now = time.time()
            if now - last_time >= interval:
                data = extract_face_embeddings(rgb, locs)[0]
                if data["quality_score"] >= 0.5:
                    # Check if we already have a similar face angle
                    unique = all(abs(f["faceAngle"]["yaw"] - angle["yaw"]) >= 10 for f in captured) or len(captured) < 3
                    
                    if unique or len(captured) % 3 == 0:  # Always capture first few frames or every 3rd frame
                        face_img = rgb[top:bottom, left:right]
                        fname = os.path.join(staff_dir, f"frame_{len(captured):03d}.jpg")
                        cv2.imwrite(fname, cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
                        
                        # Save landmarks along with other data
                        captured.append({
                            "embeddingId": f"emb_{staff_id}_{len(captured):03d}",
                            "vector": data["embedding"],
                            "faceAngle": angle,
                            "qualityScore": data["quality_score"],
                            "captureDate": datetime.now().isoformat(),
                            "sourceFrame": os.path.basename(fname),
                            "keypoints": landmarks["full_landmarks"]  # Store the actual landmark points
                        })
                        
                        last_time = now
                        
                        # Flash green to indicate capture
                        cv2.rectangle(display_frame, (0, 0), (w, h), (0, 255, 0), 10)
        else:
            # Guide user to position face
            if len(locs) == 0:
                msg = "No face detected. Please position yourself in front of the camera."
            else:
                msg = "Multiple faces detected. Please ensure only one person is in frame."
                
            # Add message with shadow for better readability
            cv2.putText(display_frame, msg, (31, 71), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, msg, (30, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, RED, 2, cv2.LINE_AA)
        
        # Combine frame and info panel
        combined = np.vstack([display_frame, info_panel])
        
        # Show the frame
        cv2.imshow("Staff Enrollment - Face Capture", combined)
        
        # Check for user quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Stopping capture.")
            break
            
        # Auto-stop if we have enough faces
        if len(captured) >= TARGET_FRAMES:
            print(f"Captured {len(captured)} face angles. Auto-stopping.")
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    return captured


def enroll_staff(staff_id, first_name, last_name, email, position, department, captured):
    db = load_database()
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
    
    embeddings_record = {
        "_id": str(uuid.uuid4()),
        "staffId": staff_id,
        "embeddings": [],
        "modelVersion": "face_recognition_v1",
        "vectorDimension": 128,
        "lastUpdated": datetime.now().isoformat()
    }
    
    staff_dir = os.path.join(FRAMES_DIR, staff_id)
    all_frames = sorted([f for f in os.listdir(staff_dir) if f.endswith('.jpg')])
    print("Processing face embeddings...")
    
    for i, frame_file in enumerate(tqdm(all_frames)):
        path = os.path.join(staff_dir, frame_file)
        frame = cv2.imread(path)
        if frame is None: continue
        
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Use MTCNN for detection
        locs, _, _ = detect_faces_mtcnn(rgb)
        if not locs: continue
        
        data = extract_face_embeddings(rgb, locs)
        if not data: continue
        
        landmarks = get_facial_landmarks(rgb, locs[0])
        angle = estimate_face_angle(landmarks)
        
        # Store both embeddings and landmarks
        # Ensure embeddings are stored as numpy arrays converted to lists
        embeddings_record["embeddings"].append({
            "embeddingId": f"emb_{staff_id}_{i:03d}",
            "vector": data[0]["embedding"],  # This is already a list from extract_face_embeddings
            "faceAngle": angle,
            "qualityScore": data[0]["quality_score"],
            "captureDate": datetime.fromtimestamp(os.path.getmtime(path)).isoformat(),
            "sourceFrame": frame_file,
            "keypoints": landmarks["full_landmarks"]  # Store facial keypoints
        })
    
    # Add records to database
    db["staff"].append(staff_record)
    db["staff_embeddings"].append(embeddings_record)
    save_database(db)
    
    print(f"Enrollment complete! {len(embeddings_record['embeddings'])} face embeddings stored.")
    print(f"Staff ID: {staff_id} has been successfully enrolled.")


def run():
    print("=== Staff Enrollment System ===")
    print("This system will capture your face from multiple angles for enrollment.")
    
    # Load database to check for existing staff IDs
    db = load_database()
    
    # Generate a unique staff ID
    staff_id = generate_staff_id()
    while any(s["staffId"] == staff_id for s in db["staff"]):
        staff_id = generate_staff_id()
    
    print(f"Auto-generated Staff ID: {staff_id}")
    change_id = input("Would you like to use a different ID? (y/n): ").strip().lower()
    
    if change_id == 'y':
        custom_id = input("Enter custom Staff ID: ").strip()
        while any(s["staffId"] == custom_id for s in db["staff"]):
            custom_id = input("ID already exists. Enter a unique Staff ID: ").strip()
        staff_id = custom_id
    
    # Collect staff information
    print("\nEnter staff information:")
    first_name = input("First Name: ").strip()
    last_name = input("Last Name: ").strip()
    email = input("Email: ").strip()
    position = input("Position: ").strip()
    department = input("Department: ").strip()
    
    print("\nPreparation for face capture:")
    print("- Ensure good lighting")
    print("- Remove glasses, hats, or other face coverings")
    print("- Follow on-screen instructions for head positioning")
    print("- Try to maintain a neutral expression")
    input("\nPress Enter to begin face capture...")
    
    # Capture video and face data
    captured = capture_video(staff_id)
    
    # Check if we have enough good quality images
    if len(captured) < MIN_FACE_FRAMES:
        print(f"Not enough face images captured ({len(captured)}). Need at least {MIN_FACE_FRAMES}.")
        return
    
    # Confirm enrollment
    print(f"\nCaptured {len(captured)} face angles.")
    confirm = input("Proceed with enrollment? (y/n): ").strip().lower()
    
    if confirm != 'y':
        print("Enrollment canceled.")
        return
    
    # Process and store the enrollment data
    print("Processing enrollment data...")
    enroll_staff(staff_id, first_name, last_name, email, position, department, captured)
    
    # Show enrolled staff list
    print("\nCurrently enrolled staff:")
    for idx, staff in enumerate(db["staff"], 1):
        print(f"{idx}. {staff['firstName']} {staff['lastName']} ({staff['staffId']})")
    
    print("\nThank you for using the Staff Enrollment System!")


if __name__ == "__main__":
    run()