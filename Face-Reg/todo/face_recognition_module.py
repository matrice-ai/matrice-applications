"""
Face Recognition Module

This module implements real-time face recognition and tracking using data 
from the enrollment database. It processes video streams, detects faces,
and matches them against known staff and customer databases.
"""

import cv2
import face_recognition
import numpy as np
import os
import json
import time
import uuid
from datetime import datetime
from collections import defaultdict, deque
import threading
import logging

# Configuration
STORAGE_DIR = "face_recognition_data"
DB_FILE = f"{STORAGE_DIR}/staff_database.json"
CUSTOMER_DB_FILE = f"{STORAGE_DIR}/customer_database.json"
VISIT_LOG_FILE = f"{STORAGE_DIR}/visit_logs.json"
FACE_DETECTION_MODEL = "hog"  # Options: "hog" (CPU) or "cnn" (GPU)
FRAME_PROCESS_INTERVAL = 2  # Process every n frames
FACE_MATCH_THRESHOLD = 0.6  # Lower is more strict (0.6 recommended)
MIN_FACE_QUALITY = 0.5  # Minimum quality threshold for face tracking
UNKNOWN_FACE_TRACK_SECONDS = 1800  # 30 minutes before unknown face enrollment trigger
UNKNOWN_FACE_RESET_SECONDS = 300  # 5 minutes before resetting tracking if face disappears

# Colors for visualization
BLUE = (255, 0, 0)        # BGR format
GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)
PURPLE = (255, 0, 255)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename=f'{STORAGE_DIR}/face_recognition.log',
    filemode='a'
)
logger = logging.getLogger('face_recognition')

# Create directories if they don't exist
os.makedirs(STORAGE_DIR, exist_ok=True)

# Initialize databases
def initialize_customer_db():
    if os.path.exists(CUSTOMER_DB_FILE):
        with open(CUSTOMER_DB_FILE, 'r') as f:
            return json.load(f)
    else:
        db = {
            "customers": [],
            "customer_embeddings": []
        }
        with open(CUSTOMER_DB_FILE, 'w') as f:
            json.dump(db, f, indent=4)
        return db

def initialize_visit_log():
    if os.path.exists(VISIT_LOG_FILE):
        with open(VISIT_LOG_FILE, 'r') as f:
            return json.load(f)
    else:
        logs = {
            "visits": [],
            "statistics": {
                "total_visits": 0,
                "unique_customers": 0,
                "repeat_customers": 0,
                "staff_entries": 0
            }
        }
        with open(VISIT_LOG_FILE, 'w') as f:
            json.dump(logs, f, indent=4)
        return logs

def load_staff_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE, 'r') as f:
            return json.load(f)
    else:
        logger.warning(f"Staff database not found at {DB_FILE}")
        return {"staff": [], "staff_embeddings": []}

def save_customer_db(db):
    with open(CUSTOMER_DB_FILE, 'w') as f:
        json.dump(db, f, indent=4)

def save_visit_log(logs):
    with open(VISIT_LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=4)

def get_face_quality_score(face_image):
    """Calculate quality score for a face (size, blur, etc)"""
    height, width = face_image.shape[:2]
    size_score = min(1.0, (height * width) / (100 * 100))
    
    # Blur detection
    try:
        gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
        blur_variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, blur_variance / 500)
    except Exception:
        blur_score = 0.3  # Default if conversion fails
    
    return 0.5 * size_score + 0.5 * blur_score

def extract_face_embeddings(frame, face_locations):
    """Extract face embeddings from detected faces"""
    face_data = []
    if not face_locations:
        return face_data
    
    try:
        # Get face encodings
        encodings = face_recognition.face_encodings(frame, face_locations)
        
        for encoding, loc in zip(encodings, face_locations):
            top, right, bottom, left = loc
            face_image = frame[top:bottom, left:right]
            
            # Calculate quality score
            quality = get_face_quality_score(face_image)
            
            face_data.append({
                "embedding": encoding,  # Keep as numpy array for faster comparison
                "embedding_list": encoding.tolist(),  # JSON serializable version
                "quality_score": quality,
                "location": {"top": top, "right": right, "bottom": bottom, "left": left},
                "size": {"width": right - left, "height": bottom - top}
            })
    except Exception as e:
        logger.error(f"Error extracting face embeddings: {str(e)}")
    
    return face_data

def calculate_face_distance(known_embedding, face_embedding):
    """Calculate the face distance between embeddings"""
    if isinstance(known_embedding, list):
        known_embedding = np.array(known_embedding)
    
    return face_recognition.face_distance([known_embedding], face_embedding)[0]

def find_matching_staff(staff_db, face_embedding, threshold=FACE_MATCH_THRESHOLD):
    """Find matching staff in the database for a given face embedding"""
    best_match = None
    best_distance = float('inf')
    best_embedding_id = None
    
    # Go through each staff member's embeddings
    for staff_emb in staff_db.get("staff_embeddings", []):
        for embedding in staff_emb.get("embeddings", []):
            vector = embedding.get("vector", [])
            if not vector:
                continue
                
            # Calculate face distance
            distance = calculate_face_distance(vector, face_embedding)
            
            # Update best match if this is better
            if distance < best_distance:
                best_distance = distance
                best_embedding_id = embedding.get("embeddingId")
                
                # Find the staff details
                staff_id = staff_emb.get("staffId")
                staff_info = next((s for s in staff_db.get("staff", []) if s.get("staffId") == staff_id), None)
                
                if staff_info:
                    best_match = {
                        "staffId": staff_id,
                        "firstName": staff_info.get("firstName", ""),
                        "lastName": staff_info.get("lastName", ""),
                        "position": staff_info.get("position", ""),
                        "department": staff_info.get("department", ""),
                        "embeddingId": best_embedding_id,
                        "confidence": 1.0 - distance  # Convert distance to confidence score
                    }
    
    # Return the best match if it meets the threshold
    if best_match and best_distance <= threshold:
        return best_match
    
    return None

def find_matching_customer(customer_db, face_embedding, threshold=FACE_MATCH_THRESHOLD):
    """Find matching customer in the database for a given face embedding"""
    best_match = None
    best_distance = float('inf')
    best_embedding_id = None
    
    # Go through each customer's embeddings
    for cust_emb in customer_db.get("customer_embeddings", []):
        for embedding in cust_emb.get("embeddings", []):
            vector = embedding.get("vector", [])
            if not vector:
                continue
                
            # Calculate face distance
            distance = calculate_face_distance(vector, face_embedding)
            
            # Update best match if this is better
            if distance < best_distance:
                best_distance = distance
                best_embedding_id = embedding.get("embeddingId")
                
                # Find the customer details
                customer_id = cust_emb.get("customerId")
                customer_info = next((c for c in customer_db.get("customers", []) if c.get("customerId") == customer_id), None)
                
                if customer_info:
                    best_match = {
                        "customerId": customer_id,
                        "visitCount": customer_info.get("visitCount", 0),
                        "firstSeen": customer_info.get("firstSeen", ""),
                        "embeddingId": best_embedding_id,
                        "confidence": 1.0 - distance  # Convert distance to confidence score
                    }
    
    # Return the best match if it meets the threshold
    if best_match and best_distance <= threshold:
        return best_match
    
    return None

def add_or_update_customer(customer_db, face_data, location="Unknown"):
    """Add a new customer or update an existing one"""
    # Generate a new customer ID
    customer_id = f"CUST_{uuid.uuid4().hex[:8]}"
    now = datetime.now().isoformat()
    
    # Create customer record
    customer_record = {
        "_id": str(uuid.uuid4()),
        "customerId": customer_id,
        "type": "anonymous",
        "firstSeen": now,
        "lastSeen": now,
        "visitCount": 1,
        "visitHistory": [{
            "timestamp": now,
            "location": location,
            "duration": 0
        }],
        "status": "active"
    }
    
    # Create embedding record
    embedding_record = {
        "_id": str(uuid.uuid4()),
        "customerId": customer_id,
        "embeddings": [{
            "embeddingId": f"emb_{customer_id}_001",
            "vector": face_data["embedding_list"],
            "qualityScore": face_data["quality_score"],
            "captureDate": now,
            "location": location
        }],
        "modelVersion": "face_recognition_v1",
        "vectorDimension": 128,
        "lastUpdated": now
    }
    
    # Add to database
    customer_db["customers"].append(customer_record)
    customer_db["customer_embeddings"].append(embedding_record)
    
    # Save the updated database
    save_customer_db(customer_db)
    
    logger.info(f"Added new customer: {customer_id}")
    return customer_record

def update_existing_customer(customer_db, customer_match, face_data, location="Unknown"):
    """Update an existing customer record with new visit information"""
    now = datetime.now().isoformat()
    customer_id = customer_match["customerId"]
    
    # Find and update customer record
    for customer in customer_db["customers"]:
        if customer["customerId"] == customer_id:
            customer["lastSeen"] = now
            customer["visitCount"] += 1
            customer["visitHistory"].append({
                "timestamp": now,
                "location": location,
                "duration": 0
            })
            break
    
    # Update embedding if quality is better
    for emb_record in customer_db["customer_embeddings"]:
        if emb_record["customerId"] == customer_id:
            # Check if this is a higher quality face
            existing_quality = max([e.get("qualityScore", 0) for e in emb_record["embeddings"]], default=0)
            
            if face_data["quality_score"] > existing_quality + 0.1:  # Only add if significantly better
                new_embedding = {
                    "embeddingId": f"emb_{customer_id}_{len(emb_record['embeddings']) + 1:03d}",
                    "vector": face_data["embedding_list"],
                    "qualityScore": face_data["quality_score"],
                    "captureDate": now,
                    "location": location
                }
                emb_record["embeddings"].append(new_embedding)
                emb_record["lastUpdated"] = now
                logger.info(f"Added new embedding for customer: {customer_id}")
            break
    
    # Save the updated database
    save_customer_db(customer_db)
    logger.info(f"Updated customer: {customer_id}, visit count: {customer['visitCount']}")
    return customer_id

def log_recognition_event(visit_logs, person_type, person_id, confidence, location="Unknown"):
    """Log a recognition event"""
    now = datetime.now().isoformat()
    
    visit_entry = {
        "visitId": str(uuid.uuid4()),
        "timestamp": now,
        "type": person_type,  # "staff" or "customer"
        "personId": person_id,
        "confidence": confidence,
        "location": location
    }
    
    # Add to visit logs
    visit_logs["visits"].append(visit_entry)
    
    # Update statistics
    if person_type == "staff":
        visit_logs["statistics"]["staff_entries"] += 1
    else:  # customer
        visit_logs["statistics"]["total_visits"] += 1
    
    # Save the updated logs
    save_visit_log(visit_logs)
    return visit_entry

def draw_face_box(frame, face_loc, text="", color=GREEN, confidence=None):
    """Draw a box around a face with optional text label"""
    top, right, bottom, left = face_loc
    
    # Draw box
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Add text background
    if text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)[0]
        cv2.rectangle(frame, (left, top - text_size[1] - 10), (left + text_size[0] + 10, top), color, -1)
        cv2.putText(frame, text, (left + 5, top - 5), cv2.FONT_HERSHEY_DUPLEX, 0.6, WHITE, 1)
    
    # Add confidence if provided
    if confidence is not None:
        conf_text = f"{confidence:.2f}"
        cv2.putText(frame, conf_text, (left + 5, bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

class FaceRecognitionSystem:
    def __init__(self, camera_id=0, location="Main Entrance"):
        self.camera_id = camera_id
        self.location = location
        
        # Load databases
        self.staff_db = load_staff_db()
        self.customer_db = initialize_customer_db()
        self.visit_logs = initialize_visit_log()
        
        # Face tracking state
        self.tracked_faces = {}  # Dictionary of tracked faces
        self.unknown_face_ids = 0  # Counter for assigning temporary IDs
        
        # Performance tracking
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.frame_count = 0
        
        # Processing locks
        self.db_lock = threading.Lock()
        
        logger.info(f"FaceRecognitionSystem initialized for camera ID {camera_id} at {location}")
    
    def track_unknown_face(self, face_data):
        """Track an unknown face - determine if it's a new tracking entry or update existing"""
        current_time = time.time()
        unknown_id = None
        
        # Calculate face center point for tracking
        loc = face_data["location"]
        face_center = (
            (loc["left"] + loc["right"]) // 2,
            (loc["top"] + loc["bottom"]) // 2
        )
        
        # Search for existing tracked faces with similar position
        best_match_id = None
        best_distance = float('inf')
        
        for face_id, tracked in list(self.tracked_faces.items()):
            # Skip already identified faces
            if tracked["identified"]:
                continue
                
            # Calculate center point distance
            tx = (tracked["bbox"]["left"] + tracked["bbox"]["right"]) // 2
            ty = (tracked["bbox"]["top"] + tracked["bbox"]["bottom"]) // 2
            distance = np.sqrt((tx - face_center[0])**2 + (ty - face_center[1])**2)
            
            # Update best match
            if distance < best_distance:
                best_distance = distance
                best_match_id = face_id
            
            # Remove expired tracks
            if current_time - tracked["last_seen"] > UNKNOWN_FACE_RESET_SECONDS:
                logger.info(f"Removing expired tracking for {face_id}")
                del self.tracked_faces[face_id]
        
        # Use existing track if good match found (within reasonable distance)
        face_width = loc["right"] - loc["left"]
        if best_match_id and best_distance < face_width * 0.8:
            unknown_id = best_match_id
            
            # Update tracking info
            self.tracked_faces[unknown_id].update({
                "bbox": loc,
                "last_seen": current_time,
                "frames_tracked": self.tracked_faces[unknown_id]["frames_tracked"] + 1,
                "total_time": current_time - self.tracked_faces[unknown_id]["first_seen"],
                "embedding": face_data["embedding"],  # Update to the latest embedding
                "quality_score": max(face_data["quality_score"], self.tracked_faces[unknown_id]["quality_score"])
            })
            
            # Check if it's time to enroll this face as a customer
            if (self.tracked_faces[unknown_id]["total_time"] > UNKNOWN_FACE_TRACK_SECONDS and
                self.tracked_faces[unknown_id]["frames_tracked"] > 50 and
                self.tracked_faces[unknown_id]["quality_score"] > MIN_FACE_QUALITY):
                logger.info(f"Unknown face {unknown_id} tracked for sufficient time, enrolling as customer")
                
                with self.db_lock:
                    # Add as a new customer
                    customer = add_or_update_customer(
                        self.customer_db, 
                        face_data,
                        self.location
                    )
                    # Mark as identified
                    self.tracked_faces[unknown_id]["identified"] = True
                    self.tracked_faces[unknown_id]["identified_as"] = {
                        "type": "customer",
                        "id": customer["customerId"],
                        "is_new": True
                    }
        else:
            # Create new tracking entry
            self.unknown_face_ids += 1
            unknown_id = f"unknown_{self.unknown_face_ids}"
            
            self.tracked_faces[unknown_id] = {
                "bbox": loc,
                "first_seen": current_time,
                "last_seen": current_time,
                "frames_tracked": 1,
                "total_time": 0.0,
                "embedding": face_data["embedding"],
                "quality_score": face_data["quality_score"],
                "identified": False,
                "identified_as": None
            }
            logger.debug(f"Started tracking new unknown face: {unknown_id}")
            
        return unknown_id
    
    def process_frame(self, frame):
        """Process a single frame for face recognition"""
        # Increment frame counter
        self.frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - self.last_frame_time
        if elapsed > 0:
            fps = 1 / elapsed
            self.fps_history.append(fps)
        self.last_frame_time = current_time
        
        # Skip frames based on interval for performance
        if self.frame_count % FRAME_PROCESS_INTERVAL != 0:
            # Still draw existing faces, but don't detect new ones
            return self.draw_tracked_faces(frame.copy())
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame, model=FACE_DETECTION_MODEL)
        
        # Extract face data
        face_data_list = extract_face_embeddings(rgb_frame, face_locations)
        
        # Process each face
        recognized_faces = []
        for face_data in face_data_list:
            face_loc = (
                face_data["location"]["top"],
                face_data["location"]["right"],
                face_data["location"]["bottom"],
                face_data["location"]["left"]
            )
            
            # Step 1: Check if this is a staff member
            staff_match = find_matching_staff(self.staff_db, face_data["embedding"])
            
            if staff_match:
                # Found a staff match
                staff_id = staff_match["staffId"]
                confidence = staff_match["confidence"]
                
                # Log recognition event
                with self.db_lock:
                    log_recognition_event(
                        self.visit_logs,
                        "staff",
                        staff_id,
                        confidence,
                        self.location
                    )
                
                # Add to recognized faces for this frame
                recognized_faces.append({
                    "type": "staff",
                    "id": staff_id,
                    "name": f"{staff_match['firstName']} {staff_match['lastName']}",
                    "confidence": confidence,
                    "location": face_loc
                })
                
                logger.info(f"Recognized staff: {staff_match['firstName']} {staff_match['lastName']} ({confidence:.2f})")
                continue
            
            # Step 2: Check if this is a known customer
            customer_match = find_matching_customer(self.customer_db, face_data["embedding"])
            
            if customer_match:
                # Found a customer match
                customer_id = customer_match["customerId"]
                confidence = customer_match["confidence"]
                
                # Update customer information
                with self.db_lock:
                    update_existing_customer(
                        self.customer_db,
                        customer_match,
                        face_data,
                        self.location
                    )
                    
                    # Log recognition event
                    log_recognition_event(
                        self.visit_logs,
                        "customer",
                        customer_id,
                        confidence,
                        self.location
                    )
                
                # Add to recognized faces for this frame
                recognized_faces.append({
                    "type": "customer",
                    "id": customer_id,
                    "name": f"Customer #{customer_match['visitCount']}",
                    "confidence": confidence,
                    "location": face_loc
                })
                
                logger.info(f"Recognized customer: {customer_id} ({confidence:.2f})")
                continue
            
            # Step 3: This is an unknown face, track it
            unknown_id = self.track_unknown_face(face_data)
            
            # Add to recognized faces list
            recognized_faces.append({
                "type": "unknown",
                "id": unknown_id,
                "name": f"Unknown",
                "confidence": 0.0,
                "location": face_loc,
                "tracked_time": self.tracked_faces[unknown_id]["total_time"] if unknown_id in self.tracked_faces else 0.0
            })
        
        # Draw results on frame
        result_frame = self.draw_recognition_results(frame.copy(), recognized_faces)
        
        return result_frame
    
    def draw_recognition_results(self, frame, recognized_faces):
        """Draw recognition results on the frame"""
        for face in recognized_faces:
            face_loc = face["location"]
            
            # Different colors for different types
            if face["type"] == "staff":
                color = GREEN
                label = f"Staff: {face['name']}"
            elif face["type"] == "customer":
                color = BLUE
                label = f"Customer: {face['id'][-4:]}"
            else:  # unknown
                color = RED
                # Show tracking time for unknown faces
                tracked_time = face.get("tracked_time", 0)
                if tracked_time > 60:
                    minutes = int(tracked_time // 60)
                    label = f"Unknown ({minutes}m)"
                else:
                    label = "Unknown"
            
            # Draw box and label
            draw_face_box(frame, face_loc, label, color, face["confidence"])
        
        # Add FPS and other info
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        # Add counts
        with self.db_lock:
            staff_count = sum(1 for face in recognized_faces if face["type"] == "staff")
            customer_count = sum(1 for face in recognized_faces if face["type"] == "customer")
            unknown_count = sum(1 for face in recognized_faces if face["type"] == "unknown")
            
            cv2.putText(frame, f"Staff: {staff_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, GREEN, 2)
            cv2.putText(frame, f"Customers: {customer_count}", (10, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BLUE, 2)
            cv2.putText(frame, f"Unknown: {unknown_count}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, RED, 2)
        
        return frame
    
    def draw_tracked_faces(self, frame):
        """Draw previously tracked faces on frames we're skipping for detection"""
        # Draw existing tracked faces
        for face_id, tracked in self.tracked_faces.items():
            # Only draw recently seen faces (within 2 seconds)
            if time.time() - tracked["last_seen"] > 2.0:
                continue
                
            # Get location
            loc = tracked["bbox"]
            face_loc = (loc["top"], loc["right"], loc["bottom"], loc["left"])
            
            # Determine label and color
            if tracked["identified"]:
                if tracked["identified_as"]["type"] == "staff":
                    color = GREEN
                    label = f"Staff: {tracked['identified_as']['id']}"
                else:
                    color = BLUE
                    label = f"Customer: {tracked['identified_as']['id'][-4:]}"
            else:
                color = RED
                # Show tracking time
                tracked_time = tracked["total_time"]
                if tracked_time > 60:
                    minutes = int(tracked_time // 60)
                    label = f"Unknown ({minutes}m)"
                else:
                    label = "Unknown"
            
            # Draw box
            draw_face_box(frame, face_loc, label, color)
        
        # Add FPS
        avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
        cv2.putText(frame, f"FPS: {avg_fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, WHITE, 2)
        
        return frame
    
    def run(self):
        """Run the face recognition system on a video feed"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera with ID {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Starting face recognition at {self.location}...")
        print("Press 'q' to quit")
        
        try:
            while True:
                # Read frame
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    break
                
                # Process frame
                result_frame = self.process_frame(frame)
                
                # Display result
                cv2.imshow("Face Recognition System", result_frame)
                
                # Check for user quit
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Stopping face recognition.")
                    break
        
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Face recognition system stopped")


def run(camera_id=0, location="Main Entrance"):
    """Run the face recognition system"""
    system = FaceRecognitionSystem(camera_id, location)
    system.run()


if __name__ == "__main__":
    run() 