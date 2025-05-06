"""
recognition_system.py - Main Face Recognition System

This is the main orchestrator that integrates all components of the face recognition system.
It handles video input, coordinates staff and customer recognition, and displays results.
"""

import cv2
import numpy as np
import time
import os
import logging
import threading
import queue
import random
from datetime import datetime
from collections import deque

from models import get_face_detector, get_embedding_model
from face_processor import FaceProcessor
from staff_recognizer import StaffRecognizer
from customer_tracker import CustomerTracker
from db_manager import db_manager, STAFF_MATCH_THRESHOLD, CUSTOMER_MATCH_THRESHOLD
import utils
from utils import draw_stylish_box

# Initialize logger
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    filename='face_recognition_data/face_recognition.log',
    filemode='a'
)
logger = logging.getLogger('face_recognition.system')

# Ensure storage directory exists
os.makedirs('face_recognition_data', exist_ok=True)

# Window name for OpenCV
WINDOW_NAME = "Face Recognition System"

# Maximum processing time
MAX_PROCESSING_TIME = 0.2  # 200ms max per frame

# Function to generate a unique 4-digit ID
def generate_unique_id():
    """Generate a unique 4-digit ID for output files"""
    return f"{random.randint(1000, 9999)}"

class FaceRecognitionSystem:
    """Main face recognition system that orchestrates all components"""
    
    def __init__(self, camera_id=0, location="Main Entrance", 
                detector_type="mtcnn", embedding_model="facenet", 
                staff_threshold=STAFF_MATCH_THRESHOLD, customer_threshold=CUSTOMER_MATCH_THRESHOLD):
        """
        Initialize the face recognition system
        
        Args:
            camera_id: Camera device ID to use
            location: Location identifier for this camera
            detector_type: Face detection model to use
            embedding_model: Face embedding model to use
            staff_threshold: Matching threshold for staff recognition
            customer_threshold: Matching threshold for customer recognition
        """
        self.camera_id = camera_id
        self.location = location
        self.detector_type = detector_type
        
        # Initialize models and processors
        logger.info(f"Initializing face recognition system at {location} with camera ID {camera_id}")
        logger.info(f"Using detector: {detector_type}, embedding model: {embedding_model}")
        logger.info(f"Staff threshold: {staff_threshold}, Customer threshold: {customer_threshold}")
        
        # Initialize components
        self.staff_recognizer = StaffRecognizer(
            detector_type=detector_type,
            embedding_model=embedding_model,
            match_threshold=staff_threshold
        )
        
        self.customer_tracker = CustomerTracker(
            detector_type=detector_type,
            embedding_model=embedding_model,
            match_threshold=customer_threshold
        )
        
        # For optimizing detection frequency
        self.skip_frames = 5  # Process every N frames for performance
        self.last_detection_frame = 0  # Track the last frame where detection was performed
        
        # Visualization options
        self.show_staff = True
        self.show_customers = True
        self.show_unknown = True
        self.display_mode = 0  # 0=normal, 1=debug
        
        # Performance tracking
        self.threaded_mode = False
        self.fast_mode = False
        self.processing_thread = None
        self.thread_running = False
        self.frame_queue = queue.Queue(maxsize=1)
        self.last_result_frame = None
        
        # Recognition results
        self.recognized_staff = []
        self.tracked_customers = []
        self.results_ready = False
        
        # Thread safety
        self.settings_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        # Background processing
        self.running = True
        self.use_threading = True
        
        # For FPS calculation
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.last_detection_time = 0
        
        # Display settings
        self.show_staff = True
        self.show_customers = True
        self.display_mode = "all"  # "all", "staff_only", "customers_only"
        
        # Thread safety
        self.settings_lock = threading.Lock()
        
        # Detection mode toggle
        self.fast_mode = False  # False = MTCNN, True = HOG
    
    def toggle_fast_mode(self):
        """Toggle between MTCNN and HOG detection modes"""
        self.fast_mode = not self.fast_mode
        new_detector = "hog" if self.fast_mode else self.detector_type
        
        # Re-initialize the recognizers with new detector
        with self.results_lock:
            self.staff_recognizer = StaffRecognizer(
                detector_type=new_detector,
                embedding_model="facenet",
                match_threshold=STAFF_MATCH_THRESHOLD
            )
            
            self.customer_tracker = CustomerTracker(
                detector_type=new_detector,
                embedding_model="facenet",
                match_threshold=CUSTOMER_MATCH_THRESHOLD
            )
            
            # Clear results
            self.recognized_staff = []
            self.tracked_customers = []
            self.results_ready = False
        
        logger.info(f"Switched to {'fast' if self.fast_mode else 'quality'} mode using {new_detector} detector")
        return self.fast_mode
    
    def start_processing_thread(self):
        """Start the background processing thread"""
        if self.processing_thread is None or not self.processing_thread.is_alive():
            self.running = True
            self.processing_thread = threading.Thread(target=self._process_frames_thread)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Started background processing thread")
    
    def stop_processing_thread(self):
        """Stop the background processing thread"""
        self.running = False
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)
            logger.info("Stopped background processing thread")
    
    def _process_frames_thread(self):
        """Background thread to process frames"""
        while self.running:
            try:
                # Get a frame from the queue with timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                start_time = time.time()
                
                # Detect faces in this thread
                staff, customers = self._detect_faces(frame)
                
                # Store the results with thread safety
                with self.results_lock:
                    self.recognized_staff = staff
                    self.tracked_customers = customers
                    self.results_ready = True
                    self.last_detection_time = time.time()
                
                processing_time = time.time() - start_time
                if processing_time > MAX_PROCESSING_TIME:
                    logger.warning(f"Face detection took {processing_time:.2f}s")
                
            except queue.Empty:
                # No frames in queue, just wait
                time.sleep(0.01)
            except Exception as e:
                logger.error(f"Error in background thread: {str(e)}")
                time.sleep(0.1)  # Sleep to avoid CPU spinning
    
    def _detect_faces(self, frame):
        """Perform face detection and recognition (called from background thread)"""
        try:
            # Detect and recognize staff
            recognized_staff, face_locations = self.staff_recognizer.detect_and_recognize_staff(
                frame, self.location
            )
            
            # Filter out staff faces for customer detection
            if recognized_staff:
                # Create a copy for customer detection
                customer_frame = frame.copy()
            else:
                customer_frame = frame
            
            # Detect and track customers
            tracked_customers, _ = self.customer_tracker.detect_and_track_customers(
                customer_frame, self.location
            )
            
            return recognized_staff, tracked_customers
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return [], []
    
    def process_frame(self, frame):
        """
        Process a single frame for display (main thread)
        
        Args:
            frame: Input video frame
            
        Returns:
            Processed frame with visualization
        """
        try:
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Calculate FPS
            current_time = time.time()
            elapsed = current_time - self.last_frame_time
            if elapsed > 0:
                fps = 1 / elapsed
                self.fps_history.append(fps)
            self.last_frame_time = current_time
            
            # Submit frame for background processing
            if self.use_threading:
                # Add current frame to processing queue, replacing any older frame
                if not self.frame_queue.full():
                    self.frame_queue.put_nowait(frame)
                else:
                    try:
                        # Clear the queue and add new frame
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(frame)
                    except:
                        pass
            # Non-threaded processing
            else:
                # Increment frame counter for frame skipping
                self.frame_count += 1
                
                # Only process every nth frame
                if self.frame_count % self.skip_frames == 0:
                    # Process directly in main thread
                    recognized_staff, tracked_customers = self._detect_faces(frame)
                    
                    # Store results
                    with self.results_lock:
                        self.recognized_staff = recognized_staff
                        self.tracked_customers = tracked_customers
                        self.results_ready = True
                        self.last_detection_time = time.time()
            
            # Get current detection results with thread safety
            with self.results_lock:
                recognized_staff = self.recognized_staff
                tracked_customers = self.tracked_customers
                results_ready = self.results_ready
            
            # Draw results based on display settings
            with self.settings_lock:
                # Which types to display
                staff_enabled = self.show_staff
                customers_enabled = self.show_customers
                
                # Display mode filtering
                if self.display_mode == "staff_only":
                    customers_enabled = False
                elif self.display_mode == "customers_only":
                    staff_enabled = False
            
            # Draw staff recognition results
            if results_ready and staff_enabled and recognized_staff:
                display_frame = self.staff_recognizer.draw_staff_recognition(display_frame, recognized_staff)
            
            # Draw customer tracking results
            if results_ready and customers_enabled and tracked_customers:
                display_frame = self.customer_tracker.draw_customer_tracking(display_frame, tracked_customers)
            
            # Calculate average FPS
            avg_fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
            
            # Get statistics for display
            staff_stats = self.staff_recognizer.get_recognition_stats()
            customer_stats = self.customer_tracker.get_recognition_stats()
            tracking_stats = self.customer_tracker.get_tracking_stats()
            
            # Build status panel info
            display_stats = {
                "Staff": f"{len(recognized_staff)}",
                "Customers": f"{tracking_stats['customer_faces']}",
                "FPS": f"{avg_fps:.1f}",
                "Mode": f"{'Fast' if self.fast_mode else 'Quality'}-{'Threaded' if self.use_threading else 'Direct'}"
            }
            
            # Add status panel with stats
            display_frame = utils.add_status_panel(
                display_frame,
                f"Face Recognition - {self.location}",
                avg_fps,
                display_stats
            )
            
            # Store the result
            self.last_result_frame = display_frame
            
            return display_frame
        
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            # Return last valid frame or original frame
            return self.last_result_frame if self.last_result_frame is not None else frame
    
    def toggle_display_mode(self):
        """Toggle between display modes"""
        with self.settings_lock:
            if self.display_mode == "all":
                self.display_mode = "staff_only"
            elif self.display_mode == "staff_only":
                self.display_mode = "customers_only"
            else:
                self.display_mode = "all"
        
        logger.info(f"Display mode changed to: {self.display_mode}")
        return self.display_mode
    
    def toggle_staff_display(self):
        """Toggle staff recognition display"""
        with self.settings_lock:
            self.show_staff = not self.show_staff
        return self.show_staff
    
    def toggle_customer_display(self):
        """Toggle customer recognition display"""
        with self.settings_lock:
            self.show_customers = not self.show_customers
        return self.show_customers
    
    def toggle_unknown_display(self):
        """Toggle unknown faces display (deprecated, retained for compatibility)"""
        logger.info("Unknown faces display toggle is deprecated - all faces are now customers")
        return True
    
    def toggle_threading(self):
        """Toggle between threaded and non-threaded processing"""
        self.use_threading = not self.use_threading
        logger.info(f"Threaded processing: {'enabled' if self.use_threading else 'disabled'}")
        return self.use_threading
    
    def set_skip_frames(self, skip_frames):
        """Set frame skipping value (higher = better performance, lower = more responsive)"""
        if skip_frames < 1:
            skip_frames = 1
        self.skip_frames = skip_frames
        logger.info(f"Frame skipping set to {skip_frames}")
    
    def run(self):
        """Run the face recognition system on the camera feed"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            logger.error(f"Error: Could not open camera with ID {self.camera_id}")
            print(f"Error: Could not open camera with ID {self.camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Start background processing thread
        self.start_processing_thread()
        
        print(f"Starting face recognition at {self.location}...")
        print("Controls:")
        print("  q - Quit")
        print("  m - Toggle display mode (all/staff/customers)")
        print("  s - Toggle staff display")
        print("  c - Toggle customer display")
        print("  t - Toggle threaded/non-threaded processing")
        print("  f - Toggle fullscreen mode")
        print("  h - Toggle fast mode (HOG) / quality mode (MTCNN)")
        print("  + - Decrease frame skipping (more frequent detection)")
        print("  - - Increase frame skipping (better performance)")
        
        # Create window with specific properties
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(WINDOW_NAME, 800, 600)
        
        fullscreen = False
        skip_display = 0  # Counter to occasionally skip display updates for smoother UI
        
        try:
            while True:
                # Process UI events first to keep responsive
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("Stopping face recognition.")
                    break
                elif key == ord('m'):
                    mode = self.toggle_display_mode()
                    print(f"Display mode: {mode}")
                elif key == ord('s'):
                    show = self.toggle_staff_display()
                    print(f"Staff display: {'On' if show else 'Off'}")
                elif key == ord('c'):
                    show = self.toggle_customer_display()
                    print(f"Customer display: {'On' if show else 'Off'}")
                elif key == ord('t'):
                    enabled = self.toggle_threading()
                    print(f"Threaded processing: {'On' if enabled else 'Off'}")
                elif key == ord('h'):
                    fast_mode = self.toggle_fast_mode()
                    print(f"{'Fast mode' if fast_mode else 'Quality mode'} enabled")
                elif key == ord('+'):
                    self.set_skip_frames(max(1, self.skip_frames - 1))
                    print(f"Processing every {self.skip_frames} frames")
                elif key == ord('-'):
                    self.set_skip_frames(self.skip_frames + 1)
                    print(f"Processing every {self.skip_frames} frames")
                elif key == ord('f'):
                    fullscreen = not fullscreen
                    if fullscreen:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    else:
                        cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                
                # Read frame with timeout
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to capture frame")
                    # Attempt to reopen the camera
                    cap.release()
                    time.sleep(0.5)
                    cap = cv2.VideoCapture(self.camera_id)
                    if not cap.isOpened():
                        break
                    
                    # Display last valid frame while waiting for camera to reconnect
                    if self.last_result_frame is not None:
                        try:
                            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                                cv2.imshow(WINDOW_NAME, self.last_result_frame)
                                cv2.waitKey(1)  # Process UI updates
                        except:
                            pass
                    
                    continue
                
                # Process frame for display
                try:
                    result_frame = self.process_frame(frame)
                    
                    # Skip some display updates if detection is running
                    skip_display = (skip_display + 1) % 3
                    
                    # Only update the display every few frames if detection is active
                    if skip_display == 0 or not self.results_ready:
                        # Check if the window still exists
                        try:
                            # Display result - only if window exists
                            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) >= 1:
                                cv2.imshow(WINDOW_NAME, result_frame)
                            else:
                                # Window was closed, recreate it
                                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                                cv2.imshow(WINDOW_NAME, result_frame)
                        except cv2.error:
                            # Window might be closed, recreate it
                            try:
                                cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
                                cv2.imshow(WINDOW_NAME, result_frame)
                            except:
                                logger.error("Failed to recreate window, will try again")
                    
                    # Make sure UI events are processed
                    cv2.waitKey(1)
                    
                except Exception as e:
                    logger.error(f"Error processing or displaying frame: {e}")
        
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
            print(f"Error: {str(e)}")
        
        finally:
            # Stop background processing
            self.stop_processing_thread()
            
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            logger.info("Face recognition system stopped")

    def process_video_frame(self, frame, frame_count):
        """
        Process a single frame from a video file
        
        Args:
            frame: Input video frame
            frame_count: Current frame number for skipping frames
            
        Returns:
            Processed frame with visualization
        """
        try:
            # Make a copy for drawing
            display_frame = frame.copy()
            
            # Process with detection only on certain frames
            if frame_count % self.skip_frames == 0:
                # Detect faces
                recognized_staff, tracked_customers = self._detect_faces(frame)
                
                # Store results
                with self.results_lock:
                    self.recognized_staff = recognized_staff
                    self.tracked_customers = tracked_customers
                    self.results_ready = True
                    # Mark this as a detection frame (not interpolated)
                    self.last_detection_frame = frame_count
            else:
                # For in-between frames, update predicted positions for smooth visualization
                with self.results_lock:
                    if self.results_ready and self.tracked_customers:
                        # Calculate how many frames since last detection
                        frames_since_detection = frame_count - self.last_detection_frame
                        
                        # Update predicted positions for each tracked customer
                        for customer in self.tracked_customers:
                            # Get current bbox
                            bbox = customer.get('bbox', None)
                            track_id = customer.get('track_id', None)
                            
                            if bbox is not None and track_id is not None:
                                # Check if we have the track in the customer tracker
                                if track_id in self.customer_tracker.tracked_faces:
                                    tracked_face = self.customer_tracker.tracked_faces[track_id]
                                    
                                    # Predict next location based on motion history
                                    predicted_bbox = tracked_face.predict_next_location()
                                    
                                    # For smoother transitions, blend current and predicted bbox
                                    # based on how many frames since last detection
                                    alpha = min(frames_since_detection / self.skip_frames, 0.9)
                                    
                                    # Calculate the weighted average of current and predicted bbox
                                    top, right, bottom, left = bbox
                                    pred_top, pred_right, pred_bottom, pred_left = predicted_bbox
                                    
                                    smooth_bbox = (
                                        int((1-alpha) * top + alpha * pred_top),
                                        int((1-alpha) * right + alpha * pred_right),
                                        int((1-alpha) * bottom + alpha * pred_bottom),
                                        int((1-alpha) * left + alpha * pred_left)
                                    )
                                    
                                    # Update bbox in customer data
                                    customer['bbox'] = smooth_bbox
            
            # Get current detection results with thread safety
            with self.results_lock:
                recognized_staff = self.recognized_staff
                tracked_customers = self.tracked_customers
                results_ready = self.results_ready
            
            # Draw staff recognition results
            if results_ready and recognized_staff:
                display_frame = self.staff_recognizer.draw_staff_recognition(display_frame, recognized_staff)
            
            # Draw customer tracking results
            if results_ready and tracked_customers:
                display_frame = self.customer_tracker.draw_customer_tracking(display_frame, tracked_customers)
            
            # Add a stylish status panel
            stats = {
                "Staff": f"{len(recognized_staff)}",
                "Customers": f"{len(tracked_customers)}"
            }
            
            # Draw timestamp on video
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            h, w = display_frame.shape[:2]
            
            # Add semi-transparent gradient overlay at the bottom
            overlay = display_frame.copy()
            cv2.rectangle(overlay, (0, h-40), (w, h), (20, 20, 20), -1)
            cv2.addWeighted(overlay, 0.5, display_frame, 0.5, 0, display_frame)
            
            # Add timestamp with shadow
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, timestamp, (w-210, h-15), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, timestamp, (w-210, h-15), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add frame count
            frame_text = f"Frame: {frame_count}"
            cv2.putText(display_frame, frame_text, (20, h-15), font, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, frame_text, (20, h-15), font, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            
            # Add custom status panel
            display_frame = utils.add_status_panel(
                display_frame,
                f"Face Recognition - {self.location}",
                0,  # No FPS tracking for video files
                stats
            )
            
            return display_frame
        
        except Exception as e:
            logger.error(f"Error processing video frame: {str(e)}")
            # Return original frame if error
            return frame

def process_video(video_path, output_dir="output_videos", location="Video Processing", 
                 detector_type="mtcnn", embedding_model="facenet", frame_interval=5, output_fps=0):
    """
    Process a video file for face recognition and save the annotated output
    
    Args:
        video_path: Path to input video file
        output_dir: Directory to save output video
        location: Location identifier for this processing
        detector_type: Face detector type to use
        embedding_model: Face embedding model to use
        frame_interval: Process every N frames (lower = smoother, higher = faster)
        output_fps: Output video FPS (0 = use input FPS, lower values reduce flickering)
    """
    print(f"Starting video processing on: {video_path}")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate unique ID for output file
    unique_id = f"{random.randint(1000, 9999)}"
    
    # Create output path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    input_filename = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{input_filename}_{unique_id}_{timestamp}.mp4")
    
    print(f"Output will be saved to: {output_path}")
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    print("[DIAG] Video file opened successfully")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"[DIAG] Video properties: {width}x{height}, {fps} fps, {total_frames} frames")
    
    # Use output_fps if specified, otherwise use input fps
    if output_fps > 0:
        write_fps = output_fps
        print(f"[DIAG] Using reduced output FPS: {write_fps} (original: {fps})")
    else:
        write_fps = fps
        print(f"[DIAG] Using original FPS: {write_fps}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, write_fps, (width, height))
    
    print("[DIAG] Video writer initialized")
    
    # Initialize detector and embedding model directly
    detector = get_face_detector(detector_type)
    model = get_embedding_model(embedding_model)
    
    print(f"[DIAG] Detector ({detector_type}) and embedding model ({embedding_model}) initialized")
    
    # Define skipping to process fewer frames (for speed)
    skip_frames = frame_interval  # Process every N frames
    
    # Initialize processing system - the video system will handle frame skipping
    system = FaceRecognitionSystem(
        camera_id=0,  # Not used for video processing
        location=location,
        detector_type=detector_type,
        embedding_model=embedding_model  # Pass the string name, not the model object
    )
    system.skip_frames = skip_frames
    system.last_detection_frame = 0  # Initialize last detection frame
    
    print(f"Total frames: {total_frames}")
    print(f"Processing every {skip_frames} frames")
    print("This may take a while...")
    
    # Import tqdm for progress bar
    from tqdm import tqdm
    
    # Set colors for visualization
    STAFF_COLOR = (0, 230, 0)  # BGR
    CUSTOMER_COLOR = (255, 128, 0)  # BGR
    
    # Process frames
    pbar = tqdm(total=total_frames, desc="Processing video", unit="frames")
    frame_count = 0
    staff_detected = set()
    customer_detected = set()
    
    print("[DIAG] Starting frame processing loop")
    
    # Simple frame-by-frame processing
    while True:
        # Read frame
        ret, frame = cap.read()
        if not ret:
            print("[DIAG] End of video reached")
            break
        
        frame_count += 1
        pbar.update(1)
        
        # Periodic progress update
        if frame_count % 100 == 0:
            print(f"Processing frame {frame_count}/{total_frames}")
        
        # Process only every nth frame (skip_frames)
        if frame_count % skip_frames == 0:
            print(f"[DIAG] Processing frame {frame_count}")
            
            # Make a copy for drawing
            draw_frame = frame.copy()
            
            try:
                # 1. Detect faces with the model
                print(f"[DIAG] Frame {frame_count}: Detecting faces")
                face_locations, probs, landmarks = detector.detect_faces(frame)
                print(f"[DIAG] Frame {frame_count}: Detected {len(face_locations)} faces")
                
                # 2. If faces found, get embeddings
                if face_locations:
                    print(f"[DIAG] Frame {frame_count}: Getting embeddings for {len(face_locations)} faces")
                    face_embeddings = model.get_embeddings(frame, face_locations)
                    print(f"[DIAG] Frame {frame_count}: Got {len(face_embeddings)} embeddings")
                    
                    # 3. Process each face
                    for i, (face_loc, embedding) in enumerate(zip(face_locations, face_embeddings)):
                        print(f"[DIAG] Frame {frame_count}: Processing face {i+1}/{len(face_locations)}")
                        
                        # Skip if embedding is invalid
                        if embedding is None or len(embedding) == 0:
                            print(f"[DIAG] Frame {frame_count}: Face {i+1} has invalid embedding, skipping")
                            continue
                        
                        # 4. Try staff match first
                        print(f"[DIAG] Frame {frame_count}: Face {i+1} - Checking staff match")
                        staff_match = db_manager.find_matching_staff(embedding, STAFF_MATCH_THRESHOLD)
                        print(f"[DIAG] Frame {frame_count}: Face {i+1} - Staff match result: {staff_match is not None}")
                        
                        if staff_match:
                            # Found staff match
                            staff_id = staff_match["staffId"]
                            name = f"{staff_match['firstName']} {staff_match['lastName']}"
                            position = staff_match.get("position", "")
                            confidence = staff_match["confidence"]
                            
                            print(f"[DIAG] Frame {frame_count}: Face {i+1} - Drawing staff box for {name}")
                            
                            # Draw staff box
                            top, right, bottom, left = face_loc
                            cv2.rectangle(draw_frame, (left, top), (right, bottom), STAFF_COLOR, 2, cv2.LINE_AA)
                            
                            # Add text
                            label = f"{name} ({confidence:.2f})"
                            cv2.putText(draw_frame, label, (left, top - 10), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, STAFF_COLOR, 1, cv2.LINE_AA)
                            
                            # Track unique staff members
                            if staff_id not in staff_detected:
                                staff_detected.add(staff_id)
                                print(f"Detected staff: {name}")
                            
                            # Log the match
                            print(f"[DIAG] Frame {frame_count}: Face {i+1} - Logging staff recognition event")
                            db_manager.log_recognition_event("staff", staff_id, confidence, location)
                            print(f"[DIAG] Frame {frame_count}: Face {i+1} - Staff recognition logged")
                            
                        else:
                            # Try customer match
                            print(f"[DIAG] Frame {frame_count}: Face {i+1} - Checking customer match")
                            customer_match = db_manager.find_matching_customer(embedding, CUSTOMER_MATCH_THRESHOLD)
                            print(f"[DIAG] Frame {frame_count}: Face {i+1} - Customer match result: {customer_match is not None}")
                            
                            if customer_match:
                                # Found existing customer match
                                customer_id = customer_match["customerId"]
                                confidence = customer_match["confidence"]
                                
                                print(f"[DIAG] Frame {frame_count}: Face {i+1} - Drawing customer box for {customer_id}")
                                
                                # Draw customer box
                                top, right, bottom, left = face_loc
                                cv2.rectangle(draw_frame, (left, top), (right, bottom), CUSTOMER_COLOR, 2, cv2.LINE_AA)
                                
                                # Add text
                                label = f"Customer {customer_id} ({confidence:.2f})"
                                cv2.putText(draw_frame, label, (left, top - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CUSTOMER_COLOR, 1, cv2.LINE_AA)
                                
                                # Track unique customers
                                if customer_id not in customer_detected:
                                    customer_detected.add(customer_id)
                                    print(f"Detected customer: {customer_id}")
                                
                                # Log the match
                                print(f"[DIAG] Frame {frame_count}: Face {i+1} - Logging customer recognition event")
                                db_manager.log_recognition_event("customer", customer_id, confidence, location)
                                print(f"[DIAG] Frame {frame_count}: Face {i+1} - Customer recognition logged")
                                
                            else:
                                # New customer - register them
                                print(f"[DIAG] Frame {frame_count}: Face {i+1} - Registering new customer")
                                face_data = {
                                    "embedding": embedding,
                                    "embedding_list": embedding.tolist(),
                                    "quality_score": 0.5  # Default quality score
                                }
                                
                                # Add as new customer
                                print(f"[DIAG] Frame {frame_count}: Face {i+1} - Adding to customer database")
                                customer = db_manager.add_new_customer(face_data, location)
                                customer_id = customer["customerId"]
                                print(f"[DIAG] Frame {frame_count}: Face {i+1} - New customer added with ID {customer_id}")
                                
                                # Draw box
                                top, right, bottom, left = face_loc
                                cv2.rectangle(draw_frame, (left, top), (right, bottom), CUSTOMER_COLOR, 2, cv2.LINE_AA)
                                
                                # Add text
                                label = f"New Customer {customer_id}"
                                cv2.putText(draw_frame, label, (left, top - 10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, CUSTOMER_COLOR, 1, cv2.LINE_AA)
                                
                                customer_detected.add(customer_id)
                                print(f"Registered new customer: {customer_id}")
                
                # 5. Add timestamp
                print(f"[DIAG] Frame {frame_count}: Adding timestamp and frame info")
                timestamp_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(draw_frame, timestamp_text, (width - 200, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # 6. Add frame number
                frame_text = f"Frame: {frame_count}"
                cv2.putText(draw_frame, frame_text, (10, height - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                
                # Write the processed frame
                print(f"[DIAG] Frame {frame_count}: Writing processed frame to output")
                out.write(draw_frame)
                print(f"[DIAG] Frame {frame_count}: Frame written successfully")
                
            except Exception as e:
                print(f"Error processing frame {frame_count}: {str(e)}")
                print(f"[DIAG] Frame {frame_count}: Exception occurred, writing original frame")
                # Write original frame if there's an error
                out.write(frame)
        else:
            # For skipped frames, just write the original
            out.write(frame)
    
    # Cleanup
    print("[DIAG] Processing complete, cleaning up resources")
    pbar.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    
    # Print summary
    print(f"Video processing complete!")
    print(f"Detected {len(staff_detected)} unique staff members and {len(customer_detected)} unique customers")
    print(f"Output saved to: {output_path}")
    print(f"Unique ID: {unique_id}")
    return output_path

def run(camera_id=0, location="Main Entrance", detector_type="mtcnn", embedding_model="facenet"):
    """Run the face recognition system"""
    system = FaceRecognitionSystem(
        camera_id=camera_id, 
        location=location,
        detector_type=detector_type,
        embedding_model=embedding_model,
        staff_threshold=STAFF_MATCH_THRESHOLD,
        customer_threshold=CUSTOMER_MATCH_THRESHOLD
    )
    system.run()


if __name__ == "__main__":
    run() 