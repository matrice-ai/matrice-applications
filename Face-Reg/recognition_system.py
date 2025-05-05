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
from datetime import datetime
from collections import deque

from models import get_face_detector, get_embedding_model
from face_processor import FaceProcessor
from staff_recognizer import StaffRecognizer
from customer_tracker import CustomerTracker
from db_manager import db_manager, STAFF_MATCH_THRESHOLD, CUSTOMER_MATCH_THRESHOLD
import utils

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
        self.frame_count = 0
        self.skip_frames = 3  # Process every nth frame
        
        # For FPS calculation
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        self.last_detection_time = 0
        
        # Display settings
        self.show_staff = True
        self.show_customers = True
        self.show_unknowns = True
        self.display_mode = "all"  # "all", "staff_only", "customers_only"
        
        # Thread safety
        self.settings_lock = threading.Lock()
        self.results_lock = threading.Lock()
        
        # Background processing
        self.frame_queue = queue.Queue(maxsize=1)  # Only keep the latest frame
        self.running = True
        self.processing_thread = None
        self.use_threading = True
        
        # Store the last results
        self.recognized_staff = []
        self.tracked_customers = []
        self.results_ready = False
        self.last_result_frame = None
        
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
                unknowns_enabled = self.show_unknowns
                
                # Display mode filtering
                if self.display_mode == "staff_only":
                    customers_enabled = False
                    unknowns_enabled = False
                elif self.display_mode == "customers_only":
                    staff_enabled = False
            
            # Draw staff recognition results
            if results_ready and staff_enabled and recognized_staff:
                display_frame = self.staff_recognizer.draw_staff_recognition(display_frame, recognized_staff)
            
            # Draw customer tracking results
            if results_ready and (customers_enabled or unknowns_enabled) and tracked_customers:
                # Filter by type if needed
                if customers_enabled and not unknowns_enabled:
                    # Show only known customers
                    filtered_customers = [c for c in tracked_customers if c["type"] != "unknown"]
                    display_frame = self.customer_tracker.draw_customer_tracking(display_frame, filtered_customers)
                elif unknowns_enabled and not customers_enabled:
                    # Show only unknowns
                    filtered_customers = [c for c in tracked_customers if c["type"] == "unknown"]
                    display_frame = self.customer_tracker.draw_customer_tracking(display_frame, filtered_customers)
                else:
                    # Show all
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
                "Customers": f"{tracking_stats['identified_faces'] - len(recognized_staff)}",
                "Unknown": f"{tracking_stats['unknown_faces']}",
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
        """Toggle unknown faces display"""
        with self.settings_lock:
            self.show_unknowns = not self.show_unknowns
        return self.show_unknowns
    
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
        print("  u - Toggle unknown faces display")
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
                elif key == ord('u'):
                    show = self.toggle_unknown_display()
                    print(f"Unknown faces display: {'On' if show else 'Off'}")
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