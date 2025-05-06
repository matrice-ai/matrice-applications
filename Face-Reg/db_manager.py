"""
db_manager.py - Database Operations Manager

Handles:
- Staff database operations
- Customer database operations
- Visit logs
- Database utilities
"""

import os
import json
import time
import uuid
import logging
import threading
import numpy as np
from datetime import datetime

# Initialize logger
logger = logging.getLogger('face_recognition.db')

# Configuration
STORAGE_DIR = "face_recognition_data"
STAFF_DB_FILE = f"{STORAGE_DIR}/staff_database.json"
CUSTOMER_DB_FILE = f"{STORAGE_DIR}/customer_database.json"
VISIT_LOG_FILE = f"{STORAGE_DIR}/visit_logs.json"

# Ensure paths are absolute for reliable file operations
STORAGE_DIR_ABS = os.path.abspath(STORAGE_DIR)
CUSTOMER_DB_FILE = os.path.join(STORAGE_DIR_ABS, "customer_database.json")
STAFF_DB_FILE = os.path.join(STORAGE_DIR_ABS, "staff_database.json")
VISIT_LOG_FILE = os.path.join(STORAGE_DIR_ABS, "visit_logs.json")

# Default thresholds - RELAXED for better matching
STAFF_MATCH_THRESHOLD = 0.5  # Lower threshold means stricter matching (was 0.8)
CUSTOMER_MATCH_THRESHOLD = 0.7  # Higher threshold means more lenient matching

# Ensure directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

def ensure_file_permissions():
    """Ensure all database files have proper permissions"""
    # Make sure the directory exists and is writeable
    if not os.path.exists(STORAGE_DIR_ABS):
        os.makedirs(STORAGE_DIR_ABS, exist_ok=True)
        print(f"Created storage directory: {STORAGE_DIR_ABS}")

    # Check if customer database file exists, if not create it with an empty structure
    if not os.path.exists(CUSTOMER_DB_FILE):
        try:
            with open(CUSTOMER_DB_FILE, 'w') as f:
                json.dump({"customers": [], "customer_embeddings": []}, f, indent=4)
            print(f"Created empty customer database file: {CUSTOMER_DB_FILE}")
        except Exception as e:
            print(f"Error creating customer database file: {e}")
            
    # Check if files are writeable
    for file_path in [CUSTOMER_DB_FILE, STAFF_DB_FILE, VISIT_LOG_FILE]:
        if os.path.exists(file_path):
            # Check if file is writeable
            if not os.access(file_path, os.W_OK):
                try:
                    # Try to make the file writeable
                    os.chmod(file_path, 0o644)
                    print(f"Updated permissions for: {file_path}")
                except Exception as e:
                    print(f"Error updating permissions for {file_path}: {e}")

# Run permission check at startup
ensure_file_permissions()

class DatabaseManager:
    """Manager for all database operations"""
    
    def __init__(self):
        """Initialize database manager with locks for thread safety"""
        self.staff_db_lock = threading.Lock()
        self.customer_db_lock = threading.Lock()
        self.visit_log_lock = threading.Lock()
    
    def load_staff_db(self):
        """Load staff database from file"""
        with self.staff_db_lock:
            if os.path.exists(STAFF_DB_FILE):
                try:
                    with open(STAFF_DB_FILE, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading staff database: {str(e)}")
                    return {"staff": [], "staff_embeddings": []}
            else:
                logger.warning(f"Staff database not found at {STAFF_DB_FILE}")
                return {"staff": [], "staff_embeddings": []}
    
    def load_customer_db(self):
        """Load customer database from file"""
        print(f"[CUSTOMER-DEBUG] Starting load_customer_db")
        with self.customer_db_lock:
            print(f"[CUSTOMER-DEBUG] Acquired customer_db_lock for loading")
            if os.path.exists(CUSTOMER_DB_FILE):
                print(f"[CUSTOMER-DEBUG] Customer database file exists: {CUSTOMER_DB_FILE}")
                try:
                    print(f"[CUSTOMER-DEBUG] Opening file for reading")
                    with open(CUSTOMER_DB_FILE, 'r') as f:
                        print(f"[CUSTOMER-DEBUG] Loading JSON data")
                        data = json.load(f)
                        print(f"[CUSTOMER-DEBUG] Successfully loaded customer database with {len(data.get('customers', []))} customers")
                        return data
                except Exception as e:
                    error_msg = f"Error loading customer database: {str(e)}"
                    logger.error(error_msg)
                    print(f"[CUSTOMER-DEBUG] ERROR: {error_msg}")
                    print(f"[CUSTOMER-DEBUG] Initializing new customer database")
                    return self._initialize_customer_db()
            else:
                print(f"[CUSTOMER-DEBUG] Customer database file not found, initializing new one")
                return self._initialize_customer_db()
    
    def load_visit_logs(self):
        """Load visit logs from file"""
        print(f"[VISIT-DEBUG] Starting load_visit_logs")
        with self.visit_log_lock:
            print(f"[VISIT-DEBUG] Acquired visit_log_lock for loading")
            if os.path.exists(VISIT_LOG_FILE):
                print(f"[VISIT-DEBUG] Visit logs file exists: {VISIT_LOG_FILE}")
                try:
                    print(f"[VISIT-DEBUG] Opening file for reading")
                    with open(VISIT_LOG_FILE, 'r') as f:
                        print(f"[VISIT-DEBUG] Loading JSON data")
                        data = json.load(f)
                        print(f"[VISIT-DEBUG] Successfully loaded visit logs with {len(data.get('visits', []))} entries")
                        return data
                except Exception as e:
                    error_msg = f"Error loading visit logs: {str(e)}"
                    logger.error(error_msg)
                    print(f"[VISIT-DEBUG] ERROR: {error_msg}")
                    print(f"[VISIT-DEBUG] Initializing new visit logs")
                    return self._initialize_visit_logs()
            else:
                print(f"[VISIT-DEBUG] Visit logs file not found, initializing new one")
                return self._initialize_visit_logs()
    
    def _initialize_customer_db(self):
        """Initialize empty customer database"""
        print(f"[CUSTOMER-DEBUG] Initializing empty customer database")
        db = {
            "customers": [],
            "customer_embeddings": []
        }
        try:
            print(f"[CUSTOMER-DEBUG] Creating empty database file")
            # Ensure directory exists
            if not os.path.exists(os.path.dirname(CUSTOMER_DB_FILE)):
                print(f"[CUSTOMER-DEBUG] Creating directory: {os.path.dirname(CUSTOMER_DB_FILE)}")
                os.makedirs(os.path.dirname(CUSTOMER_DB_FILE), exist_ok=True)
                
            with open(CUSTOMER_DB_FILE, 'w') as f:
                json.dump(db, f, indent=4)
                print(f"[CUSTOMER-DEBUG] Successfully created empty database file")
            return db
        except Exception as e:
            print(f"[CUSTOMER-DEBUG] ERROR initializing database: {str(e)}")
            # Return empty structure anyway so the system can continue
            return db
    
    def _initialize_visit_logs(self):
        """Initialize empty visit logs"""
        print(f"[VISIT-DEBUG] Initializing empty visit logs")
        logs = {
            "visits": [],
            "statistics": {
                "total_visits": 0,
                "unique_customers": 0,
                "repeat_customers": 0,
                "staff_entries": 0
            }
        }
        try:
            print(f"[VISIT-DEBUG] Creating empty visit logs file")
            # Ensure directory exists
            if not os.path.exists(os.path.dirname(VISIT_LOG_FILE)):
                print(f"[VISIT-DEBUG] Creating directory: {os.path.dirname(VISIT_LOG_FILE)}")
                os.makedirs(os.path.dirname(VISIT_LOG_FILE), exist_ok=True)
                
            with open(VISIT_LOG_FILE, 'w') as f:
                json.dump(logs, f, indent=4)
                print(f"[VISIT-DEBUG] Successfully created empty visit logs file")
            return logs
        except Exception as e:
            print(f"[VISIT-DEBUG] ERROR initializing visit logs: {str(e)}")
            # Return empty structure anyway so the system can continue
            return logs
    
    def save_customer_db(self, db):
        """Save customer database to file"""
        print(f"[CUSTOMER-DEBUG] Starting save_customer_db with {len(db['customers'])} customers")
        
        # Create a debug log file to trace the issue
        debug_log_path = f"{STORAGE_DIR}/customer_debug.log"
        with open(debug_log_path, 'a') as debug_log:
            debug_log.write(f"\n\n=== SAVE ATTEMPT at {datetime.now().isoformat()} ===\n")
            debug_log.write(f"Customers count: {len(db['customers'])}\n")
            debug_log.write(f"Embeddings count: {len(db.get('customer_embeddings', []))}\n")
            
            # Log first customer details if available
            if db['customers']:
                debug_log.write(f"First customer ID: {db['customers'][0].get('customerId', 'unknown')}\n")
        
        def convert_numpy_to_list(obj):
            """Convert numpy arrays to lists in a nested structure"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        with self.customer_db_lock:
            try:
                print(f"[CUSTOMER-DEBUG] Acquired customer_db_lock for saving")
                # Create directory if it doesn't exist
                if not os.path.exists(os.path.dirname(CUSTOMER_DB_FILE)):
                    print(f"[CUSTOMER-DEBUG] Creating directory: {os.path.dirname(CUSTOMER_DB_FILE)}")
                    os.makedirs(os.path.dirname(CUSTOMER_DB_FILE), exist_ok=True)
                
                print(f"[CUSTOMER-DEBUG] Converting data to serializable format")
                # Convert all numpy arrays to regular lists before serialization
                db_cleaned = convert_numpy_to_list(db)
                
                # Add a backup file with the raw data
                backup_file = f"{STORAGE_DIR}/customer_db_backup_{int(time.time())}.json"
                print(f"[CUSTOMER-DEBUG] Saving backup to: {backup_file}")
                with open(backup_file, 'w') as f:
                    json.dump(db_cleaned, f, indent=4)
                    print(f"[CUSTOMER-DEBUG] Backup file created successfully")
                
                print(f"[CUSTOMER-DEBUG] Opening file for writing: {CUSTOMER_DB_FILE}")
                with open(CUSTOMER_DB_FILE, 'w') as f:
                    print(f"[CUSTOMER-DEBUG] Dumping JSON to file")
                    json.dump(db_cleaned, f, indent=4)
                    print(f"[CUSTOMER-DEBUG] Successfully wrote customer database to file with {len(db_cleaned['customers'])} customers")
                
                # Verify the saved file
                with open(debug_log_path, 'a') as debug_log:
                    debug_log.write(f"Save completed. Verifying file...\n")
                    if os.path.exists(CUSTOMER_DB_FILE):
                        debug_log.write(f"File exists: {CUSTOMER_DB_FILE}, size: {os.path.getsize(CUSTOMER_DB_FILE)} bytes\n")
                        try:
                            with open(CUSTOMER_DB_FILE, 'r') as verify_file:
                                verify_data = json.load(verify_file)
                                debug_log.write(f"File loaded successfully. Contains {len(verify_data.get('customers', []))} customers\n")
                        except Exception as e:
                            debug_log.write(f"Error loading file for verification: {str(e)}\n")
                    else:
                        debug_log.write(f"ERROR: File does not exist after saving!\n")
                
                return True
            except Exception as e:
                error_msg = f"Error saving customer database: {str(e)}"
                logger.error(error_msg)
                print(f"[CUSTOMER-DEBUG] ERROR in save_customer_db: {error_msg}")
                
                # Log the error to the debug file
                with open(debug_log_path, 'a') as debug_log:
                    debug_log.write(f"ERROR in save_customer_db: {str(e)}\n")
                    debug_log.write(f"Exception type: {type(e)}\n")
                    # Get stack trace
                    import traceback
                    debug_log.write(f"Stack trace:\n{traceback.format_exc()}\n")
                
                # Try to diagnose the issue
                print(f"[CUSTOMER-DEBUG] Diagnosing save error:")
                try:
                    # Check if the directory exists and is writable
                    dir_path = os.path.dirname(CUSTOMER_DB_FILE)
                    print(f"[CUSTOMER-DEBUG] Directory exists: {os.path.exists(dir_path)}")
                    print(f"[CUSTOMER-DEBUG] Directory writable: {os.access(dir_path, os.W_OK)}")
                    
                    # Check if we have a very large database that might be causing issues
                    customers_count = len(db.get("customers", []))
                    embeddings_count = len(db.get("customer_embeddings", []))
                    print(f"[CUSTOMER-DEBUG] Database size: {customers_count} customers, {embeddings_count} embedding records")
                except Exception as diag_e:
                    print(f"[CUSTOMER-DEBUG] Error during diagnosis: {str(diag_e)}")
                
                return False
    
    def save_visit_logs(self, logs):
        """Save visit logs to file"""
        print(f"[VISIT-DEBUG] Starting save_visit_logs with {len(logs['visits'])} visits")
        
        # Create a debug log file to trace the issue
        debug_log_path = f"{STORAGE_DIR}/visit_debug.log"
        with open(debug_log_path, 'a') as debug_log:
            debug_log.write(f"\n\n=== SAVE ATTEMPT at {datetime.now().isoformat()} ===\n")
            debug_log.write(f"Visits count: {len(logs.get('visits', []))}\n")
            
            # Log statistics details
            stats = logs.get('statistics', {})
            debug_log.write(f"Statistics: total_visits={stats.get('total_visits', 0)}, " +
                           f"unique_customers={stats.get('unique_customers', 0)}, " +
                           f"repeat_customers={stats.get('repeat_customers', 0)}, " +
                           f"staff_entries={stats.get('staff_entries', 0)}\n")
        
        def convert_numpy_to_list(obj):
            """Convert numpy arrays to lists in a nested structure"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_to_list(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_to_list(item) for item in obj]
            else:
                return obj
        
        with self.visit_log_lock:
            try:
                print(f"[VISIT-DEBUG] Acquired visit_log_lock for saving")
                # Create directory if it doesn't exist
                if not os.path.exists(os.path.dirname(VISIT_LOG_FILE)):
                    print(f"[VISIT-DEBUG] Creating directory: {os.path.dirname(VISIT_LOG_FILE)}")
                    os.makedirs(os.path.dirname(VISIT_LOG_FILE), exist_ok=True)
                
                print(f"[VISIT-DEBUG] Converting data to serializable format")
                # Convert all numpy arrays to regular lists before serialization
                logs_cleaned = convert_numpy_to_list(logs)
                
                # Comment out backup creation to save disk space and processing time
                # # Add a backup file with the raw data
                # backup_file = f"{STORAGE_DIR}/visit_logs_backup_{int(time.time())}.json"
                # print(f"[VISIT-DEBUG] Saving backup to: {backup_file}")
                # with open(backup_file, 'w') as f:
                #     json.dump(logs_cleaned, f, indent=4)
                #     print(f"[VISIT-DEBUG] Backup file created successfully")
                
                print(f"[VISIT-DEBUG] Opening file for writing: {VISIT_LOG_FILE}")
                with open(VISIT_LOG_FILE, 'w') as f:
                    print(f"[VISIT-DEBUG] Dumping JSON to file")
                    json.dump(logs_cleaned, f, indent=4)
                    print(f"[VISIT-DEBUG] Successfully wrote visit logs to file with {len(logs_cleaned['visits'])} visits")
                
                # Verify the saved file
                with open(debug_log_path, 'a') as debug_log:
                    debug_log.write(f"Save completed. Verifying file...\n")
                    if os.path.exists(VISIT_LOG_FILE):
                        debug_log.write(f"File exists: {VISIT_LOG_FILE}, size: {os.path.getsize(VISIT_LOG_FILE)} bytes\n")
                        try:
                            with open(VISIT_LOG_FILE, 'r') as verify_file:
                                verify_data = json.load(verify_file)
                                debug_log.write(f"File loaded successfully. Contains {len(verify_data.get('visits', []))} visits\n")
                        except Exception as e:
                            debug_log.write(f"Error loading file for verification: {str(e)}\n")
                    else:
                        debug_log.write(f"ERROR: File does not exist after saving!\n")
                
                return True
            except Exception as e:
                error_msg = f"Error saving visit logs: {str(e)}"
                logger.error(error_msg)
                print(f"[VISIT-DEBUG] ERROR in save_visit_logs: {error_msg}")
                
                # Log the error to the debug file
                with open(debug_log_path, 'a') as debug_log:
                    debug_log.write(f"ERROR in save_visit_logs: {str(e)}\n")
                    debug_log.write(f"Exception type: {type(e)}\n")
                    # Get stack trace
                    import traceback
                    debug_log.write(f"Stack trace:\n{traceback.format_exc()}\n")
                
                # Try to diagnose the issue
                print(f"[VISIT-DEBUG] Diagnosing save error:")
                try:
                    # Check if the directory exists and is writable
                    dir_path = os.path.dirname(VISIT_LOG_FILE)
                    print(f"[VISIT-DEBUG] Directory exists: {os.path.exists(dir_path)}")
                    print(f"[VISIT-DEBUG] Directory writable: {os.access(dir_path, os.W_OK)}")
                    
                    # Check if we have a very large file that might be causing issues
                    visits_count = len(logs.get("visits", []))
                    print(f"[VISIT-DEBUG] Logs size: {visits_count} visits")
                except Exception as diag_e:
                    print(f"[VISIT-DEBUG] Error during diagnosis: {str(diag_e)}")
                
                return False
    
    def find_matching_staff(self, face_embedding, threshold=STAFF_MATCH_THRESHOLD):
        """
        Find matching staff in the database for a given face embedding
        
        Args:
            face_embedding: Face embedding vector to match
            threshold: Similarity threshold (higher is more lenient)
            
        Returns:
            Dictionary with best matching staff info or None
        """
        staff_db = self.load_staff_db()
        
        best_match = None
        best_distance = float('inf')
        best_embedding_id = None
        
        # Ensure the face_embedding is in the right format (numpy array)
        if isinstance(face_embedding, list):
            face_embedding_array = np.array(face_embedding)
        else:
            face_embedding_array = face_embedding
        
        # Debug log for troubleshooting
        logger.debug(f"Looking for staff match with embedding shape: {face_embedding_array.shape}")
        
        # Go through each staff member's embeddings
        for staff_emb in staff_db.get("staff_embeddings", []):
            for embedding in staff_emb.get("embeddings", []):
                vector = embedding.get("vector", [])
                if not vector:
                    continue
                    
                # Calculate face distance
                try:
                    # Ensure vector is numpy array
                    if isinstance(vector, list):
                        vector_array = np.array(vector)
                    else:
                        vector_array = vector
                    
                    # Ensure shapes match
                    if vector_array.shape != face_embedding_array.shape:
                        logger.warning(f"Shape mismatch: {vector_array.shape} vs {face_embedding_array.shape}")
                        continue
                    
                    # Use face_recognition's distance if available
                    try:
                        import face_recognition
                        # The known face encodings must be the first argument
                        distance = face_recognition.face_distance([vector_array], face_embedding_array)[0]
                        logger.debug(f"Face distance: {distance}, threshold: {threshold}")
                    except Exception as e:
                        # Fallback to euclidean distance
                        from scipy.spatial import distance as dist
                        distance = dist.euclidean(vector_array, face_embedding_array)
                        # Normalize to 0-1 range (approximating face_recognition's behavior)
                        distance = min(1.0, distance / 2.0)
                        logger.debug(f"Euclidean distance: {distance}, threshold: {threshold}")
                except Exception as e:
                    logger.error(f"Error calculating face distance: {str(e)}")
                    continue
                
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
                            "distance": best_distance,
                            "confidence": 1.0 - best_distance  # Convert distance to confidence score
                        }
        
        # Return the best match if it meets the threshold
        if best_match and best_distance <= threshold:
            logger.info(f"Found staff match: {best_match['firstName']} {best_match['lastName']} with confidence {best_match['confidence']:.2f}")
            return best_match
        elif best_match:
            logger.info(f"Found potential staff match but below threshold: {best_match['firstName']} {best_match['lastName']} with distance {best_distance:.2f} (threshold: {threshold})")
        else:
            logger.info("No staff match found")
        
        return None
    
    def find_matching_customer(self, face_embedding, threshold=CUSTOMER_MATCH_THRESHOLD):
        """
        Find matching customer in the database for a given face embedding
        
        Args:
            face_embedding: Face embedding vector to match
            threshold: Similarity threshold (higher is more lenient)
            
        Returns:
            Dictionary with best matching customer info or None
        """
        customer_db = self.load_customer_db()
        
        best_match = None
        best_distance = float('inf')
        best_embedding_id = None
        
        # Ensure the face_embedding is in the right format (numpy array)
        if isinstance(face_embedding, list):
            face_embedding_array = np.array(face_embedding)
        else:
            face_embedding_array = face_embedding
        
        # Go through each customer's embeddings
        for cust_emb in customer_db.get("customer_embeddings", []):
            for embedding in cust_emb.get("embeddings", []):
                vector = embedding.get("vector", [])
                if not vector:
                    continue
                    
                # Calculate face distance
                try:
                    # Ensure vector is numpy array
                    if isinstance(vector, list):
                        vector_array = np.array(vector)
                    else:
                        vector_array = vector
                    
                    # Use face_recognition's distance if available
                    try:
                        import face_recognition
                        # The known face encodings must be the first argument
                        distance = face_recognition.face_distance([vector_array], face_embedding_array)[0]
                    except Exception as e:
                        # Fallback to euclidean distance
                        from scipy.spatial import distance as dist
                        distance = dist.euclidean(vector_array, face_embedding_array)
                        # Normalize to 0-1 range (approximating face_recognition's behavior)
                        distance = min(1.0, distance / 2.0)
                except Exception as e:
                    logger.error(f"Error calculating face distance: {str(e)}")
                    continue
                
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
                            "lastSeen": customer_info.get("lastSeen", ""),
                            "embeddingId": best_embedding_id,
                            "distance": best_distance,
                            "confidence": 1.0 - best_distance  # Convert distance to confidence score
                        }
        
        # Return the best match if it meets the threshold
        if best_match and best_distance <= threshold:
            logger.info(f"Found customer match: {best_match['customerId']} with confidence {best_match['confidence']:.2f}")
            return best_match
        
        return None
    
    def add_new_customer(self, face_data, location="Unknown"):
        """
        Add a new customer to the database
        
        Args:
            face_data: Dictionary with face embedding and quality score
            location: Location where customer was detected
            
        Returns:
            The created customer record
        """
        print(f"[CUSTOMER-DEBUG] Starting add_new_customer")
        
        try:
            print(f"[CUSTOMER-DEBUG] Loading customer database")
            customer_db = self.load_customer_db()
            print(f"[CUSTOMER-DEBUG] Customer database loaded successfully")
            
            # Generate a new customer ID
            customer_id = f"CUST_{uuid.uuid4().hex[:8]}"
            now = datetime.now().isoformat()
            print(f"[CUSTOMER-DEBUG] Generated new customer ID: {customer_id}")
            
            # Create customer record
            print(f"[CUSTOMER-DEBUG] Creating customer record")
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
            print(f"[CUSTOMER-DEBUG] Creating embedding record")
            try:
                # Check embedding format
                if "embedding_list" in face_data:
                    print(f"[CUSTOMER-DEBUG] Using embedding_list from face_data - type: {type(face_data['embedding_list'])}")
                    vector = face_data["embedding_list"]
                else:
                    print(f"[CUSTOMER-DEBUG] Using embedding from face_data - type: {type(face_data['embedding'])}")
                    # If it's a numpy array, convert to list
                    if hasattr(face_data["embedding"], "tolist"):
                        vector = face_data["embedding"].tolist()
                        print(f"[CUSTOMER-DEBUG] Converted numpy array to list")
                    else:
                        vector = face_data["embedding"]
                
                # Check if vector is valid
                print(f"[CUSTOMER-DEBUG] Vector type: {type(vector)}, length: {len(vector) if hasattr(vector, '__len__') else 'unknown'}")
                
                embedding_record = {
                    "_id": str(uuid.uuid4()),
                    "customerId": customer_id,
                    "embeddings": [{
                        "embeddingId": f"emb_{customer_id}_001",
                        "vector": vector,
                        "qualityScore": face_data.get("quality_score", 0.5),
                        "captureDate": now,
                        "location": location
                    }],
                    "modelVersion": "face_recognition_v1",
                    "vectorDimension": 128,
                    "lastUpdated": now
                }
                print(f"[CUSTOMER-DEBUG] Embedding record created successfully")
            except Exception as e:
                print(f"[CUSTOMER-DEBUG] Error creating embedding record: {str(e)}")
                embedding_record = {
                    "_id": str(uuid.uuid4()),
                    "customerId": customer_id,
                    "embeddings": [{
                        "embeddingId": f"emb_{customer_id}_001",
                        "vector": [],  # Empty vector as fallback
                        "qualityScore": 0.1,
                        "captureDate": now,
                        "location": location
                    }],
                    "modelVersion": "face_recognition_v1",
                    "vectorDimension": 128,
                    "lastUpdated": now
                }
                print(f"[CUSTOMER-DEBUG] Created fallback embedding record due to error")
            
            # Add to database
            print(f"[CUSTOMER-DEBUG] Adding records to database")
            # Ensure the structures exist
            if "customers" not in customer_db:
                print(f"[CUSTOMER-DEBUG] Creating missing 'customers' list")
                customer_db["customers"] = []
            if "customer_embeddings" not in customer_db:
                print(f"[CUSTOMER-DEBUG] Creating missing 'customer_embeddings' list")
                customer_db["customer_embeddings"] = []
                
            customer_db["customers"].append(customer_record)
            customer_db["customer_embeddings"].append(embedding_record)
            
            # Save database directly, without using a lock (already in a function)
            print(f"[CUSTOMER-DEBUG] Saving customer database")
            save_result = self.save_customer_db(customer_db)
            print(f"[CUSTOMER-DEBUG] Save result: {save_result}")
            
            # Verify the customer was added successfully
            print(f"[CUSTOMER-DEBUG] Verifying customer was added")
            verify_db = self.load_customer_db()
            found = False
            for customer in verify_db.get("customers", []):
                if customer.get("customerId") == customer_id:
                    found = True
                    print(f"[CUSTOMER-DEBUG] Verification successful: customer {customer_id} found in database")
                    break
            if not found:
                print(f"[CUSTOMER-DEBUG] WARNING: Customer {customer_id} not found in database after save!")
                
            # Update statistics
            print(f"[CUSTOMER-DEBUG] Updating statistics")
            with self.visit_log_lock:
                print(f"[CUSTOMER-DEBUG] Loading visit logs")
                visit_logs = self.load_visit_logs()
                print(f"[CUSTOMER-DEBUG] Incrementing unique_customers count")
                visit_logs["statistics"]["unique_customers"] += 1
                print(f"[CUSTOMER-DEBUG] Saving visit logs")
                save_logs_result = self.save_visit_logs(visit_logs)
                print(f"[CUSTOMER-DEBUG] Save logs result: {save_logs_result}")
            
            print(f"[CUSTOMER-DEBUG] Successfully added new customer: {customer_id}")
            return customer_record
        except Exception as e:
            print(f"[CUSTOMER-DEBUG] Critical error in add_new_customer: {str(e)}")
            # Return a minimal valid customer record to prevent further errors
            basic_id = f"CUST_ERROR_{uuid.uuid4().hex[:8]}"
            print(f"[CUSTOMER-DEBUG] Returning emergency fallback customer record: {basic_id}")
            return {
                "customerId": basic_id,
                "type": "error",
                "firstSeen": datetime.now().isoformat(),
                "lastSeen": datetime.now().isoformat(),
                "visitCount": 1,
                "status": "error"
            }
    
    def update_customer(self, customer_match, face_data, location="Unknown"):
        """
        Update an existing customer record with new visit information
        
        Args:
            customer_match: Matched customer record
            face_data: Dictionary with face embedding and quality score
            location: Location where customer was detected
            
        Returns:
            The updated customer ID
        """
        customer_db = self.load_customer_db()
        now = datetime.now().isoformat()
        customer_id = customer_match["customerId"]
        
        # Find and update customer record
        customer_updated = False
        is_repeat_customer = False
        
        with self.customer_db_lock:
            for customer in customer_db["customers"]:
                if customer["customerId"] == customer_id:
                    # Check if this is a new visit (e.g., more than 1 hour since last seen)
                    try:
                        last_seen = datetime.fromisoformat(customer["lastSeen"])
                        current = datetime.fromisoformat(now)
                        time_diff = (current - last_seen).total_seconds()
                        
                        if time_diff > 3600:  # More than 1 hour = new visit
                            customer["visitCount"] += 1
                            customer["visitHistory"].append({
                                "timestamp": now,
                                "location": location,
                                "duration": 0
                            })
                            is_repeat_customer = True
                    except:
                        # If date parsing fails, just update
                        customer["visitCount"] += 1
                    
                    customer["lastSeen"] = now
                    customer_updated = True
                    break
            
            # Update embedding if quality is better
            for emb_record in customer_db["customer_embeddings"]:
                if emb_record["customerId"] == customer_id:
                    # Check if this is a higher quality face
                    existing_quality = max([e.get("qualityScore", 0) for e in emb_record["embeddings"]], default=0)
                    current_quality = face_data.get("quality_score", 0.5)
                    
                    if current_quality > existing_quality + 0.1:  # Only add if significantly better
                        new_embedding = {
                            "embeddingId": f"emb_{customer_id}_{len(emb_record['embeddings']) + 1:03d}",
                            "vector": face_data["embedding_list"] if "embedding_list" in face_data else face_data["embedding"],
                            "qualityScore": current_quality,
                            "captureDate": now,
                            "location": location
                        }
                        emb_record["embeddings"].append(new_embedding)
                        emb_record["lastUpdated"] = now
                        logger.info(f"Added new embedding for customer: {customer_id}")
                    break
            
            # Save the updated database
            self.save_customer_db(customer_db)
        
        # Update statistics if this was a repeat visit
        if is_repeat_customer:
            with self.visit_log_lock:
                visit_logs = self.load_visit_logs()
                visit_logs["statistics"]["repeat_customers"] += 1
                self.save_visit_logs(visit_logs)
        
        if customer_updated:
            logger.info(f"Updated customer: {customer_id}")
        
        return customer_id
    
    def log_recognition_event(self, person_type, person_id, confidence, location="Unknown"):
        """
        Log a recognition event
        
        Args:
            person_type: "staff" or "customer"
            person_id: ID of the recognized person
            confidence: Recognition confidence score
            location: Location where person was detected
            
        Returns:
            The created visit entry
        """
        print(f"[VISIT-DEBUG] Starting log_recognition_event for {person_type} {person_id}")
        
        try:
            print(f"[VISIT-DEBUG] Loading visit logs")
            visit_logs = self.load_visit_logs()
            print(f"[VISIT-DEBUG] Visit logs loaded successfully")
            
            # Ensure structures exist
            if "visits" not in visit_logs:
                print(f"[VISIT-DEBUG] Creating missing 'visits' list")
                visit_logs["visits"] = []
            if "statistics" not in visit_logs:
                print(f"[VISIT-DEBUG] Creating missing 'statistics' dictionary")
                visit_logs["statistics"] = {
                    "total_visits": 0,
                    "unique_customers": 0,
                    "repeat_customers": 0,
                    "staff_entries": 0
                }
                
            now = datetime.now().isoformat()
            print(f"[VISIT-DEBUG] Created timestamp: {now}")
            
            visit_entry = {
                "visitId": str(uuid.uuid4()),
                "timestamp": now,
                "type": person_type,
                "personId": person_id,
                "confidence": confidence,
                "location": location
            }
            
            print(f"[VISIT-DEBUG] Created visit record with ID {visit_entry['visitId']}")
            
            # Add to visit logs
            visit_logs["visits"].append(visit_entry)
            
            # Update statistics
            if person_type == "staff":
                visit_logs["statistics"]["staff_entries"] += 1
            else:  # customer
                visit_logs["statistics"]["total_visits"] += 1
            
            print(f"[VISIT-DEBUG] Updated statistics")
            
            # Save the updated logs
            print(f"[VISIT-DEBUG] Saving visit logs")
            result = self.save_visit_logs(visit_logs)
            print(f"[VISIT-DEBUG] Save result: {result}")
            
            # Verify the visit was logged successfully
            print(f"[VISIT-DEBUG] Verifying visit was logged")
            verify_logs = self.load_visit_logs()
            found = False
            for visit in verify_logs.get("visits", []):
                if visit.get("visitId") == visit_entry["visitId"]:
                    found = True
                    print(f"[VISIT-DEBUG] Verification successful: visit {visit_entry['visitId']} found in logs")
                    break
            if not found:
                print(f"[VISIT-DEBUG] WARNING: Visit {visit_entry['visitId']} not found in logs after save!")
            
            return visit_entry
        except Exception as e:
            print(f"[VISIT-DEBUG] Critical error in log_recognition_event: {str(e)}")
            # Return a minimal valid visit entry to prevent further errors
            emergency_id = str(uuid.uuid4())
            print(f"[VISIT-DEBUG] Returning emergency fallback visit record: {emergency_id}")
            return {
                "visitId": emergency_id,
                "timestamp": datetime.now().isoformat(),
                "type": person_type,
                "personId": person_id,
                "status": "error"
            }
    
    def get_staff_info(self, staff_id):
        """
        Get detailed information about a staff member
        
        Args:
            staff_id: ID of the staff member
            
        Returns:
            Dictionary with staff information or None
        """
        staff_db = self.load_staff_db()
        
        # Find staff record
        staff_info = next((s for s in staff_db.get("staff", []) if s.get("staffId") == staff_id), None)
        
        return staff_info
    
    def get_customer_info(self, customer_id):
        """
        Get detailed information about a customer
        
        Args:
            customer_id: ID of the customer
            
        Returns:
            Dictionary with customer information or None
        """
        customer_db = self.load_customer_db()
        
        # Find customer record
        customer_info = next((c for c in customer_db.get("customers", []) if c.get("customerId") == customer_id), None)
        
        return customer_info
    
    def get_customer_visit_stats(self, customer_id=None):
        """
        Get visit statistics for a customer or all customers
        
        Args:
            customer_id: Optional ID of the customer, if None returns aggregated stats
            
        Returns:
            Dictionary with visit statistics
        """
        visit_logs = self.load_visit_logs()
        customer_db = self.load_customer_db()
        
        if customer_id:
            # Get stats for specific customer
            customer_visits = [v for v in visit_logs.get("visits", []) if v.get("type") == "customer" and v.get("personId") == customer_id]
            
            # Find customer record
            customer = next((c for c in customer_db.get("customers", []) if c.get("customerId") == customer_id), None)
            
            if customer:
                return {
                    "customerId": customer_id,
                    "visitCount": customer.get("visitCount", 0),
                    "firstSeen": customer.get("firstSeen", ""),
                    "lastSeen": customer.get("lastSeen", ""),
                    "visitsToday": len([v for v in customer_visits if self._is_today(v.get("timestamp", ""))]),
                    "visitsThisWeek": len([v for v in customer_visits if self._is_this_week(v.get("timestamp", ""))]),
                    "visitsThisMonth": len([v for v in customer_visits if self._is_this_month(v.get("timestamp", ""))])
                }
            else:
                return None
        else:
            # Get aggregated stats
            return visit_logs.get("statistics", {})
    
    def _is_today(self, timestamp):
        """Check if timestamp is from today"""
        try:
            dt = datetime.fromisoformat(timestamp)
            today = datetime.now().date()
            return dt.date() == today
        except:
            return False
    
    def _is_this_week(self, timestamp):
        """Check if timestamp is from this week"""
        try:
            dt = datetime.fromisoformat(timestamp)
            today = datetime.now()
            # Get start of week (Monday)
            start_of_week = today.replace(hour=0, minute=0, second=0, microsecond=0)
            days_since_monday = today.weekday()
            start_of_week = start_of_week - timedelta(days=days_since_monday)
            return dt >= start_of_week
        except:
            return False
    
    def _is_this_month(self, timestamp):
        """Check if timestamp is from this month"""
        try:
            dt = datetime.fromisoformat(timestamp)
            today = datetime.now()
            return dt.year == today.year and dt.month == today.month
        except:
            return False

# Create a global instance for easy import
db_manager = DatabaseManager() 