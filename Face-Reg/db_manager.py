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

# Default thresholds - RELAXED for better matching
STAFF_MATCH_THRESHOLD = 0.8  # Higher threshold means more lenient matching (was 0.6)
CUSTOMER_MATCH_THRESHOLD = 0.7  # Higher threshold means more lenient matching

# Ensure directory exists
os.makedirs(STORAGE_DIR, exist_ok=True)

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
        with self.customer_db_lock:
            if os.path.exists(CUSTOMER_DB_FILE):
                try:
                    with open(CUSTOMER_DB_FILE, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading customer database: {str(e)}")
                    self._initialize_customer_db()
                    return {"customers": [], "customer_embeddings": []}
            else:
                return self._initialize_customer_db()
    
    def load_visit_logs(self):
        """Load visit logs from file"""
        with self.visit_log_lock:
            if os.path.exists(VISIT_LOG_FILE):
                try:
                    with open(VISIT_LOG_FILE, 'r') as f:
                        return json.load(f)
                except Exception as e:
                    logger.error(f"Error loading visit logs: {str(e)}")
                    self._initialize_visit_logs()
                    return {"visits": [], "statistics": {"total_visits": 0, "unique_customers": 0, "repeat_customers": 0, "staff_entries": 0}}
            else:
                return self._initialize_visit_logs()
    
    def _initialize_customer_db(self):
        """Initialize empty customer database"""
        db = {
            "customers": [],
            "customer_embeddings": []
        }
        with open(CUSTOMER_DB_FILE, 'w') as f:
            json.dump(db, f, indent=4)
        return db
    
    def _initialize_visit_logs(self):
        """Initialize empty visit logs"""
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
    
    def save_customer_db(self, db):
        """Save customer database to file"""
        with self.customer_db_lock:
            try:
                with open(CUSTOMER_DB_FILE, 'w') as f:
                    json.dump(db, f, indent=4)
                return True
            except Exception as e:
                logger.error(f"Error saving customer database: {str(e)}")
                return False
    
    def save_visit_logs(self, logs):
        """Save visit logs to file"""
        with self.visit_log_lock:
            try:
                with open(VISIT_LOG_FILE, 'w') as f:
                    json.dump(logs, f, indent=4)
                return True
            except Exception as e:
                logger.error(f"Error saving visit logs: {str(e)}")
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
        customer_db = self.load_customer_db()
        
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
                "vector": face_data["embedding_list"] if "embedding_list" in face_data else face_data["embedding"],
                "qualityScore": face_data.get("quality_score", 0.5),
                "captureDate": now,
                "location": location
            }],
            "modelVersion": "face_recognition_v1",
            "vectorDimension": 128,
            "lastUpdated": now
        }
        
        # Add to database
        with self.customer_db_lock:
            customer_db["customers"].append(customer_record)
            customer_db["customer_embeddings"].append(embedding_record)
            self.save_customer_db(customer_db)
        
        # Update statistics
        with self.visit_log_lock:
            visit_logs = self.load_visit_logs()
            visit_logs["statistics"]["unique_customers"] += 1
            self.save_visit_logs(visit_logs)
        
        logger.info(f"Added new customer: {customer_id}")
        return customer_record
    
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
        with self.visit_log_lock:
            visit_logs = self.load_visit_logs()
            now = datetime.now().isoformat()
            
            visit_entry = {
                "visitId": str(uuid.uuid4()),
                "timestamp": now,
                "type": person_type,
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
            self.save_visit_logs(visit_logs)
            return visit_entry
    
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