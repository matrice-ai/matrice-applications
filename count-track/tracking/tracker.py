import cv2
import numpy as np
from typing import List, Dict, Tuple
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
from collections import defaultdict

def convert_bbox_to_z(bbox):
    """
    Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
    """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w/2.
    y = bbox[1] + h/2.
    s = w * h    #scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))

def convert_x_to_bbox(x,score=None):
    """
    Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
    """
    w = np.sqrt(x[2] * x[3])
    h = x[2] / w
    if(score==None):
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
    else:
        return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))

class KalmanBoxTracker(object):
    """
    This class represents the internal state of individual tracked objects observed as bbox.
    """
    count = 0
    def __init__(self,bbox,class_name):
        """
        Initialises a tracker using initial bounding box.
        """
        #define constant velocity model
        self.kf = KalmanFilter(dim_x=7, dim_z=4) 
        self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
        self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

        self.kf.R[2:,2:] *= 10.
        self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
        self.kf.P *= 10.
        self.kf.Q[-1,-1] *= 0.01
        self.kf.Q[4:,4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox)
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.class_name = class_name

    def update(self,bbox):
        """
        Updates the state vector with observed bbox.
        """
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))

    def predict(self):
        """
        Advances the state vector and returns the predicted bounding box estimate.
        """
        if((self.kf.x[6]+self.kf.x[2])<=0):
            self.kf.x[6] *= 0.0
        self.kf.predict()
        self.age += 1
        if(self.time_since_update>0):
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self):
        """
        Returns the current bounding box estimate.
        """
        return convert_x_to_bbox(self.kf.x)

class ObjectTracker:
    def __init__(self, max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for SORT
        """
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.total_counts = defaultdict(int)  # Track total unique objects
        self.class_tracks = defaultdict(set)  # Track unique IDs per class
        
    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracks with new detections.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame (not used in this implementation)
            
        Returns:
            List of detections with added track IDs
        """
        self.frame_count += 1
        
        # Convert detection bboxes to numpy arrays
        for det in detections:
            det['bbox'] = np.array(det['bbox'])
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)
            
        # Associate detections to trackers
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections, trks)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.trackers[m[1]].update(detections[m[0]]['bbox'])
            detections[m[0]]['track_id'] = self.trackers[m[1]].id
            self.class_tracks[detections[m[0]]['class']].add(self.trackers[m[1]].id)
            self.total_counts[detections[m[0]]['class']] = len(self.class_tracks[detections[m[0]]['class']])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            trk = KalmanBoxTracker(detections[i]['bbox'], detections[i]['class'])
            self.trackers.append(trk)
            detections[i]['track_id'] = trk.id
            self.class_tracks[detections[i]['class']].add(trk.id)
            self.total_counts[detections[i]['class']] = len(self.class_tracks[detections[i]['class']])
            
        # Remove dead trackers
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1))
            i -= 1
            if(trk.time_since_update > self.max_age):
                self.trackers.pop(i)
                
        return detections
    
    def _iou(self, bb_det, bb_trk):
        """
        Computes IOU between two bboxes in the form [x1,y1,x2,y2]
        """
        bb_trk = bb_trk.reshape(-1)
        bb_det = np.array(bb_det)  # Convert list to numpy array
        
        xx1 = np.maximum(bb_det[0], bb_trk[0])
        yy1 = np.maximum(bb_det[1], bb_trk[1])
        xx2 = np.minimum(bb_det[2], bb_trk[2])
        yy2 = np.minimum(bb_det[3], bb_trk[3])
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        wh = w * h
        o = wh / ((bb_det[2]-bb_det[0])*(bb_det[3]-bb_det[1]) + (bb_trk[2]-bb_trk[0])*(bb_trk[3]-bb_trk[1]) - wh)
        return(o)
    
    def _associate_detections_to_trackers(self, detections, trks, iou_threshold=0.3):
        """
        Assigns detections to tracked object (both represented as bounding boxes)
        Returns 3 lists of matches, unmatched_detections and unmatched_trackers
        """
        if(len(trks)==0):
            return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
            
        iou_matrix = np.zeros((len(detections),len(trks)),dtype=np.float32)
        
        for d,det in enumerate(detections):
            for t,trk in enumerate(trks):
                iou_matrix[d,t] = self._iou(det['bbox'], trk)
                
        row_ind, col_ind = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(row_ind, col_ind)))
        
        unmatched_detections = []
        for d, det in enumerate(detections):
            if(d not in matched_indices[:,0]):
                unmatched_detections.append(d)
        unmatched_trackers = []
        for t, trk in enumerate(trks):
            if(t not in matched_indices[:,1]):
                unmatched_trackers.append(t)
                
        #filter out matched with low IOU
        matches = []
        for m in matched_indices:
            if(iou_matrix[m[0],m[1]]<iou_threshold):
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1,2))
        if(len(matches)==0):
            matches = np.empty((0,2),dtype=int)
        else:
            matches = np.concatenate(matches,axis=0)
            
        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)
    
    def get_track_counts(self) -> Dict[str, int]:
        """
        Get count of unique tracks for each class.
        
        Returns:
            Dictionary mapping class names to track counts
        """
        return dict(self.total_counts)
    
    def reset(self):
        """Reset the tracker."""
        self.trackers = []
        self.frame_count = 0
        self.total_counts.clear()
        self.class_tracks.clear()
        KalmanBoxTracker.count = 0 