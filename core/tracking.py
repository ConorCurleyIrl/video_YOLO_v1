"""
Core Tracking Module - Framework Agnostic
==========================================
Pure tracking logic with no UI dependencies.
Can be imported by any framework (Streamlit, Flask, FastAPI).
"""

import numpy as np
from collections import deque, defaultdict
from datetime import datetime


class TemporalTracker:
    """
    Temporal smoothing and velocity prediction for stable tracking.
    Pure logic class - no framework dependencies.
    """
    
    def __init__(self, smoothing_factor=0.7, prediction_frames=3):
        """
        Initialize temporal tracker.
        
        Args:
            smoothing_factor: Exponential smoothing weight (0-1)
            prediction_frames: Number of frames to predict ahead
        """
        self.smoothing_factor = smoothing_factor
        self.prediction_frames = prediction_frames
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
    
    def smooth_position(self, track_id, new_position):
        """
        Apply exponential smoothing with velocity prediction.
        
        Args:
            track_id: Unique identifier for tracked object
            new_position: Tuple of (x, y) coordinates
            
        Returns:
            Tuple of smoothed (x, y) coordinates
        """
        history = self.track_history[track_id]
        
        if len(history) > 0:
            prev_pos = history[-1]
            smoothed = (
                self.smoothing_factor * np.array(prev_pos) + 
                (1 - self.smoothing_factor) * np.array(new_position)
            )
            
            # Velocity-based prediction
            if len(history) >= 2:
                velocity = np.array(history[-1]) - np.array(history[-2])
                self.velocity_history[track_id].append(velocity)
                
                if len(self.velocity_history[track_id]) > 0:
                    avg_velocity = np.mean(list(self.velocity_history[track_id]), axis=0)
                    predicted = smoothed + avg_velocity * 0.5
                    return tuple(predicted.astype(int))
            
            return tuple(smoothed.astype(int))
        
        return new_position
    
    def clear_track(self, track_id):
        """Remove tracking history for specific track ID"""
        if track_id in self.track_history:
            del self.track_history[track_id]
        if track_id in self.velocity_history:
            del self.velocity_history[track_id]


class SimpleTracker:
    """
    Centroid-based object tracker with temporal consistency.
    Framework-agnostic implementation.
    """
    
    def __init__(self, max_disappeared=30, max_distance=100, use_temporal=True):
        """
        Initialize the tracker.
        
        Args:
            max_disappeared: Frames before removing a tracked object
            max_distance: Maximum pixel distance for matching
            use_temporal: Whether to use temporal smoothing
        """
        self.next_object_id = 0
        self.objects = {}  # object_id: (x, y) centroid
        self.disappeared = {}  # object_id: frames_disappeared
        self.first_seen = {}  # object_id: timestamp
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.use_temporal = use_temporal
        
        if use_temporal:
            self.temporal_tracker = TemporalTracker()
    
    def register(self, centroid, timestamp=None):
        """
        Register a new object with unique ID.
        
        Args:
            centroid: Tuple of (x, y) coordinates
            timestamp: Optional datetime object
            
        Returns:
            Integer object ID
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = timestamp
        
        object_id = self.next_object_id
        self.next_object_id += 1
        return object_id
    
    def deregister(self, object_id):
        """
        Remove an object from tracking.
        
        Args:
            object_id: Integer ID to remove
        """
        for d in [self.objects, self.disappeared, self.first_seen]:
            d.pop(object_id, None)
        
        if self.use_temporal:
            self.temporal_tracker.clear_track(object_id)
    
    def update(self, detections, timestamp=None):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y) centroids from current frame
            timestamp: Optional datetime object
            
        Returns:
            Dictionary of currently tracked objects {id: (x, y)}
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Handle no detections
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects.copy()
        
        # No objects being tracked - register all
        if len(self.objects) == 0:
            for centroid in detections:
                self.register(centroid, timestamp)
            return self.objects.copy()
        
        # Match existing objects with new detections
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())
        
        # Calculate distance matrix
        distances = np.zeros((len(object_centroids), len(detections)))
        for i, obj_centroid in enumerate(object_centroids):
            for j, det_centroid in enumerate(detections):
                distances[i, j] = np.linalg.norm(
                    np.array(obj_centroid) - np.array(det_centroid)
                )
        
        # Greedy matching (simplified Hungarian algorithm)
        rows = distances.min(axis=1).argsort()
        cols = distances.argmin(axis=1)[rows]
        used_rows, used_cols = set(), set()
        
        # Update matched objects
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if distances[row, col] > self.max_distance:
                continue
            
            object_id = object_ids[row]
            
            # Apply temporal smoothing if enabled
            if self.use_temporal:
                smoothed_pos = self.temporal_tracker.smooth_position(
                    object_id, detections[col]
                )
                self.objects[object_id] = smoothed_pos
            else:
                self.objects[object_id] = detections[col]
            
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)
        
        # Mark unmatched objects as disappeared
        unused_rows = set(range(len(object_centroids))) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        # Register new detections
        unused_cols = set(range(len(detections))) - used_cols
        for col in unused_cols:
            self.register(detections[col], timestamp)
        
        return self.objects.copy()
    
    def get_active_tracks(self):
        """
        Get all currently active track IDs.
        
        Returns:
            List of active track IDs
        """
        return list(self.objects.keys())
    
    def get_track_age(self, track_id, current_time=None):
        """
        Get age of a track in seconds.
        
        Args:
            track_id: Track ID to query
            current_time: Optional current datetime
            
        Returns:
            Float seconds or None if track doesn't exist
        """
        if track_id not in self.first_seen:
            return None
        
        if current_time is None:
            current_time = datetime.now()
        
        return (current_time - self.first_seen[track_id]).total_seconds()
    
    def reset(self):
        """Reset tracker to initial state"""
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.first_seen = {}
        
        if self.use_temporal:
            self.temporal_tracker = TemporalTracker()
    
    def get_statistics(self):
        """
        Get tracker statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        return {
            'active_tracks': len(self.objects),
            'total_registered': self.next_object_id,
            'disappeared_tracks': len([v for v in self.disappeared.values() if v > 0]),
            'avg_disappeared_frames': np.mean(list(self.disappeared.values())) if self.disappeared else 0
        }