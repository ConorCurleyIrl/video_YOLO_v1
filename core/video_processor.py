"""
Core Video Processor - Framework Agnostic
==========================================
Main processing engine that orchestrates tracking and detection.
"""

import cv2
import numpy as np
from datetime import datetime
from collections import deque
from typing import Dict, List, Optional, Tuple

from core.tracking import SimpleTracker
from core.detection import FrameAnalyzer


class VideoProcessor:
    """
    Main video processing engine - framework agnostic.
    Orchestrates detection, tracking, and analysis.
    """
    
    def __init__(self, model, config: Dict):
        """
        Initialize video processor.
        
        Args:
            model: YOLO model instance
            config: Configuration dictionary
        """
        self.model = model
        self.config = config
        
        # Initialize tracking
        self.tracker = SimpleTracker(
            max_disappeared=config.get('max_disappeared_frames', 30),
            max_distance=config.get('max_tracking_distance', 100),
            use_temporal=config.get('temporal_smoothing', True)
        )
        
        # Initialize frame analyzer
        self.frame_analyzer = FrameAnalyzer(
            model=model,
            detection_filters=config.get('detection_filters', {})
        )
        
        # Session statistics
        self.seen_ids = set()
        self.session_start_time = datetime.now()
        self.peak_count = 0
        self.total_detections = 0
        self.frame_count = 0
        self.processed_count = 0
        
        # Performance metrics
        self.processing_times = deque(maxlen=50)
        self.detection_accuracy_scores = deque(maxlen=100)
        
        # Heatmap data
        self.heatmap_accumulator = None
        
        # Time series data
        self.time_series_data = deque(maxlen=500)
        self.object_time_series = deque(maxlen=500)
    
    def process_frame(self, frame: np.ndarray, timestamp: Optional[datetime] = None) -> Dict:
        """
        Process single frame with detection and tracking.
        
        Args:
            frame: OpenCV frame (numpy array)
            timestamp: Optional datetime for the frame
            
        Returns:
            Dictionary with processing results:
                - detections: Object counts
                - detailed_detections: Full detection list
                - tracking: Tracking information
                - metrics: Session metrics
                - annotated_frame: Frame with visualizations
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        start_time = datetime.now()
        self.frame_count += 1
        
        # Analyze frame
        analysis_results = self.frame_analyzer.analyze_frame(
            frame=frame,
            conf_threshold=self.config.get('confidence_threshold', 0.5),
            resize_factor=self.config.get('resize_factor', 1.0),
            tracker_type=self.config.get('tracker_type', 'opencv'),
            enhanced_filtering=self.config.get('enhanced_filtering', True),
            nms_threshold=self.config.get('nms_threshold', 0.4),
            min_area=self.config.get('min_detection_area', 100)
        )
        
        detections = analysis_results['detections']
        detailed = analysis_results['detailed_detections']
        person_centroids = analysis_results['person_centroids']
        tracked_objects = analysis_results['tracked_objects']
        
        # Update statistics
        self.total_detections += len(detailed)
        self.processed_count += 1
        
        if detailed:
            avg_confidence = np.mean([det['confidence'] for det in detailed])
            self.detection_accuracy_scores.append(avg_confidence)
        
        # Update tracking
        tracker_type = self.config.get('tracker_type', 'opencv').lower()
        track_people_only = self.config.get('track_people_only', True)
        
        if tracker_type == "opencv":
            if track_people_only:
                self.tracker.update(person_centroids, timestamp)
                current_people = len(self.tracker.objects)
                self.seen_ids.update(self.tracker.objects.keys())
            else:
                all_centroids = [(det['center']['x'], det['center']['y']) for det in detailed]
                self.tracker.update(all_centroids, timestamp)
                current_people = len([d for d in detailed if d['object_type'] == 'person'])
                self.seen_ids.update(self.tracker.objects.keys())
        else:
            # Using YOLO built-in tracking
            if track_people_only:
                current_people = len(tracked_objects)
                self.seen_ids.update(tracked_objects.keys())
            else:
                current_people = len([d for d in detailed if d['object_type'] == 'person'])
                self.seen_ids.update(tracked_objects.keys())
        
        # Update peak count
        if current_people > self.peak_count:
            self.peak_count = current_people
        
        # Calculate tracking quality
        tracking_quality = self.frame_analyzer.calculate_tracking_quality(
            self.tracker, detailed
        )
        
        # Update heatmap if enabled
        if self.config.get('heatmap_enabled', False):
            self._update_heatmap(detailed, frame.shape)
        
        # Calculate metrics
        elapsed_minutes = (timestamp - self.session_start_time).total_seconds() / 60
        rate = len(self.seen_ids) / elapsed_minutes if elapsed_minutes > 0 else 0
        
        processing_time = (datetime.now() - start_time).total_seconds()
        self.processing_times.append(processing_time)
        fps = 1.0 / np.mean(self.processing_times) if len(self.processing_times) > 0 else 0
        
        estimated_accuracy = np.mean(self.detection_accuracy_scores) if self.detection_accuracy_scores else 0.0
        
        # Store time series data
        self.time_series_data.append({
            'timestamp': timestamp,
            'count': current_people,
            'unique_total': len(self.seen_ids),
            'tracking_quality': tracking_quality
        })
        
        self.object_time_series.append({
            'timestamp': timestamp,
            'person': detections.get('person', 0),
            'car': detections.get('car', 0),
            'bicycle': detections.get('bicycle', 0),
            'motorcycle': detections.get('motorcycle', 0),
            'bus': detections.get('bus', 0),
            'truck': detections.get('truck', 0),
            'boat': detections.get('boat', 0)
        })
        
        # Create annotated frame
        annotated_frame = self._draw_visualization(
            frame, detailed, tracked_objects, tracker_type
        )
        
        # Build result dictionary
        return {
            'detections': detections,
            'detailed_detections': detailed,
            'tracking': {
                'active_tracks': self.tracker.objects if tracker_type == "opencv" else tracked_objects,
                'tracker_type': tracker_type,
                'quality': tracking_quality
            },
            'metrics': {
                'unique_count': len(self.seen_ids),
                'current_visible': current_people,
                'peak_count': self.peak_count,
                'rate_per_minute': rate,
                'processing_fps': fps,
                'total_detections': self.total_detections,
                'detection_accuracy': estimated_accuracy,
                'tracking_quality': tracking_quality
            },
            'annotated_frame': annotated_frame,
            'timestamp': timestamp
        }
    
    def _update_heatmap(self, detailed_detections: List[Dict], frame_shape: Tuple):
        """Update heatmap accumulator"""
        object_type = self.config.get('heatmap_object_type', 'person')
        decay_factor = self.config.get('heatmap_decay', 0.95)
        
        if self.heatmap_accumulator is None:
            self.heatmap_accumulator = np.zeros(
                (frame_shape[0]//4, frame_shape[1]//4), dtype=np.float32
            )
        
        self.heatmap_accumulator *= decay_factor
        
        for det in detailed_detections:
            if object_type == "all" or det['object_type'] == object_type:
                hx = int(det['center']['x'] // 4)
                hy = int(det['center']['y'] // 4)
                
                if (0 <= hx < self.heatmap_accumulator.shape[1] and 
                    0 <= hy < self.heatmap_accumulator.shape[0]):
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            nx, ny = hx + dx, hy + dy
                            if (0 <= nx < self.heatmap_accumulator.shape[1] and 
                                0 <= ny < self.heatmap_accumulator.shape[0]):
                                distance = np.sqrt(dx*dx + dy*dy)
                                intensity = np.exp(-distance/2) * det['confidence']
                                self.heatmap_accumulator[ny, nx] += intensity
    
    def _draw_visualization(self, frame: np.ndarray, detailed_detections: List[Dict],
                           tracked_objects: Dict, tracker_type: str) -> np.ndarray:
        """
        Draw visualizations on frame.
        
        Args:
            frame: Original frame
            detailed_detections: List of detection dictionaries
            tracked_objects: Dictionary of tracked object positions
            tracker_type: Type of tracker being used
            
        Returns:
            Annotated frame
        """
        annotated = frame.copy()
        
        # Apply heatmap overlay if enabled
        if self.config.get('heatmap_enabled', False) and self.heatmap_accumulator is not None:
            if self.heatmap_accumulator.max() > 0:
                normalized = (self.heatmap_accumulator / self.heatmap_accumulator.max() * 255).astype(np.uint8)
                heatmap_resized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]))
                heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
                annotated = cv2.addWeighted(annotated, 0.4, heatmap_colored, 0.6, 0)
        
        # Color mapping for different objects
        colors = {
            'person': (0, 255, 0), 'car': (255, 0, 0), 'bicycle': (0, 255, 255),
            'motorcycle': (255, 0, 255), 'bus': (255, 128, 0), 'truck': (128, 0, 255),
            'boat': (0, 128, 255)
        }
        
        # Draw detections
        for det in detailed_detections:
            bbox = det['bbox']
            obj_type = det['object_type']
            confidence = det['confidence']
            track_id = det.get('track_id')
            
            color = colors.get(obj_type, (255, 255, 255))
            thickness = 3 if track_id is not None else 2
            
            cv2.rectangle(annotated, (bbox['x1'], bbox['y1']), 
                         (bbox['x2'], bbox['y2']), color, thickness)
            
            # Label
            label = f"{obj_type}"
            if track_id is not None:
                label += f" ID:{track_id}"
            label += f" {confidence:.2f}"
            
            (label_w, label_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            
            overlay = annotated.copy()
            cv2.rectangle(overlay, (bbox['x1'], bbox['y1'] - label_h - baseline - 10),
                         (bbox['x1'] + label_w + 10, bbox['y1']), color, -1)
            cv2.addWeighted(overlay, 0.7, annotated, 0.3, 0, annotated)
            
            cv2.putText(annotated, label, (bbox['x1'] + 5, bbox['y1'] - baseline - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Draw center point for tracked people
            if track_id is not None and obj_type == 'person':
                center = det['center']
                cv2.circle(annotated, (center['x'], center['y']), 5, (0, 255, 0), -1)
                cv2.circle(annotated, (center['x'], center['y']), 8, (255, 255, 255), 2)
        
        # Draw tracker overlays
        if tracker_type == "opencv" and hasattr(self.tracker, 'objects'):
            for object_id, centroid in self.tracker.objects.items():
                cv2.circle(annotated, tuple(map(int, centroid)), 8, (0, 255, 0), -1)
                cv2.circle(annotated, tuple(map(int, centroid)), 12, (255, 255, 255), 2)
        elif tracker_type in ["bytetrack", "botsort"] and tracked_objects:
            for track_id, (cx, cy) in tracked_objects.items():
                cv2.circle(annotated, (cx, cy), 6, (0, 255, 255), -1)
                cv2.circle(annotated, (cx, cy), 10, (255, 255, 255), 2)
        
        # Add info overlay
        track_count = len(self.tracker.objects) if tracker_type == "opencv" else len(tracked_objects)
        info_text = f"Tracking: {tracker_type.upper()} ({track_count} active)"
        (text_w, text_h), _ = cv2.getTextSize(info_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (10, 10), (text_w + 20, text_h + 20), (0, 0, 0), -1)
        cv2.putText(annotated, info_text, (15, text_h + 15),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return annotated
    
    def get_session_stats(self) -> Dict:
        """
        Get current session statistics.
        
        Returns:
            Dictionary with session statistics
        """
        elapsed_time = (datetime.now() - self.session_start_time).total_seconds()
        
        return {
            'unique_count': len(self.seen_ids),
            'peak_count': self.peak_count,
            'total_detections': self.total_detections,
            'frames_processed': self.processed_count,
            'elapsed_seconds': elapsed_time,
            'tracker_stats': self.tracker.get_statistics(),
            'avg_fps': 1.0 / np.mean(self.processing_times) if self.processing_times else 0,
            'avg_accuracy': np.mean(self.detection_accuracy_scores) if self.detection_accuracy_scores else 0
        }
    
    def get_time_series_data(self) -> List[Dict]:
        """Get time series data as list of dictionaries"""
        return list(self.time_series_data)
    
    def get_object_time_series(self) -> List[Dict]:
        """Get object-specific time series data"""
        return list(self.object_time_series)
    
    def reset(self):
        """Reset processor to initial state"""
        self.tracker.reset()
        self.seen_ids = set()
        self.session_start_time = datetime.now()
        self.peak_count = 0
        self.total_detections = 0
        self.frame_count = 0
        self.processed_count = 0
        self.processing_times = deque(maxlen=50)
        self.detection_accuracy_scores = deque(maxlen=100)
        self.heatmap_accumulator = None
        self.time_series_data = deque(maxlen=500)
        self.object_time_series = deque(maxlen=500)