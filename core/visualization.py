"""
Visualization utilities - Framework agnostic
"""
import cv2
import numpy as np
from typing import Dict, List, Any, Optional, Tuple


def draw_enhanced_visualization(frame: np.ndarray, tracker, detailed_detections: List[Dict[str, Any]], 
                               tracked_objects: Optional[Dict[int, Tuple[int, int]]] = None, 
                               show_heatmap: bool = False, heatmap_data: Optional[np.ndarray] = None,
                               tracker_type: str = "OpenCV") -> np.ndarray:
    """Draw enhanced visualization with tracking overlays"""
    from .detection import create_heatmap_overlay
    
    annotated_frame = frame.copy()
    
    # Apply heatmap overlay if enabled
    if show_heatmap and heatmap_data is not None:
        annotated_frame = create_heatmap_overlay(annotated_frame, heatmap_data)
    
    # Color mapping for different object types
    colors = {
        'person': (0, 255, 0), 'car': (255, 0, 0), 'bicycle': (0, 255, 255),
        'motorcycle': (255, 0, 255), 'bus': (255, 128, 0), 'truck': (128, 0, 255),
        'boat': (0, 128, 255)
    }
    
    # Draw detections
    for det in detailed_detections:
        x1, y1, x2, y2 = det['bbox_x1'], det['bbox_y1'], det['bbox_x2'], det['bbox_y2']
        obj_type = det['object_type']
        confidence = det['confidence']
        track_id = det.get('track_id')
        
        color = colors.get(obj_type, (255, 255, 255))
        thickness = 3 if track_id is not None else 2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Create label
        label = f"{obj_type} ID:{track_id} {confidence:.2f}" if track_id else f"{obj_type} {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        # Draw label background
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (x1, y1 - label_height - baseline - 10), 
                     (x1 + label_width + 10, y1), color, -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Draw label text
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw center point for people
        if track_id is not None and obj_type == 'person':
            cv2.circle(annotated_frame, (det['center_x'], det['center_y']), 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (det['center_x'], det['center_y']), 8, (255, 255, 255), 2)
    
    # Draw tracker overlays
    if tracker_type == "OpenCV" and tracker and hasattr(tracker, 'objects'):
        for object_id, centroid in tracker.objects.items():
            # Enhanced tracking visualization
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 8, (0, 255, 0), -1)
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 12, (255, 255, 255), 2)
            
            track_text = f"ID:{object_id}"
            (text_width, text_height), _ = cv2.getTextSize(track_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            
            # Enhanced ID label
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, 
                         (int(centroid[0]) - text_width//2 - 5, int(centroid[1]) + 15),
                         (int(centroid[0]) + text_width//2 + 5, int(centroid[1]) + text_height + 20),
                         (0, 255, 0), -1)
            cv2.addWeighted(overlay, 0.8, annotated_frame, 0.2, 0, annotated_frame)
            
            cv2.putText(annotated_frame, track_text, 
                       (int(centroid[0]) - text_width//2, int(centroid[1]) + text_height + 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    
    elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"] and tracked_objects:
        for track_id, (center_x, center_y) in tracked_objects.items():
            cv2.circle(annotated_frame, (center_x, center_y), 8, (255, 255, 0), -1)
            cv2.circle(annotated_frame, (center_x, center_y), 12, (0, 0, 255), 2)
            
            track_text = f"YT:{track_id}"
            (text_width, text_height), _ = cv2.getTextSize(track_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.putText(annotated_frame, track_text, 
                       (center_x - text_width//2, center_y + text_height + 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add tracking info overlay
    track_count = len(tracker.objects) if tracker and hasattr(tracker, 'objects') else len(tracked_objects) if tracked_objects else 0
    algo_text = f"Tracking: {tracker_type} ({track_count} active)"
    (algo_width, algo_height), _ = cv2.getTextSize(algo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated_frame, (10, 10), (algo_width + 20, algo_height + 20), (0, 0, 0), -1)
    cv2.putText(annotated_frame, algo_text, (15, algo_height + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame


def calculate_tracking_quality(tracker, detailed_detections: List[Dict[str, Any]]) -> float:
    """Calculate tracking quality score based on various factors"""
    if not hasattr(tracker, 'objects') or len(tracker.objects) == 0:
        return 0.0
    
    total_detections = len(detailed_detections)
    tracked_objects = len(tracker.objects)
    
    # Quality factors
    tracking_ratio = min(tracked_objects / max(total_detections, 1), 1.0)
    confidence_avg = np.mean([det['confidence'] for det in detailed_detections]) if detailed_detections else 0.0
    stability_factor = 0.8  # Could be enhanced with ID consistency tracking
    
    quality_score = (tracking_ratio * 0.4 + confidence_avg * 0.4 + stability_factor * 0.2)
    return quality_score