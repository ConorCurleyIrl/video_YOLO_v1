"""
Core detection and filtering algorithms - Framework agnostic
"""
import numpy as np
import torch
import torchvision
import cv2
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime


class EnhancedDetectionFilter:
    """Multi-step detection validation and filtering"""
    
    @staticmethod
    def filter_detections(results, conf_threshold: float = 0.5, nms_threshold: float = 0.4, 
                         min_area: int = 100) -> List[int]:
        """Apply enhanced filtering to YOLO results"""
        if results.boxes is None:
            return []
        
        # Step 1: Confidence filtering
        high_conf_mask = results.boxes.conf > conf_threshold
        if not high_conf_mask.any():
            return []
        
        # Step 2: Area filtering
        boxes = results.boxes.xyxy[high_conf_mask]
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        area_mask = areas > min_area
        if not area_mask.any():
            return []
        
        # Step 3: Aspect ratio filtering for people
        classes = results.boxes.cls[high_conf_mask][area_mask]
        class_names = [results.names[int(cls)] for cls in classes]
        final_mask = torch.ones(area_mask.sum(), dtype=torch.bool)
        
        for i, class_name in enumerate(class_names):
            if class_name == 'person':
                box = boxes[area_mask][i]
                width = box[2] - box[0]
                height = box[3] - box[1]
                aspect_ratio = height / (width + 1e-6)
                if not (1.2 <= aspect_ratio <= 4.0):
                    final_mask[i] = False
        
        temp_indices = torch.arange(len(results.boxes))[high_conf_mask][area_mask][final_mask]
        
        # Step 4: Non-Maximum Suppression
        if len(temp_indices) > 1:
            final_boxes = results.boxes[temp_indices]
            keep_indices = torchvision.ops.nms(final_boxes.xyxy, final_boxes.conf, nms_threshold)
            return temp_indices[keep_indices].tolist()
        
        return temp_indices.tolist()


class HeatmapProcessor:
    """Process and manage heatmap data"""
    
    def __init__(self, frame_shape: Tuple[int, int], scale_factor: int = 4):
        self.scale_factor = scale_factor
        self.accumulator = np.zeros((frame_shape[0]//scale_factor, frame_shape[1]//scale_factor), 
                                   dtype=np.float32)
    
    def update(self, detections: List[Dict[str, Any]], object_type: str = "person", 
               decay_factor: float = 0.95) -> None:
        """Update heatmap accumulator with new detections"""
        self.accumulator *= decay_factor
        
        for det in detections:
            if object_type == "all" or det['object_type'] == object_type:
                hx = int(det['center_x'] // self.scale_factor)
                hy = int(det['center_y'] // self.scale_factor)
                
                if (0 <= hx < self.accumulator.shape[1] and 
                    0 <= hy < self.accumulator.shape[0]):
                    
                    # Apply Gaussian-like kernel
                    for dy in range(-3, 4):
                        for dx in range(-3, 4):
                            nx, ny = hx + dx, hy + dy
                            if (0 <= nx < self.accumulator.shape[1] and 
                                0 <= ny < self.accumulator.shape[0]):
                                distance = np.sqrt(dx*dx + dy*dy)
                                intensity = np.exp(-distance/2) * det['confidence']
                                self.accumulator[ny, nx] += intensity
    
    def get_heatmap(self) -> np.ndarray:
        """Return current heatmap data"""
        return self.accumulator.copy()
    
    def reset(self) -> None:
        """Reset heatmap accumulator"""
        self.accumulator.fill(0)


class FrameAnalyzer:
    """Main frame analysis class - framework agnostic"""
    
    def __init__(self, model, detection_config: Dict[str, Any]):
        self.model = model
        self.detection_config = detection_config
        self.filter = EnhancedDetectionFilter()
    
    def analyze_frame(self, frame: np.ndarray, filters: Dict[str, bool], 
                     resize_factor: float = 1.0, tracker_type: str = "OpenCV") -> Tuple[Dict[str, int], List[Dict[str, Any]], List[Tuple[int, int]], Dict[int, Tuple[int, int]]]:
        """
        Analyze frame with YOLO detection and optional filtering
        Returns: (detections_count, detailed_detections, person_centroids, tracked_objects)
        """
        # Resize for performance
        if resize_factor < 1.0:
            height, width = frame.shape[:2]
            new_width, new_height = int(width * resize_factor), int(height * resize_factor)
            frame_resized = cv2.resize(frame, (new_width, new_height))
            scale_x, scale_y = width / new_width, height / new_height
        else:
            frame_resized = frame
            scale_x = scale_y = 1.0
        
        # Run YOLO with appropriate tracker
        try:
            conf_threshold = self.detection_config.get('confidence_threshold', 0.5)
            
            if tracker_type == "ByteTrack (YOLO)":
                results = self.model.track(frame_resized, conf=conf_threshold, 
                                         tracker="bytetrack.yaml", verbose=False)[0]
            elif tracker_type == "BoT-SORT (YOLO)":
                results = self.model.track(frame_resized, conf=conf_threshold, 
                                         tracker="botsort.yaml", verbose=False)[0]
            else:
                results = self.model(frame_resized, conf=conf_threshold, verbose=False)[0]
        except Exception:
            results = self.model(frame_resized, conf=conf_threshold, verbose=False)[0]
        
        # Apply filtering
        enhanced_filtering = self.detection_config.get('enhanced_filtering', True)
        if enhanced_filtering:
            valid_indices = self.filter.filter_detections(
                results, 
                conf_threshold,
                self.detection_config.get('nms_threshold', 0.4),
                self.detection_config.get('min_area', 100)
            )
            filtered_boxes = results.boxes[valid_indices] if valid_indices else []
        else:
            filtered_boxes = results.boxes if results.boxes is not None else []
        
        # Process detections
        return self._process_detections(results, filtered_boxes, filters, scale_x, scale_y, enhanced_filtering)
    
    def _process_detections(self, results, filtered_boxes, filters, scale_x, scale_y, enhanced_filtering):
        """Process filtered detections into structured data"""
        detections = {}
        detailed_detections = []
        person_centroids = []
        tracked_objects = {}
        
        if len(filtered_boxes) > 0:
            for box in filtered_boxes:
                try:
                    class_id = int(box.cls[0])
                    class_name = results.names[class_id]
                    confidence = float(box.conf[0])
                    
                    # Check if this object type is enabled
                    if class_name in filters and filters[class_name]:
                        detections[class_name] = detections.get(class_name, 0) + 1
                        
                        bbox_tensor = box.xyxy[0].cpu().numpy()
                        if len(bbox_tensor) >= 4:
                            x1, y1, x2, y2 = [int(float(v) * s) for v, s in 
                                              zip(bbox_tensor[:4], [scale_x, scale_y, scale_x, scale_y])]
                            
                            if x2 <= x1 or y2 <= y1:
                                continue
                            
                            center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                            
                            # Extract tracking ID if available
                            track_id = None
                            if hasattr(box, 'id') and box.id is not None:
                                try:
                                    track_id = int(box.id[0])
                                    tracked_objects[track_id] = (center_x, center_y)
                                except (ValueError, IndexError, TypeError):
                                    pass
                            
                            # Collect person centroids for OpenCV tracking
                            if class_name == 'person':
                                person_centroids.append((center_x, center_y))
                            
                            # Create detailed detection record
                            detailed_detections.append({
                                'object_type': class_name,
                                'confidence': float(confidence),
                                'confidence_pct': f"{confidence:.1%}",
                                'bbox_x1': x1, 'bbox_y1': y1, 'bbox_x2': x2, 'bbox_y2': y2,
                                'center_x': center_x, 'center_y': center_y,
                                'width': x2 - x1, 'height': y2 - y1,
                                'area_pixels': (x2 - x1) * (y2 - y1),
                                'track_id': track_id,
                                'enhanced_filtered': enhanced_filtering
                            })
                except Exception:
                    continue
        
        return detections, detailed_detections, person_centroids, tracked_objects


def create_heatmap_overlay(frame: np.ndarray, heatmap_data: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    """Create heatmap overlay on frame"""
    if heatmap_data.max() == 0:
        return frame
    
    normalized = (heatmap_data / heatmap_data.max() * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)