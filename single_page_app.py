"""
===================================================================================
STREET TRAFFIC ANALYZER WITH OBJECT TRACKING - OPTIMIZED VERSION
===================================================================================
Developer: Conor Curley
Version: 2.0
License: MIT

Cleaned up steps 

Step 1: Imports & Environment Setup
Step 2: Page Configuration
Step 3: Session State Initialization
Step 4: Core Detection & Tracking Classes
Step 5: Helper Functions
Step 6: Preset Configurations
Step 7: UI Header
Step 8: Sidebar Configuration
Step 9: Stream Selection
Step 10: Visualization Area
Step 11: Main Analysis Loop
Step 12: Data Tabs
Step 13: Run Analysis

(Split to Multiple Files)
core/ - Pure Business Logic (No UI dependencies)
    
    * Can be imported by ANY framework
    * No Streamlit/Flask/FastAPI imports
    * Returns data structures (dicts, DataFrames, numpy arrays)
    * No session state or framework-specific features
    tracking.py (Framework-agnostic)
    
utils/ - Framework-Independent Helpers
    * Configuration management
    * Stream extraction
    * Model loading
    * No UI code
    utils/config.py (Pure configuration)
    
interfaces/ - Framework-Specific UI Layers - Streamlit / Flask / Fastapi
    * Each file imports from core/ and utils/
    * Handles its own framework's specifics
    * Translates between framework and core logic
    streamlit_app.py (future)
"""

# ===================================================================================
# STEP 1: IMPORTS AND ENVIRONMENT SETUP
# ===================================================================================

import os
import sys
import subprocess
import streamlit as st
import time
from datetime import datetime
from collections import deque, defaultdict

# Environment configuration for cloud deployment
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def install_missing_packages():
    """Install missing packages with error handling"""
    try:
        import cv2
    except ImportError:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless"])
        except subprocess.CalledProcessError as e:
            st.error(f"Failed to install opencv: {e}")

install_missing_packages()

# Import required packages
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import yt_dlp
    from PIL import Image
    import pandas as pd
    import torch
    import torchvision
except ImportError as e:
    st.error(f"Import error: {e}")
    st.info("Install: pip install opencv-python-headless ultralytics streamlit yt-dlp pandas torch torchvision")
    st.stop()

# ===================================================================================
# STEP 2: PAGE CONFIGURATION
# ===================================================================================

st.set_page_config(page_title="Street Traffic Analyzer", layout="wide")

# Header with refresh button
col1, col3 = st.columns([8, 1])
with col3:
    if st.button("🔄 Refresh", key="refresh"):
        st.session_state.clear()
        st.rerun()

# ===================================================================================
# STEP 3: SESSION STATE INITIALIZATION
# ===================================================================================

def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        'analyzing': False,
        'detection_log': [],
        'all_detections_df': pd.DataFrame(),
        'tracking_history': [],
        'time_series_data': deque(maxlen=500),
        'object_time_series': deque(maxlen=500),
        'unique_people_count': 0,
        'heatmap_data': [],
        'heatmap_enabled': False,
        'tracking_quality': 0.0,
        'current_url': ""
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# ===================================================================================
# STEP 4: CORE DETECTION & TRACKING CLASSES
# ===================================================================================

class EnhancedDetectionFilter:
    """Multi-step detection validation and filtering"""
    
    @staticmethod
    def filter_detections(results, conf_threshold=0.5, nms_threshold=0.4, min_area=100):
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


class TemporalTracker:
    """Temporal smoothing and velocity prediction for stable tracking"""
    
    def __init__(self, smoothing_factor=0.7, prediction_frames=3):
        self.smoothing_factor = smoothing_factor
        self.prediction_frames = prediction_frames
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
    
    def smooth_position(self, track_id, new_position):
        """Apply exponential smoothing with velocity prediction"""
        history = self.track_history[track_id]
        
        if len(history) > 0:
            prev_pos = history[-1]
            smoothed = (
                self.smoothing_factor * np.array(prev_pos) + 
                (1 - self.smoothing_factor) * np.array(new_position)
            )
            
            if len(history) >= 2:
                velocity = np.array(history[-1]) - np.array(history[-2])
                self.velocity_history[track_id].append(velocity)
                
                if len(self.velocity_history[track_id]) > 0:
                    avg_velocity = np.mean(list(self.velocity_history[track_id]), axis=0)
                    predicted = smoothed + avg_velocity * 0.5
                    return tuple(predicted.astype(int))
            
            return tuple(smoothed.astype(int))
        
        return new_position


class SimpleTracker:
    """Centroid-based object tracker with temporal consistency"""
    
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_object_id = 0
        self.objects = {}
        self.disappeared = {}
        self.first_seen = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.temporal_tracker = TemporalTracker()
    
    def register(self, centroid, timestamp):
        """Register new object with unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = timestamp
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove object from tracking"""
        for d in [self.objects, self.disappeared, self.first_seen]:
            d.pop(object_id, None)
    
    def update(self, detections, timestamp):
        """Update tracker with new detections"""
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        if len(self.objects) == 0:
            for centroid in detections:
                self.register(centroid, timestamp)
        else:
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distance matrix
            distances = np.zeros((len(object_centroids), len(detections)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, det_centroid in enumerate(detections):
                    distances[i, j] = np.linalg.norm(np.array(obj_centroid) - np.array(det_centroid))
            
            # Greedy matching
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            used_rows, used_cols = set(), set()
            
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols or distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                smoothed_pos = self.temporal_tracker.smooth_position(object_id, detections[col])
                self.objects[object_id] = smoothed_pos
                self.disappeared[object_id] = 0
                used_rows.add(row)
                used_cols.add(col)
            
            # Handle disappeared objects
            for row in set(range(len(object_centroids))) - used_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections
            for col in set(range(len(detections))) - used_cols:
                self.register(detections[col], timestamp)
        
        return self.objects


# ===================================================================================
# STEP 5: HELPER FUNCTIONS
# ===================================================================================

@st.cache_resource
def load_model(model_name):
    """Load and cache YOLO model"""
    try:
        return YOLO(model_name)
    except Exception as e:
        st.error(f"Failed to load {model_name}: {e}")
        return YOLO('yolov8n.pt')


def get_stream_url(youtube_url):
    """Extract stream URL from YouTube with fallback strategies"""
    ydl_opts = {
        'format': 'best[height<=720]/best[height<=480]/best',
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'referer': 'https://www.youtube.com/',
        'socket_timeout': 30,
        'youtube_include_dash_manifest': False,
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            st.info("📄 Extracting stream information...")
            info = ydl.extract_info(youtube_url, download=False)
            
            if 'url' in info and info['url']:
                st.success("✅ Found direct stream URL")
                return info['url']
            
            if 'formats' in info and info['formats']:
                for fmt in info['formats']:
                    if (fmt.get('url') and fmt.get('vcodec') != 'none' and 
                        fmt.get('protocol') in ['https', 'http'] and fmt.get('height', 0) <= 720):
                        #st.success(f"✅ Selected format: {fmt.get('format_note', 'Unknown')}")
                        return fmt['url']
                
                for fmt in info['formats']:
                    if fmt.get('url') and fmt.get('vcodec') != 'none':
                        st.warning("⚠️ Using fallback format")
                        return fmt['url']
            
            return None
    except Exception as e:
        st.error(f"Stream extraction failed: {str(e)}")
        return None


def analyze_frame(frame, model, filters, conf_threshold, resize_factor, tracker_type, 
                  enhanced_filtering=True, nms_threshold=0.4, min_area=100):
    """Analyze frame with YOLO detection and optional filtering"""
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
        if tracker_type == "ByteTrack (YOLO)":
            results = model.track(frame_resized, conf=conf_threshold, tracker="bytetrack.yaml", verbose=False)[0]
        elif tracker_type == "BoT-SORT (YOLO)":
            results = model.track(frame_resized, conf=conf_threshold, tracker="botsort.yaml", verbose=False)[0]
        else:
            results = model(frame_resized, conf=conf_threshold, verbose=False)[0]
    except Exception:
        results = model(frame_resized, conf=conf_threshold, verbose=False)[0]
    
    # Apply filtering
    if enhanced_filtering:
        valid_indices = EnhancedDetectionFilter.filter_detections(results, conf_threshold, nms_threshold, min_area)
        filtered_boxes = results.boxes[valid_indices] if valid_indices else []
    else:
        filtered_boxes = results.boxes if results.boxes is not None else []
    
    # Process detections
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
                
                if class_name in filters and filters[class_name]:
                    detections[class_name] = detections.get(class_name, 0) + 1
                    
                    bbox_tensor = box.xyxy[0].cpu().numpy()
                    if len(bbox_tensor) >= 4:
                        x1, y1, x2, y2 = [int(float(v) * s) for v, s in 
                                          zip(bbox_tensor[:4], [scale_x, scale_y, scale_x, scale_y])]
                        
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                        
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            try:
                                track_id = int(box.id[0])
                                tracked_objects[track_id] = (center_x, center_y)
                            except (ValueError, IndexError, TypeError):
                                pass
                        
                        if class_name == 'person':
                            person_centroids.append((center_x, center_y))
                        
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


def update_heatmap(detailed_detections, frame_shape, object_type="person", decay_factor=0.95):
    """Update heatmap accumulator with new detections"""
    if 'heatmap_accumulator' not in st.session_state:
        st.session_state.heatmap_accumulator = np.zeros((frame_shape[0]//4, frame_shape[1]//4), dtype=np.float32)
    
    st.session_state.heatmap_accumulator *= decay_factor
    
    for det in detailed_detections:
        if object_type == "all" or det['object_type'] == object_type:
            hx, hy = int(det['center_x'] // 4), int(det['center_y'] // 4)
            if 0 <= hx < st.session_state.heatmap_accumulator.shape[1] and 0 <= hy < st.session_state.heatmap_accumulator.shape[0]:
                for dy in range(-3, 4):
                    for dx in range(-3, 4):
                        nx, ny = hx + dx, hy + dy
                        if 0 <= nx < st.session_state.heatmap_accumulator.shape[1] and 0 <= ny < st.session_state.heatmap_accumulator.shape[0]:
                            distance = np.sqrt(dx*dx + dy*dy)
                            intensity = np.exp(-distance/2) * det['confidence']
                            st.session_state.heatmap_accumulator[ny, nx] += intensity


def create_heatmap_overlay(frame, heatmap_data, alpha=0.6):
    """Create heatmap overlay on frame"""
    if heatmap_data.max() == 0:
        return frame
    
    normalized = (heatmap_data / heatmap_data.max() * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]))
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)


def calculate_tracking_quality(tracker, detailed_detections):
    """Calculate tracking quality score"""
    if not hasattr(tracker, 'objects') or len(tracker.objects) == 0:
        return 0.0
    
    total_detections = len(detailed_detections)
    tracked_objects = len(tracker.objects)
    tracking_ratio = min(tracked_objects / max(total_detections, 1), 1.0)
    confidence_avg = np.mean([det['confidence'] for det in detailed_detections]) if detailed_detections else 0.0
    stability_factor = 0.8
    
    return tracking_ratio * 0.4 + confidence_avg * 0.4 + stability_factor * 0.2


def draw_enhanced_visualization(frame, tracker, detailed_detections, tracked_objects=None, 
                                show_heatmap=False, tracker_type="OpenCV"):
    """Draw enhanced visualization with tracking overlays"""
    annotated_frame = frame.copy()
    
    if show_heatmap and 'heatmap_accumulator' in st.session_state:
        annotated_frame = create_heatmap_overlay(annotated_frame, st.session_state.heatmap_accumulator)
    
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
        
        label = f"{obj_type} ID:{track_id} {confidence:.2f}" if track_id else f"{obj_type} {confidence:.2f}"
        (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        
        overlay = annotated_frame.copy()
        cv2.rectangle(overlay, (x1, y1 - label_height - baseline - 10), 
                     (x1 + label_width + 10, y1), color, -1)
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        cv2.putText(annotated_frame, label, (x1 + 5, y1 - baseline - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        if track_id is not None and obj_type == 'person':
            cv2.circle(annotated_frame, (det['center_x'], det['center_y']), 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (det['center_x'], det['center_y']), 8, (255, 255, 255), 2)
    
    # Draw tracker overlays
    if tracker_type == "OpenCV" and tracker and hasattr(tracker, 'objects'):
        for object_id, centroid in tracker.objects.items():
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 8, (0, 255, 0), -1)
            cv2.circle(annotated_frame, tuple(map(int, centroid)), 12, (255, 255, 255), 2)
    elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"] and tracked_objects:
        for track_id, (center_x, center_y) in tracked_objects.items():
            cv2.circle(annotated_frame, (center_x, center_y), 6, (0, 255, 255), -1)
            cv2.circle(annotated_frame, (center_x, center_y), 10, (255, 255, 255), 2)
    
    # Add tracking info overlay
    track_count = len(tracker.objects) if tracker and hasattr(tracker, 'objects') else len(tracked_objects) if tracked_objects else 0
    algo_text = f"Tracking: {tracker_type} ({track_count} active)"
    (algo_width, algo_height), _ = cv2.getTextSize(algo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated_frame, (10, 10), (algo_width + 20, algo_height + 20), (0, 0, 0), -1)
    cv2.putText(annotated_frame, algo_text, (15, algo_height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return annotated_frame


# ===================================================================================
# STEP 6: PRESET CONFIGURATIONS
# ===================================================================================

PRESET_CONFIGS = {
    'optimal': {
        'frame_skip': 1, 'resize_factor': 0.75, 'model_size': 7, 'tracker_type': 1,
        'temporal_smoothing': False, 'smoothing_factor': 0.7, 'enhanced_filtering': True,
        'nms_threshold': 0.3, 'min_detection_area': 150, 'heatmap_enabled': False,
        'analysis_interval': 1, 'track_people_only': True, 'max_tracking_distance': 150,
        'max_disappeared_frames': 45, 'confidence_threshold': 0.6
    },
    'speed': {
        'frame_skip': 2, 'resize_factor': 0.25, 'model_size': 3, 'tracker_type': 1,
        'temporal_smoothing': False, 'smoothing_factor': 0.5, 'enhanced_filtering': False,
        'nms_threshold': 0.5, 'min_detection_area': 200, 'heatmap_enabled': False,
        'analysis_interval': 3, 'track_people_only': True, 'max_tracking_distance': 80,
        'max_disappeared_frames': 15, 'confidence_threshold': 0.7
    },
    'accuracy': {
        'frame_skip': 1, 'resize_factor': 1.0, 'model_size': 8, 'tracker_type': 2,
        'temporal_smoothing': False, 'smoothing_factor': 0.8, 'enhanced_filtering': True,
        'nms_threshold': 0.2, 'min_detection_area': 50, 'heatmap_enabled': True,
        'analysis_interval': 1, 'track_people_only': False, 'max_tracking_distance': 200,
        'max_disappeared_frames': 60, 'confidence_threshold': 0.4
    }
}

DETECTION_DEFAULTS = {
    'optimal': {'person': True, 'bicycle': True, 'car': True, 'motorcycle': True, 'bus': True, 'truck': True, 'boat': False},
    'speed': {'person': True, 'bicycle': False, 'car': True, 'motorcycle': False, 'bus': False, 'truck': False, 'boat': False},
    'accuracy': {'person': True, 'bicycle': True, 'car': True, 'motorcycle': True, 'bus': True, 'truck': True, 'boat': True}
}

# ===================================================================================
# STEP 7: USER INTERFACE - HEADER
# ===================================================================================

st.markdown("<h1 style='text-align: center;'>🚶 Live Video Analyzer with Enhanced Tracking</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze YouTube live streams with advanced tracking and detection filtering</p>", unsafe_allow_html=True)

# ===================================================================================
# STEP 8: SIDEBAR - CONFIGURATION
# ===================================================================================

with st.sidebar:
    st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 1: Configure Settings</h2>", unsafe_allow_html=True)
    
    # Preset buttons
    with st.container(border=True):
        st.subheader("⚙️ Quick Setup", divider="blue")
        col1, col2, col3 = st.columns(3)
        
        preset_buttons = {
            'optimal': (col1, "🎯 Optimal", "Best balance of accuracy and performance"),
            'speed': (col2, "⚡ Speed", "Maximum performance"),
            'accuracy': (col3, "🏆 Accuracy", "Maximum accuracy")
        }
        
        for preset_name, (col, label, help_text) in preset_buttons.items():
            with col:
                if st.button(label, use_container_width=True, help=help_text):
                    for key, value in PRESET_CONFIGS[preset_name].items():
                        st.session_state[f'{key}_default'] = value
                    for key, value in DETECTION_DEFAULTS[preset_name].items():
                        st.session_state[f'detect_{key}_default'] = value
                    st.session_state.preset_mode = preset_name
                    st.session_state.defaults_set = True
                    st.rerun()
        
        if st.session_state.get('defaults_set', False):
            preset_labels = {'optimal': "🎯 Optimal", 'speed': "⚡ Speed", 'accuracy': "🏆 Accuracy"}
            
    # Performance settings
    with st.container(border=True):
        st.subheader("⚡ Model & Performance", divider="rainbow")
        
        frame_skip = st.slider("Frame Skip", 1, 10, st.session_state.get('frame_skip_default', 1),
                              help="Process every Nth frame")
        resize_factor = st.slider("Image Resize Factor", 0.25, 1.0, st.session_state.get('resize_factor_default', 0.5), 0.25,
                                  help="Resize frames before processing")
        
        model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov10n.pt", "yolov10s.pt", 
                        "yolov10m.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
        model_size = st.selectbox("YOLO Model", model_options, st.session_state.get('model_size_default', 1),
                                 help="v8=stable, v10=faster, v11=newest")
        
        tracker_options = ["OpenCV", "ByteTrack (YOLO)", "BoT-SORT (YOLO)"]
        tracker_type = st.selectbox("Tracking Algorithm", tracker_options, st.session_state.get('tracker_type_default', 0),
                                   help="OpenCV=default, ByteTrack=fast, BoT-SORT=accurate")
        
        if tracker_type == "OpenCV":
            temporal_smoothing = st.checkbox("Temporal Smoothing", st.session_state.get('temporal_smoothing_default', True))
            if temporal_smoothing:
                smoothing_factor = st.slider("Smoothing Factor", 0.5, 0.9, st.session_state.get('smoothing_factor_default', 0.7), 0.1)
        
        st.subheader("🔍 Detection Settings")
        enhanced_filtering = st.checkbox("Enhanced Filtering", st.session_state.get('enhanced_filtering_default', True))
        
        if enhanced_filtering:
            nms_threshold = st.slider("NMS Threshold", 0.1, 0.8, st.session_state.get('nms_threshold_default', 0.4), 0.1)
            min_detection_area = st.slider("Min Detection Area", 50, 500, st.session_state.get('min_detection_area_default', 100), 50)
        
        heatmap_enabled = st.checkbox("Generate Heatmap", st.session_state.get('heatmap_enabled_default', False))
        if heatmap_enabled:
            heatmap_object_type = st.selectbox("Heatmap Object", ["person", "car", "bicycle", "all"], 
                                              st.session_state.get('heatmap_object_type_default', 0))
            heatmap_decay = st.slider("Heatmap Decay", 0.90, 0.99, st.session_state.get('heatmap_decay_default', 0.95), 0.01)
        
        analysis_interval = st.slider("Analysis Interval (sec)", 1, 10, st.session_state.get('analysis_interval_default', 2))
    
    # Tracking settings
    with st.container(border=True):
        st.subheader("🎯 Tracking & Detection", divider="rainbow")
        
        track_people_only = st.checkbox("Track People Only", st.session_state.get('track_people_only_default', True))
        max_tracking_distance = st.slider("Max Tracking Distance", 25, 300, st.session_state.get('max_tracking_distance_default', 100))
        max_disappeared_frames = st.slider("Max Disappeared Frames", 5, 60, st.session_state.get('max_disappeared_frames_default', 30))
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, st.session_state.get('confidence_threshold_default', 0.1), 0.1)
        
        st.subheader("🔍 Detection Filters")
        detect_person = st.checkbox("👤 People", st.session_state.get('detect_person_default', True))
        detect_bicycle = st.checkbox("🚲 Bicycles", st.session_state.get('detect_bicycle_default', True))
        detect_car = st.checkbox("🚗 Cars", st.session_state.get('detect_car_default', True))
        detect_motorcycle = st.checkbox("🏍️ Motorcycles", st.session_state.get('detect_motorcycle_default', True))
        detect_bus = st.checkbox("🚌 Buses", st.session_state.get('detect_bus_default', True))
        detect_truck = st.checkbox("🚚 Trucks", st.session_state.get('detect_truck_default', True))
        detect_boat = st.checkbox("⛵ Boats", st.session_state.get('detect_boat_default', True))
        
        detection_filters = {
            'person': detect_person, 'bicycle': detect_bicycle, 'car': detect_car,
            'motorcycle': detect_motorcycle, 'bus': detect_bus, 'truck': detect_truck, 'boat': detect_boat
        }
    
    # Technical details
    with st.expander("🚀 Technical Details", expanded=False):
        st.markdown("""
        ## 🔥 Tracking Algorithms
        - **OpenCV**: Temporal smoothing, balanced approach
        - **ByteTrack/BoT-SORT**: Built-in YOLO tracking
        
        ## 📊 Metrics
        - **🎯 Quality**: 🟢>80% Excellent, 🟡60-80% Good, 🔴<60% Poor
        - **⚡ FPS**: Target >10 for real-time
        
        ## 🛠️ Models
        **YOLOv8**: Stable (v8s recommended for tracking)
        **YOLOv10**: 20-30% faster inference
        **YOLOv11**: Best accuracy (v11s best overall)
        """)
        st.markdown("---")
        st.markdown("Developed by Conor Curley | [LinkedIn](https://www.linkedin.com/in/ccurleyds/) | License: MIT")

# ===================================================================================
# STEP 9: STREAM SELECTION
# ===================================================================================

st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 2: Select Live Stream</h2>", unsafe_allow_html=True)

PRESET_STREAMS = {
    "Dublin, Ireland - Temple Bar": "https://www.youtube.com/watch?v=u4UZ4UvZXrg",
    "Melbourne, Australia - Intersection": "https://www.youtube.com/watch?v=fOiFXweVdrE",
    "London, UK - Abbey Road": "https://www.youtube.com/watch?v=57w2gYXjRic",
    "Sydney, Australia - Harbour Bridge": "https://www.youtube.com/watch?v=5uZa3-RMFos"
}

col1, col2, col3 = st.columns([1, 2, 1])

with col1:
    with st.container(border=True):
        st.subheader("📹 Stream Selection", divider="grey")
        st.write("Select a preset stream or enter custom URL:")
        
        if st.button("☘️ Dublin - Temple Bar", use_container_width=True, key="dublin_btn"):
            st.session_state.current_url = PRESET_STREAMS["Dublin, Ireland - Temple Bar"]
            st.rerun()
        if st.button("🚌 London - Abbey Road", use_container_width=True, key="london_btn"):
            st.session_state.current_url = PRESET_STREAMS["London, UK - Abbey Road"]
            st.rerun()
        if st.button("🚃 Melbourne - Intersection", use_container_width=True, key="melbourne_btn"):
            st.session_state.current_url = PRESET_STREAMS["Melbourne, Australia - Intersection"]
            st.rerun()
        if st.button("🚢 Sydney - Harbour Bridge", use_container_width=True, key="sydney_btn"):
            st.session_state.current_url = PRESET_STREAMS["Sydney, Australia - Harbour Bridge"]
            st.rerun()
        
        youtube_url = st.text_input("Or enter YouTube URL", value=st.session_state.current_url,
                                    placeholder="https://youtube.com/watch?v=...", key="url_input")

youtube_url = st.session_state.current_url if youtube_url == "" else youtube_url
if youtube_url != st.session_state.current_url:
    st.session_state.current_url = youtube_url

with col2:
    with st.container(border=True):
        st.subheader("📺 Live Stream", divider="grey")
        video_placeholder = st.empty()
        
        if youtube_url and ('youtube.com' in youtube_url or 'youtu.be' in youtube_url):
            video_id = youtube_url.split('watch?v=')[-1].split('&')[0] if 'watch?v=' in youtube_url else youtube_url.split('/')[-1]
            video_placeholder.markdown(
                f'<iframe width="100%" height="357" src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1" frameborder="0" allowfullscreen></iframe>',
                unsafe_allow_html=True
            )

with col3:
    with st.container(border=True):
        st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 3: Punch it!</h2>", unsafe_allow_html=True)
        st.subheader("▶️ Control Panel", divider="grey")
        
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            if st.button("▶️ Start", disabled=st.session_state.analyzing, use_container_width=True, type="primary", key="start_btn"):
                if youtube_url:
                    st.session_state.analyzing = True
                    st.rerun()
                else:
                    st.error("❌ Enter YouTube URL first!")
        
        with btn_col2:
            if st.button("⏹️ Stop", disabled=not st.session_state.analyzing, use_container_width=True, type="secondary", key="stop_btn"):
                st.session_state.analyzing = False
                st.info("⏹️ Analysis stopped")
                st.rerun()
        
        
        if  st.session_state.analyzing:
            st.success("🟢 **Status:** Running")

        if st.button("🗑️ Clear All Data", use_container_width=True, type="secondary", key="clear_btn"):
            for key in ['detection_log', 'tracking_history', 'heatmap_data', 'unique_people_count', 'tracking_quality']:
                if key in st.session_state:
                    if key in ['detection_log', 'tracking_history', 'heatmap_data']:
                        st.session_state[key] = []
                    elif key == 'all_detections_df':
                        st.session_state[key] = pd.DataFrame()
                    else:
                        st.session_state[key] = 0 if key == 'unique_people_count' else 0.0
            st.session_state.time_series_data = deque(maxlen=500)
            st.session_state.object_time_series = deque(maxlen=500)
            st.success("🗑️ Data cleared!")
            st.rerun()

# ===================================================================================
# STEP 10: VISUALIZATION AREA
# ===================================================================================

with st.container(border=True):
    col2, col3 = st.columns([1, 1])
    
    with col2:
        st.subheader("🎯 Detection & Tracking", divider="grey")
        annotated_frame_placeholder = st.empty()
        tracking_quality_placeholder = st.empty()
    
    with col3:
        st.subheader("📈 Multi-Object Tracking", divider="grey")
        time_series_placeholder = st.empty()
    
    summary_placeholder = st.empty()
    summary_placeholder.markdown("**Configure settings and start analysis.**")
    summary_placeholder.markdown("---")
    
    # Metrics
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    unique_metric = col_m1.metric("🆔 Unique People", 0)
    current_metric = col_m2.metric("👁️ Currently Visible", 0)
    peak_metric = col_m3.metric("📈 Peak Count", 0)
    rate_metric = col_m4.metric("⚡ Rate (people/min)", "0.0")
    
    col_m5, col_m6, col_m7, col_m8 = st.columns(4)
    quality_metric = col_m5.metric("🎯 Tracking Quality", f"{st.session_state.tracking_quality:.1%}")
    fps_metric = col_m6.metric("📊 Processing FPS", "0.0")
    detection_metric = col_m7.metric("🔍 Total Detections", 0)
    accuracy_metric = col_m8.metric("✅ Detection Accuracy", "0.0%")
    
    st.markdown("---")

# ===================================================================================
# STEP 11: MAIN ANALYSIS LOOP
# ===================================================================================

def run_enhanced_analysis():
    """Main analysis loop with tracking and visualization"""
    
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        st.session_state.analyzing = False
        return
    
    status_placeholder = st.empty()
    status_placeholder.info("🚀 Initializing analysis...")
    
    # Initialize tracker
    tracker = SimpleTracker(max_disappeared=max_disappeared_frames, max_distance=max_tracking_distance)
    if tracker_type == "OpenCV" and 'temporal_smoothing' in locals() and temporal_smoothing:
        if 'smoothing_factor' in locals():
            tracker.temporal_tracker.smoothing_factor = smoothing_factor
    
    # Load model
    status_placeholder.info("🤖 Loading YOLO model...")
    model = load_model(model_size)
    
    # Get stream URL
    status_placeholder.info("🔗 Extracting stream URL...")
    stream_url = get_stream_url(youtube_url)
    
    if not stream_url:
        st.error("❌ Failed to extract stream URL")
        st.info("""
        **💡 Troubleshooting:**
        - Try a different YouTube live stream
        - Some streams are geo-restricted
        - The stream might not be live
        - YouTube may be blocking automated access
        """)
        st.session_state.analyzing = False
        return
    
    st.success("✅ Stream URL extracted!")
    
    # Open video capture
    status_placeholder.info("📹 Opening video stream...")
    cap = None
    
    for attempt in range(3):
        try:
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 15)
            
            if cap.isOpened():
                ret, test_frame = cap.read()
                if ret and test_frame is not None:
                    st.success("✅ Stream connected!")
                    break
                cap.release()
                cap = None
            
            st.warning(f"⚠️ Attempt {attempt + 1}/3 failed")
            if attempt < 2:
                time.sleep(3)
        except Exception as e:
            st.warning(f"⚠️ Error: {e}")
            if cap:
                cap.release()
                cap = None
            time.sleep(2)
    
    if not cap or not cap.isOpened():
        st.error("❌ Failed to connect to stream")
        st.session_state.analyzing = False
        return
    
    # Initialize counters
    frame_count = processed_count = 0
    last_analysis_time = time.time()
    session_start_time = datetime.now()
    peak_count = total_detections = consecutive_errors = 0
    seen_ids = set()
    processing_times = deque(maxlen=50)
    detection_accuracy_scores = deque(maxlen=100)
    
    # Main loop
    while st.session_state.analyzing:
        frame_start_time = time.time()
        ret, frame = cap.read()
        
        if not ret:
            consecutive_errors += 1
            if consecutive_errors >= 5:
                st.warning("❌ Multiple stream errors. Stopping.")
                break
            st.warning(f"⚠️ Error {consecutive_errors}/5. Retrying...")
            time.sleep(1)
            continue
        
        consecutive_errors = 0
        frame_count += 1
        current_time = time.time()
        
        if frame_count % frame_skip != 0:
            continue
        
        if current_time - last_analysis_time >= analysis_interval:
            processed_count += 1
            timestamp = datetime.now()
            
            try:
                # Analyze frame
                if enhanced_filtering:
                    detections, detailed, person_centroids, tracked_objects = analyze_frame(
                        frame, model, detection_filters, confidence_threshold, resize_factor,
                        tracker_type, enhanced_filtering, nms_threshold, min_detection_area
                    )
                else:
                    detections, detailed, person_centroids, tracked_objects = analyze_frame(
                        frame, model, detection_filters, confidence_threshold, resize_factor, tracker_type
                    )
                
                total_detections += len(detailed)
                
                if detailed:
                    avg_confidence = np.mean([det['confidence'] for det in detailed])
                    detection_accuracy_scores.append(avg_confidence)
                
                # Update heatmap
                if heatmap_enabled:
                    update_heatmap(detailed, frame.shape, heatmap_object_type, heatmap_decay)
                
                # Update tracking
                if tracker_type == "OpenCV":
                    if track_people_only:
                        tracker.update(person_centroids, timestamp)
                        current_people = len(tracker.objects)
                        seen_ids.update(tracker.objects.keys())
                    else:
                        all_centroids = [(det['center_x'], det['center_y']) for det in detailed]
                        tracker.update(all_centroids, timestamp)
                        current_people = len([det for det in detailed if det['object_type'] == 'person'])
                        seen_ids.update(tracker.objects.keys())
                elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
                    if track_people_only:
                        person_tracks = {tid: pos for tid, pos in tracked_objects.items()}
                        current_people = len(person_tracks)
                        seen_ids.update(person_tracks.keys())
                    else:
                        current_people = len([det for det in detailed if det['object_type'] == 'person'])
                        seen_ids.update(tracked_objects.keys())
                else:
                    current_people = len([det for det in detailed if det['object_type'] == 'person'])
                
                # Calculate tracking quality
                if hasattr(tracker, 'objects') and len(tracker.objects) > 0:
                    st.session_state.tracking_quality = calculate_tracking_quality(tracker, detailed)
                else:
                    st.session_state.tracking_quality = 0.8 if tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"] else 0.6
                
                # Draw visualization
                try:
                    if tracker_type == "OpenCV":
                        annotated_frame = draw_enhanced_visualization(frame, tracker, detailed, None, heatmap_enabled, tracker_type)
                    elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
                        annotated_frame = draw_enhanced_visualization(frame, None, detailed, tracked_objects, heatmap_enabled, tracker_type)
                    else:
                        annotated_frame = draw_enhanced_visualization(frame, None, detailed, None, heatmap_enabled, tracker_type)
                except Exception:
                    annotated_frame = frame
                
                # Display frame
                if annotated_frame.shape[1] > 800:
                    display_height = int(annotated_frame.shape[0] * (800 / annotated_frame.shape[1]))
                    display_frame = cv2.resize(annotated_frame, (800, display_height))
                else:
                    display_frame = annotated_frame
                
                annotated_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                annotated_frame_placeholder.image(Image.fromarray(annotated_rgb), use_container_width=True)
                
                # Update metrics
                st.session_state.unique_people_count = len(seen_ids)
                if current_people > peak_count:
                    peak_count = current_people
                
                elapsed_minutes = (timestamp - session_start_time).total_seconds() / 60
                rate = st.session_state.unique_people_count / elapsed_minutes if elapsed_minutes > 0 else 0
                
                processing_time = time.time() - frame_start_time
                processing_times.append(processing_time)
                fps = 1.0 / np.mean(processing_times) if len(processing_times) > 0 else 0
                
                estimated_accuracy = np.mean(detection_accuracy_scores) if detection_accuracy_scores else 0.0
                
                # Update UI metrics
                unique_metric.metric("🆔 Unique People", st.session_state.unique_people_count)
                current_metric.metric("👁️ Currently Visible", current_people)
                peak_metric.metric("📈 Peak Count", peak_count)
                rate_metric.metric("⚡ Rate (people/min)", f"{rate:.1f}")
                quality_metric.metric("🎯 Tracking Quality", f"{st.session_state.tracking_quality:.1%}")
                fps_metric.metric("📊 Processing FPS", f"{fps:.1f}")
                detection_metric.metric("🔍 Total Detections", total_detections)
                accuracy_metric.metric("✅ Detection Accuracy", f"{estimated_accuracy:.1%}")
                
                # Quality indicator
                if st.session_state.tracking_quality > 0.8:
                    quality_text = "🟢 Excellent"
                elif st.session_state.tracking_quality > 0.6:
                    quality_text = "🟡 Good"
                else:
                    quality_text = "🔴 Poor"
                tracking_quality_placeholder.markdown(f"**Tracking Quality:** {quality_text}")
                
                # Summary text
                summary_text = f"**Analysis - {timestamp.strftime('%H:%M:%S')}**\n\n"
                summary_text += f"**Frames:** {processed_count} | **FPS:** {fps:.1f}\n\n"
                summary_text += f"**Tracker:** {tracker_type} | **Quality:** {quality_text}\n\n"
                
                if detections:
                    summary_text += "**Current Detections:**\n"
                    for obj, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
                        summary_text += f"- {obj.title()}: {count}\n"
                else:
                    summary_text += "*No objects detected*\n"
                
                summary_placeholder.markdown(summary_text)
                
                # Update time series
                st.session_state.time_series_data.append({
                    'timestamp': timestamp,
                    'count': current_people,
                    'unique_total': st.session_state.unique_people_count,
                    'tracking_quality': st.session_state.tracking_quality
                })
                
                st.session_state.object_time_series.append({
                    'timestamp': timestamp,
                    'person': detections.get('person', 0),
                    'car': detections.get('car', 0),
                    'bicycle': detections.get('bicycle', 0),
                    'motorcycle': detections.get('motorcycle', 0),
                    'bus': detections.get('bus', 0),
                    'truck': detections.get('truck', 0),
                    'boat': detections.get('boat', 0)
                })
                
                # Update chart
                if len(st.session_state.object_time_series) > 1:
                    obj_df = pd.DataFrame(list(st.session_state.object_time_series))
                    plot_columns = [col for col in ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'boat'] 
                                   if obj_df[col].sum() > 0]
                    if plot_columns:
                        time_series_placeholder.line_chart(obj_df.set_index('timestamp')[plot_columns], height=300)
                
                # Update tracking history
                st.session_state.tracking_history.insert(0, {
                    'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'current_visible': int(current_people),
                    'unique_session': int(st.session_state.unique_people_count),
                    'peak_count': int(peak_count),
                    'tracking_quality': float(st.session_state.tracking_quality),
                    'processing_fps': float(fps),
                    'tracker_algorithm': str(tracker_type)
                })
                
                if len(st.session_state.tracking_history) > 500:
                    st.session_state.tracking_history = st.session_state.tracking_history[:500]
                
                # Update detections dataframe
                if detailed:
                    for det in detailed:
                        det['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        det['frame_number'] = int(frame_count)
                    
                    try:
                        new_df = pd.DataFrame(detailed)
                        if st.session_state.all_detections_df.empty:
                            st.session_state.all_detections_df = new_df
                        else:
                            st.session_state.all_detections_df = pd.concat([new_df, st.session_state.all_detections_df], ignore_index=True)
                        
                        if len(st.session_state.all_detections_df) > 1000:
                            st.session_state.all_detections_df = st.session_state.all_detections_df.head(1000)
                    except Exception:
                        pass
                
                last_analysis_time = current_time
                
            except Exception as e:
                st.warning(f"Analysis error: {e}")
                consecutive_errors += 1
                if consecutive_errors >= 5:
                    break
        
        time.sleep(0.02)
    
    if 'cap' in locals():
        cap.release()
    st.session_state.analyzing = False
    status_placeholder.success("✅ Analysis completed!")


# ===================================================================================
# STEP 12: DATA TABS
# ===================================================================================

st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 4: Explore Data & Export</h2>", unsafe_allow_html=True)
st.subheader("📊 Detailed Analysis", divider="grey")

tab1, tab2, tab3 = st.tabs(["📋 Tracking History", "📊 Detection Data", "📈 Statistics"])

with tab1:
    st.markdown("### 📊 Tracking History")
    if st.session_state.tracking_history:
        tracking_df = pd.DataFrame(st.session_state.tracking_history)
        st.dataframe(tracking_df.head(50), use_container_width=True, hide_index=True)
        
        csv = tracking_df.to_csv(index=False)
        st.download_button("📥 Download", csv, f"tracking_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("🎯 No tracking data yet. Start analysis to collect data.")

with tab2:
    st.markdown("### 🔍 Detection Data")
    if not st.session_state.all_detections_df.empty:
        df = st.session_state.all_detections_df
        st.dataframe(df.head(100), use_container_width=True, hide_index=True)
        
        csv = df.to_csv(index=False)
        st.download_button("📥 Download", csv, f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", "text/csv")
    else:
        st.info("🔍 No detection data yet. Start analysis to populate.")

with tab3:
    st.markdown("### 📈 Session Analytics")
    if st.session_state.tracking_history:
        tracking_df = pd.DataFrame(st.session_state.tracking_history)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Observations", len(tracking_df))
            st.metric("Avg Visible", f"{tracking_df['current_visible'].mean():.1f}")
        with col2:
            st.metric("Peak Visible", int(tracking_df['current_visible'].max()))
            st.metric("Unique People", st.session_state.unique_people_count)
        
        if len(tracking_df) > 1:
            st.line_chart(tracking_df['current_visible'].tail(30))
    else:
        st.info("📈 Statistics will appear once analysis starts.")

# ===================================================================================
# STEP 13: RUN ANALYSIS
# ===================================================================================

if st.session_state.get('analyzing', False):
    run_enhanced_analysis()