"""
===================================================================================
STREET TRAFFIC ANALYZER WITH OBJECT TRACKING
===================================================================================
"""
import os
import sys
import subprocess
import streamlit as st
# Environment setup for cloud deployment
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

def install_missing_packages():
    """Install missing packages with proper error handling"""
    packages_to_install = []
    
    # Check for OpenCV
    try:
        import cv2
    except ImportError:
        packages_to_install.append('opencv-python-headless')
    
    # Install missing packages
    if packages_to_install:
        for package in packages_to_install:
            try:
                print(f"📦 Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")

# Install missing packages
install_missing_packages()

# Now import all required packages
try:
    import cv2
    import numpy as np
    from ultralytics import YOLO
    import yt_dlp
    from PIL import Image
    import time
    from datetime import datetime
    import pandas as pd
    from collections import deque, defaultdict
    import torch
    import torchvision
    print("✅ All packages imported successfully")
except ImportError as e:
    st.error(f"❌ Import error: {e}")
    st.info("Please install missing packages using: pip install opencv-python-headless ultralytics streamlit")
    st.stop()
# ===================================================================================
# PAGE CONFIGURATION
# ===================================================================================

st.set_page_config(page_title="Street Traffic Analyzer", layout="wide")

# ===================================================================================
col1, col3 = st.columns([8,1])
with col3:
    if st.button("🔄 Refresh", key="refresh"):
        st.session_state.clear()
        st.rerun()
# ===================================================================================
# SESSION STATE INITIALIZATION
# ===================================================================================

if 'analyzing' not in st.session_state:
    st.session_state.analyzing = False
if 'detection_log' not in st.session_state:
    st.session_state.detection_log = []
if 'all_detections_df' not in st.session_state:
    st.session_state.all_detections_df = pd.DataFrame()
if 'tracking_history' not in st.session_state:
    st.session_state.tracking_history = []
if 'time_series_data' not in st.session_state:
    st.session_state.time_series_data = deque(maxlen=500)
if 'object_time_series' not in st.session_state:
    st.session_state.object_time_series = deque(maxlen=500)
if 'unique_people_count' not in st.session_state:
    st.session_state.unique_people_count = 0

# ===================================================================================
# DETECTION FILTERING
# ===================================================================================

def enhanced_detection_filtering(results, conf_threshold=0.5, nms_threshold=0.4, min_area=100):
    """Enhanced detection filtering with multiple validation steps"""
    
    if results.boxes is None:
        return []
    
    # Step 1: Confidence filtering
    high_conf_mask = results.boxes.conf > conf_threshold
    
    if not high_conf_mask.any():
        return []
    
    # Step 2: Area filtering (remove tiny detections)
    boxes = results.boxes.xyxy[high_conf_mask]
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_mask = areas > min_area
    
    if not area_mask.any():
        return []
    
    # Step 3: Aspect ratio filtering for people (height > width typically)
    classes = results.boxes.cls[high_conf_mask][area_mask]
    class_names = [results.names[int(cls)] for cls in classes]
    
    # Create final mask
    final_mask = torch.ones(area_mask.sum(), dtype=torch.bool)
    
    for i, class_name in enumerate(class_names):
        if class_name == 'person':
            # People typically have aspect ratio between 1.2 and 4.0
            box = boxes[area_mask][i]
            width = box[2] - box[0]
            height = box[3] - box[1]
            aspect_ratio = height / (width + 1e-6)
            
            if not (1.2 <= aspect_ratio <= 4.0):
                final_mask[i] = False
    
    # Get final indices
    temp_indices = torch.arange(len(results.boxes))[high_conf_mask][area_mask][final_mask]
    
    # Step 4: Apply NMS if needed
    if len(temp_indices) > 1:
        final_boxes = results.boxes[temp_indices]
        keep_indices = torchvision.ops.nms(
            final_boxes.xyxy, 
            final_boxes.conf, 
            nms_threshold
        )
        final_indices = temp_indices[keep_indices]
    else:
        final_indices = temp_indices
    
    return final_indices.tolist()

# ===================================================================================
# Part 1: Tracking Algos
# ===================================================================================

class TemporalTracker:
    """Temporal smoothing and prediction for tracking stability"""
    
    def __init__(self, smoothing_factor=0.7, prediction_frames=3):
        self.smoothing_factor = smoothing_factor
        self.prediction_frames = prediction_frames
        self.track_history = defaultdict(lambda: deque(maxlen=10))
        self.velocity_history = defaultdict(lambda: deque(maxlen=5))
    
    def smooth_position(self, track_id, new_position):
        """Apply temporal smoothing to position"""
        history = self.track_history[track_id]
        
        if len(history) > 0:
            # Exponential smoothing
            prev_pos = history[-1]
            smoothed = (
                self.smoothing_factor * np.array(prev_pos) + 
                (1 - self.smoothing_factor) * np.array(new_position)
            )
            
            # Velocity-based prediction
            if len(history) >= 2:
                velocity = np.array(history[-1]) - np.array(history[-2])
                self.velocity_history[track_id].append(velocity)
                
                # Average velocity for smoother prediction
                if len(self.velocity_history[track_id]) > 0:
                    avg_velocity = np.mean(list(self.velocity_history[track_id]), axis=0)
                    predicted = smoothed + avg_velocity * 0.5  # Damped prediction
                    return tuple(predicted.astype(int))
            
            return tuple(smoothed.astype(int))
        
        return new_position
    
    def update_tracking_with_prediction(self, detections):
        """Apply temporal smoothing to detections"""
        smoothed_detections = []
        
        for det in detections:
            if 'track_id' in det and det['track_id'] is not None:
                track_id = det['track_id']
                current_pos = (det['center_x'], det['center_y'])
                smoothed_pos = self.smooth_position(track_id, current_pos)
                
                det['center_x'], det['center_y'] = smoothed_pos
                det['bbox_x1'] = smoothed_pos[0] - det['width'] // 2
                det['bbox_x2'] = smoothed_pos[0] + det['width'] // 2
                det['bbox_y1'] = smoothed_pos[1] - det['height'] // 2
                det['bbox_y2'] = smoothed_pos[1] + det['height'] // 2
                det['smoothed'] = True
                
                # Update history
                self.track_history[track_id].append(smoothed_pos)
            
            smoothed_detections.append(det)
        
        return smoothed_detections


class SimpleTracker:
    """
    Enhanced simple centroid-based object tracker with temporal consistency.
    """
    
    def __init__(self, max_disappeared=30, max_distance=100):
        """
        Initialize the tracker.
        
        Args:
            max_disappeared: Number of frames before removing a tracked object
            max_distance: Maximum pixel distance to consider a match
        """
        self.next_object_id = 0
        self.objects = {}  # object_id: (x, y) centroid
        self.disappeared = {}  # object_id: frames_disappeared
        self.first_seen = {}  # object_id: timestamp
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.temporal_tracker = TemporalTracker()
    
    def register(self, centroid, timestamp):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = timestamp
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
            del self.disappeared[object_id]
            del self.first_seen[object_id]
    
    def update(self, detections, timestamp):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y) centroids from current frame
            timestamp: Current timestamp
            
        Returns:
            Dictionary of currently tracked objects {id: centroid}
        """
        # Mark all objects as disappeared by default
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects
        
        # If no objects being tracked, register all detections
        if len(self.objects) == 0:
            for centroid in detections:
                self.register(centroid, timestamp)
        else:
            # Match existing objects with new detections
            object_ids = list(self.objects.keys())
            object_centroids = list(self.objects.values())
            
            # Calculate distances between all pairs
            distances = np.zeros((len(object_centroids), len(detections)))
            for i, obj_centroid in enumerate(object_centroids):
                for j, det_centroid in enumerate(detections):
                    distances[i, j] = np.linalg.norm(
                        np.array(obj_centroid) - np.array(det_centroid)
                    )
            
            # Match using greedy approach (simplified Hungarian algorithm)
            rows = distances.min(axis=1).argsort()
            cols = distances.argmin(axis=1)[rows]
            
            used_rows = set()
            used_cols = set()
            
            # Update matched objects with temporal smoothing
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                
                # Apply temporal smoothing
                smoothed_pos = self.temporal_tracker.smooth_position(object_id, detections[col])
                self.objects[object_id] = smoothed_pos
                self.disappeared[object_id] = 0
                
                used_rows.add(row)
                used_cols.add(col)
            
            # Mark unmatched objects as disappeared
            unused_rows = set(range(distances.shape[0])) - used_rows
            for row in unused_rows:
                object_id = object_ids[row]
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            
            # Register new detections
            unused_cols = set(range(len(detections))) - used_cols
            for col in unused_cols:
                self.register(detections[col], timestamp)
        
        return self.objects



# ===================================================================================
# Part 2: UI - HEADER & TITLE
# ===================================================================================
st.markdown("<h1 style='text-align: center;'>🚶 Live Video Analyzer with Enhanced Tracking</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze YouTube live streams with Deep SORT tracking, temporal consistency, and enhanced detection filtering</p>", unsafe_allow_html=True)

if 'heatmap_data' not in st.session_state:
    st.session_state.heatmap_data = []
if 'heatmap_enabled' not in st.session_state:
    st.session_state.heatmap_enabled = False

# ===================================================================================
# SIDEBAR - ENHANCED CONFIGURATION
# ===================================================================================

with st.sidebar:
    
    st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 1: Configure Enhanced Settings</h2>", unsafe_allow_html=True)
    
    # -------------------------------------------------------------------------------
    # DEFAULT SETTINGS BUTTON
    # -------------------------------------------------------------------------------
    cont_defaults = st.container(border=True)
    with cont_defaults:
        st.subheader("⚙️ Quick Setup", divider="blue")
        
        col_def1, col_def2, col_def3 = st.columns(3)  
        
        with col_def1:
            if st.button("🎯 Optimal", use_container_width=True, help="Best balance of accuracy and performance"):
                # UPDATED: Better tracker choice and optimized settings
                st.session_state.frame_skip_default = 1
                st.session_state.resize_factor_default = 0.75  # CHANGED: Higher quality for better tracking
                st.session_state.model_size_default = 7  # CHANGED: yolo11s.pt (best accuracy)
                st.session_state.tracker_type_default = 1  # CHANGED: ByteTrack (YOLO) - more reliable
                st.session_state.temporal_smoothing_default = False  # CHANGED: ByteTrack handles this
                st.session_state.smoothing_factor_default = 0.7
                st.session_state.enhanced_filtering_default = True
                st.session_state.nms_threshold_default = 0.3  # CHANGED: Stricter for better quality
                st.session_state.min_detection_area_default = 150  # CHANGED: Larger for stability
                st.session_state.heatmap_enabled_default = False
                st.session_state.heatmap_object_type_default = 0
                st.session_state.heatmap_decay_default = 0.95
                st.session_state.analysis_interval_default = 1  # CHANGED: More responsive
                st.session_state.track_people_only_default = True
                st.session_state.max_tracking_distance_default = 150  # CHANGED: Increased for better continuity
                st.session_state.max_disappeared_frames_default = 45  # CHANGED: More persistence
                st.session_state.confidence_threshold_default = 0.6  # CHANGED: Higher for quality
                st.session_state.detect_person_default = True
                st.session_state.detect_bicycle_default = True
                st.session_state.detect_car_default = True
                st.session_state.detect_motorcycle_default = True
                st.session_state.detect_bus_default = True
                st.session_state.detect_truck_default = True
                st.session_state.detect_boat_default = False  # CHANGED: Most streams don't have boats
                st.session_state.preset_mode = "optimal"
                st.session_state.defaults_set = True
                st.rerun()
        
        with col_def2:
            if st.button("⚡ Speed", use_container_width=True, help="Maximum performance for real-time analysis"):
                # UPDATED: Optimized for pure speed
                st.session_state.frame_skip_default = 2  # CHANGED: Still responsive but faster
                st.session_state.resize_factor_default = 0.25
                st.session_state.model_size_default = 3  # KEEP: yolov10n.pt (fastest)
                st.session_state.tracker_type_default = 1  # KEEP: ByteTrack (YOLO) - fastest built-in
                st.session_state.temporal_smoothing_default = False
                st.session_state.smoothing_factor_default = 0.5  # CHANGED: Minimal if needed
                st.session_state.enhanced_filtering_default = False  # KEEP: Skip extra processing
                st.session_state.nms_threshold_default = 0.5  # CHANGED: Faster processing
                st.session_state.min_detection_area_default = 200  # CHANGED: Larger = fewer objects = faster
                st.session_state.heatmap_enabled_default = False
                st.session_state.heatmap_object_type_default = 0
                st.session_state.heatmap_decay_default = 0.95
                st.session_state.analysis_interval_default = 3  # CHANGED: Less frequent analysis
                st.session_state.track_people_only_default = True
                st.session_state.max_tracking_distance_default = 80  # CHANGED: Shorter = faster matching
                st.session_state.max_disappeared_frames_default = 15  # CHANGED: Drop faster = less overhead
                st.session_state.confidence_threshold_default = 0.7  # CHANGED: Higher = fewer detections = faster
                st.session_state.detect_person_default = True
                st.session_state.detect_bicycle_default = False  # CHANGED: Focus on essentials only
                st.session_state.detect_car_default = True
                st.session_state.detect_motorcycle_default = False
                st.session_state.detect_bus_default = False
                st.session_state.detect_truck_default = False
                st.session_state.detect_boat_default = False
                st.session_state.preset_mode = "speed"
                st.session_state.defaults_set = True
                st.rerun()
                
        with col_def3:
            if st.button("🏆 Accuracy", use_container_width=True, help="Maximum accuracy for detailed analysis"):
                # NEW: High-accuracy preset for when quality matters most
                st.session_state.frame_skip_default = 1  # Process every frame
                st.session_state.resize_factor_default = 1.0  # CHANGED: Full resolution
                st.session_state.model_size_default = 8  # CHANGED: yolo11m.pt (most accurate)
                st.session_state.tracker_type_default = 2  # CHANGED: BoT-SORT (YOLO) - most accurate
                st.session_state.temporal_smoothing_default = False  # BoT-SORT handles this better
                st.session_state.smoothing_factor_default = 0.8
                st.session_state.enhanced_filtering_default = True
                st.session_state.nms_threshold_default = 0.2  # CHANGED: Very strict overlap removal
                st.session_state.min_detection_area_default = 50   # CHANGED: Catch smaller objects
                st.session_state.heatmap_enabled_default = True   # CHANGED: Enable for detailed analysis
                st.session_state.heatmap_object_type_default = 0
                st.session_state.heatmap_decay_default = 0.98     # CHANGED: Longer memory
                st.session_state.analysis_interval_default = 1    # Analyze every second
                st.session_state.track_people_only_default = False # CHANGED: Track all objects
                st.session_state.max_tracking_distance_default = 200 # CHANGED: Very persistent
                st.session_state.max_disappeared_frames_default = 60  # CHANGED: Very persistent
                st.session_state.confidence_threshold_default = 0.4   # CHANGED: Lower for more detections
                st.session_state.detect_person_default = True
                st.session_state.detect_bicycle_default = True
                st.session_state.detect_car_default = True
                st.session_state.detect_motorcycle_default = True
                st.session_state.detect_bus_default = True
                st.session_state.detect_truck_default = True
                st.session_state.detect_boat_default = True  # CHANGED: Detect everything
                st.session_state.preset_mode = "accuracy"
                st.session_state.defaults_set = True
                st.rerun()
        
        # Show current preset if defaults were set
        if st.session_state.get('defaults_set', False):
            preset_mode = st.session_state.get('preset_mode', 'optimal')
            if preset_mode == "optimal":
                st.success("🎯 Optimal Defaults Applied")
            elif preset_mode == "performance":
                st.success("⚡ Performance Mode Applied")
            elif preset_mode == "ultra_light":
                st.success("💡 Ultra Light Mode Applied")

    # -------------------------------------------------------------------------------
    # Performance Settings (Updated with default values)
    # -------------------------------------------------------------------------------
    cont1 = st.container(border=True)
    with cont1:
        st.subheader("⚡ Model & Performance Settings", divider="rainbow")
        
        frame_skip = st.slider(
            "Frame Skip", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get('frame_skip_default', 1),
            help="Process every Nth frame. Higher = faster but less frequent updates"
        )
        
        resize_factor = st.slider(
            "Image Resize Factor", 
            min_value=0.25, 
            max_value=1.0, 
            value=st.session_state.get('resize_factor_default', 0.5), 
            step=0.25,
            help="Resize frames before processing. Higher = better accuracy for tracking"
        )
        
        model_options = ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov10n.pt", "yolov10s.pt", "yolov10m.pt", "yolo11n.pt", "yolo11s.pt", "yolo11m.pt"]
        model_size = st.selectbox(
            "YOLO Model Size", 
            options=model_options,
            index=st.session_state.get('model_size_default', 1),  # Default to yolov8s for better tracking
            help="v8=stable, v10=faster, v11=newest & most accurate"
        )
        
        tracker_options = ["OpenCV", "ByteTrack (YOLO)", "BoT-SORT (YOLO)"]
        tracker_type = st.selectbox(
            "Tracking Algorithm",
            options=tracker_options,
            index=st.session_state.get('tracker_type_default', 0),
            help="OpenCV = default tracking, ByteTrack = fast & robust, BoT-SORT = most accurate"
        )

        # Enhanced tracking settings
        if tracker_type == "OpenCV":
            temporal_smoothing = st.checkbox(
                "Temporal Smoothing", 
                value=st.session_state.get('temporal_smoothing_default', True), 
                help="Apply temporal consistency for smoother tracking"
            )
            
            if temporal_smoothing:
                smoothing_factor = st.slider(
                    "Smoothing Factor",
                    min_value=0.5,
                    max_value=0.9,
                    value=st.session_state.get('smoothing_factor_default', 0.7),
                    step=0.1,
                    help="Higher = more smoothing (0.7 recommended)"
                )

        # Enhanced detection filtering
        st.subheader("🔍 Enhanced Detection Settings")
        
        enhanced_filtering = st.checkbox(
            "Enhanced Detection Filtering", 
            value=st.session_state.get('enhanced_filtering_default', True), 
            help="Apply multi-step validation for better detection quality"
        )
        
        if enhanced_filtering:
            nms_threshold = st.slider(
                "NMS Threshold",
                min_value=0.1,
                max_value=0.8,
                value=st.session_state.get('nms_threshold_default', 0.4),
                step=0.1,
                help="Non-Maximum Suppression threshold (lower = less overlap)"
            )
            
            min_detection_area = st.slider(
                "Min Detection Area (pixels)",
                min_value=50,
                max_value=500,
                value=st.session_state.get('min_detection_area_default', 100),
                step=50,
                help="Minimum area to consider a valid detection"
            )

        heatmap_enabled = st.checkbox(
            "Generate Heatmap", 
            value=st.session_state.get('heatmap_enabled_default', False), 
            help="Track object concentration areas"
        )
        
        if heatmap_enabled:
            heatmap_object_options = ["person", "car", "bicycle", "all"]
            heatmap_object_type = st.selectbox(
                "Heatmap Object Type",
                options=heatmap_object_options,
                index=st.session_state.get('heatmap_object_type_default', 0),
                help="Which objects to include in heatmap"
            )
            heatmap_decay = st.slider(
                "Heatmap Decay Factor",
                min_value=0.90,
                max_value=0.99,
                value=st.session_state.get('heatmap_decay_default', 0.95),
                step=0.01,
                help="How quickly heat fades (higher = longer lasting)"
            )

        

        analysis_interval = st.slider(
            "Analysis Interval (seconds)", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get('analysis_interval_default', 2),
            help="How often to analyze frames (lower = more responsive)"
        )
    
    # -------------------------------------------------------------------------------
    # Enhanced Tracking Settings (Updated with default values)
    # -------------------------------------------------------------------------------
    
    cont2 = st.container(border=True)
    with cont2:
        st.subheader("🎯 Enhanced Tracking & Detection Settings", divider="rainbow")
        
        track_people_only = st.checkbox(
            "Track People Only", 
            value=st.session_state.get('track_people_only_default', True),
            help="Focus tracking on pedestrians for better accuracy"
        )
        
        max_tracking_distance = st.slider(
            "Max Tracking Distance (pixels)", 
            min_value=25, 
            max_value=300, 
            value=st.session_state.get('max_tracking_distance_default', 100),  # Increased default for better tracking
            help="Maximum distance to match objects between frames"
        )
        
        max_disappeared_frames = st.slider(
            "Max Disappeared Frames", 
            min_value=5, 
            max_value=60, 
            value=st.session_state.get('max_disappeared_frames_default', 30),  # Increased default for better persistence
            help="Frames before considering object as left scene"
        )
        
        # Enhanced detection confidence
        confidence_threshold = st.slider(
            "Detection Confidence Threshold", 
            min_value=0.1, 
            max_value=0.9, 
            value=st.session_state.get('confidence_threshold_default', 0.1), 
            step=0.1,
            help="Minimum confidence for detections (higher = fewer false positives)"
        )
        
        # -------------------------------------------------------------------------------
        # Detection Filters (Updated with default values)
        # -------------------------------------------------------------------------------
        st.subheader("🔍 Detection Filters")
        
        detect_person = st.checkbox(
            "👤 Detect People", 
            value=st.session_state.get('detect_person_default', True)
        )
        detect_bicycle = st.checkbox(
            "🚲 Detect Bicycles", 
            value=st.session_state.get('detect_bicycle_default', True)
        )
        detect_car = st.checkbox(
            "🚗 Detect Cars", 
            value=st.session_state.get('detect_car_default', True)
        )
        detect_motorcycle = st.checkbox(
            "🏍️ Detect Motorcycles", 
            value=st.session_state.get('detect_motorcycle_default', True)
        )
        detect_bus = st.checkbox(
            "🚌 Detect Buses", 
            value=st.session_state.get('detect_bus_default', True)
        )
        detect_truck = st.checkbox(
            "🚚 Detect Trucks", 
            value=st.session_state.get('detect_truck_default', True)
        )
        detect_boat = st.checkbox(
            "⛵ Detect Boats", 
            value=st.session_state.get('detect_boat_default', True)
        )

        # Create detection filters dictionary (unchanged)
        detection_filters = {
            'person': detect_person,
            'bicycle': detect_bicycle,
            'car': detect_car,
            'motorcycle': detect_motorcycle,
            'bus': detect_bus,
            'truck': detect_truck,
            'boat': detect_boat,
        }
        # Add this after your current detection filters section
        with st.expander("🔍 **View All Detectable Objects** (80+ object types available)", expanded=False):
            st.markdown("### 🎯 **Complete YOLO Detection Capabilities**")
            st.info("💡 **Currently using 7 objects optimized for traffic analysis**. Additional objects can be enabled for specialized use cases.")
            
            # Create organized tabs for different categories
            tab_transport, tab_urban, tab_people, tab_animals, tab_other = st.tabs([
                "🚗 Transportation", "🏙️ Urban & Infrastructure", "👥 People & Items", "🐾 Animals", "🔧 Other Objects"
            ])
            
            with tab_transport:
                st.markdown("#### 🚗 Transportation & Vehicles")
                transport_col1, transport_col2 = st.columns(2)
                
                with transport_col1:
                    st.markdown("**✅ Currently Enabled:**")
                    st.markdown("""
                    - 👥 **person** - Pedestrians
                    - 🚗 **car** - Automobiles  
                    - 🚲 **bicycle** - Bikes
                    - 🏍️ **motorcycle** - Motorcycles
                    """)
                
                with transport_col2:
                    st.markdown("**🔧 Available Options:**")
                    st.markdown("""
                    - 🚌 **bus** - Public buses ✅
                    - 🚛 **truck** - Trucks/lorries ✅  
                    - ⛵ **boat** - Watercraft ✅
                    - ✈️ **airplane** - Aircraft
                    - 🚂 **train** - Trains/locomotives
                    - 🛹 **skateboard** - Alternative transport
                    """)
            
            with tab_urban:
                st.markdown("#### 🏙️ Urban Infrastructure & Street Elements")
                urban_col1, urban_col2 = st.columns(2)
                
                with urban_col1:
                    st.markdown("**🚦 Traffic Management:**")
                    st.markdown("""
                    - 🚦 **traffic light** - Signal analysis
                    - 🛑 **stop sign** - Traffic control
                    - 🅿️ **parking meter** - Parking zones
                    """)
                    st.markdown("**🏗️ Street Furniture:**")
                    st.markdown("""
                    - 🪑 **bench** - Seating areas
                    - 🚒 **fire hydrant** - Safety infrastructure
                    - 🕐 **clock** - Public displays
                    """)
                
                with urban_col2:
                    st.markdown("**💡 High Value for Traffic Analysis:**")
                    st.info("""
                    **🚦 Traffic Lights** - Monitor signal compliance
                    **🛑 Stop Signs** - Track intersection behavior  
                    **🪑 Benches** - Identify waiting/rest areas
                    **🚒 Fire Hydrants** - Map street infrastructure
                    """)
            
            with tab_people:
                st.markdown("#### 👥 People & Personal Items")
                people_col1, people_col2 = st.columns(2)
                
                with people_col1:
                    st.markdown("**👜 Personal Items:**")
                    st.markdown("""
                    - 🎒 **backpack** - Student/commuter tracking
                    - 👜 **handbag** - Personal belongings
                    - 🧳 **suitcase** - Travel behavior
                    - ☂️ **umbrella** - Weather response
                    - 📱 **cell phone** - Device usage
                    """)
                
                with people_col2:
                    st.markdown("**🍕 Food & Beverages:**")
                    st.markdown("""
                    - 🍼 **bottle** - Beverage containers
                    - ☕ **cup** - Drinks
                    - 🍎 **apple, banana, orange** - Fruits
                    - 🥪 **sandwich** - Street food
                    - 🍕 **pizza, hot dog** - Fast food
                    """)
            
            with tab_animals:
                st.markdown("#### 🐾 Animals & Wildlife")
                animal_col1, animal_col2 = st.columns(2)
                
                with animal_col1:
                    st.markdown("**🏙️ Urban Animals:**")
                    st.markdown("""
                    - 🐕 **dog** - Pet tracking
                    - 🐱 **cat** - Stray/pet cats
                    - 🐦 **bird** - Urban wildlife
                    """)
                
                with animal_col2:
                    st.markdown("**🌿 Other Animals:**")
                    st.markdown("""
                    - 🐎 **horse** - Mounted police/transport
                    - 🐄 **cow, sheep** - Livestock
                    - 🐘 **elephant, zebra, giraffe** - Zoo/safari
                    - 🐻 **bear** - Wildlife areas
                    """)
            
            with tab_other:
                st.markdown("#### 🔧 Other Detectable Objects")
                other_col1, other_col2 = st.columns(2)
                
                with other_col1:
                    st.markdown("**⚽ Sports & Recreation:**")
                    st.markdown("""
                    - ⚽ **sports ball** - Various balls
                    - 🪁 **kite** - Park activities  
                    - ⚾ **baseball bat/glove** - Sports
                    - 🎾 **tennis racket** - Court sports
                    - 🏄 **surfboard** - Water sports
                    - 🎿 **skis, snowboard** - Winter sports
                    """)
                
                with other_col2:
                    st.markdown("**🏠 Indoor/Furniture:**")
                    st.markdown("""
                    - 📺 **tv** - Outdoor displays
                    - 💻 **laptop** - Mobile computing
                    - 🪑 **chair, couch** - Outdoor furniture
                    - 🍽️ **dining table** - Outdoor dining
                    - 📚 **book** - Reading materials
                    - 🧸 **teddy bear** - Toys/entertainment
                    """)
            
            # Usage recommendations
            st.warning("⚠️ **Performance Note**: More object types = slower processing. Current 7-object configuration is optimized for real-time traffic analysis.")
            

    with st.expander("🚀 Technical Details", expanded=False):
        st.markdown("""
            ## 🔥 Tracking Algorithms
            - **Open CV**: Used OpenCV for temporal smoothing, good balance
            - **ByteTrack/BoT-SORT**: Built-in YOLO tracking
            
            ## 📊 Metrics Explained
            - **🎯 Tracking Quality**: 🟢>80% Excellent, 🟡60-80% Good, 🔴<60% Poor
            - **⚡ Processing FPS**: Target >10 for real-time
            - **✅ Detection Accuracy**: Average confidence score
            
            ## 🛠️ YOLO Model Options
                    
            **YOLOv8**: Stable and reliable for most tasks
            - v8n: 6MB, ~45 FPS, good for real-time
            - v8s: 22MB, ~35 FPS, **recommended for tracking**  
            - v8m: 50MB, ~25 FPS, highest accuracy
            
            **YOLOv10**: Optimized for speed (20-30% faster inference)
            - v10n: 5MB, ~60 FPS, fastest option
            - v10s: 16MB, ~45 FPS, good balance
            - v10m: 32MB, ~35 FPS, accurate & fast
            
            **YOLOv11**: Latest with best accuracy (10-15% improvement)
            - v11n: 5MB, ~50 FPS, newest nano
            - v11s: 20MB, ~40 FPS, **best overall choice**
            - v11m: 45MB, ~30 FPS, most accurate
            
        """)
        st.markdown("---")

        st.markdown("Developed by Conor Curley | [LinkedIn](https://www.linkedin.com/in/ccurleyds/) | License: MIT")

# ===================================================================================
# MAIN CONTENT AREA - STREAM SELECTION (UNCHANGED)
# ===================================================================================

st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 2: Select Live Stream for Analysis</h2>", unsafe_allow_html=True)

preset_streams = {
    "Dublin, Ireland - Temple Bar": "https://www.youtube.com/watch?v=u4UZ4UvZXrg",
    "Melbourne, Australia - Intersection": "https://www.youtube.com/watch?v=fOiFXweVdrE",
    "London, UK - Abbey Road": "https://www.youtube.com/watch?v=57w2gYXjRic",
    "Sydney, Australia - Harbour Bridge": "https://www.youtube.com/watch?v=5uZa3-RMFos"
}

if 'current_url' not in st.session_state:
    st.session_state.current_url = ""

col1, col2, col3 = st.columns([1,2,1])

with col1:
    cont3 = st.container(border=True)
    with cont3:
        st.subheader("📹 Stream Selection",divider="grey")
        st.write("")
        st.write("Select a preset stream or enter a custom URL:")

        if st.button("☘️ Dublin - Temple Bar", width='stretch'):
            st.session_state.current_url = preset_streams["Dublin, Ireland - Temple Bar"]
            st.rerun()

        if st.button(":bus: London - Abbey Road", width='stretch'):
            st.session_state.current_url = preset_streams["London, UK - Abbey Road"]
            st.rerun()

        if st.button("🚃 Melbourne - Intersection", width='stretch'):
            st.session_state.current_url = preset_streams["Melbourne, Australia - Intersection"]
            st.rerun()

        if st.button(":ship: Sydney - Harbour Bridge", width='stretch'):
            st.session_state.current_url = preset_streams["Sydney, Australia - Harbour Bridge"]
            st.rerun()

        st.write("")
        youtube_url = st.text_input(
            "Or enter YouTube URL", 
            value=st.session_state.current_url,
            placeholder="https://youtube.com/watch?v=...",
            key="url_input"
        )
        

# Sync URL with session state
youtube_url = st.session_state.current_url if youtube_url == "" else youtube_url
if youtube_url != st.session_state.current_url:
    st.session_state.current_url = youtube_url

with col2:
    cont4 = st.container(border=True)
    with cont4:
        st.subheader("📺 Live Stream", divider="grey")
        video_placeholder = st.empty()
        
        # Embed YouTube video if URL provided
        if youtube_url:
            if 'youtube.com' in youtube_url or 'youtu.be' in youtube_url:
                video_id = youtube_url.split('watch?v=')[-1].split('&')[0] if 'watch?v=' in youtube_url else youtube_url.split('/')[-1]
                video_placeholder.markdown(
                    f'<iframe width="100%" height="357" src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1" frameborder="0" allowfullscreen></iframe>', 
                    unsafe_allow_html=True
                )
        st.write("")

with col3:

    cont7 = st.container(border=True)
    with cont7:
        st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 3: Punch it!</h2>", unsafe_allow_html=True)

        st.subheader("▶️ Control Panel", divider="grey")
        st.write("Use the buttons below to start/stop analysis or clear data.")
        
        # Initialize analyzing state if not exists
        if 'analyzing' not in st.session_state:
            st.session_state.analyzing = False
        
        # Create columns for better button layout
        btn_col1, btn_col2 = st.columns(2)
        
        with btn_col1:
            # Start button - only enabled when not analyzing
            if st.button(
                "▶️ Start Analysis", 
                disabled=st.session_state.analyzing,
                width='stretch',
                type="primary",
                key="start_btn"
            ):
                if youtube_url:  # Check if URL is provided
                    st.session_state.analyzing = True
                    st.rerun()
                else:
                    st.error("❌ Please enter a YouTube URL first!")

        with btn_col2:
            # Stop button - only enabled when analyzing
            if st.button(
                "⏹️ Stop Analysis", 
                disabled=not st.session_state.analyzing,
                width='stretch',
                type="secondary",
                key="stop_btn"
            ):
                st.session_state.analyzing = False
                st.info("⏹️ Analysis stopped by user")
                st.rerun()

        # Status indicator
        if st.session_state.analyzing:
            st.success("🟢 **Status:** Analysis Running")
        else:
            st.info("🔴 **Status:** Analysis Stopped")
    
        # Clear data button (always available)
        if st.button(
            "🗑️ Clear All Data",
            width='stretch',
            type="secondary",
            key="clear_btn"
        ):
            # Clear all session state data
            keys_to_clear = [
                'detection_log', 'all_detections_df', 'tracking_history', 
                'time_series_data', 'object_time_series', 'unique_people_count',
                'heatmap_data', 'heatmap_accumulator', 'tracking_quality'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    if key in ['time_series_data', 'object_time_series']:
                        st.session_state[key] = deque(maxlen=500)
                    elif key == 'all_detections_df':
                        st.session_state[key] = pd.DataFrame()
                    elif key in ['detection_log', 'tracking_history', 'heatmap_data']:
                        st.session_state[key] = []
                    elif key == 'unique_people_count':
                        st.session_state[key] = 0
                    elif key == 'tracking_quality':
                        st.session_state[key] = 0.0
            
            st.success("🗑️ All data cleared successfully!")
            st.rerun()

    st.markdown("---")
# ===================================================================================
# ENHANCED MAIN CONTENT AREA - VIDEO & VISUALIZATION
# ===================================================================================

cont5 = st.container(border=True)
with cont5:
    col2, col3 = st.columns([1, 1])

    with col2:
        st.subheader("🎯 Enhanced Detection & Tracking Visualization", divider="grey")
        annotated_frame_placeholder = st.empty()
        
        # Add tracking quality indicator
        tracking_quality_placeholder = st.empty()
        st.write("")

    with col3:
        st.subheader("📈 Multi-Object Tracking Over Time", divider="grey")
        time_series_placeholder = st.empty()
        st.write("")

    # Enhanced summary text with tracking details
    summary_placeholder = st.empty()
    summary_placeholder.markdown("**No analysis running. Configure enhanced settings and start analysis.**")
    summary_placeholder.markdown("---")

    # Enhanced top metrics
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    unique_metric = col_metric1.metric("🆔 Unique People (Session)", 0, help="Total unique individuals tracked")
    current_metric = col_metric2.metric("👁️ Currently Visible", 0, help="People currently in frame")
    peak_metric = col_metric3.metric("📈 Peak Count", 0, help="Maximum people visible at once")
    rate_metric = col_metric4.metric("⚡ Rate (people/min)", "0.0", help="Average unique people per minute")

    # Additional enhanced metrics
    col_metric5, col_metric6, col_metric7, col_metric8 = st.columns(4)
    if 'tracking_quality' not in st.session_state:
        st.session_state.tracking_quality = 0.0
    quality_metric = col_metric5.metric("🎯 Tracking Quality", f"{st.session_state.tracking_quality:.1%}", help="Tracking stability score")
    fps_metric = col_metric6.metric("📊 Processing FPS", "0.0", help="Frames processed per second")
    detection_metric = col_metric7.metric("🔍 Total Detections", 0, help="Total objects detected this session")
    accuracy_metric = col_metric8.metric("✅ Detection Accuracy", "0.0%", help="Estimated detection accuracy")

    st.markdown("---")

# ===================================================================================
# ENHANCED HELPER FUNCTIONS
# ===================================================================================

@st.cache_resource
def load_model(model_name):
    """Load and cache YOLO model"""
    return YOLO(model_name)

def get_stream_url(youtube_url):
    """Extract direct video stream URL from YouTube with better error handling"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best[height<=720]/best',  # Multiple fallbacks
        'quiet': True,
        'no_warnings': True,
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'referer': 'https://www.youtube.com/',
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            
            # Try different URL extraction strategies
            if 'url' in info:
                return info['url']
            elif 'formats' in info and info['formats']:
                # Get best available format
                for fmt in reversed(info['formats']):
                    if fmt.get('url') and fmt.get('vcodec') != 'none':
                        return fmt['url']
            
            return None
    except Exception as e:
        st.error(f"Error getting stream URL: {str(e)}")
        return None
def analyze_frame(frame, model, filters, conf_threshold, resize_factor, tracker_type, enhanced_filtering=True, nms_threshold=0.4, min_area=100):
    """
    Enhanced frame analysis with improved detection filtering and tracking support.
    """
    # Resize frame for faster processing
    if resize_factor < 1.0:
        height, width = frame.shape[:2]
        new_width = int(width * resize_factor)
        new_height = int(height * resize_factor)
        frame_resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        scale_x = width / new_width
        scale_y = height / new_height
    else:
        frame_resized = frame
        scale_x = scale_y = 1.0
    
    # Run YOLO with tracking if selected
    try:
        if tracker_type == "ByteTrack (YOLO)":
            results = model.track(frame_resized, conf=conf_threshold, tracker="bytetrack.yaml", verbose=False)[0]
        elif tracker_type == "BoT-SORT (YOLO)":
            results = model.track(frame_resized, conf=conf_threshold, tracker="botsort.yaml", verbose=False)[0]
        else:
            results = model(frame_resized, conf=conf_threshold, verbose=False)[0]
    except Exception as e:
        # Fallback to regular detection if tracking fails
        st.warning(f"YOLO tracking failed, using detection only: {e}")
        results = model(frame_resized, conf=conf_threshold, verbose=False)[0]
    
    # Apply enhanced detection filtering
    if enhanced_filtering:
        valid_indices = enhanced_detection_filtering(results, conf_threshold, nms_threshold, min_area)
        if valid_indices:
            # Create filtered results
            filtered_boxes = results.boxes[valid_indices]
        else:
            filtered_boxes = []
    else:
        filtered_boxes = results.boxes if results.boxes is not None else []
    
    detections = {}
    detailed_detections = []
    person_centroids = []
    tracked_objects = {}
    
    # Process filtered detections with enhanced error handling
    if len(filtered_boxes) > 0:
        for i, box in enumerate(filtered_boxes):
            try:
                class_id = int(box.cls[0])
                class_name = results.names[class_id]
                confidence = float(box.conf[0])
                
                # Check if this object type should be detected
                if class_name in filters and filters[class_name]:
                    if class_name not in detections:
                        detections[class_name] = 0
                    detections[class_name] += 1
                    
                    # Get bounding box and scale back to original size with safe conversion
                    bbox_tensor = box.xyxy[0].cpu().numpy()
                    
                    # Ensure we have 4 coordinates
                    if len(bbox_tensor) >= 4:
                        x1, y1, x2, y2 = bbox_tensor[:4]
                        
                        # Convert to integers safely
                        x1 = int(float(x1) * scale_x)
                        y1 = int(float(y1) * scale_y)
                        x2 = int(float(x2) * scale_x)
                        y2 = int(float(y2) * scale_y)
                        
                        # Validate bbox
                        if x2 <= x1 or y2 <= y1:
                            continue
                            
                        # Calculate center point
                        center_x = (x1 + x2) // 2
                        center_y = (y1 + y2) // 2
                        area = (x2 - x1) * (y2 - y1)
                        
                        # FIXED: Get tracking ID if available
                        track_id = None
                        if hasattr(box, 'id') and box.id is not None:
                            try:
                                track_id = int(box.id[0])
                                # Store tracked objects for visualization (all objects, not just people)
                                tracked_objects[track_id] = (center_x, center_y)
                            except (ValueError, IndexError, TypeError) as e:
                                # Handle cases where ID extraction fails
                                track_id = None
                                
                        
                        # Collect person centroids for custom tracking fallback
                        if class_name == 'person':
                            person_centroids.append((center_x, center_y))
                        
                        # Store detailed detection info with safe values
                        detection_info = {
                            'object_type': class_name,
                            'confidence': float(confidence),
                            'confidence_pct': f"{confidence:.1%}",
                            'bbox_x1': x1,
                            'bbox_y1': y1,
                            'bbox_x2': x2,
                            'bbox_y2': y2,
                            'center_x': center_x,
                            'center_y': center_y,
                            'width': x2 - x1,
                            'height': y2 - y1,
                            'area_pixels': area,
                            'track_id': track_id,
                            'enhanced_filtered': enhanced_filtering
                        }
                        detailed_detections.append(detection_info)
                        
            except Exception as detection_error:
                # Skip problematic detections
                continue
    
    return detections, detailed_detections, person_centroids, tracked_objects

def update_heatmap(detailed_detections, frame_shape, object_type="person", decay_factor=0.95):
    """Update heatmap data with new detections"""
    if 'heatmap_accumulator' not in st.session_state:
        st.session_state.heatmap_accumulator = np.zeros((frame_shape[0]//4, frame_shape[1]//4), dtype=np.float32)
    
    # Decay existing heatmap
    st.session_state.heatmap_accumulator *= decay_factor
    
    # Add new detections
    for det in detailed_detections:
        if object_type == "all" or det['object_type'] == object_type:
            # Scale coordinates to heatmap size
            hx = int(det['center_x'] // 4)
            hy = int(det['center_y'] // 4)
            
            if 0 <= hx < st.session_state.heatmap_accumulator.shape[1] and 0 <= hy < st.session_state.heatmap_accumulator.shape[0]:
                # Add gaussian blob around detection
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
    
    # Normalize and resize heatmap to frame size
    normalized = (heatmap_data / heatmap_data.max() * 255).astype(np.uint8)
    heatmap_resized = cv2.resize(normalized, (frame.shape[1], frame.shape[0]))
    
    # Apply colormap
    heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
    
    # Blend with original frame
    return cv2.addWeighted(frame, 1-alpha, heatmap_colored, alpha, 0)

def calculate_tracking_quality(tracker, detailed_detections):
    """Calculate tracking quality score based on various factors"""
    if not hasattr(tracker, 'objects') or len(tracker.objects) == 0:
        return 0.0
    
    # Factors for quality calculation
    total_detections = len(detailed_detections)
    tracked_objects = len(tracker.objects)
    
    # Quality factors
    tracking_ratio = min(tracked_objects / max(total_detections, 1), 1.0)
    confidence_avg = np.mean([det['confidence'] for det in detailed_detections]) if detailed_detections else 0.0
    
    # Stability factor (how consistent tracking IDs are)
    stability_factor = 0.8  # Placeholder - could be enhanced with ID consistency tracking
    
    quality_score = (tracking_ratio * 0.4 + confidence_avg * 0.4 + stability_factor * 0.2)
    return quality_score


def draw_enhanced_visualization(frame, tracker, detailed_detections, tracked_objects=None, show_heatmap=False, tracker_type="OpenCV"):
    """
    Enhanced visualization with improved visual elements and tracking information.
    """
    annotated_frame = frame.copy()
    
    # Add heatmap overlay if enabled
    if show_heatmap and 'heatmap_accumulator' in st.session_state:
        annotated_frame = create_heatmap_overlay(annotated_frame, st.session_state.heatmap_accumulator)
    
    # Enhanced colors for different object types
    colors = {
        'person': (0, 255, 0),        # Green
        'car': (255, 0, 0),           # Blue
        'bicycle': (0, 255, 255),     # Yellow
        'motorcycle': (255, 0, 255),  # Magenta
        'bus': (255, 128, 0),         # Orange
        'truck': (128, 0, 255),       # Purple
        'boat': (0, 128, 255),        # Light Blue
    }
    
    # Draw all detected objects with enhanced visual elements
    for det in detailed_detections:
        x1, y1, x2, y2 = det['bbox_x1'], det['bbox_y1'], det['bbox_x2'], det['bbox_y2']
        obj_type = det['object_type']
        confidence = det['confidence']
        track_id = det.get('track_id')
        enhanced_filtered = det.get('enhanced_filtered', False)
        
        # Get color for this object type
        color = colors.get(obj_type, (255, 255, 255))
        
        # Enhanced bounding box - thicker for tracked objects
        thickness = 3 if track_id is not None else 2
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, thickness)
        
        # Enhanced label with more information
        if track_id is not None:
            label = f"{obj_type} ID:{track_id} {confidence:.2f}"
            if enhanced_filtered:
                label += " ✓"
        else:
            label = f"{obj_type} {confidence:.2f}"
            if enhanced_filtered:
                label += " ✓"
        
        # Enhanced label background with transparency effect
        (label_width, label_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )
        
        # Draw semi-transparent label background
        overlay = annotated_frame.copy()
        cv2.rectangle(
            overlay,
            (x1, y1 - label_height - baseline - 10),
            (x1 + label_width + 10, y1),
            color,
            -1
        )
        cv2.addWeighted(overlay, 0.7, annotated_frame, 0.3, 0, annotated_frame)
        
        # Draw label text with better visibility
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, y1 - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )
        
        # Draw center point for tracked objects
        if track_id is not None and obj_type == 'person':
            cv2.circle(annotated_frame, (det['center_x'], det['center_y']), 5, (0, 255, 0), -1)
            cv2.circle(annotated_frame, (det['center_x'], det['center_y']), 8, (255, 255, 255), 2)
    
    # FIXED: Handle different tracking types properly
    if tracker_type == "OpenCV":
        # Draw custom tracker objects for OpenCV tracking
        if tracker is not None and hasattr(tracker, 'objects'):
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
    
    elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
        # FIXED: Draw tracked objects for ByteTrack/BoT-SORT
        if tracked_objects is not None and len(tracked_objects) > 0:
            for track_id, (center_x, center_y) in tracked_objects.items():
                # Draw tracking indicator - different style from OpenCV
                cv2.circle(annotated_frame, (center_x, center_y), 6, (0, 255, 255), -1)  # Yellow center
                cv2.circle(annotated_frame, (center_x, center_y), 10, (255, 255, 255), 2)  # White border
                cv2.circle(annotated_frame, (center_x, center_y), 14, (0, 255, 255), 1)   # Outer yellow ring
                
                # Draw track ID with distinctive style
                track_text = f"T:{track_id}"
                (text_width, text_height), _ = cv2.getTextSize(track_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                
                # Enhanced ID label for YOLO trackers - different color
                overlay = annotated_frame.copy()
                cv2.rectangle(overlay, 
                             (center_x - text_width//2 - 5, center_y + 18),
                             (center_x + text_width//2 + 5, center_y + text_height + 23),
                             (0, 255, 255), -1)  # Yellow background
                cv2.addWeighted(overlay, 0.9, annotated_frame, 0.1, 0, annotated_frame)
                
                cv2.putText(annotated_frame, track_text, 
                           (center_x - text_width//2, center_y + text_height + 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Add tracking algorithm indicator with track count
    if tracker_type == "OpenCV" and tracker is not None and hasattr(tracker, 'objects'):
        track_count = len(tracker.objects)
        algo_text = f"Tracking: {tracker_type} ({track_count} active)"
    elif tracked_objects is not None:
        track_count = len(tracked_objects)
        algo_text = f"Tracking: {tracker_type} ({track_count} active)"
    else:
        algo_text = f"Tracking: {tracker_type} (0 active)"
    
    (algo_width, algo_height), _ = cv2.getTextSize(algo_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(annotated_frame, (10, 10), (algo_width + 20, algo_height + 20), (0, 0, 0), -1)
    cv2.putText(annotated_frame, algo_text, (15, algo_height + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Add tracker type indicator in top-right
    tracker_indicator = "🎯 OpenCV" if tracker_type == "OpenCV" else "⚡ YOLO"
    (ind_width, ind_height), _ = cv2.getTextSize(tracker_indicator, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(annotated_frame, 
                  (annotated_frame.shape[1] - ind_width - 20, 10), 
                  (annotated_frame.shape[1] - 10, ind_height + 20), 
                  (50, 50, 50), -1)
    cv2.putText(annotated_frame, tracker_indicator, 
                (annotated_frame.shape[1] - ind_width - 15, ind_height + 15), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame


# ===================================================================================
# ENHANCED MAIN ANALYSIS FUNCTION
# ===================================================================================


def run_enhanced_analysis():
    """
    Enhanced main analysis loop with improved tracking, detection filtering, and temporal consistency.
    """
    
    # Validate URL
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        st.session_state.analyzing = False
        return
    
    # Add analysis status indicator
    status_placeholder = st.empty()
    status_placeholder.info("🚀 Initializing enhanced analysis...")
    
    # Load and initialize tracker
    tracker = None
    tracker_initialized = False
    
    try:
        if tracker_type == "OpenCV":
            tracker = SimpleTracker(
                max_disappeared=max_disappeared_frames,
                max_distance=max_tracking_distance
            )
            # Apply temporal smoothing if enabled
            if 'temporal_smoothing' in locals() and temporal_smoothing:
                if 'smoothing_factor' in locals():
                    tracker.temporal_tracker.smoothing_factor = smoothing_factor
            st.write("🎯 Using OpenCV tracking algorithm")
            tracker_initialized = True

        elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
            # Built-in YOLO trackers - use simple tracker for fallback
            tracker = SimpleTracker(
                max_disappeared=max_disappeared_frames,
                max_distance=max_tracking_distance
            )
            st.write(f"⚡ Using {tracker_type} tracking algorithm")
            tracker_initialized = True
            
        else:
            # Fallback to Enhanced Custom
            tracker = SimpleTracker(
                max_disappeared=max_disappeared_frames,
                max_distance=max_tracking_distance
            )
            #st.info("🎯 Using OpenCV tracking algorithm")
            tracker_initialized = True
        
        if not tracker_initialized:
            st.error("Failed to initialize tracker. Stopping analysis.")
            st.session_state.analyzing = False
            return
        
        # Load YOLO model
        status_placeholder.info("🤖 Loading YOLO model...")
        with st.spinner(f"Loading Enhanced YOLO model ({model_size})..."):
            model = load_model(model_size)
        
        # Get stream URL
        status_placeholder.info("📹 Opening video stream...")
        
        max_retries = 3
        cap = None
        
        for attempt in range(max_retries):
            try:
                cap = cv2.VideoCapture(youtube_url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if cap.isOpened():
                    # Test if we can read a frame
                    ret, test_frame = cap.read()
                    if ret and test_frame is not None:
                        st.success(f"✅ Stream connected successfully!")
                        break
                    else:
                        cap.release()
                        cap = None
                
                st.warning(f"⚠️ Connection attempt {attempt + 1}/{max_retries} failed")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    
            except Exception as e:
                st.warning(f"⚠️ Stream error on attempt {attempt + 1}: {e}")
                if cap:
                    cap.release()
                    cap = None
        
        if not cap or not cap.isOpened():
            st.error("❌ Failed to open video stream after multiple attempts")
            st.info("💡 **Try these solutions:**")
            st.markdown("""
            - ✅ Check if the YouTube URL is live and working
            - ✅ Try a different YouTube live stream  
            - ✅ Refresh the page and try again
            - ✅ Some streams may be geo-restricted
            """)
            st.session_state.analyzing = False
            return
        
        
        # Initialize enhanced counters and metrics
        frame_count = 0
        processed_count = 0
        last_analysis_time = time.time()
        session_start_time = datetime.now()
        peak_count = 0
        seen_ids = set()  # Track all unique IDs seen
        total_detections = 0
        processing_times = deque(maxlen=50)  # For FPS calculation
        detection_accuracy_scores = deque(maxlen=100)  # For accuracy estimation
        
        # Enhanced error handling and recovery
        consecutive_errors = 0
        max_consecutive_errors = 5
        
        # Main enhanced processing loop
        while st.session_state.analyzing:
            frame_start_time = time.time()
            
            ret, frame = cap.read()
            
            if not ret:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    st.warning("❌ Multiple stream errors detected. Stopping analysis.")
                    break
                st.warning(f"⚠️ Stream error {consecutive_errors}/{max_consecutive_errors}. Retrying...")
                time.sleep(1)
                continue
            
            # Reset error counter on successful frame
            consecutive_errors = 0
            frame_count += 1
            current_time = time.time()
            
            # Skip frames based on frame_skip setting
            if frame_count % frame_skip != 0:
                continue
            
            # Process frame at specified interval
            if current_time - last_analysis_time >= analysis_interval:
                processed_count += 1
                timestamp = datetime.now()
                
                try:
                    # Run enhanced YOLO detection with improved filtering
                    if enhanced_filtering:
                        detections, detailed, person_centroids, tracked_objects = analyze_frame(
                            frame, model, detection_filters, confidence_threshold, resize_factor, 
                            tracker_type, enhanced_filtering, 
                            nms_threshold,
                            min_detection_area
                        )
                    else:
                        detections, detailed, person_centroids, tracked_objects = analyze_frame(
                            frame, model, detection_filters, confidence_threshold, resize_factor, tracker_type
                        )
                    
                    # Update total detection count
                    total_detections += len(detailed)
                    
                    # Calculate detection quality score
                    if detailed:
                        avg_confidence = np.mean([det['confidence'] for det in detailed])
                        detection_accuracy_scores.append(avg_confidence)
                    
                    # Update heatmap if enabled
                    if heatmap_enabled:
                        decay_factor = heatmap_decay if heatmap_enabled else 0.95
                        update_heatmap(detailed, frame.shape, heatmap_object_type, decay_factor)
                    
                    # Enhanced tracking update - simplified without Deep SORT
                    if tracker_type == "OpenCV":
                        # Enhanced custom tracking with temporal smoothing
                        if track_people_only:
                            tracker.update(person_centroids, timestamp)
                            current_people = len(tracker.objects)
                            seen_ids.update(tracker.objects.keys())
                        else:
                            # Track all objects
                            all_centroids = [(det['center_x'], det['center_y']) for det in detailed]
                            tracker.update(all_centroids, timestamp)
                            current_people = len([det for det in detailed if det['object_type'] == 'person'])
                            seen_ids.update(tracker.objects.keys())

                    elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
                        # FIXED: Properly handle YOLO tracking data and update detections
                        if track_people_only:
                            person_tracks = {tid: pos for tid, pos in tracked_objects.items()}
                            current_people = len(person_tracks)
                            seen_ids.update(person_tracks.keys())
                        else:
                            current_people = len([det for det in detailed if det['object_type'] == 'person'])
                            seen_ids.update(tracked_objects.keys())
                        
                        # CRITICAL FIX: Update detailed detections with track IDs for visualization
                        for det in detailed:
                            if det['track_id'] is not None and det['object_type'] == 'person':
                                # Ensure track_id is properly set for visualization
                                det['tracked_by_yolo'] = True
                                # Add to tracked_objects if not already there
                                if det['track_id'] not in tracked_objects:
                                    tracked_objects[det['track_id']] = (det['center_x'], det['center_y'])
                    
                    else:
                        # Fallback - just count detections without tracking
                        current_people = len([det for det in detailed if det['object_type'] == 'person'])
                        for i, det in enumerate([d for d in detailed if d['object_type'] == 'person']):
                            seen_ids.add(f"fallback_{processed_count}_{i}")
                    
                    # Calculate enhanced tracking quality (simplified)
                    if hasattr(tracker, 'objects') and len(tracker.objects) > 0:
                        tracking_quality = calculate_tracking_quality(tracker, detailed)
                        st.session_state.tracking_quality = tracking_quality
                    else:
                        # Default quality for built-in trackers
                        st.session_state.tracking_quality = 0.8 if tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"] else 0.6

                    # Draw enhanced visualization with better error handling
                    try:
                        if tracker_type == "OpenCV":
                            annotated_frame = draw_enhanced_visualization(
                                frame, tracker, detailed, tracked_objects=None, 
                                show_heatmap=heatmap_enabled, tracker_type=tracker_type
                            )
                        elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
                            # FIXED: Pass tracked_objects properly for YOLO trackers
                            annotated_frame = draw_enhanced_visualization(
                                frame, None, detailed, tracked_objects=tracked_objects, 
                                show_heatmap=heatmap_enabled, tracker_type=tracker_type
                            )
                        else:
                            # Fallback visualization
                            annotated_frame = draw_enhanced_visualization(
                                frame, None, detailed, tracked_objects=None, 
                                show_heatmap=heatmap_enabled, tracker_type=tracker_type
                            )
                    except Exception as viz_error:
                        st.warning(f"Visualization error: {viz_error}")
                        annotated_frame = frame  # Fallback to original frame
                    
                    
                    # Resize for display if needed
                    if annotated_frame.shape[1] > 800:
                        display_height = int(annotated_frame.shape[0] * (800 / annotated_frame.shape[1]))
                        display_frame = cv2.resize(annotated_frame, (800, display_height))
                    else:
                        display_frame = annotated_frame
                    
                    # Convert to RGB and display
                    annotated_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    annotated_pil = Image.fromarray(annotated_rgb)
                    annotated_frame_placeholder.image(annotated_pil, width='stretch')
                    
                    # Update enhanced tracking statistics
                    st.session_state.unique_people_count = len(seen_ids)
                    
                    if current_people > peak_count:
                        peak_count = current_people
                    
                    # Calculate enhanced metrics
                    elapsed_minutes = (timestamp - session_start_time).total_seconds() / 60
                    rate = st.session_state.unique_people_count / elapsed_minutes if elapsed_minutes > 0 else 0
                    
                    # Calculate processing FPS
                    processing_time = time.time() - frame_start_time
                    processing_times.append(processing_time)
                    avg_processing_time = np.mean(processing_times)
                    fps = 1.0 / avg_processing_time if avg_processing_time > 0 else 0
                    
                    # Calculate estimated detection accuracy
                    if detection_accuracy_scores:
                        estimated_accuracy = np.mean(detection_accuracy_scores)
                    else:
                        estimated_accuracy = 0.0
                    
                    # Update enhanced metrics
                    unique_metric.metric(
                        "🆔 Unique People (Session)", 
                        st.session_state.unique_people_count,
                        help="Total unique individuals tracked this session"
                    )
                    current_metric.metric(
                        "👁️ Currently Visible", 
                        current_people,
                        help="People currently in frame"
                    )
                    peak_metric.metric(
                        "📈 Peak Count", 
                        peak_count,
                        help="Maximum people visible at once"
                    )
                    rate_metric.metric(
                        "⚡ Rate (people/min)", 
                        f"{rate:.1f}",
                        help="Average unique people per minute"
                    )
                    
                    # Update enhanced additional metrics
                    quality_metric.metric(
                        "🎯 Tracking Quality", 
                        f"{st.session_state.tracking_quality:.1%}",
                        help="Tracking stability and accuracy score"
                    )
                    fps_metric.metric(
                        "📊 Processing FPS", 
                        f"{fps:.1f}",
                        help="Frames processed per second"
                    )
                    detection_metric.metric(
                        "🔍 Total Detections", 
                        total_detections,
                        help="Total objects detected this session"
                    )
                    accuracy_metric.metric(
                        "✅ Detection Accuracy", 
                        f"{estimated_accuracy:.1%}",
                        help="Estimated detection confidence average"
                    )
                    
                    # Display tracking quality indicator
                    if st.session_state.tracking_quality > 0.8:
                        quality_color = "🟢"
                        quality_text = "Excellent"
                    elif st.session_state.tracking_quality > 0.6:
                        quality_color = "🟡"
                        quality_text = "Good"
                    else:
                        quality_color = "🔴"
                        quality_text = "Poor"
                    
                    tracking_quality_placeholder.markdown(
                        f"**Tracking Quality:** {quality_color} {quality_text} ({st.session_state.tracking_quality:.1%})"
                    )
                    
                    # Update enhanced summary text
                    summary_text = f"**Enhanced Analysis - {timestamp.strftime('%H:%M:%S')}**\n\n"
                    summary_text += f"**Frames Processed:** {processed_count} | **Processing FPS:** {fps:.1f}\n\n"
                    summary_text += f"**Tracker:** {tracker_type} | **Quality:** {quality_color} {quality_text}\n\n"
                    
                    if detections:
                        summary_text += "**Current Detections:**\n"
                        for obj, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
                            summary_text += f"- {obj.title()}: {count}\n"
                        
                        if 'enhanced_filtering' in locals() and enhanced_filtering:
                            summary_text += f"\n*Enhanced filtering active with NMS threshold {nms_threshold if 'nms_threshold' in locals() else 0.4}*\n"
                    else:
                        summary_text += "*No objects detected this frame*\n"
                    
                    if heatmap_enabled:
                        summary_text += f"\n*Heatmap tracking {heatmap_object_type} with {heatmap_decay:.2f} decay*\n"
                    
                    summary_placeholder.markdown(summary_text)
                    
                    # Enhanced time series data collection
                    enhanced_time_entry = {
                        'timestamp': timestamp,
                        'count': current_people,
                        'unique_total': st.session_state.unique_people_count,
                        'tracking_quality': st.session_state.tracking_quality,
                        'processing_fps': fps,
                        'detection_accuracy': estimated_accuracy
                    }
                    st.session_state.time_series_data.append(enhanced_time_entry)
                    
                    # Enhanced object tracking with additional metadata
                    enhanced_object_counts = {
                        'timestamp': timestamp,
                        'person': detections.get('person', 0),
                        'car': detections.get('car', 0),
                        'bicycle': detections.get('bicycle', 0),
                        'motorcycle': detections.get('motorcycle', 0),
                        'bus': detections.get('bus', 0),
                        'truck': detections.get('truck', 0),
                        'boat': detections.get('boat', 0),
                        'total_detections': len(detailed),
                        'avg_confidence': np.mean([det['confidence'] for det in detailed]) if detailed else 0.0,
                        'tracking_algorithm': tracker_type,
                        'enhanced_filtering': enhanced_filtering if 'enhanced_filtering' in locals() else False
                    }
                    st.session_state.object_time_series.append(enhanced_object_counts)
                    
                    # Update enhanced time series visualization
                    if len(st.session_state.object_time_series) > 1:
                        obj_df = pd.DataFrame(list(st.session_state.object_time_series))
                        
                        # Create enhanced visualization with only detected objects
                        plot_columns = []
                        for col in ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'boat']:
                            if obj_df[col].sum() > 0:
                                plot_columns.append(col)
                        
                        if plot_columns:
                            # Enhanced chart with better styling
                            chart_data = obj_df.set_index('timestamp')[plot_columns]
                            time_series_placeholder.line_chart(
                                chart_data,
                                width='stretch',
                                height=300
                            )
                        else:
                            time_series_placeholder.info("⏳ Waiting for object detections...")
                    
                    # Enhanced tracking history with more detailed information - FIXED
                    enhanced_tracking_entry = {
                        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        'current_visible': int(current_people),  # Ensure integer
                        'unique_session': int(st.session_state.unique_people_count),  # Ensure integer
                        'peak_count': int(peak_count),  # Ensure integer
                        'tracking_quality': float(st.session_state.tracking_quality),  # Ensure float
                        'processing_fps': float(fps),  # Ensure float
                        'tracker_algorithm': str(tracker_type),  # Ensure string
                        'total_detections': int(len(detailed)),  # Ensure integer
                        'avg_confidence': float(np.mean([det['confidence'] for det in detailed])) if detailed else 0.0,
                        'enhanced_filtering': bool(enhanced_filtering if 'enhanced_filtering' in locals() else False)
                    }
                    
                    # Insert at beginning for chronological order
                    st.session_state.tracking_history.insert(0, enhanced_tracking_entry)
                    
                    # Limit history size for performance
                    if len(st.session_state.tracking_history) > 500:  # Reduced from 1000
                        st.session_state.tracking_history = st.session_state.tracking_history[:500]
                    
                    # Enhanced detailed detections with proper data types - FIXED
                    if detailed:
                        for det in detailed:
                            det['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            det['frame_number'] = int(frame_count)
                            det['processing_fps'] = float(fps)
                            det['tracking_algorithm'] = str(tracker_type)
                            det['enhanced_filtered'] = bool(enhanced_filtering if 'enhanced_filtering' in locals() else False)
                            det['tracking_quality'] = float(st.session_state.tracking_quality)
                            det['session_unique_count'] = int(st.session_state.unique_people_count)
                            
                            # Ensure numeric types for bbox coordinates
                            det['bbox_x1'] = int(det['bbox_x1'])
                            det['bbox_y1'] = int(det['bbox_y1'])
                            det['bbox_x2'] = int(det['bbox_x2'])
                            det['bbox_y2'] = int(det['bbox_y2'])
                            det['center_x'] = int(det['center_x'])
                            det['center_y'] = int(det['center_y'])
                            det['confidence'] = float(det['confidence'])
                        
                        # Add to dataframe with error handling
                        try:
                            new_df = pd.DataFrame(detailed)
                            if st.session_state.all_detections_df.empty:
                                st.session_state.all_detections_df = new_df
                            else:
                                st.session_state.all_detections_df = pd.concat(
                                    [new_df, st.session_state.all_detections_df], 
                                    ignore_index=True
                                )
                            
                            # Limit dataframe size for performance
                            if len(st.session_state.all_detections_df) > 1000:  # Reduced from 2000
                                st.session_state.all_detections_df = st.session_state.all_detections_df.head(1000)
                                
                        except Exception as df_error:
                            st.warning(f"Error updating detection dataframe: {df_error}")

                    last_analysis_time = current_time
                    
                except Exception as analysis_error:
                    st.warning(f"Analysis error: {analysis_error}")
                    consecutive_errors += 1
                    if consecutive_errors >= max_consecutive_errors:
                        st.error("Too many analysis errors. Stopping.")
                        break
            
            # Small delay to prevent overwhelming the system
            time.sleep(0.02)
            
    except KeyboardInterrupt:
        st.info("⏹️ Analysis interrupted by user")
    except Exception as e:
        st.error(f"❌ Unexpected error: {str(e)}")
    finally:
        # Cleanup
        if 'cap' in locals():
            cap.release()
        st.session_state.analyzing = False
        status_placeholder.success("✅ Enhanced analysis completed successfully!")


# ===================================================================================
# ENHANCED DATA TABS WITH IMPROVED FUNCTIONALITY - FIXED VERSION
# ===================================================================================
st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 4: Explore Data & Export Results</h2>", unsafe_allow_html=True)
st.subheader("📊 Detailed Analysis", divider="grey")

# Initialize session state variables if they don't exist
if 'tracking_history' not in st.session_state:
    st.session_state.tracking_history = []
if 'all_detections_df' not in st.session_state:
    st.session_state.all_detections_df = pd.DataFrame()
if 'object_time_series' not in st.session_state:
    st.session_state.object_time_series = deque(maxlen=500)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["📋 Tracking History", "📊 Detection Data", "📈 Statistics", "🌍 Spatial Analysis"])

# TAB 1: Enhanced Tracking History - FIXED
with tab1:
    st.markdown("### 📊 Enhanced Tracking History")
    
    if st.session_state.tracking_history and len(st.session_state.tracking_history) > 0:
        try:
            tracking_df = pd.DataFrame(st.session_state.tracking_history)
            
            # Enhanced summary metrics
            col_t1, col_t2, col_t3, col_t4, col_t5 = st.columns(5)
            with col_t1:
                st.metric("📋 Total Records", len(tracking_df))
            with col_t2:
                if 'current_visible' in tracking_df.columns:
                    avg_visible = tracking_df['current_visible'].astype(float).mean()
                    st.metric("👥 Avg Visible", f"{avg_visible:.1f}")
                else:
                    st.metric("👥 Avg Visible", "0")
            with col_t3:
                if 'current_visible' in tracking_df.columns:
                    max_visible = tracking_df['current_visible'].astype(float).max()
                    st.metric("📈 Max Visible", int(max_visible))
                else:
                    st.metric("📈 Max Visible", "0")
            with col_t4:
                if 'unique_session' in tracking_df.columns:
                    unique_max = tracking_df['unique_session'].astype(float).max()
                    st.metric("🆔 Unique People", int(unique_max))
                else:
                    st.metric("🆔 Unique People", st.session_state.get('unique_people_count', 0))
            with col_t5:
                if 'tracking_quality' in tracking_df.columns:
                    quality_values = pd.to_numeric(tracking_df['tracking_quality'], errors='coerce')
                    if not quality_values.isna().all():
                        avg_quality = quality_values.mean()
                        st.metric("🎯 Avg Quality", f"{avg_quality:.1%}")
                    else:
                        st.metric("🎯 Quality", "N/A")
                else:
                    st.metric("🎯 Quality", f"{st.session_state.get('tracking_quality', 0):.1%}")
            
            # Filter options
            col_f1, col_f2 = st.columns(2)
            with col_f1:
                show_last_n = st.selectbox("Show Last N Records", [10, 25, 50, 100, "All"], index=1, key="tracking_filter")
            with col_f2:
                if 'tracker_algorithm' in tracking_df.columns:
                    unique_algos = tracking_df['tracker_algorithm'].unique().tolist()
                    algo_filter = st.selectbox("Filter by Algorithm", ["All"] + unique_algos, key="algo_filter")
                else:
                    algo_filter = "All"
            
            # Apply filters
            display_df = tracking_df.copy()
            if algo_filter != "All" and 'tracker_algorithm' in display_df.columns:
                display_df = display_df[display_df['tracker_algorithm'] == algo_filter]
            
            if show_last_n != "All":
                display_df = display_df.head(show_last_n)
            
            # Display table
            st.dataframe(display_df, width='stretch', hide_index=True)
            
            # Download button
            if len(display_df) > 0:
                csv = display_df.to_csv(index=False)
                filename = f"tracking_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "📥 Download Tracking Data",
                    csv,
                    filename,
                    "text/csv",
                    width='stretch',
                    key="download_tracking"
                )
                
        except Exception as e:
            st.error(f"Error displaying tracking history: {str(e)}")
            st.info("Raw tracking history length: " + str(len(st.session_state.tracking_history)))
    else:
        st.info("🎯 No tracking data yet. Start analysis to begin collecting data.")
        if st.session_state.get('analyzing', False):
            st.info("⏳ Analysis is running - data will appear here shortly...")

# TAB 2: Enhanced Detection Data - FIXED
with tab2:
    st.markdown("### 🔍 Enhanced Detection Analysis")
    
    if not st.session_state.all_detections_df.empty:
        try:
            df = st.session_state.all_detections_df.copy()
            
            # Summary metrics
            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            col_s1.metric("🔍 Total Detections", len(df))
            if 'confidence' in df.columns:
                col_s2.metric("📊 Avg Confidence", f"{df['confidence'].mean():.2f}")
            if 'object_type' in df.columns:
                col_s3.metric("🎯 Object Types", df['object_type'].nunique())
            if 'enhanced_filtered' in df.columns:
                enhanced_count = df['enhanced_filtered'].sum() if df['enhanced_filtered'].dtype == bool else 0
                col_s4.metric("✨ Enhanced Filtered", enhanced_count)
            
            # Filter options
            col_f1, col_f2, col_f3 = st.columns(3)
            
            with col_f1:
                if 'object_type' in df.columns:
                    unique_objects = df['object_type'].unique().tolist()
                    filter_object = st.multiselect(
                        "🎯 Object Types",
                        options=unique_objects,
                        default=unique_objects[:3] if len(unique_objects) > 3 else unique_objects,
                        key="object_filter"
                    )
                else:
                    filter_object = []
            
            with col_f2:
                if 'confidence' in df.columns:
                    min_confidence = st.slider(
                        "🎚️ Min Confidence",
                        min_value=0.0,
                        max_value=1.0,
                        value=0.0,
                        step=0.1,
                        key="conf_filter"
                    )
                else:
                    min_confidence = 0.0
            
            with col_f3:
                max_records = st.selectbox(
                    "📊 Max Records",
                    [50, 100, 250, 500, "All"],
                    index=1,
                    key="max_records_filter"
                )
            
            # Apply filters
            filtered_df = df.copy()
            
            if filter_object and 'object_type' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['object_type'].isin(filter_object)]
            
            if 'confidence' in filtered_df.columns:
                filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
            
            if max_records != "All":
                filtered_df = filtered_df.head(max_records)
            
            # Display filtered results
            if len(filtered_df) > 0:
                st.info(f"Showing {len(filtered_df)} of {len(df)} total detections")
                
                # Select key columns for display
                display_columns = ['object_type', 'confidence', 'timestamp']
                if 'track_id' in filtered_df.columns:
                    display_columns.append('track_id')
                if 'center_x' in filtered_df.columns and 'center_y' in filtered_df.columns:
                    display_columns.extend(['center_x', 'center_y'])
                
                # Only show columns that exist
                available_columns = [col for col in display_columns if col in filtered_df.columns]
                
                st.dataframe(filtered_df[available_columns], width='stretch', hide_index=True)
                
                # Download button
                csv = filtered_df.to_csv(index=False)
                filename = f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "📥 Download Detection Data",
                    csv,
                    filename,
                    "text/csv",
                    width='stretch',
                    key="download_detections"
                )
                
                # Quick visualization
                if 'object_type' in filtered_df.columns and len(filtered_df) > 5:
                    st.markdown("#### 📊 Object Distribution")
                    obj_counts = filtered_df['object_type'].value_counts()
                    st.bar_chart(obj_counts)
                    
            else:
                st.warning("No detections match the current filters.")
                
        except Exception as e:
            st.error(f"Error displaying detection data: {str(e)}")
            st.info("Raw detections dataframe shape: " + str(st.session_state.all_detections_df.shape))
    else:
        st.info("🔍 No detection data available. Start analysis to populate detection information.")
        if st.session_state.get('analyzing', False):
            st.info("⏳ Analysis is running - detection data will appear here shortly...")

# TAB 3: Enhanced Statistics - FIXED  
with tab3:
    st.markdown("### 📈 Enhanced Session Analytics")
    
    # Check if we have any data
    has_tracking = st.session_state.tracking_history and len(st.session_state.tracking_history) > 0
    has_detections = not st.session_state.all_detections_df.empty
    has_timeseries = st.session_state.object_time_series and len(st.session_state.object_time_series) > 0
    
    if has_tracking or has_detections or has_timeseries:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### 👥 Session Summary")
            
            # Basic session stats
            session_stats = {
                'Unique People Tracked': st.session_state.get('unique_people_count', 0),
                'Current Tracking Quality': f"{st.session_state.get('tracking_quality', 0):.1%}",
                'Analysis Status': "🟢 Running" if st.session_state.get('analyzing', False) else "🔴 Stopped"
            }
            
            if has_tracking:
                try:
                    tracking_df = pd.DataFrame(st.session_state.tracking_history)
                    if 'current_visible' in tracking_df.columns:
                        session_stats['Average Visible'] = f"{tracking_df['current_visible'].astype(float).mean():.1f}"
                        session_stats['Peak Visible'] = str(int(tracking_df['current_visible'].astype(float).max()))
                        session_stats['Total Observations'] = str(len(tracking_df))
                except Exception as e:
                    st.warning(f"Error processing tracking stats: {e}")
            
            # Display stats
            for stat_name, stat_value in session_stats.items():
                st.metric(stat_name, stat_value)
        
        with col_s2:
            st.markdown("#### 📊 Live Charts")
            
            # Simple time series if we have tracking data
            if has_tracking:
                try:
                    tracking_df = pd.DataFrame(st.session_state.tracking_history)
                    if 'current_visible' in tracking_df.columns and len(tracking_df) > 1:
                        # Simple line chart of people count over time
                        people_data = tracking_df['current_visible'].astype(float).tail(20)  # Last 20 points
                        st.line_chart(people_data, height=200)
                    else:
                        st.info("Not enough data points for chart yet")
                except Exception as e:
                    st.warning(f"Error creating chart: {e}")
            
            # Object type breakdown if we have detection data
            if has_detections:
                try:
                    df = st.session_state.all_detections_df
                    if 'object_type' in df.columns and len(df) > 0:
                        st.markdown("**Object Types Detected:**")
                        obj_counts = df['object_type'].value_counts().head(5)  # Top 5
                        st.bar_chart(obj_counts, height=200)
                except Exception as e:
                    st.warning(f"Error creating object chart: {e}")
        
        # Time series visualization if available
        if has_timeseries:
            st.markdown("#### ⏰ Detection Timeline")
            try:
                timeline_df = pd.DataFrame(list(st.session_state.object_time_series))
                
                if 'timestamp' in timeline_df.columns:
                    # Show recent activity
                    recent_data = timeline_df.tail(30)  # Last 30 data points
                    
                    # Find columns with actual data
                    object_cols = ['person', 'car', 'bicycle', 'motorcycle', 'bus', 'truck', 'boat']
                    active_cols = []
                    for col in object_cols:
                        if col in recent_data.columns and recent_data[col].sum() > 0:
                            active_cols.append(col)
                    
                    if active_cols:
                        chart_data = recent_data.set_index('timestamp')[active_cols]
                        st.area_chart(chart_data, height=250)
                    else:
                        st.info("No object activity detected yet")
                        
            except Exception as e:
                st.warning(f"Error creating timeline: {e}")
                
    else:
        if st.session_state.get('analyzing', False):
            st.info("⏳ Analysis is running - statistics will populate as data is collected...")
        
        # Show some helpful info about what will appear
        st.markdown("""
        **What you'll see here once analysis starts:**
        - 👥 People tracking statistics  
        - 📊 Object detection breakdowns
        - ⏰ Activity timeline charts
        - 🎯 Tracking quality metrics
        - 📈 Session performance data
        """)

with tab4:
    st.markdown("### 🌍 Spatial Intelligence & Movement Analysis")
    
    if not st.session_state.all_detections_df.empty:
        try:
            df = st.session_state.all_detections_df.copy()
            
            # Spatial metrics overview
            col_sp1, col_sp2, col_sp3, col_sp4 = st.columns(4)
            
            if 'center_x' in df.columns and 'center_y' in df.columns:
                # Calculate coverage area
                x_range = df['center_x'].max() - df['center_x'].min()
                y_range = df['center_y'].max() - df['center_y'].min()
                coverage_area = x_range * y_range
                
                col_sp1.metric("📐 Coverage Area", f"{coverage_area:,.0f} px²")
                col_sp2.metric("↔️ Width Span", f"{x_range:.0f} px")
                col_sp3.metric("↕️ Height Span", f"{y_range:.0f} px")
                
                # Activity density
                total_detections = len(df)
                density = total_detections / (coverage_area / 10000) if coverage_area > 0 else 0
                col_sp4.metric("🔥 Activity Density", f"{density:.1f}/100m²")
            
            # Spatial analysis options
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                spatial_object = st.selectbox(
                    "🎯 Analyze Object Type",
                    options=["All Objects"] + df['object_type'].unique().tolist() if 'object_type' in df.columns else ["All Objects"],
                    key="spatial_object_filter"
                )
            
            with col_opt2:
                analysis_type = st.selectbox(
                    "📊 Analysis Type",
                    options=["Heat Zones", "Movement Paths", "Zone Activity", "Congestion Points"],
                    key="spatial_analysis_type"
                )
            
            # Filter data for selected object
            if spatial_object != "All Objects" and 'object_type' in df.columns:
                spatial_df = df[df['object_type'] == spatial_object].copy()
            else:
                spatial_df = df.copy()
            
            if len(spatial_df) > 0 and 'center_x' in spatial_df.columns and 'center_y' in spatial_df.columns:
                
                # HEAT ZONES ANALYSIS (unchanged)
                if analysis_type == "Heat Zones":
                    st.markdown("#### 🔥 Activity Heat Zones")
                    
                    # Create grid-based heat zones
                    x_min, x_max = spatial_df['center_x'].min(), spatial_df['center_x'].max()
                    y_min, y_max = spatial_df['center_y'].min(), spatial_df['center_y'].max()
                    
                    # Create grid (10x10)
                    grid_size = 10
                    x_bins = np.linspace(x_min, x_max, grid_size + 1)
                    y_bins = np.linspace(y_min, y_max, grid_size + 1)
                    
                    # Calculate heat map
                    heat_map, _, _ = np.histogram2d(spatial_df['center_x'], spatial_df['center_y'], 
                                                   bins=[x_bins, y_bins])
                    
                    # Find hottest zones
                    top_zones = []
                    for i in range(grid_size):
                        for j in range(grid_size):
                            if heat_map[i, j] > 0:
                                zone_activity = heat_map[i, j]
                                zone_center_x = (x_bins[i] + x_bins[i+1]) / 2
                                zone_center_y = (y_bins[j] + y_bins[j+1]) / 2
                                top_zones.append({
                                    'Zone': f"Zone_{i}_{j}",
                                    'Activity_Count': int(zone_activity),
                                    'Center_X': int(zone_center_x),
                                    'Center_Y': int(zone_center_y),
                                    'Activity_Level': 'High' if zone_activity > np.percentile(heat_map.flatten(), 75) else 'Medium' if zone_activity > np.percentile(heat_map.flatten(), 50) else 'Low'
                                })
                    
                    if top_zones:
                        zones_df = pd.DataFrame(top_zones).sort_values('Activity_Count', ascending=False)
                        
                        col_heat1, col_heat2 = st.columns(2)
                        
                        with col_heat1:
                            st.markdown("**🏆 Top Activity Zones**")
                            st.dataframe(zones_df.head(10), hide_index=True)
                        
                        with col_heat2:
                            st.markdown("**📊 Zone Activity Distribution**")
                            activity_levels = zones_df['Activity_Level'].value_counts()
                            st.bar_chart(activity_levels)
                
                # MOVEMENT PATHS ANALYSIS - FIXED
                elif analysis_type == "Movement Paths":
                    st.markdown("#### 🛤️ Movement Path Analysis")
                    
                    if 'track_id' in spatial_df.columns:
                        # Analyze paths for tracked objects
                        tracked_objects_df = spatial_df[spatial_df['track_id'].notna()]
                        
                        if len(tracked_objects_df) > 0:
                            path_analysis = []
                            
                            for track_id in tracked_objects_df['track_id'].unique():
                                track_data = tracked_objects_df[tracked_objects_df['track_id'] == track_id].sort_values('timestamp')
                                
                                if len(track_data) >= 2:
                                    # Calculate path metrics
                                    start_pos = (track_data.iloc[0]['center_x'], track_data.iloc[0]['center_y'])
                                    end_pos = (track_data.iloc[-1]['center_x'], track_data.iloc[-1]['center_y'])
                                    
                                    # Calculate total distance traveled
                                    total_distance = 0
                                    for i in range(1, len(track_data)):
                                        prev_pos = (track_data.iloc[i-1]['center_x'], track_data.iloc[i-1]['center_y'])
                                        curr_pos = (track_data.iloc[i]['center_x'], track_data.iloc[i]['center_y'])
                                        total_distance += np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
                                    
                                    # Calculate straight-line distance
                                    straight_distance = np.sqrt((end_pos[0] - start_pos[0])**2 + (end_pos[1] - start_pos[1])**2)
                                    
                                    # Path efficiency (how direct the path is)
                                    efficiency = straight_distance / total_distance if total_distance > 0 else 0
                                    
                                    path_analysis.append({
                                        'Track_ID': int(track_id),
                                        'Object_Type': track_data.iloc[0]['object_type'],
                                        'Duration_Points': len(track_data),
                                        'Total_Distance': f"{total_distance:.0f} px",
                                        'Straight_Distance': f"{straight_distance:.0f} px",
                                        'Path_Efficiency': f"{efficiency:.2f}",
                                        'Start_X': int(start_pos[0]),
                                        'Start_Y': int(start_pos[1]),
                                        'End_X': int(end_pos[0]),
                                        'End_Y': int(end_pos[1])
                                    })
                            
                            if path_analysis:
                                paths_df = pd.DataFrame(path_analysis)
                                
                                col_path1, col_path2 = st.columns(2)
                                
                                with col_path1:
                                    st.markdown("**🎯 Individual Path Analysis**")
                                    st.dataframe(paths_df, hide_index=True)
                                
                                with col_path2:
                                    st.markdown("**📈 Path Efficiency Analysis**")
                                    if 'Path_Efficiency' in paths_df.columns:
                                        efficiency_data = pd.to_numeric(paths_df['Path_Efficiency'], errors='coerce')
                                        # FIXED: Enhanced efficiency visualization without st.histogram
                                        if not efficiency_data.dropna().empty:
                                            # Create efficiency categories
                                            efficiency_categories = []
                                            for eff in efficiency_data.dropna():
                                                if eff >= 0.8:
                                                    efficiency_categories.append("Very Direct (0.8+)")
                                                elif eff >= 0.6:
                                                    efficiency_categories.append("Direct (0.6-0.8)")
                                                elif eff >= 0.4:
                                                    efficiency_categories.append("Moderate (0.4-0.6)")
                                                elif eff >= 0.2:
                                                    efficiency_categories.append("Indirect (0.2-0.4)")
                                                else:
                                                    efficiency_categories.append("Very Indirect (<0.2)")
                                            
                                            # Count categories and display
                                            cat_counts = pd.Series(efficiency_categories).value_counts()
                                            st.bar_chart(cat_counts)
                                            
                                            # Show efficiency statistics
                                            st.markdown("**📊 Efficiency Stats:**")
                                            st.write(f"• Average: {efficiency_data.mean():.2f}")
                                            st.write(f"• Best: {efficiency_data.max():.2f}")
                                            st.write(f"• Worst: {efficiency_data.min():.2f}")
                                        else:
                                            st.info("No efficiency data available")
                        else:
                            st.info("No tracked movement paths available yet.")
                    else:
                        st.info("Movement path analysis requires object tracking to be enabled.")
                
                # ZONE ACTIVITY ANALYSIS (unchanged)
                elif analysis_type == "Zone Activity":
                    st.markdown("#### 🏙️ Zone-Based Activity Analysis")
                    
                    # Define zones based on image quadrants
                    if 'center_x' in spatial_df.columns and 'center_y' in spatial_df.columns:
                        x_mid = (spatial_df['center_x'].min() + spatial_df['center_x'].max()) / 2
                        y_mid = (spatial_df['center_y'].min() + spatial_df['center_y'].max()) / 2
                        
                        # Assign zones
                        def assign_zone(row):
                            if row['center_x'] < x_mid and row['center_y'] < y_mid:
                                return "Top-Left"
                            elif row['center_x'] >= x_mid and row['center_y'] < y_mid:
                                return "Top-Right"
                            elif row['center_x'] < x_mid and row['center_y'] >= y_mid:
                                return "Bottom-Left"
                            else:
                                return "Bottom-Right"
                        
                        spatial_df['Zone'] = spatial_df.apply(assign_zone, axis=1)
                        
                        # Zone analysis
                        zone_stats = spatial_df.groupby(['Zone', 'object_type']).size().reset_index(name='Count')
                        zone_summary = spatial_df.groupby('Zone').agg({
                            'object_type': 'count',
                            'confidence': 'mean'
                        }).round(3)
                        zone_summary.columns = ['Total_Objects', 'Avg_Confidence']
                        
                        col_zone1, col_zone2 = st.columns(2)
                        
                        with col_zone1:
                            st.markdown("**📊 Zone Summary**")
                            st.dataframe(zone_summary)
                        
                        with col_zone2:
                            st.markdown("**🎯 Activity by Zone**")
                            zone_totals = spatial_df['Zone'].value_counts()
                            st.bar_chart(zone_totals)
                        
                        # Detailed breakdown
                        st.markdown("**🔍 Detailed Zone Activity**")
                        pivot_table = zone_stats.pivot(index='Zone', columns='object_type', values='Count').fillna(0)
                        st.dataframe(pivot_table)
                
                # CONGESTION POINTS ANALYSIS - FIXED (removed scipy dependency)
                elif analysis_type == "Congestion Points":
                    st.markdown("#### 🚦 Congestion & Density Analysis")
                    
                    # FIXED: Simple clustering without scipy
                    if len(spatial_df) >= 2:
                        coordinates = spatial_df[['center_x', 'center_y']].values
                        cluster_threshold = 100  # pixels
                        
                        congestion_points = []
                        processed = set()
                        
                        for i, coord in enumerate(coordinates):
                            if i in processed:
                                continue
                                
                            # Find all points within threshold
                            cluster = [i]
                            for j, other_coord in enumerate(coordinates):
                                if j != i and j not in processed:
                                    distance = np.sqrt(np.sum((coord - other_coord)**2))
                                    if distance <= cluster_threshold:
                                        cluster.append(j)
                            
                            if len(cluster) >= 3:  # Only consider clusters with 3+ objects
                                cluster_coords = coordinates[cluster]
                                center_x = np.mean(cluster_coords[:, 0])
                                center_y = np.mean(cluster_coords[:, 1])
                                
                                # Get object types in this cluster
                                cluster_objects = spatial_df.iloc[cluster]['object_type'].value_counts()
                                main_object = cluster_objects.index[0]
                                
                                congestion_points.append({
                                    'Cluster_ID': len(congestion_points) + 1,
                                    'Objects_Count': len(cluster),
                                    'Center_X': int(center_x),
                                    'Center_Y': int(center_y),
                                    'Main_Object_Type': main_object,
                                    'Density_Level': 'High' if len(cluster) > 5 else 'Medium'
                                })
                                
                                # Mark as processed
                                processed.update(cluster)
                        
                        if congestion_points:
                            congestion_df = pd.DataFrame(congestion_points)
                            
                            col_cong1, col_cong2 = st.columns(2)
                            
                            with col_cong1:
                                st.markdown("**🚦 Congestion Points**")
                                st.dataframe(congestion_df, hide_index=True)
                            
                            with col_cong2:
                                st.markdown("**📊 Congestion Distribution**")
                                density_levels = congestion_df['Density_Level'].value_counts()
                                st.bar_chart(density_levels)
                            
                            # Summary statistics
                            st.markdown("**📈 Congestion Summary**")
                            total_clustered = congestion_df['Objects_Count'].sum()
                            total_objects = len(spatial_df)
                            clustering_rate = (total_clustered / total_objects) * 100
                            
                            cong_col1, cong_col2, cong_col3 = st.columns(3)
                            cong_col1.metric("🎯 Congestion Points", len(congestion_points))
                            cong_col2.metric("👥 Objects in Clusters", total_clustered)
                            cong_col3.metric("📊 Clustering Rate", f"{clustering_rate:.1f}%")
                        else:
                            st.info("No significant congestion points detected.")
                    else:
                        st.info("Need more detection data to analyze congestion patterns.")
            
            else:
                st.info("No spatial data available for the selected object type.")
            
            # Export spatial analysis
            if len(spatial_df) > 0:
                st.markdown("---")
                spatial_export = spatial_df[['object_type', 'center_x', 'center_y', 'timestamp', 'confidence']].copy()
                csv = spatial_export.to_csv(index=False)
                filename = f"spatial_analysis_{spatial_object}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                st.download_button(
                    "📥 Download Spatial Data",
                    csv,
                    filename,
                    "text/csv",
                    key="download_spatial"
                )
                    
        except Exception as e:
            st.error(f"Error in spatial analysis: {str(e)}")
            # Add debug info
            st.write("Debug info:", str(e))
    else:
        st.info("🌍 No spatial data available. Start analysis to begin collecting spatial information.")
        if st.session_state.get('analyzing', False):
            st.info("⏳ Analysis is running - spatial data will appear here shortly...")
        
        # Show preview of what will be available
        st.markdown("""
        **🌍 Spatial Analysis Features:**
        - 🔥 **Heat Zones**: Identify high-activity areas
        - 🛤️ **Movement Paths**: Track object trajectories and efficiency
        - 🏙️ **Zone Activity**: Compare activity across different areas
        - 🚦 **Congestion Points**: Find areas of high object density
        - 📐 **Coverage Analysis**: Measure spatial spread of activity
        - 📊 **Density Metrics**: Calculate objects per area unit
        """)
# ===================================================================================
if st.session_state.get('analyzing', False):
    # Only run analysis if we're in analyzing state
    run_enhanced_analysis()