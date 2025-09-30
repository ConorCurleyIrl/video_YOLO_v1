"""
===================================================================================
STREET TRAFFIC ANALYZER WITH OBJECT TRACKING
===================================================================================
A Streamlit application for analyzing YouTube live streams using YOLO object detection
and tracking unique pedestrians over time.

Features:
- Real-time object detection (people, vehicles, boats, etc.)
- Unique person tracking across frames
- Time-series analytics
- Detailed logging and statistics
- CSV export capabilities

Author: Created by Conor Curley | LinkedIn: https://www.linkedin.com/in/ccurleyds/
License: MIT License
===================================================================================
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import yt_dlp
from PIL import Image
import time
from datetime import datetime
import pandas as pd
from collections import deque, defaultdict

# ===================================================================================
# PAGE CONFIGURATION
# ===================================================================================

st.set_page_config(page_title="Street Traffic Analyzer", layout="wide")

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
if 'unique_people_count' not in st.session_state:
    st.session_state.unique_people_count = 0

# ===================================================================================
# OBJECT TRACKER CLASS
# ===================================================================================

class SimpleTracker:
    """
    Simple centroid-based object tracker for tracking people across frames.
    Uses distance-based matching to associate detections with existing tracked objects.
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
    
    def register(self, centroid, timestamp):
        """Register a new object with a unique ID"""
        self.objects[self.next_object_id] = centroid
        self.disappeared[self.next_object_id] = 0
        self.first_seen[self.next_object_id] = timestamp
        self.next_object_id += 1
    
    def deregister(self, object_id):
        """Remove an object from tracking"""
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
            
            # Update matched objects
            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                if distances[row, col] > self.max_distance:
                    continue
                
                object_id = object_ids[row]
                self.objects[object_id] = detections[col]
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
# HEADER & TITLE
# ===================================================================================
#centre
st.markdown("<h1 style='text-align: center;'>🚶 Live Video Analyzer with Tracking</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Analyze YouTube live streams with YOLO object detection and track unique pedestrians over time</p>", unsafe_allow_html=True)


# ===================================================================================
# SIDEBAR - CONFIGURATION
# ===================================================================================


with st.sidebar:
    
    st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 1: Configure Settings</h2>", unsafe_allow_html=True)

    # -------------------------------------------------------------------------------
    # Stream Selection
    # -------------------------------------------------------------------------------
   
    
    # -------------------------------------------------------------------------------
    # Performance Settings
    # -------------------------------------------------------------------------------
    cont1 = st.container(border=True)
    with cont1:
        st.subheader("⚡ Model & Performance Settings", divider="rainbow")
        
        frame_skip = st.slider(
            "Frame Skip", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="Process every Nth frame. Higher = faster but less frequent updates"
        )
        
        resize_factor = st.slider(
            "Image Resize Factor", 
            min_value=0.25, 
            max_value=1.0, 
            value=0.5, 
            step=0.25,
            help="Resize frames before processing. Lower = faster but less accurate"
        )
        
        model_size = st.selectbox(
            "YOLO Model Size", 
            options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
            index=0,
            help="n=fastest, s=balanced, m=most accurate"
        )
        
        analysis_interval = st.slider(
            "Analysis Interval (seconds)", 
            min_value=1, 
            max_value=10, 
            value=3,
            help="How often to analyze frames"
        )
    
     
    # -------------------------------------------------------------------------------
    # Tracking Settings
    # -------------------------------------------------------------------------------
    
    cont2 = st.container(border=True)
    with cont2:
        st.subheader("🎯 Tracking & Detection Settings", divider="rainbow")
        
        track_people_only = st.checkbox(
            "Track People Only", 
            value=True,
            help="Focus tracking on pedestrians for better accuracy"
        )
        
        max_tracking_distance = st.slider(
            "Max Tracking Distance (pixels)", 
            min_value=50, 
            max_value=300, 
            value=100,
            help="Maximum distance to match objects between frames"
        )
        
        max_disappeared_frames = st.slider(
            "Max Disappeared Frames", 
            min_value=10, 
            max_value=60, 
            value=30,
            help="Frames before considering object as left scene"
        )
        

        
        # -------------------------------------------------------------------------------
        # Detection Filters
        # -------------------------------------------------------------------------------
        st.subheader("🔍 Detection Filters")
        
        detect_person = st.checkbox("👤 Detect People", value=True)
        detect_bicycle = st.checkbox("🚲 Detect Bicycles", value=True)
        detect_car = st.checkbox("🚗 Detect Cars", value=True)
        detect_motorcycle = st.checkbox("🏍️ Detect Motorcycles", value=True)
        detect_bus = st.checkbox("🚌 Detect Buses", value=True)
        detect_truck = st.checkbox("🚚 Detect Trucks", value=True)
        detect_boat = st.checkbox("⛵ Detect Boats", value=True)
        
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.0, 
            max_value=1.0, 
            value=0.5, 
            step=0.05,
            help="Minimum confidence for detections"
        )

        # Create detection filters dictionary
        detection_filters = {
            'person': detect_person,
            'bicycle': detect_bicycle,
            'car': detect_car,
            'motorcycle': detect_motorcycle,
            'bus': detect_bus,
            'truck': detect_truck,
            'boat': detect_boat,
        }

# ===================================================================================
# MAIN CONTENT AREA - TOP METRICS
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

# Quick select buttons


col1, col2 = st.columns([1,1])

# Manual URL input


with col1:
    cont3 = st.container(border=True)
    with cont3:
        st.subheader("📹 Stream Selection",divider="grey")
        st.write("")
        st.write("Select a preset stream or enter a custom URL:")

        if st.button("☘️ Dublin - Temple Bar", use_container_width=True):
            st.session_state.current_url = preset_streams["Dublin, Ireland - Temple Bar"]
            st.rerun()


        if st.button(":bus: London - Abbey Road", use_container_width=True):
            st.session_state.current_url = preset_streams["London, UK - Abbey Road"]
            st.rerun()


        if st.button("🚃 Melbourne - Intersection", use_container_width=True):
            st.session_state.current_url = preset_streams["Melbourne, Australia - Intersection"]
            st.rerun()


        if st.button(":ship: Sydney - Harbour Bridge", use_container_width=True):
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
youtube_url = st.session_state.current_url

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

st.markdown("---")
# ===================================================================================
# MAIN CONTENT AREA - VIDEO & VISUALIZATION
# ===================================================================================

col2, col3 = st.columns([1, 1])

with col2:
    cont5 = st.container(border=True)
    with cont5:
        st.subheader("🎯 Detection & Tracking Visualization", divider="grey")
        annotated_frame_placeholder = st.empty()
        st.write("")

with col3:
    cont6 = st.container(border=True)
    with cont6:
        st.subheader("📈 Pedestrian Count Over Time", divider="grey")
        time_series_placeholder = st.empty()
        st.write("")
        

# Summary text
summary_placeholder = st.empty()
summary_placeholder.markdown("**No analysis running. Configure settings and start analysis.**")
summary_placeholder.markdown("---")

# Top metrics
col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
unique_metric = col_metric1.metric("🆔 Unique People (Session)", 0, help="Total unique individuals tracked")
current_metric = col_metric2.metric("👁️ Currently Visible", 0, help="People currently in frame")
peak_metric = col_metric3.metric("📈 Peak Count", 0, help="Maximum people visible at once")
rate_metric = col_metric4.metric("⚡ Rate (people/min)", "0.0", help="Average unique people per minute")

st.markdown("---")

# ===================================================================================
# CONTROL BUTTONS
# ===================================================================================
st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 3: Start Analysis</h2>", unsafe_allow_html=True)


st.subheader("▶️ Control Panel", divider="grey")

cont7 = st.container(border=True)
with cont7:
    st.write("Use the buttons below to start/stop analysis or clear data.")
    col_btn1, col_btn2, col_btn3 = st.columns(3)

    with col_btn1:
        analyze_btn = st.button(
            "▶️ Start Analysis", 
            disabled=not youtube_url or st.session_state.analyzing,
            use_container_width=True
        )

    with col_btn2:
        stop_btn = st.button(
            "⏹️ Stop Analysis", 
            disabled=not st.session_state.analyzing,
            use_container_width=True
        )

    with col_btn3:
        clear_log_btn = st.button(
            "🗑️ Clear All Data",
            use_container_width=True
        )

    # Handle clear button
    if clear_log_btn:
        st.session_state.detection_log = []
        st.session_state.all_detections_df = pd.DataFrame()
        st.session_state.tracking_history = []
        st.session_state.time_series_data = deque(maxlen=500)
        st.session_state.unique_people_count = 0
        summary_placeholder.markdown("**Data cleared. Configure settings and start analysis.**")
        annotated_frame_placeholder.empty()

        st.rerun()

st.markdown("---")

# ===================================================================================
# DATA TABS
# ===================================================================================

# Center and colour the step sub header
st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 4: Explore Data & Export Results</h2>", unsafe_allow_html=True)


st.subheader("📊 Detailed Analysis", divider="grey")
tab1, tab2, tab3 = st.tabs(["📋 Tracking History", "📊 Detection Data", "📈 Statistics"])

# ===================================================================================
# HELPER FUNCTIONS
# ===================================================================================

@st.cache_resource
def load_model(model_name):
    """Load and cache YOLO model"""
    return YOLO(model_name)

def get_stream_url(youtube_url):
    """Extract direct video stream URL from YouTube"""
    ydl_opts = {
        'format': 'best[ext=mp4]/best',
        'quiet': True,
        'no_warnings': True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=False)
            return info['url']
    except Exception as e:
        st.error(f"Error getting stream URL: {str(e)}")
        return None

def analyze_frame(frame, model, filters, conf_threshold, resize_factor):
    """
    Analyze a single frame with YOLO detection.
    
    Args:
        frame: Input video frame
        model: YOLO model instance
        filters: Dictionary of object types to detect
        conf_threshold: Confidence threshold for detections
        resize_factor: Factor to resize frame before processing
        
    Returns:
        detections: Dictionary of detected object counts
        detailed_detections: List of detailed detection info
        person_centroids: List of (x,y) centroids for people
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
    
    # Run YOLO inference
    results = model(frame_resized, conf=conf_threshold, verbose=False)[0]
    
    detections = {}
    detailed_detections = []
    person_centroids = []
    
    # Process each detection
    for box in results.boxes:
        class_id = int(box.cls[0])
        class_name = results.names[class_id]
        confidence = float(box.conf[0])
        
        # Check if this object type should be detected
        if class_name in filters and filters[class_name]:
            if class_name not in detections:
                detections[class_name] = 0
            detections[class_name] += 1
            
            # Get bounding box and scale back to original size
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            x1, y1, x2, y2 = int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)
            
            # Calculate center point
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)
            
            # Collect person centroids for tracking
            if class_name == 'person':
                person_centroids.append((center_x, center_y))
            
            # Store detailed detection info
            detailed_detections.append({
                'object_type': class_name,
                'confidence': confidence,
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
            })
    
    return detections, detailed_detections, person_centroids

def draw_tracking_visualization(frame, tracker):
    """
    Draw tracking IDs and bounding boxes on frame.
    
    Args:
        frame: Input video frame
        tracker: SimpleTracker instance
        
    Returns:
        Annotated frame with tracking visualization
    """
    annotated_frame = frame.copy()
    
    # Draw each tracked object
    for object_id, centroid in tracker.objects.items():
        # Draw centroid
        cv2.circle(annotated_frame, tuple(map(int, centroid)), 8, (0, 255, 0), -1)
        
        # Draw ID label
        text = f"ID: {object_id}"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(annotated_frame, 
                     (int(centroid[0]) - 5, int(centroid[1]) - text_height - 25),
                     (int(centroid[0]) + text_width + 5, int(centroid[1]) - 15),
                     (0, 255, 0), -1)
        
        # Text
        cv2.putText(annotated_frame, text, 
                   (int(centroid[0]), int(centroid[1]) - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return annotated_frame

# ===================================================================================
# MAIN ANALYSIS FUNCTION
# ===================================================================================

def run_analysis():
    """
    Main analysis loop that:
    1. Connects to YouTube stream
    2. Processes frames with YOLO
    3. Tracks unique people
    4. Updates visualizations and metrics
    5. Logs data for export
    """
    
    # Validate URL
    if not youtube_url:
        st.error("Please enter a YouTube URL")
        return
    
    st.session_state.analyzing = True
    
    # Initialize tracker
    tracker = SimpleTracker(
        max_disappeared=max_disappeared_frames,
        max_distance=max_tracking_distance
    )
    
    # Load YOLO model
    with st.spinner(f"Loading YOLO model ({model_size})..."):
        model = load_model(model_size)
    
    # Get stream URL
    with st.spinner("Connecting to stream..."):
        stream_url = get_stream_url(youtube_url)
    
    if not stream_url:
        st.session_state.analyzing = False
        return
    
    # Open video capture
    cap = cv2.VideoCapture(stream_url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering
    
    if not cap.isOpened():
        st.error("Failed to open video stream")
        st.session_state.analyzing = False
        return
    
    st.success("✅ Analysis started with tracking enabled!")
    
    # Initialize counters
    frame_count = 0
    processed_count = 0
    last_analysis_time = time.time()
    session_start_time = datetime.now()
    peak_count = 0
    seen_ids = set()  # Track all unique IDs seen
    
    try:
        # Main processing loop
        while st.session_state.analyzing:
            ret, frame = cap.read()
            
            if not ret:
                st.warning("Stream ended or connection lost")
                break
            
            frame_count += 1
            current_time = time.time()
            
            # Skip frames based on frame_skip setting
            if frame_count % frame_skip != 0:
                continue
            
            # Process frame at specified interval
            if current_time - last_analysis_time >= analysis_interval:
                processed_count += 1
                timestamp = datetime.now()
                
                # Run YOLO detection
                detections, detailed, person_centroids = analyze_frame(
                    frame, model, detection_filters, confidence_threshold, resize_factor
                )
                
                # Update tracker with person detections
                if track_people_only:
                    tracker.update(person_centroids, timestamp)
                
                # Draw tracking visualization
                annotated_frame = draw_tracking_visualization(frame, tracker)
                
                # Resize for display if needed
                if annotated_frame.shape[1] > 800:
                    display_height = int(annotated_frame.shape[0] * (800 / annotated_frame.shape[1]))
                    display_frame = cv2.resize(annotated_frame, (800, display_height))
                else:
                    display_frame = annotated_frame
                
                # Convert to RGB and display
                annotated_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                annotated_pil = Image.fromarray(annotated_rgb)
                annotated_frame_placeholder.image(annotated_pil, use_container_width=True)
                
                # Update tracking statistics
                current_people = len(tracker.objects)
                seen_ids.update(tracker.objects.keys())
                st.session_state.unique_people_count = len(seen_ids)
                
                if current_people > peak_count:
                    peak_count = current_people
                
                # Calculate rate
                elapsed_minutes = (timestamp - session_start_time).total_seconds() / 60
                rate = st.session_state.unique_people_count / elapsed_minutes if elapsed_minutes > 0 else 0
                
                # Update top metrics
                unique_metric.metric(
                    "🆔 Unique People (Session)", 
                    st.session_state.unique_people_count,
                    help="Total unique individuals tracked"
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
                
                # Update summary text
                summary_text = f"**Timestamp:** {timestamp.strftime('%H:%M:%S')}\n\n"
                summary_text += f"**Frames Processed:** {processed_count}\n\n"
                
                if detections:
                    summary_text += "**Detections:**\n"
                    for obj, count in sorted(detections.items(), key=lambda x: x[1], reverse=True):
                        summary_text += f"- {obj.title()}: {count}\n"
                else:
                    summary_text += "*No objects detected*\n"
                
                summary_placeholder.markdown(summary_text)
                
                # Add to time series data
                st.session_state.time_series_data.append({
                    'timestamp': timestamp,
                    'count': current_people,
                })
                
                # Update time series graph
                if len(st.session_state.time_series_data) > 1:
                    ts_df = pd.DataFrame(list(st.session_state.time_series_data))
                    ts_df['moving_avg'] = ts_df['count'].rolling(window=5, min_periods=1).mean()
                    
                    time_series_placeholder.line_chart(
                        ts_df.set_index('timestamp')[['count', 'moving_avg']],
                        use_container_width=True
                    )
                
                # Add to tracking history
                tracking_entry = {
                    'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    'current_visible': current_people,
                    'unique_session': st.session_state.unique_people_count,
                    'peak_count': peak_count,
                }
                st.session_state.tracking_history.insert(0, tracking_entry)
                
                if len(st.session_state.tracking_history) > 500:
                    st.session_state.tracking_history = st.session_state.tracking_history[:500]
                
                # Add detailed detections to dataframe
                if detailed:
                    for det in detailed:
                        det['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                        det['frame_number'] = frame_count
                    
                    new_df = pd.DataFrame(detailed)
                    if st.session_state.all_detections_df.empty:
                        st.session_state.all_detections_df = new_df
                    else:
                        st.session_state.all_detections_df = pd.concat(
                            [new_df, st.session_state.all_detections_df], 
                            ignore_index=True
                        )
                    
                    if len(st.session_state.all_detections_df) > 1000:
                        st.session_state.all_detections_df = st.session_state.all_detections_df.head(1000)
                
                last_analysis_time = current_time
            
            time.sleep(0.05)
            
    except Exception as e:
        st.error(f"Analysis error: {str(e)}")
    finally:
        cap.release()
        st.session_state.analyzing = False

# ===================================================================================
# TAB 1: TRACKING HISTORY
# ===================================================================================

with tab1:
    if st.session_state.tracking_history:
        st.markdown("### 📊 Tracking History")
        
        tracking_df = pd.DataFrame(st.session_state.tracking_history)
        
        # Summary metrics
        col_t1, col_t2, col_t3, col_t4 = st.columns(4)
        with col_t1:
            st.metric("Total Snapshots", len(tracking_df))
        with col_t2:
            st.metric("Avg Visible", f"{tracking_df['current_visible'].mean():.1f}")
        with col_t3:
            st.metric("Max Visible", tracking_df['current_visible'].max())
        with col_t4:
            st.metric("Unique People", tracking_df['unique_session'].max())
        
        # Display table
        st.dataframe(tracking_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = tracking_df.to_csv(index=False)
        st.download_button(
            "📥 Download Tracking Data",
            csv,
            f"tracking_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("No tracking data yet. Start analysis to begin tracking.")

# ===================================================================================
# TAB 2: DETECTION DATA
# ===================================================================================

with tab2:
    if not st.session_state.all_detections_df.empty:
        st.markdown("### 📊 All Detections")
        
        # Filter options
        filter_object = st.multiselect(
            "Filter by Object Type",
            options=st.session_state.all_detections_df['object_type'].unique().tolist(),
            default=st.session_state.all_detections_df['object_type'].unique().tolist()
        )
        
        # Apply filter
        filtered_df = st.session_state.all_detections_df[
            st.session_state.all_detections_df['object_type'].isin(filter_object)
        ].head(200)
        
        # Display table
        st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Detection Data",
            csv,
            f"detections_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            "text/csv",
            use_container_width=True
        )
    else:
        st.info("No detection data yet. Start analysis to populate this table.")

# ===================================================================================
# TAB 3: STATISTICS
# ===================================================================================

with tab3:
    if st.session_state.tracking_history:
        st.markdown("### 📈 Session Statistics")
        
        tracking_df = pd.DataFrame(st.session_state.tracking_history)
        
        # Two column layout for charts
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown("#### People Count Over Time")
            st.line_chart(
                tracking_df.set_index('timestamp')['current_visible'],
                use_container_width=True
            )
            
            st.markdown("#### Summary Statistics")
            stats_data = {
                'Metric': [
                    'Total Unique People',
                    'Average Visible',
                    'Peak Visible',
                    'Total Observations'
                ],
                'Value': [
                    st.session_state.unique_people_count,
                    f"{tracking_df['current_visible'].mean():.1f}",
                    tracking_df['current_visible'].max(),
                    len(tracking_df)
                ]
            }
            st.dataframe(pd.DataFrame(stats_data), use_container_width=True, hide_index=True)
        
        with col_s2:
            st.markdown("#### Cumulative Unique People")
            st.line_chart(
                tracking_df.set_index('timestamp')['unique_session'],
                use_container_width=True
            )
            
            st.markdown("#### Peak Analysis")
            if len(tracking_df) > 0:
                busiest = tracking_df.loc[tracking_df['current_visible'].idxmax()]
                st.info(f"**Busiest Moment:** {busiest['current_visible']} people at {busiest['timestamp']}")
                
                # Object type distribution if available
                if not st.session_state.all_detections_df.empty:
                    st.markdown("#### Object Type Distribution")
                    obj_counts = st.session_state.all_detections_df['object_type'].value_counts()
                    st.bar_chart(obj_counts)
    else:
        st.info("No statistics available yet. Start analysis to see statistics.")

# ===================================================================================
# HANDLE BUTTON CLICKS
# ===================================================================================

if analyze_btn:
    run_analysis()

if stop_btn:
    st.session_state.analyzing = False
    st.rerun()

# ===================================================================================
# INSTRUCTIONS & HELP
# ===================================================================================

with st.expander("ℹ️ How to Use This Application"):
    st.markdown("""
    ## 🎯 Quick Start Guide
    
    ### 1. Select a Stream
    - Click one of the preset location buttons (Dublin, Melbourne, London, Sydney)
    - OR paste your own YouTube live stream URL
    
    ### 2. Configure Settings (Sidebar)
    
    **Performance Settings:**
    - **Frame Skip**: Higher values = faster processing but less frequent updates
    - **Resize Factor**: Lower values = faster processing but lower accuracy
    - **Model Size**: 
      - `yolov8n.pt` - Fastest, good for real-time
      - `yolov8s.pt` - Balanced speed and accuracy
      - `yolov8m.pt` - Most accurate, slower
    
    **Tracking Settings:**
    - **Track People Only**: Recommended for pedestrian analysis
    - **Max Tracking Distance**: Increase for slow frame rates or crowded scenes
    - **Max Disappeared Frames**: How long to wait before considering someone "left"
    
    **Detection Filters:**
    - Check/uncheck objects you want to detect
    - Lower confidence threshold = more detections (but more false positives)
    
    ### 3. Start Analysis
    - Click "▶️ Start Analysis" button
    - Watch the metrics update in real-time
    - View the tracking visualization showing unique IDs
    
    ### 4. View Results
    
    **Live Metrics (Top):**
    - 🆔 **Unique People**: Total different individuals tracked this session
    - 👁️ **Currently Visible**: How many people are in frame right now
    - 📈 **Peak Count**: Maximum people visible at any moment
    - ⚡ **Rate**: Average unique people per minute
    
    **Time Series Graph:**
    - Blue line: Current people count
    - Orange line: 5-point moving average (smoothed)
    
    **Three Data Tabs:**
    1. **Tracking History**: Timestamped log of people counts
    2. **Detection Data**: Raw detection data with bounding boxes
    3. **Statistics**: Charts and summary statistics
    
    ### 5. Export Data
    - Each tab has a download button for CSV export
    - Data includes timestamps, counts, positions, and confidence scores
    
    ## 💡 Tips for Best Results
    
    ### For Accurate Tracking:
    - Use lower frame skip (1-2)
    - Use higher resize factor (0.75-1.0)
    - Increase max tracking distance in crowded scenes (150-200)
    - Use yolov8s or yolov8m model
    
    ### For Fast Performance:
    - Use higher frame skip (3-5)
    - Use lower resize factor (0.25-0.5)
    - Use yolov8n model
    - Increase analysis interval (5-10 seconds)
    
    ### For Crowded Scenes:
    - Increase max tracking distance to 150-200 pixels
    - Increase max disappeared frames to 40-60
    - Lower confidence threshold to 0.3-0.4
    
    ### For Sparse Scenes:
    - Decrease max tracking distance to 50-80 pixels
    - Decrease max disappeared frames to 15-20
    - Increase confidence threshold to 0.6-0.7
    
    ## 🔧 Troubleshooting
    
    **Problem: Stream won't load**
    - Check the YouTube URL is correct
    - Ensure the video is public and not age-restricted
    - Try a different stream
    
    **Problem: Analysis is too slow**
    - Increase frame skip to 5-10
    - Reduce resize factor to 0.25
    - Use yolov8n model
    - Increase analysis interval to 5-10 seconds
    
    **Problem: Tracking loses people**
    - Increase max tracking distance
    - Decrease frame skip
    - Increase resize factor for better detection
    
    **Problem: Too many false detections**
    - Increase confidence threshold
    - Use a larger model (yolov8s or yolov8m)
    - Uncheck object types you don't need
    
    ## 📊 Understanding the Data
    
    **Unique People vs Current Count:**
    - **Unique People**: Total different individuals seen (counts each person once)
    - **Current Count**: How many people are visible right now
    - Example: If 10 people walk by one at a time, unique = 10, current = 1
    
    **Tracking IDs:**
    - Each person gets a unique ID number
    - IDs persist as long as the person is tracked
    - When someone leaves and returns, they get a new ID
    
    **Rate (people/min):**
    - Total unique people divided by elapsed time
    - Useful for understanding traffic flow
    - Example: 60 people/min = busy pedestrian area
    
    ## 🚀 Advanced Usage
    
    **Monitoring Multiple Locations:**
    - Use the preset buttons to quickly switch between streams
    - Clear data between locations for separate analysis
    - Export CSV data for each location
    
    **Long-term Monitoring:**
    - The app stores up to 500 tracking snapshots
    - For longer sessions, export data periodically
    - Consider running analysis at longer intervals (5-10 sec)
    
    **Custom Streams:**
    - Works with any public YouTube live stream
    - Good for: streets, beaches, harbors, parks, events
    - Can detect boats in harbor/beach scenes
    
    ## 📝 Notes
    
    - First run downloads YOLO model (~6-20MB depending on size)
    - YouTube streams have 10-30 second delay (inherent to YouTube)
    - Tracking is approximate - accuracy depends on video quality and settings
    - Data is stored in browser session - refresh clears everything
    - Export data regularly if you need to keep it
    
    ## 🆘 Need Help?
    
    If you're having issues:
    1. Try the default settings first
    2. Test with one of the preset streams
    3. Check the troubleshooting section above
    4. Adjust one setting at a time to find optimal configuration
    """)

# ===================================================================================
# END OF APPLICATION
# ===================================================================================