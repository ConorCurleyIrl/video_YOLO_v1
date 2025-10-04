"""
Streamlit-specific UI implementation
Imports from core/ and utils/ modules
Following the framework-agnostic architectural patterns
"""
import streamlit as st
import os
import sys
import time
from datetime import datetime
from collections import deque
import pandas as pd
import numpy as np
from PIL import Image
import cv2

# Environment setup for cloud deployment
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import our core modules (framework-agnostic)
from core.tracking import SimpleTracker, TemporalTracker
from core.detection import FrameAnalyzer, HeatmapProcessor, create_heatmap_overlay
from core.visualization import draw_enhanced_visualization, calculate_tracking_quality
from utils.config import ConfigManager
from utils.model_loader import model_loader
from utils.stream_processor import StreamExtractor

class StreamlitApp:
    """Main Streamlit application class - handles only UI and Streamlit-specific logic"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.stream_extractor = StreamExtractor()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state variables"""
        defaults = {
            'analyzing': False,
            'current_url': "",
            'detection_log': [],
            'tracking_history': [],
            'all_detections_df': pd.DataFrame(),
            'time_series_data': deque(maxlen=500),
            'object_time_series': deque(maxlen=500),
            'unique_people_count': 0,
            'tracking_quality': 0.0,
            'heatmap_data': [],
            'heatmap_enabled': False
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render application header and page config"""
        st.set_page_config(page_title="Street Traffic Analyzer", layout="wide")
        
        # Header with refresh button
        col1, col3 = st.columns([8, 1])
        with col3:
            if st.button("🔄 Refresh", key="refresh"):
                st.session_state.clear()
                st.rerun()
        
        st.markdown("<h1 style='text-align: center;'>🚶 Live Video Analyzer with Enhanced Tracking</h1>", 
                   unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Analyze YouTube live streams with advanced tracking and detection filtering</p>", 
                   unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration using ConfigManager presets"""
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 1: Configure Settings</h2>", 
                       unsafe_allow_html=True)
            
            # Preset buttons using ConfigManager
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
                            self.apply_preset(preset_name)
                
                if st.session_state.get('defaults_set', False):
                    preset_labels = {'optimal': "🎯 Optimal", 'speed': "⚡ Speed", 'accuracy': "🏆 Accuracy"}
                    st.info(f"Using preset: {preset_labels.get(st.session_state.get('preset_mode', ''), 'Custom')}")
            
            # Performance settings
            performance_config = self.render_performance_settings()
            
            # Tracking settings
            tracker_config = self.render_tracker_settings()
            
            # Detection settings
            detection_config = self.render_detection_settings()
            
            # Heatmap settings
            heatmap_config = self.render_heatmap_settings()
            
            # Technical details expander
            self.render_technical_details()
            
            return {
                'performance': performance_config,
                'tracker': tracker_config,
                'detection': detection_config,
                'heatmap': heatmap_config
            }
    
    def apply_preset(self, preset_name: str):
        """Apply preset configuration from ConfigManager"""
        try:
            config = self.config_manager.get_preset_config(preset_name)
            
            # Apply performance settings
            for key, value in config['performance'].__dict__.items():
                st.session_state[f'{key}_default'] = value
            
            # Apply tracker settings
            for key, value in config['tracker'].__dict__.items():
                st.session_state[f'{key}_default'] = value
            
            # Apply detection settings
            for key, value in config['detection'].__dict__.items():
                st.session_state[f'{key}_default'] = value
            
            # Apply heatmap settings
            for key, value in config['heatmap'].__dict__.items():
                st.session_state[f'{key}_default'] = value
            
            # Apply model and tracker indices
            st.session_state['model_index_default'] = config['model_index']
            st.session_state['tracker_index_default'] = config['tracker_index']
            st.session_state['track_people_only_default'] = config['track_people_only']
            
            # Apply detection filters
            for filter_name, enabled in config['detection_filters'].items():
                st.session_state[f'detect_{filter_name}_default'] = enabled
            
            st.session_state.preset_mode = preset_name
            st.session_state.defaults_set = True
            st.rerun()
            
        except Exception as e:
            st.error(f"Error applying preset {preset_name}: {e}")
    
    def render_performance_settings(self):
        """Render performance configuration UI"""
        with st.container(border=True):
            st.subheader("⚡ Model & Performance", divider="rainbow")
            
            frame_skip = st.slider("Frame Skip", 1, 10, 
                                  st.session_state.get('frame_skip_default', 1),
                                  help="Process every Nth frame")
            
            resize_factor = st.slider("Image Resize Factor", 0.25, 1.0, 
                                     st.session_state.get('resize_factor_default', 0.5), 0.25,
                                     help="Resize frames before processing")
            
            # Use ConfigManager for model options
            model_options = [model.name for model in self.config_manager.YOLO_MODELS.values()]
            model_index = st.selectbox("YOLO Model", range(len(model_options)), 
                                      st.session_state.get('model_index_default', 1),
                                      format_func=lambda x: model_options[x],
                                      help="v8=stable, v10=faster, v11=newest")
            
            tracker_options = self.config_manager.TRACKER_TYPES
            tracker_index = st.selectbox("Tracking Algorithm", range(len(tracker_options)), 
                                        st.session_state.get('tracker_index_default', 0),
                                        format_func=lambda x: tracker_options[x],
                                        help="OpenCV=default, ByteTrack=fast, BoT-SORT=accurate")
            
            # Temporal smoothing for OpenCV tracker
            temporal_smoothing = False
            smoothing_factor = 0.7
            if tracker_options[tracker_index] == "OpenCV":
                temporal_smoothing = st.checkbox("Temporal Smoothing", 
                                                st.session_state.get('temporal_smoothing_default', True))
                if temporal_smoothing:
                    smoothing_factor = st.slider("Smoothing Factor", 0.5, 0.9, 
                                                st.session_state.get('smoothing_factor_default', 0.7), 0.1)
            
            analysis_interval = st.slider("Analysis Interval (sec)", 1, 10, 
                                         st.session_state.get('analysis_interval_default', 2))
            
            return {
                'frame_skip': frame_skip,
                'resize_factor': resize_factor,
                'model_index': model_index,
                'tracker_index': tracker_index,
                'temporal_smoothing': temporal_smoothing,
                'smoothing_factor': smoothing_factor,
                'analysis_interval': analysis_interval
            }
    
    def render_tracker_settings(self):
        """Render tracker configuration UI"""
        with st.container(border=True):
            st.subheader("🎯 Tracking & Detection", divider="rainbow")
            
            track_people_only = st.checkbox("Track People Only", 
                                           st.session_state.get('track_people_only_default', True))
            
            max_tracking_distance = st.slider("Max Tracking Distance", 25, 300, 
                                             st.session_state.get('max_distance_default', 100))
            
            max_disappeared_frames = st.slider("Max Disappeared Frames", 5, 60, 
                                              st.session_state.get('max_disappeared_default', 30))
            
            return {
                'track_people_only': track_people_only,
                'max_distance': max_tracking_distance,
                'max_disappeared': max_disappeared_frames
            }
    
    def render_detection_settings(self):
        """Render detection configuration UI"""
        st.subheader("🔍 Detection Settings")
        
        confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 
                                        st.session_state.get('confidence_threshold_default', 0.5), 0.1)
        
        enhanced_filtering = st.checkbox("Enhanced Filtering", 
                                       st.session_state.get('enhanced_filtering_default', True))
        
        nms_threshold = 0.4
        min_detection_area = 100
        
        if enhanced_filtering:
            nms_threshold = st.slider("NMS Threshold", 0.1, 0.8, 
                                    st.session_state.get('nms_threshold_default', 0.4), 0.1)
            min_detection_area = st.slider("Min Detection Area", 50, 500, 
                                         st.session_state.get('min_detection_area_default', 100), 50)
        
        # Detection filters
        st.subheader("🔍 Detection Filters")
        
        filter_objects = ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat']
        emoji_map = {'person': '👤', 'bicycle': '🚲', 'car': '🚗', 'motorcycle': '🏍️', 
                    'bus': '🚌', 'truck': '🚚', 'boat': '⛵'}
        
        detection_filters = {}
        for obj_type in filter_objects:
            detection_filters[obj_type] = st.checkbox(
                f"{emoji_map.get(obj_type, '🔸')} {obj_type.title()}s",
                st.session_state.get(f'detect_{obj_type}_default', True)
            )
        
        return {
            'confidence_threshold': confidence_threshold,
            'enhanced_filtering': enhanced_filtering,
            'nms_threshold': nms_threshold,
            'min_area': min_detection_area,
            'filters': detection_filters
        }
    
    def render_heatmap_settings(self):
        """Render heatmap configuration UI"""
        heatmap_enabled = st.checkbox("Generate Heatmap", 
                                    st.session_state.get('enabled_default', False))
        
        heatmap_object_type = "person"
        heatmap_decay = 0.95
        
        if heatmap_enabled:
            heatmap_object_type = st.selectbox("Heatmap Object", ["person", "car", "bicycle", "all"], 
                                             st.session_state.get('heatmap_object_type_default', 0))
            heatmap_decay = st.slider("Heatmap Decay", 0.90, 0.99, 
                                    st.session_state.get('decay_factor_default', 0.95), 0.01)
        
        return {
            'enabled': heatmap_enabled,
            'object_type': heatmap_object_type,
            'decay_factor': heatmap_decay
        }
    
    def render_technical_details(self):
        """Render technical details expander"""
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
    
    def render_stream_selection(self):
        """Render stream selection UI"""
        st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 2: Select Live Stream</h2>", 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            with st.container(border=True):
                st.subheader("📹 Stream Selection", divider="grey")
                st.write("Select a preset stream or enter custom URL:")
                
                # Use ConfigManager preset streams
                preset_streams = self.config_manager.PRESET_STREAMS
                
                if st.button("☘️ Dublin - Temple Bar", use_container_width=True, key="dublin_btn"):
                    st.session_state.current_url = preset_streams["Dublin, Ireland - Temple Bar"]
                    st.rerun()
                if st.button("🚌 London - Abbey Road", use_container_width=True, key="london_btn"):
                    st.session_state.current_url = preset_streams["London, UK - Abbey Road"]
                    st.rerun()
                if st.button("🚃 Melbourne - Intersection", use_container_width=True, key="melbourne_btn"):
                    st.session_state.current_url = preset_streams["Melbourne, Australia - Intersection"]
                    st.rerun()
                if st.button("🚢 Sydney - Harbour Bridge", use_container_width=True, key="sydney_btn"):
                    st.session_state.current_url = preset_streams["Sydney, Australia - Harbour Bridge"]
                    st.rerun()
                
                youtube_url = st.text_input("Or enter YouTube URL", 
                                          value=st.session_state.current_url,
                                          placeholder="https://youtube.com/watch?v=...", 
                                          key="url_input")
        
        # Update session state with current URL
        youtube_url = st.session_state.current_url if youtube_url == "" else youtube_url
        if youtube_url != st.session_state.current_url:
            st.session_state.current_url = youtube_url
        
        with col2:
            self.render_video_preview(youtube_url)
        
        with col3:
            self.render_control_panel(youtube_url)
        
        return youtube_url
    
    def render_video_preview(self, youtube_url: str):
        """Render video preview"""
        with st.container(border=True):
            st.subheader("📺 Live Stream", divider="grey")
            video_placeholder = st.empty()
            
            if youtube_url and ('youtube.com' in youtube_url or 'youtu.be' in youtube_url):
                video_id = self.stream_extractor.extract_video_id(youtube_url)
                if video_id:
                    video_placeholder.markdown(
                        f'<iframe width="100%" height="357" src="https://www.youtube.com/embed/{video_id}?autoplay=1&mute=1" frameborder="0" allowfullscreen></iframe>',
                        unsafe_allow_html=True
                    )
    
    def render_control_panel(self, youtube_url: str):
        """Render control panel"""
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 3: Punch it!</h2>", 
                       unsafe_allow_html=True)
            st.subheader("▶️ Control Panel", divider="grey")
            
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("▶️ Start", disabled=st.session_state.analyzing, 
                           use_container_width=True, type="primary", key="start_btn"):
                    if youtube_url:
                        st.session_state.analyzing = True
                        st.rerun()
                    else:
                        st.error("❌ Enter YouTube URL first!")
            
            with btn_col2:
                if st.button("⏹️ Stop", disabled=not st.session_state.analyzing, 
                           use_container_width=True, type="secondary", key="stop_btn"):
                    st.session_state.analyzing = False
                    st.info("⏹️ Analysis stopped")
                    st.rerun()
            
            if st.session_state.analyzing:
                st.success("🟢 **Status:** Running")
            
            if st.button("🗑️ Clear All Data", use_container_width=True, 
                        type="secondary", key="clear_btn"):
                self.clear_session_data()
                st.success("🗑️ Data cleared!")
                st.rerun()
    
    def clear_session_data(self):
        """Clear session data"""
        keys_to_clear = ['detection_log', 'tracking_history', 'heatmap_data', 
                        'unique_people_count', 'tracking_quality']
        
        for key in keys_to_clear:
            if key in st.session_state:
                if key in ['detection_log', 'tracking_history', 'heatmap_data']:
                    st.session_state[key] = []
                elif key == 'all_detections_df':
                    st.session_state[key] = pd.DataFrame()
                else:
                    st.session_state[key] = 0 if key == 'unique_people_count' else 0.0
        
        st.session_state.time_series_data = deque(maxlen=500)
        st.session_state.object_time_series = deque(maxlen=500)
    
    def render_visualization_area(self):
        """Render main visualization area"""
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
            
            return annotated_frame_placeholder, tracking_quality_placeholder, time_series_placeholder, summary_placeholder
    
    def render_metrics(self):
        """Render metrics display"""
        # Primary metrics
        col_m1, col_m2, col_m3, col_m4 = st.columns(4)
        unique_metric = col_m1.metric("🆔 Unique People", st.session_state.unique_people_count)
        current_metric = col_m2.metric("👁️ Currently Visible", 0)
        peak_metric = col_m3.metric("📈 Peak Count", 0)
        rate_metric = col_m4.metric("⚡ Rate (people/min)", "0.0")
        
        # Secondary metrics
        col_m5, col_m6, col_m7, col_m8 = st.columns(4)
        quality_metric = col_m5.metric("🎯 Tracking Quality", f"{st.session_state.tracking_quality:.1%}")
        fps_metric = col_m6.metric("📊 Processing FPS", "0.0")
        detection_metric = col_m7.metric("🔍 Total Detections", 0)
        accuracy_metric = col_m8.metric("✅ Detection Accuracy", "0.0%")
        
        st.markdown("---")
        
        return (unique_metric, current_metric, peak_metric, rate_metric,
                quality_metric, fps_metric, detection_metric, accuracy_metric)
    
    def run_enhanced_analysis(self, youtube_url: str, config: dict, 
                            annotated_frame_placeholder, tracking_quality_placeholder, 
                            time_series_placeholder, summary_placeholder,
                            unique_metric, current_metric, peak_metric, rate_metric,
                            quality_metric, fps_metric, detection_metric, accuracy_metric):
        """Main analysis loop using modular components"""
        
        if not youtube_url:
            st.error("Please enter a YouTube URL")
            st.session_state.analyzing = False
            return
        
        status_placeholder = st.empty()
        status_placeholder.info("🚀 Initializing analysis...")
        
        # Initialize tracker with temporal smoothing if OpenCV
        tracker_type = self.config_manager.get_tracker_name(config['performance']['tracker_index'])
        tracker = SimpleTracker(
            max_disappeared=config['tracker']['max_disappeared'], 
            max_distance=config['tracker']['max_distance']
        )
        
        if (tracker_type == "OpenCV" and 
            config['performance']['temporal_smoothing']):
            tracker.temporal_tracker.smoothing_factor = config['performance']['smoothing_factor']
        
        # Load model using ConfigManager
        status_placeholder.info("🤖 Loading YOLO model...")
        model_config = self.config_manager.get_model_config(config['performance']['model_index'])
        model = model_loader.load_yolo_model(model_config.file_path)
        
        # Initialize FrameAnalyzer
        frame_analyzer = FrameAnalyzer(model, config['detection'])
        
        # Initialize HeatmapProcessor if enabled
        heatmap_processor = None
        
        # Get stream URL using StreamExtractor
        status_placeholder.info("🔗 Extracting stream URL...")
        stream_url = self.stream_extractor.get_youtube_stream_url(youtube_url)
        
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
        
        # Initialize tracking variables
        frame_count = processed_count = 0
        last_analysis_time = time.time()
        session_start_time = datetime.now()
        peak_count = total_detections = consecutive_errors = 0
        seen_ids = set()
        processing_times = deque(maxlen=50)
        detection_accuracy_scores = deque(maxlen=100)
        
        # Main analysis loop
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
            
            if frame_count % config['performance']['frame_skip'] != 0:
                continue
            
            if current_time - last_analysis_time >= config['performance']['analysis_interval']:
                processed_count += 1
                timestamp = datetime.now()
                
                try:
                    # Initialize heatmap processor with first frame
                    if config['heatmap']['enabled'] and heatmap_processor is None:
                        heatmap_processor = HeatmapProcessor(frame.shape[:2])
                    
                    # Analyze frame using FrameAnalyzer
                    detections, detailed_detections, person_centroids, tracked_objects = frame_analyzer.analyze_frame(
                        frame, 
                        config['detection']['filters'], 
                        config['performance']['resize_factor'],
                        tracker_type
                    )
                    
                    total_detections += len(detailed_detections)
                    
                    if detailed_detections:
                        avg_confidence = np.mean([det['confidence'] for det in detailed_detections])
                        detection_accuracy_scores.append(avg_confidence)
                    
                    # Update heatmap
                    if heatmap_processor:
                        heatmap_processor.update(
                            detailed_detections, 
                            config['heatmap']['object_type'],
                            config['heatmap']['decay_factor']
                        )
                    
                    # Update tracking using framework-agnostic tracker
                    current_people = 0
                    if tracker_type == "OpenCV":
                        if config['tracker']['track_people_only']:
                            tracker.update(person_centroids, timestamp)
                            current_people = len(tracker.objects)
                            seen_ids.update(tracker.objects.keys())
                        else:
                            all_centroids = [(det['center_x'], det['center_y']) for det in detailed_detections]
                            tracker.update(all_centroids, timestamp)
                            current_people = len([det for det in detailed_detections if det['object_type'] == 'person'])
                            seen_ids.update(tracker.objects.keys())
                    elif tracker_type in ["ByteTrack (YOLO)", "BoT-SORT (YOLO)"]:
                        if config['tracker']['track_people_only']:
                            person_tracks = {tid: pos for tid, pos in tracked_objects.items()}
                            current_people = len(person_tracks)
                            seen_ids.update(person_tracks.keys())
                        else:
                            current_people = len([det for det in detailed_detections if det['object_type'] == 'person'])
                            seen_ids.update(tracked_objects.keys())
                    else:
                        current_people = len([det for det in detailed_detections if det['object_type'] == 'person'])
                    
                    # Calculate tracking quality using core function
                    st.session_state.tracking_quality = calculate_tracking_quality(tracker, detailed_detections)
                    
                    # Draw visualization using core function
                    try:
                        heatmap_data = heatmap_processor.get_heatmap() if heatmap_processor else None
                        annotated_frame = draw_enhanced_visualization(
                            frame, tracker, detailed_detections, tracked_objects, 
                            config['heatmap']['enabled'], heatmap_data, tracker_type
                        )
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
                    
                    # Update time series data (Streamlit-specific)
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
                    
                    # Update session state for data tabs
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
                    if detailed_detections:
                        for det in detailed_detections:
                            det['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            det['frame_number'] = int(frame_count)
                            det['tracking_algorithm'] = str(tracker_type)
                        
                        try:
                            new_df = pd.DataFrame(detailed_detections)
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
        
        # Cleanup
        if cap:
            cap.release()
        st.session_state.analyzing = False
        status_placeholder.success("✅ Analysis completed!")
    
    def render_data_tabs(self):
        """Render data analysis tabs"""
        st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 4: Explore Data & Export</h2>", 
                   unsafe_allow_html=True)
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
    
    def run(self):
        """Main application run method"""
        # Render UI components
        self.render_header()
        config = self.render_sidebar()
        youtube_url = self.render_stream_selection()
        
        # Visualization area
        (annotated_frame_placeholder, tracking_quality_placeholder, 
         time_series_placeholder, summary_placeholder) = self.render_visualization_area()
        
        # Metrics
        (unique_metric, current_metric, peak_metric, rate_metric,
         quality_metric, fps_metric, detection_metric, accuracy_metric) = self.render_metrics()
        
        # Data tabs
        self.render_data_tabs()
        
        # Run analysis if active
        if st.session_state.analyzing and youtube_url:
            self.run_enhanced_analysis(
                youtube_url, config, 
                annotated_frame_placeholder, tracking_quality_placeholder, 
                time_series_placeholder, summary_placeholder,
                unique_metric, current_metric, peak_metric, rate_metric,
                quality_metric, fps_metric, detection_metric, accuracy_metric
            )


# Entry point
def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()