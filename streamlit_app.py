"""
Streamlit-specific UI implementation
Imports from core/ and utils/ modules
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

# Environment setup
os.environ['OPENCV_IO_ENABLE_JASPER'] = '1'
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

# Import our core modules
from core.video_processor import VideoProcessor
from core.tracking import SimpleTracker
from core.detection import FrameAnalyzer, HeatmapProcessor
from core.visualization import draw_enhanced_visualization, calculate_tracking_quality
from utils.config import ConfigManager
from utils.model_loader import model_loader
from utils.stream_processor import StreamExtractor

class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.config_manager = ConfigManager()
        self.stream_extractor = StreamExtractor()
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
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
            'video_processor': None,
            'heatmap_processor': None
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def render_header(self):
        """Render application header"""
        st.set_page_config(page_title="Street Traffic Analyzer", layout="wide")
        
        col1, col2 = st.columns([8, 1])
        with col2:
            if st.button("🔄 Refresh", key="refresh"):
                st.session_state.clear()
                st.rerun()
        
        st.markdown("<h1 style='text-align: center;'>🚶 Live Video Analyzer with Enhanced Tracking</h1>", 
                   unsafe_allow_html=True)
        st.markdown("<p style='text-align: center;'>Analyze YouTube live streams with advanced tracking and detection filtering</p>", 
                   unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar configuration"""
        with st.sidebar:
            st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 1: Configure Settings</h2>", 
                       unsafe_allow_html=True)
            
            # Preset buttons
            self.render_preset_buttons()
            
            # Configuration sections
            performance_config = self.render_performance_settings()
            tracker_config = self.render_tracker_settings() 
            detection_config = self.render_detection_settings()
            heatmap_config = self.render_heatmap_settings()
            
            return {
                'performance': performance_config,
                'tracker': tracker_config,
                'detection': detection_config,
                'heatmap': heatmap_config
            }
    
    def render_preset_buttons(self):
        """Render preset configuration buttons"""
        with st.container(border=True):
            st.subheader("⚙️ Quick Setup", divider="blue")
            col1, col2, col3 = st.columns(3)
            
            presets = [
                (col1, "🎯 Optimal", "optimal", "Best balance of accuracy and performance"),
                (col2, "⚡ Speed", "speed", "Maximum performance"),
                (col3, "🏆 Accuracy", "accuracy", "Maximum accuracy")
            ]
            
            for col, label, preset_name, help_text in presets:
                with col:
                    if st.button(label, use_container_width=True, help=help_text, key=f"preset_{preset_name}"):
                        self.apply_preset(preset_name)
    
    def apply_preset(self, preset_name: str):
        """Apply a preset configuration"""
        try:
            config = self.config_manager.get_preset_config(preset_name)
            
            # Store preset values in session state
            for key, value in config.items():
                if key != 'detection_filters':  # Handle filters separately
                    st.session_state[f'{key}_preset'] = value
            
            # Handle detection filters
            for filter_name, enabled in config['detection_filters'].items():
                st.session_state[f'detect_{filter_name}_preset'] = enabled
            
            st.session_state.preset_applied = preset_name
            st.rerun()
            
        except Exception as e:
            st.error(f"Error applying preset {preset_name}: {e}")
    
    def render_performance_settings(self):
        """Render performance configuration UI"""
        with st.container(border=True):
            st.subheader("⚡ Model & Performance", divider="rainbow")
            
            frame_skip = st.slider("Frame Skip", 1, 10, 
                                  st.session_state.get('performance_preset', {}).get('frame_skip', 1),
                                  help="Process every Nth frame")
            
            resize_factor = st.slider("Image Resize Factor", 0.25, 1.0, 
                                     st.session_state.get('performance_preset', {}).get('resize_factor', 0.75), 0.25,
                                     help="Resize frames before processing")
            
            model_options = [model.name for model in self.config_manager.YOLO_MODELS.values()]
            model_index = st.selectbox("YOLO Model", range(len(model_options)), 
                                      st.session_state.get('model_index_preset', 1),
                                      format_func=lambda x: model_options[x],
                                      help="v8=stable, v10=faster, v11=newest")
            
            tracker_options = self.config_manager.TRACKER_TYPES
            tracker_index = st.selectbox("Tracking Algorithm", range(len(tracker_options)), 
                                        st.session_state.get('tracker_index_preset', 0),
                                        format_func=lambda x: tracker_options[x],
                                        help="OpenCV=default, ByteTrack=fast, BoT-SORT=accurate")
            
            analysis_interval = st.slider("Analysis Interval (sec)", 1, 10, 
                                         st.session_state.get('performance_preset', {}).get('analysis_interval', 2))
            
            return {
                'frame_skip': frame_skip,
                'resize_factor': resize_factor,
                'model_index': model_index,
                'tracker_index': tracker_index,
                'analysis_interval': analysis_interval
            }
    
    def render_tracker_settings(self):
        """Render tracker configuration UI"""
        with st.container(border=True):
            st.subheader("🎯 Tracking Settings", divider="rainbow")
            
            track_people_only = st.checkbox("Track People Only", 
                                           st.session_state.get('track_people_only_preset', True))
            
            max_tracking_distance = st.slider("Max Tracking Distance", 25, 300, 
                                             st.session_state.get('tracker_preset', {}).get('max_distance', 100))
            
            max_disappeared_frames = st.slider("Max Disappeared Frames", 5, 60, 
                                              st.session_state.get('tracker_preset', {}).get('max_disappeared', 30))
            
            return {
                'track_people_only': track_people_only,
                'max_distance': max_tracking_distance,
                'max_disappeared': max_disappeared_frames
            }
    
    def render_detection_settings(self):
        """Render detection configuration UI"""
        with st.container(border=True):
            st.subheader("🔍 Detection Settings", divider="rainbow")
            
            confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 
                                            st.session_state.get('detection_preset', {}).get('confidence_threshold', 0.5), 0.1)
            
            enhanced_filtering = st.checkbox("Enhanced Filtering", 
                                           st.session_state.get('detection_preset', {}).get('enhanced_filtering', True))
            
            nms_threshold = 0.4
            min_area = 100
            
            if enhanced_filtering:
                nms_threshold = st.slider("NMS Threshold", 0.1, 0.8, 
                                        st.session_state.get('detection_preset', {}).get('nms_threshold', 0.4), 0.1)
                min_area = st.slider("Min Detection Area", 50, 500, 
                                   st.session_state.get('detection_preset', {}).get('min_area', 100), 50)
            
            # Detection filters
            st.subheader("🔍 Detection Filters")
            detection_filters = {}
            for obj_type in ['person', 'bicycle', 'car', 'motorcycle', 'bus', 'truck', 'boat']:
                emoji_map = {'person': '👤', 'bicycle': '🚲', 'car': '🚗', 'motorcycle': '🏍️', 
                           'bus': '🚌', 'truck': '🚚', 'boat': '⛵'}
                detection_filters[obj_type] = st.checkbox(
                    f"{emoji_map.get(obj_type, '🔸')} {obj_type.title()}",
                    st.session_state.get(f'detect_{obj_type}_preset', True)
                )
            
            return {
                'confidence_threshold': confidence_threshold,
                'enhanced_filtering': enhanced_filtering,
                'nms_threshold': nms_threshold,
                'min_area': min_area,
                'filters': detection_filters
            }
    
    def render_heatmap_settings(self):
        """Render heatmap configuration UI"""
        heatmap_enabled = st.checkbox("Generate Heatmap", False)
        heatmap_object_type = "person"
        heatmap_decay = 0.95
        
        if heatmap_enabled:
            heatmap_object_type = st.selectbox("Heatmap Object", ["person", "car", "bicycle", "all"], 0)
            heatmap_decay = st.slider("Heatmap Decay", 0.90, 0.99, 0.95, 0.01)
        
        return {
            'enabled': heatmap_enabled,
            'object_type': heatmap_object_type,
            'decay_factor': heatmap_decay
        }
    
    def render_stream_selection(self):
        """Render stream selection UI"""
        st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 2: Select Live Stream</h2>", 
                   unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col1:
            with st.container(border=True):
                st.subheader("📹 Stream Selection", divider="grey")
                st.write("Select a preset stream or enter custom URL:")
                
                streams = self.config_manager.PRESET_STREAMS
                for name, url in streams.items():
                    button_text = name.split(",")[0]  # Get city name
                    if st.button(button_text, use_container_width=True, key=f"stream_{name}"):
                        st.session_state.current_url = url
                        st.rerun()
                
                youtube_url = st.text_input("Or enter YouTube URL", 
                                          value=st.session_state.current_url,
                                          placeholder="https://youtube.com/watch?v=...", 
                                          key="url_input")
        
        # FIXED: Always update session state with current text input value
        st.session_state.current_url = youtube_url
        
        with col2:
            self.render_video_preview(youtube_url)
        
        with col3:
            self.render_control_panel(youtube_url)  # Pass current URL directly
        
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
    
    def render_control_panel(self, current_url: str = ""):
        """Render control panel - FIXED: Use passed URL parameter"""
        with st.container(border=True):
            st.markdown("<h2 style='text-align: center; color: #FF6347;'>Step 3: Control</h2>", 
                       unsafe_allow_html=True)
            st.subheader("▶️ Control Panel", divider="grey")
            
            # FIXED: Use current_url parameter instead of session state check
            url_valid = bool(current_url and current_url.strip())
            
            # Display current URL status
            if url_valid:
                st.success(f"✅ **URL Ready:** {current_url[:50]}{'...' if len(current_url) > 50 else ''}")
            else:
                st.warning("⚠️ **No URL entered**")
            
            btn_col1, btn_col2 = st.columns(2)
            
            with btn_col1:
                if st.button("▶️ Start", disabled=st.session_state.analyzing or not url_valid, 
                           use_container_width=True, type="primary", key="start_btn"):
                    if url_valid:
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
            else:
                st.info("⏸️ **Status:** Ready")
            
            if st.button("🗑️ Clear All Data", use_container_width=True, 
                        type="secondary", key="clear_btn"):
                self.clear_session_data()
                st.success("🗑️ Data cleared!")
                st.rerun()
    
    def clear_session_data(self):
        """Clear session data"""
        keys_to_clear = ['detection_log', 'tracking_history', 'all_detections_df', 
                        'unique_people_count', 'tracking_quality']
        
        for key in keys_to_clear:
            if key in st.session_state:
                if key == 'all_detections_df':
                    st.session_state[key] = pd.DataFrame()
                elif key in ['unique_people_count', 'tracking_quality']:
                    st.session_state[key] = 0 if key == 'unique_people_count' else 0.0
                else:
                    st.session_state[key] = []
        
        st.session_state.time_series_data = deque(maxlen=500)
        st.session_state.object_time_series = deque(maxlen=500)
    
    def render_visualization_area(self):
        """Render main visualization area"""
        with st.container(border=True):
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("🎯 Detection & Tracking", divider="grey")
                annotated_frame_placeholder = st.empty()
                tracking_quality_placeholder = st.empty()
            
            with col2:
                st.subheader("📈 Multi-Object Tracking", divider="grey")
                time_series_placeholder = st.empty()
            
            return annotated_frame_placeholder, tracking_quality_placeholder, time_series_placeholder
    
    def render_metrics(self):
        """Render metrics display"""
        col1, col2, col3, col4 = st.columns(4)
        
        unique_metric = col1.metric("🆔 Unique People", st.session_state.unique_people_count)
        current_metric = col2.metric("👁️ Currently Visible", 0)
        peak_metric = col3.metric("📈 Peak Count", 0)
        rate_metric = col4.metric("⚡ Rate (people/min)", "0.0")
        
        return unique_metric, current_metric, peak_metric, rate_metric
    
    def run_analysis(self, youtube_url: str, config: dict, frame_placeholder, 
                    quality_placeholder, time_series_placeholder):
        """Run the main analysis loop using modular components"""
        
        # FIXED: Additional validation with cleaner error message
        if not youtube_url or not youtube_url.strip():
            st.error("❌ Please enter a valid YouTube URL")
            st.session_state.analyzing = False
            return
        
        # Validate URL format
        if not ('youtube.com' in youtube_url or 'youtu.be' in youtube_url):
            st.error("❌ Please enter a valid YouTube URL (youtube.com or youtu.be)")
            st.session_state.analyzing = False
            return
        
        status_placeholder = st.empty()
        status_placeholder.info("🚀 Initializing modular analysis...")
        
        try:
            # Load model with caching
            model_config = self.config_manager.get_model_config(config['performance']['model_index'])
            status_placeholder.info(f"📥 Loading {model_config.name}...")
            model = model_loader.load_yolo_model(model_config.file_path)
            
            # Initialize components
            frame_analyzer = FrameAnalyzer(model, config['detection'])
            
            # Initialize tracker
            tracker_type = self.config_manager.get_tracker_name(config['performance']['tracker_index'])
            tracker = SimpleTracker(
                max_disappeared=config['tracker']['max_disappeared'],
                max_distance=config['tracker']['max_distance']
            )
            
            # Initialize heatmap processor
            heatmap_processor = None
            if config['heatmap']['enabled']:
                # We'll initialize this when we get the first frame
                pass
            
            # Extract stream URL
            status_placeholder.info("🔗 Extracting stream URL...")
            stream_url = self.stream_extractor.get_youtube_stream_url(youtube_url)
            
            if not stream_url:
                st.error("❌ Failed to extract stream URL. Please check the YouTube URL and try again.")
                st.session_state.analyzing = False
                return
            
            # Open video capture
            status_placeholder.info("📹 Opening video stream...")
            cap = cv2.VideoCapture(stream_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            if not cap.isOpened():
                st.error("❌ Failed to open video stream. The stream may be offline or restricted.")
                st.session_state.analyzing = False
                return
            
            status_placeholder.success("✅ Stream connected successfully!")
            
            # Initialize tracking variables
            frame_count = 0
            processed_count = 0
            peak_count = 0
            seen_ids = set()
            last_analysis_time = time.time()
            session_start_time = datetime.now()
            
            # Main analysis loop
            while st.session_state.analyzing:
                ret, frame = cap.read()
                
                if not ret:
                    st.warning("⚠️ Stream ended or connection lost")
                    break
                
                frame_count += 1
                current_time = time.time()
                
                # Skip frames based on frame_skip setting
                if frame_count % config['performance']['frame_skip'] != 0:
                    continue
                
                # Process frame at specified interval
                if current_time - last_analysis_time >= config['performance']['analysis_interval']:
                    processed_count += 1
                    timestamp = datetime.now()
                    
                    # Initialize heatmap processor with first frame
                    if config['heatmap']['enabled'] and heatmap_processor is None:
                        heatmap_processor = HeatmapProcessor(frame.shape[:2])
                    
                    # Analyze frame
                    detections, detailed_detections, person_centroids, tracked_objects = frame_analyzer.analyze_frame(
                        frame, 
                        config['detection']['filters'], 
                        config['performance']['resize_factor'],
                        tracker_type
                    )
                    
                    # Update heatmap
                    if heatmap_processor:
                        heatmap_processor.update(
                            detailed_detections, 
                            config['heatmap']['object_type'],
                            config['heatmap']['decay_factor']
                        )
                    
                    # Update tracking
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
                    else:
                        # YOLO tracking
                        if config['tracker']['track_people_only']:
                            person_tracks = {tid: pos for tid, pos in tracked_objects.items()}
                            current_people = len(person_tracks)
                            seen_ids.update(person_tracks.keys())
                        else:
                            current_people = len([det for det in detailed_detections if det['object_type'] == 'person'])
                            seen_ids.update(tracked_objects.keys())
                    
                    # Calculate tracking quality
                    tracking_quality = calculate_tracking_quality(tracker, detailed_detections)
                    st.session_state.tracking_quality = tracking_quality
                    
                    # Draw visualization
                    heatmap_data = heatmap_processor.get_heatmap() if heatmap_processor else None
                    annotated_frame = draw_enhanced_visualization(
                        frame, 
                        tracker, 
                        detailed_detections, 
                        tracked_objects,
                        config['heatmap']['enabled'],
                        heatmap_data,
                        tracker_type
                    )
                    
                    # Display frame
                    if annotated_frame.shape[1] > 800:
                        display_height = int(annotated_frame.shape[0] * (800 / annotated_frame.shape[1]))
                        display_frame = cv2.resize(annotated_frame, (800, display_height))
                    else:
                        display_frame = annotated_frame
                    
                    annotated_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                    annotated_pil = Image.fromarray(annotated_rgb)
                    frame_placeholder.image(annotated_pil, width='stretch')
                    
                    # Update metrics
                    st.session_state.unique_people_count = len(seen_ids)
                    if current_people > peak_count:
                        peak_count = current_people
                    
                    # Display tracking quality
                    if tracking_quality > 0.8:
                        quality_color = "🟢"
                        quality_text = "Excellent"
                    elif tracking_quality > 0.6:
                        quality_color = "🟡"
                        quality_text = "Good"
                    else:
                        quality_color = "🔴"
                        quality_text = "Poor"
                    
                    quality_placeholder.markdown(
                        f"**Tracking Quality:** {quality_color} {quality_text} ({tracking_quality:.1%})"
                    )
                    
                    # Update session data
                    tracking_entry = {
                        'timestamp': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                        'current_visible': current_people,
                        'unique_session': len(seen_ids),
                        'peak_count': peak_count,
                        'tracking_quality': tracking_quality,
                        'processing_fps': 1.0 / max(current_time - last_analysis_time, 0.01),
                        'tracker_algorithm': tracker_type,
                        'total_detections': len(detailed_detections)
                    }
                    
                    st.session_state.tracking_history.insert(0, tracking_entry)
                    if len(st.session_state.tracking_history) > 500:
                        st.session_state.tracking_history = st.session_state.tracking_history[:500]
                    
                    # Update detection dataframe
                    if detailed_detections:
                        for det in detailed_detections:
                            det['timestamp'] = timestamp.strftime("%Y-%m-%d %H:%M:%S")
                            det['frame_number'] = frame_count
                            det['tracking_algorithm'] = tracker_type
                        
                        new_df = pd.DataFrame(detailed_detections)
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
                
                # Small delay to prevent overwhelming
                time.sleep(0.02)
            
            # Cleanup
            cap.release()
            status_placeholder.success("✅ Analysis completed!")
            
        except Exception as e:
            st.error(f"❌ Analysis error: {str(e)}")
            st.session_state.analyzing = False
            if 'cap' in locals():
                cap.release()
    
    def run(self):
        """Main application run method"""
        # Render UI components
        self.render_header()
        config = self.render_sidebar()
        youtube_url = self.render_stream_selection()
        
        # Visualization area
        frame_placeholder, quality_placeholder, time_series_placeholder = self.render_visualization_area()
        
        # Metrics
        self.render_metrics()
        
        # Render data tabs (simplified for now)
        st.markdown("---")
        tab1, tab2 = st.tabs(["📋 Tracking History", "📊 Detection Data"])
        
        with tab1:
            if st.session_state.tracking_history:
                df = pd.DataFrame(st.session_state.tracking_history)
                st.dataframe(df.head(20), use_container_width=True)
            else:
                st.info("No tracking data yet")
        
        with tab2:
            if not st.session_state.all_detections_df.empty:
                st.dataframe(st.session_state.all_detections_df.head(20), use_container_width=True)
            else:
                st.info("No detection data yet")
        
        # Run analysis if active
        if st.session_state.analyzing and youtube_url:
            self.run_analysis(youtube_url, config, frame_placeholder, quality_placeholder, time_series_placeholder)


# Entry point
def main():
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()