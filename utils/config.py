"""
Configuration management - Framework independent
Pure configuration without UI dependencies
"""
from typing import Dict, Any
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """YOLO model configuration"""
    name: str
    file_path: str
    description: str
    size_mb: int
    estimated_fps: int
    accuracy_level: str


@dataclass
class TrackerConfig:
    """Tracker configuration"""
    max_disappeared: int = 30
    max_distance: int = 100
    smoothing_factor: float = 0.7
    temporal_smoothing: bool = True


@dataclass
class DetectionConfig:
    """Detection configuration"""
    confidence_threshold: float = 0.5
    nms_threshold: float = 0.4
    min_area: int = 100
    enhanced_filtering: bool = True


@dataclass
class PerformanceConfig:
    """Performance configuration"""
    frame_skip: int = 1
    resize_factor: float = 0.75
    analysis_interval: int = 1


@dataclass
class HeatmapConfig:
    """Heatmap configuration"""
    enabled: bool = False
    object_type: str = "person"
    decay_factor: float = 0.95
    alpha: float = 0.6


class ConfigManager:
    """Centralized configuration management"""
    
    # Available YOLO models
    YOLO_MODELS = {
        0: ModelConfig("yolov8n.pt", "yolov8n.pt", "YOLOv8 Nano - Fastest", 6, 45, "Good"),
        1: ModelConfig("yolov8s.pt", "yolov8s.pt", "YOLOv8 Small - Balanced", 22, 35, "Very Good"),
        2: ModelConfig("yolov8m.pt", "yolov8m.pt", "YOLOv8 Medium - Accurate", 50, 25, "Excellent"),
        3: ModelConfig("yolov10n.pt", "yolov10n.pt", "YOLOv10 Nano - Ultra Fast", 5, 60, "Good"),
        4: ModelConfig("yolov10s.pt", "yolov10s.pt", "YOLOv10 Small - Fast", 16, 45, "Very Good"),
        5: ModelConfig("yolov10m.pt", "yolov10m.pt", "YOLOv10 Medium - Fast+Accurate", 32, 35, "Excellent"),
        6: ModelConfig("yolo11n.pt", "yolo11n.pt", "YOLOv11 Nano - Latest", 5, 50, "Very Good"),
        7: ModelConfig("yolo11s.pt", "yolo11s.pt", "YOLOv11 Small - Best Overall", 20, 40, "Excellent"),
        8: ModelConfig("yolo11m.pt", "yolo11m.pt", "YOLOv11 Medium - Most Accurate", 45, 30, "Outstanding"),
    }
    
    # Available tracking algorithms
    TRACKER_TYPES = ["OpenCV", "ByteTrack (YOLO)", "BoT-SORT (YOLO)"]
    
    # Preset configurations
    PRESET_CONFIGS = {
        'optimal': {
            'performance': PerformanceConfig(frame_skip=1, resize_factor=0.75, analysis_interval=1),
            'model_index': 7,  # yolo11s.pt
            'tracker_index': 1,  # ByteTrack
            'tracker': TrackerConfig(max_disappeared=45, max_distance=150, smoothing_factor=0.7, temporal_smoothing=False),
            'detection': DetectionConfig(confidence_threshold=0.6, nms_threshold=0.3, min_area=150, enhanced_filtering=True),
            'heatmap': HeatmapConfig(enabled=False, object_type="person", decay_factor=0.95),
            'track_people_only': True,
            'detection_filters': {
                'person': True, 'bicycle': True, 'car': True, 'motorcycle': True, 
                'bus': True, 'truck': True, 'boat': False
            }
        },
        'speed': {
            'performance': PerformanceConfig(frame_skip=2, resize_factor=0.25, analysis_interval=3),
            'model_index': 3,  # yolov10n.pt
            'tracker_index': 1,  # ByteTrack
            'tracker': TrackerConfig(max_disappeared=15, max_distance=80, smoothing_factor=0.5, temporal_smoothing=False),
            'detection': DetectionConfig(confidence_threshold=0.7, nms_threshold=0.5, min_area=200, enhanced_filtering=False),
            'heatmap': HeatmapConfig(enabled=False, object_type="person", decay_factor=0.95),
            'track_people_only': True,
            'detection_filters': {
                'person': True, 'bicycle': False, 'car': True, 'motorcycle': False, 
                'bus': False, 'truck': False, 'boat': False
            }
        },
        'accuracy': {
            'performance': PerformanceConfig(frame_skip=1, resize_factor=1.0, analysis_interval=1),
            'model_index': 8,  # yolo11m.pt
            'tracker_index': 2,  # BoT-SORT
            'tracker': TrackerConfig(max_disappeared=60, max_distance=200, smoothing_factor=0.8, temporal_smoothing=False),
            'detection': DetectionConfig(confidence_threshold=0.4, nms_threshold=0.2, min_area=50, enhanced_filtering=True),
            'heatmap': HeatmapConfig(enabled=True, object_type="person", decay_factor=0.98),
            'track_people_only': False,
            'detection_filters': {
                'person': True, 'bicycle': True, 'car': True, 'motorcycle': True, 
                'bus': True, 'truck': True, 'boat': True
            }
        }
    }
    
    # Default detection filters
    DEFAULT_DETECTION_FILTERS = {
        'person': True, 'bicycle': True, 'car': True, 'motorcycle': True,
        'bus': True, 'truck': True, 'boat': True
    }
    
    # Preset YouTube streams
    PRESET_STREAMS = {
        "Dublin, Ireland - Temple Bar": "https://www.youtube.com/watch?v=u4UZ4UvZXrg",
        "Melbourne, Australia - Intersection": "https://www.youtube.com/watch?v=fOiFXweVdrE", 
        "London, UK - Abbey Road": "https://www.youtube.com/watch?v=57w2gYXjRic",
        "Sydney, Australia - Harbour Bridge": "https://www.youtube.com/watch?v=5uZa3-RMFos"
    }
    
    @classmethod
    def get_preset_config(cls, preset_name: str) -> Dict[str, Any]:
        """Get a preset configuration by name"""
        if preset_name not in cls.PRESET_CONFIGS:
            raise ValueError(f"Unknown preset: {preset_name}")
        return cls.PRESET_CONFIGS[preset_name].copy()
    
    @classmethod
    def get_model_config(cls, model_index: int) -> ModelConfig:
        """Get model configuration by index"""
        if model_index not in cls.YOLO_MODELS:
            raise ValueError(f"Unknown model index: {model_index}")
        return cls.YOLO_MODELS[model_index]
    
    @classmethod
    def get_tracker_name(cls, tracker_index: int) -> str:
        """Get tracker name by index"""
        if tracker_index >= len(cls.TRACKER_TYPES):
            raise ValueError(f"Unknown tracker index: {tracker_index}")
        return cls.TRACKER_TYPES[tracker_index]
    
    @classmethod
    def validate_detection_filters(cls, filters: Dict[str, bool]) -> Dict[str, bool]:
        """Validate and sanitize detection filters"""
        validated = cls.DEFAULT_DETECTION_FILTERS.copy()
        for key, value in filters.items():
            if key in validated:
                validated[key] = bool(value)
        return validated


# Environment configuration for deployment
DEPLOYMENT_CONFIG = {
    'opencv_io_enable_jasper': '1',
    'qt_qpa_platform': 'offscreen',
    'streamlit_server_port': '$PORT',
    'streamlit_server_address': '0.0.0.0'
}

# Quality thresholds
QUALITY_THRESHOLDS = {
    'excellent': 0.8,
    'good': 0.6,
    'poor': 0.0
}

# Performance targets
PERFORMANCE_TARGETS = {
    'min_fps': 10,
    'max_memory_mb': 500,
    'max_response_time_ms': 100
}