# Copilot Instructions: Video YOLO Traffic Analyzer

## Architecture Overview

This is a **modular computer vision application** that analyzes live YouTube streams for object detection and tracking. The project is **in transition** from a monolithic to a clean 3-layer architecture:

```
core/           # Pure business logic (framework-agnostic)
├── tracking.py       # Centroid-based object tracking algorithms  
├── video_processor.py # Main orchestration engine
└── detection.py      # (Coming) YOLO analysis & filtering

utils/          # Framework-independent helpers
└── config.py         # (Coming) Configuration management

ui/             # Framework-specific interfaces
└── single_page_app.py # Current Streamlit monolith (to be split)
```

## Key Architectural Patterns

### 1. Framework-Agnostic Core
- **Core modules return data structures only** (dicts, DataFrames, numpy arrays)
- **Never import Streamlit/Flask/FastAPI** in core/ modules
- Use dependency injection for models: `VideoProcessor(model, config)`
- Example pattern in `core/tracking.py`:
```python
def update(self, detections, timestamp=None):
    """Returns: Dict of tracked objects {id: (x, y)}"""
```

### 2. Tracking System Design
- **Three tracking algorithms**: OpenCV centroid, ByteTrack (YOLO), BoT-SORT (YOLO)
- **Temporal smoothing**: Exponential smoothing + velocity prediction for stability
- **Detection filtering**: Multi-step validation (confidence, area, aspect ratio, NMS)
- Key classes: `SimpleTracker`, `TemporalTracker`, `EnhancedDetectionFilter`

### 3. YOLO Model Management
- **Multiple YOLO versions**: v8 (stable), v10 (faster), v11 (newest)
- **Cached model loading**: Use `@st.cache_resource` for Streamlit
- **Model files**: `yolo*.pt` files in root (committed for deployment)
- Performance tiers: `yolov8n.pt` (fast) → `yolo11m.pt` (accurate)

## Critical Development Workflows

### Running the Application
```bash
# Current entry point (monolithic)
streamlit run single_page_app.py

# Target modular entry point  
streamlit run streamlit_app.py  # (empty - to be implemented)
```

### Deployment Configuration
- **Heroku-ready**: See `Heroku-deployment-checklist.md` for complete setup
- **Environment variables**: `OPENCV_IO_ENABLE_JASPER=1`, `QT_QPA_PLATFORM=offscreen`
- **System dependencies**: `Aptfile` contains OpenCV requirements
- **Entry point**: `Procfile` expects `app.py` (needs alignment)

### Development Environment
```bash
pip install -r requirements.txt
# Key dependencies: opencv-python-headless, ultralytics, streamlit, yt-dlp
```

## Project-Specific Conventions

### 1. Configuration Patterns
- **Preset configurations**: `PRESET_CONFIGS` dict with 'optimal', 'speed', 'accuracy' modes
- **Detection filters**: Boolean dict for object types `{'person': True, 'car': False}`
- **Session state**: Streamlit state management for analysis history and metrics

### 2. Stream Processing Pipeline
1. **YouTube URL → yt-dlp extraction** (handles geo-restrictions, format selection)  
2. **Frame analysis**: Resize → YOLO detection → Enhanced filtering → Tracking
3. **Visualization**: Annotated frames with bounding boxes, tracking trails, heatmaps
4. **Data collection**: Time series, detection logs, tracking history

### 3. Error Handling Patterns
- **Graceful stream failures**: Retry logic for YouTube connection issues
- **Missing dependency installation**: Auto-install OpenCV with subprocess
- **Frame processing errors**: Continue analysis with error counting

## Integration Points

### External Dependencies
- **yt-dlp**: YouTube stream extraction (version-sensitive)
- **YOLO models**: Ultralytics library for detection/tracking
- **OpenCV**: Video processing (headless version for deployment)
- **Streamlit**: Current UI framework (plan: multi-framework support)

### Data Flow Boundaries
- **Input**: YouTube URLs → Stream URLs → Video frames
- **Processing**: Frames → Detections → Tracked objects → Analytics
- **Output**: Annotated frames, CSV exports, real-time metrics

### Performance Considerations
- **Frame skipping**: Process every Nth frame based on `frame_skip` setting
- **Image resizing**: Scale down before YOLO inference (`resize_factor`)
- **Model selection**: Balance accuracy vs speed (v8n for speed, v11m for accuracy)

## Refactoring Guidelines

When working on the modular migration:
1. **Extract logic to core/**: Move tracking/detection classes from `single_page_app.py`
2. **Create utils/**: Stream extraction, model loading, configuration management  
3. **Keep UI thin**: Only Streamlit-specific state management and rendering
4. **Test across frameworks**: Core modules should work with Flask/FastAPI too
5. **Maintain deployment compatibility**: Update Procfile when changing entry points