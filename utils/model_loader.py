"""
Model loading utilities - Framework independent
"""
from typing import Any, Optional
import os


class ModelLoader:
    """Load and manage YOLO models"""
    
    def __init__(self):
        self._loaded_models = {}
    
    def load_yolo_model(self, model_path: str) -> Any:
        """Load YOLO model with caching"""
        if model_path in self._loaded_models:
            return self._loaded_models[model_path]
        
        try:
            # Import here to avoid circular dependencies
            from ultralytics import YOLO
            
            if not os.path.exists(model_path):
                print(f"Model file {model_path} not found, downloading...")
            
            model = YOLO(model_path)
            self._loaded_models[model_path] = model
            return model
            
        except Exception as e:
            print(f"Failed to load {model_path}: {e}")
            # Fallback to smallest model
            try:
                from ultralytics import YOLO
                fallback_model = YOLO('yolov8n.pt')
                self._loaded_models[model_path] = fallback_model
                return fallback_model
            except Exception as e2:
                print(f"Fallback model loading failed: {e2}")
                raise e2
    
    def get_model_info(self, model_path: str) -> dict:
        """Get information about a model"""
        model_info = {
            'path': model_path,
            'exists': os.path.exists(model_path),
            'size_mb': 0,
            'loaded': model_path in self._loaded_models
        }
        
        if model_info['exists']:
            try:
                model_info['size_mb'] = os.path.getsize(model_path) / (1024 * 1024)
            except Exception:
                pass
        
        return model_info
    
    def clear_cache(self) -> None:
        """Clear model cache"""
        self._loaded_models.clear()


# Global model loader instance
model_loader = ModelLoader()