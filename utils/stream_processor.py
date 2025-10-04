"""
Stream processing utilities - Framework independent
"""
import yt_dlp
from typing import Optional, Dict, Any


class StreamExtractor:
    """Extract stream URLs from YouTube and other sources"""
    
    @staticmethod
    def get_youtube_stream_url(youtube_url: str) -> Optional[str]:
        """Extract stream URL from YouTube with fallback strategies"""
        ydl_opts = {
            'format': 'best[height<=720]/best[height<=480]/best',
            'quiet': True,
            'no_warnings': True,
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'referer': 'https://www.youtube.com/',
            'socket_timeout': 30,
            'youtube_include_dash_manifest': False,
            'force_generic_extractor': False,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(youtube_url, download=False)
                
                # Strategy 1: Direct URL
                if 'url' in info and info['url']:
                    return info['url']
                
                # Strategy 2: Search through formats
                if 'formats' in info and info['formats']:
                    # Look for compatible formats
                    for fmt in info['formats']:
                        if (fmt.get('url') and 
                            fmt.get('vcodec') != 'none' and 
                            fmt.get('protocol') in ['https', 'http'] and 
                            fmt.get('height', 0) <= 720):
                            return fmt['url']
                    
                    # Fallback: any video format
                    for fmt in info['formats']:
                        if fmt.get('url') and fmt.get('vcodec') != 'none':
                            return fmt['url']
                
                return None
                
        except Exception as e:
            print(f"Stream extraction error: {e}")
            return None
    
    @staticmethod
    def extract_video_id(youtube_url: str) -> Optional[str]:
        """Extract video ID from YouTube URL"""
        try:
            if 'watch?v=' in youtube_url:
                return youtube_url.split('watch?v=')[-1].split('&')[0]
            elif 'youtu.be/' in youtube_url:
                return youtube_url.split('/')[-1]
            return None
        except Exception:
            return None
    
    @staticmethod
    def validate_stream_url(url: str) -> bool:
        """Validate if URL appears to be a valid stream"""
        if not url:
            return False
        return any(url.startswith(protocol) for protocol in ['http://', 'https://'])


class StreamConnectionManager:
    """Manage video stream connections"""
    
    def __init__(self, max_retries: int = 3, timeout: int = 30):
        self.max_retries = max_retries
        self.timeout = timeout
    
    def create_capture_config(self) -> Dict[str, Any]:
        """Create OpenCV VideoCapture configuration"""
        return {
            'buffer_size': 1,
            'fps': 15,
            'timeout': self.timeout
        }
    
    def get_connection_status(self, cap) -> Dict[str, Any]:
        """Get connection status information"""
        if not cap or not cap.isOpened():
            return {'connected': False, 'error': 'Capture not opened'}
        
        try:
            ret, frame = cap.read()
            if ret and frame is not None:
                return {
                    'connected': True, 
                    'frame_shape': frame.shape,
                    'fps': cap.get(5) if hasattr(cap, 'get') else 'unknown'
                }
            else:
                return {'connected': False, 'error': 'Cannot read frame'}
        except Exception as e:
            return {'connected': False, 'error': str(e)}