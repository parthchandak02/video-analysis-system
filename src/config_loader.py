import json
import os
from pathlib import Path

DEFAULT_MODEL = 'gemini-3.8-flash'


class ConfigLoader:
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.config_dir = self.base_dir / 'config'
        self.config_file = self.config_dir / 'config.json'
        self.template_file = self.config_dir / 'config.example.json'
        self.settings = self.load_settings()

    def load_settings(self):
        """Load settings from config file."""
        try:
            with open(self.config_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            # If config.json doesn't exist, try to copy from example
            if self.template_file.exists():
                with open(self.template_file, 'r') as f:
                    settings = json.load(f)
                    
                # Save template as config.json
                with open(self.config_file, 'w') as f:
                    json.dump(settings, f, indent=4)
                return settings
            else:
                raise FileNotFoundError("Neither config.json nor config.example.json found")

    @property
    def gemini_api_key(self):
        """Gemini API key: GEMINI_API_KEY env var, else config (the example placeholder counts as unset)."""
        key = os.getenv('GEMINI_API_KEY') or self.settings.get('gemini_api_key')
        if not key or key == 'your_api_key_here':
            return None
        return key

    @property
    def model_version(self):
        """Gemini model id"""
        return self.settings.get('model_version') or DEFAULT_MODEL

    @property
    def frame_interval(self):
        """Seconds between extracted frames (frame_extraction.interval_seconds)"""
        return self.settings.get('frame_extraction', {}).get('interval_seconds') or 4

    @property
    def max_frames(self):
        """Maximum frames to extract per video, or None for no limit (frame_extraction.max_frames)"""
        return self.settings.get('frame_extraction', {}).get('max_frames')

    @property
    def video_extensions(self):
        """Get supported video extensions"""
        return self.settings.get('video_extensions', ['.mp4', '.mov', '.avi'])

    @property
    def analysis_mode(self):
        """Get analysis mode"""
        return self.settings.get('analysis_mode', 'road_safety')
