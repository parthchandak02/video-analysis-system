import json
from pathlib import Path

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
        """Get Gemini API key"""
        return self.settings.get('gemini_api_key')

    @property
    def frame_interval(self):
        """Get frame extraction interval"""
        return self.settings.get('frame_extraction_interval', 4)

    @property
    def video_extensions(self):
        """Get supported video extensions"""
        return self.settings.get('video_extensions', ['.mp4', '.mov', '.avi'])

    @property
    def analysis_mode(self):
        """Get analysis mode"""
        return self.settings.get('analysis_mode', 'road_safety')
