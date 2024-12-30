import os
import json
import cv2
import pandas as pd
from pathlib import Path
import subprocess
import logging
import argparse
import time
from gemini_video_analyzer import GeminiVideoAnalyzer, AnalysisMode
from config_loader import ConfigLoader

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class VideoProcessor:
    def __init__(self):
        self.logger = logging.getLogger('VideoProcessor')
        self.logger.info("Initializing Video Processor")
        
        self.config = ConfigLoader()
        self.base_dir = Path(__file__).parent.parent
        
        # Setup directory structure
        self.data_dir = self.base_dir / 'data'
        self.dirs = {
            'input': self.data_dir / '00_input',
            'videos': self.data_dir / '01_videos',
            'frames': self.data_dir / '02_frames',
            'captions': self.data_dir / '03_captions',
            'individual_analysis': self.data_dir / '04_individual_analysis',
            'scores': self.data_dir / '05_scores',
            'comparative': self.data_dir / '06_comparative_analysis'
        }
        
        # Create all directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        # Initialize Gemini analyzer
        self.analyzer = GeminiVideoAnalyzer(self.config.settings['gemini_api_key'])
        self.criteria_file = self.base_dir / 'config' / 'scoring_criteria.json'

    def get_video_id(self, url):
        """Extract video ID from Google Drive URL."""
        if 'drive.google.com' not in url:
            self.logger.error(f"Invalid URL format: {url}")
            return None
        file_id = url.split('/d/')[1].split('/')[0]
        self.logger.debug(f"Extracted video ID: {file_id}")
        return file_id

    def download_video(self, url, filename):
        """Download video from Google Drive."""
        # Extract file ID from Google Drive URL
        file_id = url.split('/')[-2]
        video_path = self.dirs['videos'] / f"{filename}.mp4"
        
        if video_path.exists():
            self.logger.info(f"Video already exists: {filename}")
            return str(video_path)
        
        self.logger.info(f"Downloading video: {filename}")
        gdown_path = str(Path(__file__).parent.parent / "venv" / "bin" / "gdown")
        subprocess.run([gdown_path, "--output", str(video_path), f"https://drive.google.com/uc?id={file_id}"], check=True)
        self.logger.info(f"Download complete: {filename}")
        
        return str(video_path)

    def extract_frames(self, video_path, video_name):
        """Extract frames from video."""
        self.logger.info(f"Extracting frames from video: {video_name}")
        
        # Open video file
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        self.logger.debug(f"Video FPS: {fps}")
        
        # Create directory for frames
        frames_dir = self.dirs['frames'] / video_name
        frames_dir.mkdir(parents=True, exist_ok=True)
        
        frame_count = 0
        saved_count = 0
        interval = 4  # Extract a frame every 4 seconds
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Extract frame every 'interval' seconds
            if frame_count % (fps * interval) == 0:
                frame_path = frames_dir / f"frame_{saved_count:03d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                saved_count += 1
                if saved_count % 5 == 0:  # Log progress every 5 frames
                    self.logger.info(f"Saved {saved_count} frames...")
                    
            frame_count += 1
            
        cap.release()
        self.logger.info(f"Frame extraction complete. Saved {saved_count} frames")
        return saved_count

    def extract_audio_captions(self, video_path, video_name):
        """Extract audio captions using external tool (e.g., Whisper)."""
        self.logger.info(f"Generating captions for video: {video_name}")
        caption_file = self.dirs['captions'] / f"{video_name}.txt"
        if not caption_file.exists():
            self.logger.warning("Caption extraction not implemented. Creating empty caption file.")
            with open(caption_file, 'w') as f:
                f.write("Captions not available")
        return str(caption_file)

    def process_videos_from_csv(self, csv_file, mode='all'):
        """Process videos listed in a CSV file.
        
        Args:
            csv_file: Path to CSV file containing video information
            mode: One of ['all', 'download', 'frames', 'analysis', 'compare']
                 - 'all': Run complete pipeline
                 - 'download': Only download videos
                 - 'frames': Download and extract frames
                 - 'analysis': Run individual analysis (assumes frames exist)
                 - 'compare': Only run comparative analysis (assumes individual analyses exist)
        """
        self.logger.info(f"Processing videos from CSV: {csv_file} in {mode} mode")
        df = pd.read_csv(csv_file)
        total_videos = len(df)
        self.logger.info(f"Found {total_videos} videos to process")
        
        if mode in ['all', 'download', 'frames', 'analysis']:
            batch_size = 5
            for i in range(0, total_videos, batch_size):
                batch_num = i // batch_size + 1
                end_idx = min(i + batch_size, total_videos)
                self.logger.info(f"\nProcessing batch {batch_num} ({i+1}-{end_idx} of {total_videos})")
                
                batch_df = df.iloc[i:end_idx]
                for _, row in batch_df.iterrows():
                    video_name = row['Video Name']
                    video_url = row['Video URL']
                    
                    self.logger.info(f"\nProcessing video {i+1}/{total_videos}: {video_name}")
                    try:
                        if mode in ['all', 'download']:
                            video_path = self.download_video(video_url, video_name)
                            
                        if mode in ['all', 'frames']:
                            video_path = self.dirs['videos'] / f"{video_name}.mp4"
                            if video_path.exists():
                                self.extract_frames(str(video_path), video_name)
                            
                        if mode in ['all', 'analysis']:
                            frames_dir = self.dirs['frames'] / video_name
                            if frames_dir.exists():
                                caption_file = self.extract_audio_captions(video_path, video_name)
                                self.analyzer.create_video_analysis(
                                    video_name,
                                    frames_dir,
                                    Path(caption_file),
                                    AnalysisMode.ROAD_SAFETY
                                )
                                self.analyzer.save_analysis_markdown(self.analyzer.analysis)
                                scores = self.analyzer.generate_score(self.analyzer.analysis, self.criteria_file)
                                self.analyzer.save_scores(scores)
                                
                    except Exception as e:
                        self.logger.error(f"Error processing video {video_name}: {str(e)}")
        
        if mode in ['all', 'compare']:
            # Run comparative analysis
            self.logger.info("\nGenerating final comparative analysis...")
            try:
                time.sleep(60)  # Wait 60 seconds
                self.analyzer.generate_comparative_analysis()
                self.logger.info("Comparative analysis complete!")
            except Exception as e:
                self.logger.error(f"Error generating comparative analysis: {str(e)}")
        
        self.logger.info("\nProcessing complete!")

    def clean_data(self, mode='all'):
        """Clean up data directories.
        
        Args:
            mode: One of ['all', 'frames', 'analysis', 'compare']
                 - 'all': Delete everything except input
                 - 'frames': Delete frames and everything after
                 - 'analysis': Delete analysis and everything after
                 - 'compare': Delete only comparative analysis
        """
        def delete_dir_contents(dir_path):
            if dir_path.exists():
                for item in dir_path.iterdir():
                    if item.is_file():
                        item.unlink()
                    elif item.is_dir():
                        import shutil
                        shutil.rmtree(item)
                        
        # Define cleanup levels
        cleanup_levels = {
            'compare': ['comparative'],
            'analysis': ['individual_analysis', 'scores', 'comparative'],
            'frames': ['frames', 'captions', 'individual_analysis', 'scores', 'comparative'],
            'all': ['videos', 'frames', 'captions', 'individual_analysis', 'scores', 'comparative']
        }
        
        dirs_to_clean = cleanup_levels.get(mode, [])
        if not dirs_to_clean:
            self.logger.warning(f"Invalid cleanup mode: {mode}")
            return
            
        for dir_key in dirs_to_clean:
            self.logger.info(f"Cleaning {dir_key} directory...")
            delete_dir_contents(self.dirs[dir_key])
            
        self.logger.info("Cleanup complete!")

def main():
    parser = argparse.ArgumentParser(description='Process road safety videos')
    parser.add_argument('csv_file', help='Path to CSV file containing video information')
    parser.add_argument('--mode', choices=['all', 'download', 'frames', 'analysis', 'compare'],
                      default='all', help='Processing mode')
    parser.add_argument('--clean', choices=['all', 'frames', 'analysis', 'compare'],
                      default=None, help='Cleanup mode')
    args = parser.parse_args()
    
    processor = VideoProcessor()
    if args.clean:
        processor.clean_data(args.clean)
    if args.mode == 'compare':
        # Only run comparative analysis
        processor.analyzer.generate_comparative_analysis()
    else:
        processor.process_videos_from_csv(args.csv_file, args.mode)

if __name__ == "__main__":
    main()
