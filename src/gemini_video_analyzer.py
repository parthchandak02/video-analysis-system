import os
import json
import time
from pathlib import Path
from typing import List, Dict, Optional
from enum import Enum
import google.generativeai as genai
from PIL import Image
import cv2
import numpy as np
from datetime import datetime
import logging
import pandas as pd
from config_loader import ConfigLoader

# Setup basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class AnalysisMode(Enum):
    ROAD_SAFETY = "road_safety"
    GENERAL = "general"

class GeminiVideoAnalyzer:
    def __init__(self, api_key: str):
        """Initialize the analyzer with Gemini API key."""
        self.logger = logging.getLogger('GeminiAnalyzer')
        self.logger.info("Initializing Gemini Video Analyzer")
        
        # Load configuration
        self.config = ConfigLoader()
        
        # Initialize Gemini
        genai.configure(api_key=api_key)
        
        # Use model version from config
        model_version = self.config.settings.get('model_version', 'gemini-2.0-flash-exp')
        self.logger.info(f"Using Gemini model version: {model_version}")
        self.model = genai.GenerativeModel(model_version)  # For both image and text analysis
        self.text_model = self.model  # Use same model for consistency
        
        self.base_dir = Path(__file__).parent.parent
        
        # Use new directory structure
        self.data_dir = self.base_dir / 'data'
        self.dirs = {
            'individual_analysis': self.data_dir / '04_individual_analysis',
            'scores': self.data_dir / '05_scores',
            'comparative': self.data_dir / '06_comparative_analysis'
        }
        
        # Create directories
        for dir_path in self.dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
            
        self.api_key = api_key
        
        # Rate limiting settings from config
        rate_limits = self.config.settings.get('rate_limits', {})
        self.requests_per_minute = rate_limits.get('requests_per_minute', 15)
        self.max_tokens = rate_limits.get('max_tokens_per_request', 2048)
        self.last_request_time = 0
        self.request_count = 0
        self.reset_time = time.time()
        
    def _wait_for_rate_limit(self):
        """Implement rate limiting to stay within free tier limits."""
        current_time = time.time()
        
        # Reset counter if a minute has passed
        if current_time - self.reset_time >= 60:
            self.logger.debug("Resetting rate limit counter")
            self.request_count = 0
            self.reset_time = current_time
        
        # If we're at the limit, wait until next minute
        if self.request_count >= self.requests_per_minute:
            wait_time = 60 - (current_time - self.reset_time)
            if wait_time > 0:
                self.logger.info(f"Rate limit reached. Waiting {wait_time:.1f} seconds...")
                time.sleep(wait_time)
                self.request_count = 0
                self.reset_time = time.time()
        
        # Ensure minimum time between requests
        time_since_last = current_time - self.last_request_time
        if time_since_last < 1:  # Minimum 1 second between requests
            time.sleep(1 - time_since_last)
        
        self.last_request_time = time.time()
        self.request_count += 1

    def analyze_frame(self, image_path: str, mode: AnalysisMode) -> str:
        """Analyze a single frame using Gemini Vision."""
        max_retries = 3  # Maximum number of retries
        retry_delay = 60  # Delay in seconds before retrying
        
        for attempt in range(max_retries):
            try:
                self.logger.debug(f"Opening image: {image_path}")
                image = Image.open(image_path)
                
                if mode == AnalysisMode.ROAD_SAFETY:
                    prompt = """
                    Analyze this frame from a road safety video. Focus on:
                    1. Road safety elements and practices
                    2. Potential hazards or warnings
                    3. Educational elements
                    4. Visual messaging effectiveness
                    Be specific but concise.
                    """
                else:
                    prompt = "Describe what's happening in this image, focusing on key elements and actions."
                
                self._wait_for_rate_limit()
                self.logger.debug(f"Sending request to Gemini for frame analysis with model version: {self.model.model_name}")
                try:
                    response = self.model.generate_content([prompt, image])
                    self.logger.debug(f"Received response from Gemini: {response}")
                    
                    if response.text:
                        return response.text
                    else:
                        raise ValueError("Empty response from Gemini")
                except Exception as api_error:
                    if "429" in str(api_error):  # Rate limit error
                        if attempt < max_retries - 1:  # Don't sleep on the last attempt
                            self.logger.warning(f"Rate limit hit. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}")
                            time.sleep(retry_delay)
                            continue
                    self.logger.error(f"Gemini API error: {str(api_error)}")
                    # Fallback to a simpler prompt
                    self.logger.info("Retrying with simpler prompt...")
                    response = self.model.generate_content(["Describe this image briefly.", image])
                    if response.text:
                        return response.text
                    else:
                        raise ValueError("Empty response from Gemini even with simple prompt")
                
            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:  # Rate limit error
                    self.logger.warning(f"Rate limit hit. Waiting {retry_delay} seconds before retry {attempt + 1}/{max_retries}")
                    time.sleep(retry_delay)
                    continue
                self.logger.error(f"Error analyzing frame {image_path}: {str(e)}")
                self.logger.error(f"Full error details: {repr(e)}")
                raise
                
        raise Exception(f"Failed to analyze frame after {max_retries} attempts")

    def analyze_text(self, text: str, mode: AnalysisMode) -> str:
        """Analyze text content using Gemini."""
        self.logger.info("Analyzing text content")
        try:
            if mode == AnalysisMode.ROAD_SAFETY:
                prompt = f"""
                Analyze this text from a road safety video:
                {text}
                
                Focus on:
                1. Safety messages and instructions
                2. Clarity of communication
                3. Educational value
                4. Call to action
                
                Provide a concise summary.
                """
            else:
                prompt = f"Summarize the key points from this text: {text}"
            
            self._wait_for_rate_limit()
            self.logger.debug("Sending request to Gemini for text analysis")
            response = self.text_model.generate_content(prompt)
            self.logger.debug("Received response from Gemini")
            
            return response.text
            
        except Exception as e:
            self.logger.error(f"Error analyzing text: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def create_video_analysis(self, video_name: str, frames_dir: Path, caption_file: Optional[Path], mode: AnalysisMode) -> Dict:
        """Create a comprehensive analysis of the video."""
        try:
            self.logger.info(f"Starting comprehensive analysis for video: {video_name}")
            self.analysis = {
                "video_name": video_name,
                "date_analyzed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "frame_analyses": [],
                "video_description": None,
                "overall_analysis": None
            }
            
            # Get all frames
            frames = sorted(list(frames_dir.glob("*.jpg")))
            self.logger.info(f"Found {len(frames)} frames")
            
            # Read captions if available
            self.logger.info("Reading captions")
            captions = ""
            if caption_file and caption_file.exists():
                with open(caption_file) as f:
                    captions = f.read()
            
            # Analyze each frame
            for i, frame in enumerate(frames, 1):
                self.logger.info(f"Analyzing frame {i}/{len(frames)}")
                frame_analysis = self.analyze_frame(str(frame), mode)
                self.analysis["frame_analyses"].append({
                    "frame_path": str(frame),
                    "analysis": frame_analysis
                })
            
            # Generate overall video description
            self.logger.info("Generating video description")
            description_prompt = f"""
            Based on these frame analyses from a road safety video:
            {[fa['analysis'] for fa in self.analysis['frame_analyses']]}
            
            Provide a concise but comprehensive description of what happens in the video.
            Focus on the key safety messages and visual elements.
            """
            
            self._wait_for_rate_limit()
            self.logger.debug("Sending request to Gemini for video description")
            description_response = self.text_model.generate_content(description_prompt)
            if not description_response.text:
                raise ValueError("Empty response from Gemini for video description")
            self.analysis["video_description"] = description_response.text
            
            # Generate overall analysis
            self.logger.info("Generating overall analysis")
            analysis_prompt = f"""
            Based on this road safety video description:
            {self.analysis['video_description']}
            
            Audio/Captions:
            {captions if captions else "No captions available"}
            
            Provide a comprehensive analysis focusing on:
            1. Overall message and theme
            2. Target audience and relevance
            3. Safety education value
            4. Production quality and creativity
            5. Areas of excellence
            6. Areas for improvement
            
            Structure your analysis to help with scoring the video later.
            """
            
            self._wait_for_rate_limit()
            self.logger.debug("Sending request to Gemini for overall analysis")
            analysis_response = self.text_model.generate_content(analysis_prompt)
            if not analysis_response.text:
                raise ValueError("Empty response from Gemini for overall analysis")
            self.analysis["overall_analysis"] = analysis_response.text
            
            self.logger.info("Analysis completed successfully")
            return self.analysis
            
        except Exception as e:
            self.logger.error(f"Error in video analysis: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def save_analysis_markdown(self, analysis: Dict) -> Path:
        """Save the analysis in a markdown file."""
        try:
            markdown_file = self.dirs['individual_analysis'] / f"{analysis['video_name']}_analysis.md"
            
            with open(markdown_file, 'w') as f:
                f.write(f"# Video Analysis: {analysis['video_name']}\n\n")
                f.write(f"Analysis Date: {analysis['date_analyzed']}\n\n")
                
                f.write("## Video Description\n\n")
                f.write(f"{analysis['video_description']}\n\n")
                
                f.write("## Overall Analysis\n\n")
                f.write(f"{analysis['overall_analysis']}\n")
            
            self.logger.info(f"Saved analysis to: {markdown_file}")
            return markdown_file
        except Exception as e:
            self.logger.error(f"Error saving analysis: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def generate_score(self, analysis: Dict, criteria_file: Path = None) -> Dict:
        """Generate scores based on the analysis and criteria."""
        try:
            if criteria_file is None:
                criteria_file = self.base_dir / 'config' / 'scoring_criteria.json'
            
            with open(criteria_file) as f:
                criteria = json.load(f)
            
            # Create scoring prompt
            scoring_prompt = f"""
            Based on this video analysis, generate scores according to these criteria:
            {json.dumps(criteria, indent=2)}

            Video Analysis:
            {json.dumps(analysis, indent=2)}

            Generate a JSON object with scores for each criterion. The response should be a valid JSON object with this structure:
            {{
                "scores": {{
                    "<criterion_name>": {{
                        "score": <number>,
                        "justification": "<text>",
                        "feedback": "<text>"
                    }}
                }}
            }}
            
            For each criterion:
            1. Score must be a number between 0 and the max_marks
            2. Justify the score with specific examples from the video
            3. Provide constructive feedback for improvement
            
            IMPORTANT: Return ONLY the JSON object, with NO markdown formatting or code block markers.
            """
            
            self._wait_for_rate_limit()
            self.logger.debug("Sending request to Gemini for scoring")
            response = self.text_model.generate_content(scoring_prompt)
            self.logger.debug("Received response from Gemini")
            
            try:
                # Clean the response text
                cleaned_response = response.text.strip()
                if cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response.split('\n', 1)[1]
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response.rsplit('\n', 1)[0]
                cleaned_response = cleaned_response.strip()
                if cleaned_response.startswith('json'):
                    cleaned_response = cleaned_response.split('\n', 1)[1]
                cleaned_response = cleaned_response.strip()
                
                self.logger.debug(f"Cleaned response: {cleaned_response}")
                scores = json.loads(cleaned_response)
                
            except json.JSONDecodeError as e:
                self.logger.error(f"Failed to parse Gemini response as JSON: {e}")
                self.logger.error(f"Raw response: {response.text}")
                self.logger.error(f"Cleaned response: {cleaned_response}")
                raise
            
            # Calculate total score
            total_score = sum(item['score'] for item in scores['scores'].values())
            max_possible = sum(item['max_marks'] for item in criteria['criteria'])
            
            scores.update({
                "video_name": analysis['video_name'],
                "date_scored": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "total_score": total_score,
                "max_possible_score": max_possible
            })
            
            self.logger.info(f"Generated scores for {analysis['video_name']}: {total_score}/{max_possible}")
            return scores
            
        except Exception as e:
            self.logger.error(f"Error generating scores: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def save_scores(self, scores: Dict) -> Path:
        """Save the scores to a JSON file."""
        try:
            score_file = self.dirs['scores'] / f"{scores['video_name']}_score.json"
            with open(score_file, 'w') as f:
                json.dump(scores, f, indent=4)
            self.logger.info(f"Saved scores to: {score_file}")
            return score_file
        except Exception as e:
            self.logger.error(f"Error saving scores: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def create_comparative_analysis(self, video_analyses: List[Dict], scores: List[Dict]) -> Dict:
        """Create a comparative analysis of multiple videos."""
        self.logger.info("Creating comparative analysis")
        try:
            # Prepare summaries for comparison
            video_summaries = []
            for analysis in video_analyses:
                video_summaries.append({
                    "name": analysis["video_name"],
                    "description": analysis["video_description"],
                    "analysis": analysis["overall_analysis"]
                })
            
            # Create comparison prompt
            comparison_prompt = f"""
            Compare these road safety videos based on their analyses:

            {json.dumps(video_summaries, indent=2)}

            Create a comparative analysis focusing on:
            1. Strengths and weaknesses of each video
            2. Different approaches to road safety messaging
            3. Effectiveness for target audiences
            4. Production quality comparison
            5. Overall impact and educational value

            Also create a final ranking of the videos with justification.
            """

            self._wait_for_rate_limit()
            self.logger.debug("Sending request to Gemini for comparative analysis")
            comparison_response = self.text_model.generate_content(comparison_prompt)
            
            # Create score comparison table
            score_table = {
                "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "videos": [],
                "comparative_analysis": comparison_response.text
            }

            # Add each video's scores
            for score in scores:
                video_scores = {
                    "name": score["video_name"],
                    "total_score": score["total_score"],
                    "criteria_scores": score["scores"]
                }
                score_table["videos"].append(video_scores)

            return score_table

        except Exception as e:
            self.logger.error(f"Error creating comparative analysis: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def save_comparative_analysis(self, analysis: Dict, suffix: str = "") -> tuple[Path, Path]:
        """Save the comparative analysis as markdown and CSV."""
        try:
            # Save markdown analysis
            analysis_file = self.dirs['comparative'] / f"comparative_analysis_{datetime.now().strftime('%Y%m%d')}{suffix}.md"
            
            with open(analysis_file, 'w') as f:
                f.write("# Comparative Analysis of Road Safety Videos\n\n")
                f.write(f"Analysis Date: {analysis['date']}\n\n")
                
                # Write comparative analysis
                f.write("## Overall Comparison\n\n")
                f.write(f"{analysis['comparative_analysis']}\n\n")
                
                # Write score comparison table
                f.write("## Score Comparison\n\n")
                f.write("| Video Name | Focus | Creativity | Impact | Visual | Theme | Clarity | Ethics | Relevance | Other | Total |\n")
                f.write("|------------|-------|------------|--------|---------|--------|----------|---------|-----------|--------|-------|\n")
                
                for video in analysis["videos"]:
                    scores = video["criteria_scores"]
                    f.write(f"| {video['name']} | ")
                    f.write(f"{scores['Focus']['score']} | ")
                    f.write(f"{scores['Creativity and Originality']['score']} | ")
                    f.write(f"{scores['Impact']['score']} | ")
                    f.write(f"{scores['Visual Appeal']['score']} | ")
                    f.write(f"{scores['Adherence to Theme']['score']} | ")
                    f.write(f"{scores['Clarity']['score']} | ")
                    f.write(f"{scores['Ethical Standards']['score']} | ")
                    f.write(f"{scores['Relevance']['score']} | ")
                    f.write(f"{scores['Other Remarks']['score']} | ")
                    f.write(f"{video['total_score']} |\n")
                
                # Write detailed scores and feedback
                f.write("\n## Detailed Scoring Analysis\n\n")
                for video in analysis["videos"]:
                    f.write(f"### {video['name']}\n\n")
                    for criterion, details in video["criteria_scores"].items():
                        f.write(f"#### {criterion}\n")
                        f.write(f"Score: {details['score']}\n")
                        f.write(f"Justification: {details['justification']}\n")
                        f.write(f"Feedback: {details['feedback']}\n\n")
            
            # Save CSV summary with exactly the requested columns
            csv_file = self.dirs['comparative'] / f"score_comparison_{datetime.now().strftime('%Y%m%d')}{suffix}.csv"
            with open(csv_file, 'w', newline='') as f:
                # Write header exactly as requested
                headers = [
                    "Video Name",
                    "Focus",
                    "Creativity and Originality",
                    "Impact",
                    "Visual Appeal",
                    "Adherence to Theme",
                    "Clarity",
                    "Ethical Standards",
                    "Relevance",
                    "Other(Notes explaining why the score was given - out of 10)"
                ]
                f.write(','.join(headers) + '\n')
                
                # Write scores for each video
                for video in analysis["videos"]:
                    scores = video["criteria_scores"]
                    row = [
                        video['name'],
                        str(scores['Focus']['score']),
                        str(scores['Creativity and Originality']['score']),
                        str(scores['Impact']['score']),
                        str(scores['Visual Appeal']['score']),
                        str(scores['Adherence to Theme']['score']),
                        str(scores['Clarity']['score']),
                        str(scores['Ethical Standards']['score']),
                        str(scores['Relevance']['score']),
                        # For the Other column, combine score and justification
                        f"{scores['Other Remarks']['score']}/10 - {scores['Other Remarks']['justification']}"
                    ]
                    # Properly escape any commas in the text
                    row = [f'"{item}"' if ',' in item else item for item in row]
                    f.write(','.join(row) + '\n')
            
            self.logger.info(f"Saved comparative analysis to: {analysis_file}")
            self.logger.info(f"Saved score comparison to: {csv_file}")
            return analysis_file, csv_file
            
        except Exception as e:
            self.logger.error(f"Error saving comparative analysis: {str(e)}")
            self.logger.error(f"Full error details: {repr(e)}")
            raise

    def generate_comparative_analysis(self):
        """Generate comparative analysis for all processed videos."""
        self.logger.info("Starting comparative analysis")
        
        # Get all score files in original order from CSV
        csv_path = self.base_dir / 'all_videos.csv'
        video_order = pd.read_csv(csv_path)['Video Name'].tolist()
        
        # Get all score files
        scores = []
        for video_name in video_order:
            score_file = self.dirs['scores'] / f"{video_name}_score.json"
            if score_file.exists():
                with open(score_file, 'r') as f:
                    scores.append(json.load(f))
        
        if not scores:
            self.logger.warning("No score files found for comparison")
            return
            
        # Generate markdown report
        report = "# Comparative Analysis of Road Safety Videos\n\n"
        report += f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Overall rankings (but keep original order)
        report += "## Video Analysis\n\n"
        report += "| Order | Video Name | Total Score | Focus | Creativity | Impact | Visual | Theme | Clarity | Ethics | Relevance | Other |\n"
        report += "|-------|------------|-------------|--------|------------|---------|---------|--------|----------|---------|------------|--------|\n"
        
        for idx, score in enumerate(scores, 1):
            s = score['scores']
            report += f"| {idx} | {score['video_name']} | {score['total_score']:.1f} | "
            report += f"{s['Focus']['score']} | {s['Creativity and Originality']['score']} | "
            report += f"{s['Impact']['score']} | {s['Visual Appeal']['score']} | "
            report += f"{s['Adherence to Theme']['score']} | {s['Clarity']['score']} | "
            report += f"{s['Ethical Standards']['score']} | {s['Relevance']['score']} | "
            report += f"{s['Other Remarks']['score']} |\n"
            
        # Category analysis
        categories = ['Focus', 'Creativity and Originality', 'Impact', 'Visual Appeal', 
                     'Adherence to Theme', 'Clarity', 'Ethical Standards', 'Relevance']
                     
        report += "\n## Category Analysis\n\n"
        for category in categories:
            report += f"\n### {category}\n\n"
            cat_scores = [(s['video_name'], s['scores'][category]['score'], 
                          s['scores'][category]['justification']) 
                         for s in scores]
            # Sort by score but keep videos with same score in original order
            cat_scores.sort(key=lambda x: (-x[1], video_order.index(x[0])))
            
            report += "| Video Name | Score | Justification |\n"
            report += "|------------|-------|---------------|\n"
            for name, score, just in cat_scores:
                report += f"| {name} | {score} | {just} |\n"
                
        # Save report to comparative directory
        output_file = self.dirs['comparative'] / f"comparative_analysis_{datetime.now().strftime('%Y%m%d')}.md"
        with open(output_file, 'w') as f:
            f.write(report)
            
        # Generate CSV with holistic statements
        csv_data = []
        for score in scores:
            row = {
                'Video Name': score['video_name'],
                'Total Score': score['total_score']
            }
            # Add individual category scores
            for category in categories:
                row[category] = score['scores'][category]['score']
            
            # Add holistic statement
            holistic = score['scores']['Other Remarks']
            row['Key Highlights'] = holistic['justification']
            row['Recommendations'] = holistic['feedback']
            
            csv_data.append(row)
            
        csv_file = self.dirs['comparative'] / f"score_comparison_{datetime.now().strftime('%Y%m%d')}.csv"
        pd.DataFrame(csv_data).to_csv(csv_file, index=False)
        
        self.logger.info(f"Saved comparative analysis to {output_file}")
        self.logger.info(f"Saved score comparison to {csv_file}")
