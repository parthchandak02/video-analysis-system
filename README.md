# Video Analysis System

A Python-based system for analyzing and scoring videos using Google's Gemini Vision API. This system is particularly focused on evaluating road safety videos but can be adapted for other video analysis tasks.

## Features

- 🎥 Automatic video frame extraction
- 🔍 Frame-by-frame analysis using Google Gemini Vision API
- 📊 Comprehensive scoring system with multiple criteria
- 📝 Detailed analysis reports in Markdown format
- 📈 Comparative analysis across multiple videos
- ⚡ Rate limiting and error handling for API requests
- 🔄 Batch processing support

## Setup

1. Clone the repository:
```bash
git clone https://github.com/parthchandak02/video-analysis-system.git
cd video-analysis-system
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Copy the example config and add your Gemini API key:
```bash
cp config/config.example.json config/config.json
```
Then edit `config/config.json` to add your Gemini API key.

## Usage

1. Create a CSV file with video information (see `sample_videos.csv` for format):
```csv
Video Name,Video URL
Example Video,https://example.com/video.mp4
```

2. Run the video processor:
```bash
python src/video_processor.py your_videos.csv --mode all
```

Available modes:
- `all`: Process everything (download, extract frames, analyze)
- `download`: Only download videos
- `frames`: Only extract frames
- `analysis`: Only run analysis
- `compare`: Only generate comparative analysis

## Configuration

The system can be configured via `config/config.json`:

- `gemini_api_key`: Your Google Gemini API key
- `model_version`: Gemini model version to use
- `rate_limits`: API rate limiting settings
- `frame_extraction`: Frame extraction settings

## Directory Structure

```
.
├── config/                 # Configuration files
├── data/                  # Data directories
│   ├── 01_videos/        # Downloaded videos
│   ├── 02_frames/        # Extracted frames
│   ├── 03_captions/      # Video captions
│   ├── 04_analysis/      # Individual analysis reports
│   ├── 05_scores/        # Scoring data
│   └── 06_comparative/   # Comparative analysis
└── src/                  # Source code
    ├── video_processor.py
    ├── gemini_video_analyzer.py
    └── config_loader.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
