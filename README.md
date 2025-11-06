# Anime Video Retrieval Dataset Creation

This script processes the JacobLinCool/anime-2024 spring subset to create a query dataset.

## Setup

1. Install dependencies: `uv sync`
2. Set your Gemini API key: `export GEMINI_API_KEY=your_key_here`
3. Run the script: `uv run python labeling/main.py`

## What it does

- Downloads anime videos from the spring subset
- **Fixed 1-minute segmentation**: Cuts each video into exactly 60-second segments
- Uses Gemini to generate descriptions for each segment (based on extracted frames)
- Generates 10 possible search queries per segment
- Uploads everything to TakalaWang/Anime-2024-spring-query dataset

## Technical Details

- **No ffmpeg dependency**: Uses pure OpenCV for video processing
- **Low fps output**: Videos are saved at 10 fps to reduce file size
- **Fixed segmentation**: Simple and reliable 60-second chunks
 
Note: The script now uses the system `ffmpeg` binary to perform precise and fast trimming and re-encoding of segments. Please ensure `ffmpeg` is installed and available in your PATH. On macOS you can install it with:

```bash
brew install ffmpeg
```

## Note

This is a resource-intensive process. The full dataset has many videos, so the code is currently limited to the first 5 videos for testing. Remove `.select(range(5))` to process all videos.

## API Usage

Uses the updated google-genai package for Gemini API calls.
