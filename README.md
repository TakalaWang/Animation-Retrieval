# Anime Video Retrieval Dataset Creation

This script processes the JacobLinCool/anime-2024 winter subset to create a query dataset with Hugging Face Dataset Viewer support.

## Setup

1. Install dependencies: `uv sync`
2. Set your Gemini API key: `export GEMINI_API_KEY=your_key_here`
3. Set your Hugging Face token: `export HF_TOKEN=your_token_here`
4. Run the script: `uv run python labeling/main.py`

## Dataset Structure

The script creates three Hugging Face datasets with automatic Dataset Viewer support:

### Episode Dataset (`TakalaWang/anime-2024-winter-episode-queries`)
```
📁 videos/{series_name}/
├── episode_{episode_id}.mp4    # Full episode videos
└── metadata.jsonl              # Episode metadata with queries
```

### Segment Dataset (`TakalaWang/anime-2024-winter-segment-queries`)
```
📁 videos/
├── segment_{episode_id}_seg{index}.mp4  # 60-second video segments
└── metadata.jsonl                       # Segment metadata with queries
```

### Series Dataset (`TakalaWang/anime-2024-winter-series-queries`)
```
📁 videos/
├── series_{series_name}.mp4    # Concatenated full series videos
└── series_{series_name}.json   # Series-level metadata
```

## Metadata Format

Each `metadata.jsonl` file contains JSON Lines format with the following structure:

### Episode Metadata
```json
{
  "file_name": "videos/series_name/episode_01.mp4",
  "series_name": "series_name",
  "episode_name": "01",
  "release_date": "2024-01-01",
  "duration": 1200.5,
  "model_response": {
    "main_plot": ["query1", "query2", "query3"],
    "turning_point": ["query1", "query2", "query3"],
    "relationship_change": ["query1", "query2", "query3"],
    "episode_mood": ["query1", "query2", "query3"],
    "notable_scene": ["query1", "query2", "query3"]
  }
}
```

### Segment Metadata
```json
{
  "file_name": "videos/segment_episode01_seg0.mp4",
  "series_name": "series_name",
  "episode_id": "01",
  "segment_index": 0,
  "release_date": "2024-01-01",
  "queries": {
    "visual_saliency": ["query1", "query2", "query3"],
    "character_emotion": ["query1", "query2", "query3"],
    "action_behavior": ["query1", "query2", "query3"],
    "dialogue": ["query1", "query2", "query3"],
    "symbolic_scene": ["query1", "query2", "query3"]
  }
}
```

## Dataset Viewer Support

All datasets automatically have **Dataset Viewer** enabled on Hugging Face Hub through `metadata.jsonl` files:

- **Episode Dataset**: `videos/{series_name}/episode_{id}.mp4` + `metadata.jsonl`
- **Segment Dataset**: `videos/segment_{episode_id}_seg{index}.mp4` + `metadata.jsonl`

The `metadata.jsonl` files contain one JSON object per line with a `file_name` field linking to video files, automatically enabling the Dataset Viewer without requiring additional libraries like pandas or datasets for upload.

## Usage

```python
from datasets import load_dataset

# Load episode dataset
episodes = load_dataset("TakalaWang/anime-2024-winter-episode-queries")

# Load segment dataset
segments = load_dataset("TakalaWang/anime-2024-winter-segment-queries")
```

## What it does

- Downloads anime videos from the winter subset
- **Fixed 1-minute segmentation**: Cuts each video into exactly 60-second segments
- Uses Gemini to generate descriptions for each segment (based on extracted frames)
- Generates 10 possible search queries per segment
- Uploads everything to TakalaWang/Anime-2024-winter-query dataset

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
