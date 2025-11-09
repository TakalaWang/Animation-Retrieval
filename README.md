# Anime Video Retrieval Dataset Creation

This script processes the JacobLinCool/anime-2024 winter subset to create a comprehensive query dataset with Hugging Face Dataset Viewer support across three levels: episode, segment, and series.

## Setup

1. Install dependencies: `uv sync`
2. Set your Gemini API key(s): `export GEMINI_API_KEY=key1,key2,key3` (supports multiple keys for rate limiting)
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
├── segment_{episode_id}_seg{index}.mp4  # 60-second video segments (with 5s overlap)
└── metadata.jsonl                       # Segment metadata with queries
```

### Series Dataset (`TakalaWang/anime-2024-winter-series-queries`)
```
📁 videos/
├── series_{series_name}.mp4    # Concatenated full series videos
└── metadata.jsonl              # Series metadata with queries
```

## Metadata Format

Each `metadata.jsonl` file contains JSON Lines format with the following structure:

### Episode Metadata
```json
{
  "file_name": "videos/series_name/episode_01.mp4",
  "series_name": "series_name",
  "episode_id": "01",
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

### Series Metadata
```json
{
  "file_name": "videos/series_series_name.mp4",
  "series_name": "series_name",
  "episode_count": 13,
  "release_dates": ["2024-01-01", "2024-01-08", "2024-01-15"],
  "model_response": {
    "series_overview": ["query1", "query2", "query3"],
    "character_arcs": ["query1", "query2", "query3"],
    "themes_motifs": ["query1", "query2", "query3"],
    "pacing_rhythm": ["query1", "query2", "query3"],
    "ending_climax": ["query1", "query2", "query3"]
  }
}
```

## Dataset Viewer Support

All datasets automatically have **Dataset Viewer** enabled on Hugging Face Hub through `metadata.jsonl` files:

- **Episode Dataset**: `videos/{series_name}/episode_{id}.mp4` + `metadata.jsonl`
- **Segment Dataset**: `videos/segment_{episode_id}_seg{index}.mp4` + `metadata.jsonl`
- **Series Dataset**: `videos/series_{series_name}.mp4` + `metadata.jsonl`

The `metadata.jsonl` files contain one JSON object per line with a `file_name` field linking to video files, automatically enabling the Dataset Viewer without requiring additional libraries like pandas or datasets for upload.

## Usage

```python
from datasets import load_dataset

# Load episode dataset
episodes = load_dataset("TakalaWang/anime-2024-winter-episode-queries")

# Load segment dataset
segments = load_dataset("TakalaWang/anime-2024-winter-segment-queries")

# Load series dataset
series = load_dataset("TakalaWang/anime-2024-winter-series-queries")
```

## What it does

- Downloads anime videos from the winter subset
- **Three-level processing**: Episode, Segment (60s with 5s overlap), and Series levels
- **Intelligent caching**: Avoids re-processing already analyzed videos and segments
- **Multi-API key support**: Automatically rotates through multiple Gemini API keys to handle rate limits
- **Robust error handling**: Automatic retry with key rotation for failed API calls
- Uses Gemini to generate natural language search queries for each level
- Creates comprehensive metadata with structured query categories
- Uploads videos and metadata to three separate Hugging Face datasets
- Enables Dataset Viewer on all datasets through metadata.jsonl files

## Technical Details

- **Multi-level processing**: Episode (full videos), Segment (60s chunks with 5s overlap), Series (concatenated episodes)
- **Smart caching system**: Local JSON cache prevents re-processing of analyzed content
- **API key rotation**: Supports multiple Gemini API keys with automatic failover
- **Rate limit handling**: Intelligent retry mechanism with exponential backoff
- **Video processing**: Uses MoviePy for precise video segmentation and concatenation
- **Metadata management**: Structured JSONL format for Dataset Viewer compatibility
- **File state checking**: Ensures Gemini file uploads are processed before use
- **Progress tracking**: Comprehensive progress bars for all processing stages

## Caching System

The script implements a comprehensive caching system to avoid redundant API calls:

- **Segment cache**: `./cache_gemini_video/segment_{episode_id}_seg{index}.json`
- **Episode cache**: `./cache_gemini_video/episode_{episode_id}.json`
- **Series cache**: `./cache_gemini_video/series_{series_name}.json`
- **Video cache**: Processed video segments and concatenated series videos

Cached results are automatically reused, significantly reducing processing time and API costs for subsequent runs.

## API Usage

Uses the updated google-genai package for Gemini API calls with advanced features:

- **Multi-key support**: Configure multiple API keys separated by commas for automatic rotation
- **Intelligent retry**: Automatic key switching and retry on rate limit errors
- **File processing**: Proper waiting for asynchronous file processing completion
- **Structured output**: JSON schema validation for consistent query generation
- **Rate limit management**: Built-in handling of quota exceeded errors


