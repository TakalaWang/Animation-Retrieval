# update_hf_metadata.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from huggingface_hub import HfApi

# === 你原本的設定，改這裡就好 ===
CACHE_DIR = Path("./cache_gemini_video")
METADATA_CACHE_DIR = Path("./metadata")
HF_REPO_SEGMENT = "TakalaWang/anime-2024-winter-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-winter-episode-queries"
HF_REPO_SERIES = "TakalaWang/anime-2024-winter-series-queries"
METADATA_FILENAME = "metadata.jsonl"
# ==================================



def ensure_file_name(record: Dict[str, Any], level: str) -> Dict[str, Any]:
    """確保每筆記錄都有 file_name，沒有就依照你原本的命名規則補一個"""
    if "file_name" in record and record["file_name"]:
        return record

    series_name = record.get("series_name", "unknown").replace(" ", "_").replace("/", "_")

    if level == "segment":
        ep = record.get("episode_id", "unknown")
        seg = record.get("segment_index", 0)
        record["file_name"] = f"videos/{series_name}/segment_{series_name}_{ep}_seg{seg}.mp4"
    elif level == "episode":
        ep = record.get("episode_id", "unknown")
        record["file_name"] = f"videos/{series_name}/episode_{series_name}_{ep}.mp4"
    elif level == "series":
        record["file_name"] = f"videos/series_{series_name}.mp4"

    return record


def collect_segment_metadata() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in CACHE_DIR.glob("segment_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 有些是 list（同一集所有 seg），有些是單筆
        if isinstance(data, list):
            for item in data:
                items.append(ensure_file_name(item, "segment"))
        else:
            items.append(ensure_file_name(data, "segment"))
    return items


def collect_episode_metadata() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in CACHE_DIR.glob("episode_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items.append(ensure_file_name(data, "episode"))
    return items


def collect_series_metadata() -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    for path in CACHE_DIR.glob("series_*.json"):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        items.append(ensure_file_name(data, "series"))
    return items


def write_jsonl(local_path: Path, items: List[Dict[str, Any]]):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def upload_jsonl_to_hf(repo_id: str, local_path: Path, remote_name: str, hf_token: str):
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=repo_id,
        path_in_repo=remote_name,
        repo_type="dataset",
    )
    print(f"✅ uploaded {remote_name} -> {repo_id}")


def update_segment_metadata(hf_token: str):
    items = collect_segment_metadata()
    if not items:
        print("⚠️ No segment metadata found.")
        return

    local_path = METADATA_CACHE_DIR / f"segment_{METADATA_FILENAME}"
    write_jsonl(local_path, items)
    upload_jsonl_to_hf(HF_REPO_SEGMENT, local_path, METADATA_FILENAME, hf_token)
    print(f"📝 segment: {len(items)} rows")


def update_episode_metadata(hf_token: str):
    items = collect_episode_metadata()
    if not items:
        print("⚠️ No episode metadata found.")
        return

    local_path = METADATA_CACHE_DIR / f"episode_{METADATA_FILENAME}"
    write_jsonl(local_path, items)
    upload_jsonl_to_hf(HF_REPO_EPISODE, local_path, METADATA_FILENAME, hf_token)
    print(f"📝 episode: {len(items)} rows")

def update_series_metadata(hf_token: str):
    items = collect_series_metadata()
    if not items:
        print("⚠️ No series metadata found.")
        return

    local_path = METADATA_CACHE_DIR / f"series_{METADATA_FILENAME}"
    write_jsonl(local_path, items)
    upload_jsonl_to_hf(HF_REPO_SERIES, local_path, METADATA_FILENAME, hf_token)
    print(f"📝 series: {len(items)} rows")


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("請先在 .env 設定 HF_TOKEN")

    update_segment_metadata(hf_token)
    update_episode_metadata(hf_token)
    update_series_metadata(hf_token)


if __name__ == "__main__":
    main()
