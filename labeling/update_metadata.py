# update_hf_metadata.py
import os
import json
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from huggingface_hub import HfApi

# === 基本設定 ===
CACHE_DIR = Path("./cache_gemini_video")
METADATA_CACHE_DIR = Path("./metadata")

HF_REPO_SEGMENT = "TakalaWang/anime-2024-winter-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-winter-episode-queries"
HF_REPO_SERIES = "TakalaWang/anime-2024-winter-series-queries"

METADATA_FILENAME = "metadata.jsonl"
# =================

def ensure_file_name(record: Dict[str, Any], level: str) -> Dict[str, Any]:
    """確保每筆記錄都有 file_name，沒有就依照命名規則補上"""
    if record.get("file_name"):
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


def sort_items(items: List[Dict[str, Any]], level: str) -> List[Dict[str, Any]]:
    """讓 jsonl 有固定順序"""
    if level == "segment":
        return sorted(items, key=lambda x: (x.get("series_name", ""), int(x.get("episode_id", "")), int(x.get("segment_index", 0))))
    elif level == "episode":
        return sorted(items, key=lambda x: (x.get("series_name", ""), int(x.get("episode_id", ""))))
    elif level == "series":
        return sorted(items, key=lambda x: x.get("series_name", ""))
    return items


def collect_metadata(level: str) -> List[Dict[str, Any]]:
    """掃描 cache_gemini_video 資料夾，收集 metadata"""
    pattern = {
        "segment": "segment_*.json",
        "episode": "episode_*.json",
        "series": "series_*.json",
    }[level]

    items: List[Dict[str, Any]] = []
    for path in CACHE_DIR.glob(pattern):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    items.append(ensure_file_name(item, level))
            else:
                items.append(ensure_file_name(data, level))
        except Exception as e:
            print(f"⚠️ 無法讀取 {path.name}: {e}")
    return items


def write_jsonl(local_path: Path, items: List[Dict[str, Any]]):
    """寫入 metadata.jsonl"""
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")


def upload_jsonl_to_hf(repo_id: str, local_path: Path, hf_token: str):
    """上傳至 Hugging Face"""
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=repo_id,
        path_in_repo=METADATA_FILENAME,
        repo_type="dataset",
    )
    print(f"✅ 已上傳 {METADATA_FILENAME} 至 {repo_id}")


def update_segment_metadata(hf_token: str):
    items = collect_metadata("segment")
    if not items:
        print("⚠️ 沒有 segment metadata。")
        return
    items = sort_items(items, "segment")

    local_path = METADATA_CACHE_DIR / f"segment_{METADATA_FILENAME}"
    write_jsonl(local_path, items)
    upload_jsonl_to_hf(HF_REPO_SEGMENT, local_path, hf_token)
    print(f"📝 segment metadata: {len(items)} 筆")


def update_episode_metadata(hf_token: str):
    items = collect_metadata("episode")
    if not items:
        print("⚠️ 沒有 episode metadata。")
        return
    items = sort_items(items, "episode")

    local_path = METADATA_CACHE_DIR / f"episode_{METADATA_FILENAME}"
    write_jsonl(local_path, items)
    upload_jsonl_to_hf(HF_REPO_EPISODE, local_path, hf_token)
    print(f"📝 episode metadata: {len(items)} 筆")


def update_series_metadata(hf_token: str):
    items = collect_metadata("series")
    if not items:
        print("⚠️ 沒有 series metadata。")
        return
    items = sort_items(items, "series")

    local_path = METADATA_CACHE_DIR / f"series_{METADATA_FILENAME}"
    write_jsonl(local_path, items)
    upload_jsonl_to_hf(HF_REPO_SERIES, local_path, hf_token)
    print(f"📝 series metadata: {len(items)} 筆")


def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("請先設定 HF_TOKEN")

    update_segment_metadata(hf_token)
    update_episode_metadata(hf_token)
    update_series_metadata(hf_token)


if __name__ == "__main__":
    main()
