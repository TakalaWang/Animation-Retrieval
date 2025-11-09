# update_hf_metadata.py (相容版)
import os
import json
from pathlib import Path
from typing import Any, Dict, List
from dotenv import load_dotenv
from huggingface_hub import HfApi

# === 基本設定 ===
CACHE_DIR = Path("./cache_gemini_video")
METADATA_CACHE_DIR = Path("./metadata")
SYNC_INDEX_PATH = METADATA_CACHE_DIR / ".sync_index.json"

HF_REPO_SEGMENT = "TakalaWang/anime-2024-winter-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-winter-episode-queries"
HF_REPO_SERIES = "TakalaWang/anime-2024-winter-series-queries"
METADATA_FILENAME = "metadata.jsonl"
# =================

# === 共用工具 ===
def load_sync_index() -> Dict[str, float]:
    if SYNC_INDEX_PATH.exists():
        with open(SYNC_INDEX_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

def save_sync_index(index: Dict[str, float]):
    SYNC_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SYNC_INDEX_PATH, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)

def ensure_file_name(record: Dict[str, Any], level: str) -> Dict[str, Any]:
    if record.get("file_name"):  # 已存在則不動
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
    if level == "segment":
        return sorted(items, key=lambda x: (
            x.get("series_name", ""),
            str(x.get("episode_id", "")),
            int(x.get("segment_index", 0)),
        ))
    elif level == "episode":
        return sorted(items, key=lambda x: (x.get("series_name", ""), str(x.get("episode_id", ""))))
    elif level == "series":
        return sorted(items, key=lambda x: x.get("series_name", ""))
    return items

def write_jsonl(local_path: Path, items: List[Dict[str, Any]]):
    local_path.parent.mkdir(parents=True, exist_ok=True)
    with open(local_path, "w", encoding="utf-8") as f:
        for item in items:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

def upload_jsonl_to_hf(repo_id: str, local_path: Path, hf_token: str):
    api = HfApi(token=hf_token)
    api.upload_file(
        path_or_fileobj=str(local_path),
        repo_id=repo_id,
        path_in_repo=METADATA_FILENAME,
        repo_type="dataset",
    )
    print(f"✅ {local_path.name} → {repo_id} 已上傳")

# === 主邏輯 ===
def collect_metadata(level: str, sync_index: Dict[str, float]) -> List[Dict[str, Any]]:
    """只收集有變更的 metadata"""
    pattern = {
        "segment": "segment_*.json",
        "episode": "episode_*.json",
        "series": "series_*.json",
    }[level]

    new_items: List[Dict[str, Any]] = []
    for path in CACHE_DIR.glob(pattern):
        try:
            mtime = path.stat().st_mtime
            if sync_index.get(str(path)) == mtime:
                continue  # 未變更
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    new_items.append(ensure_file_name(item, level))
            else:
                new_items.append(ensure_file_name(data, level))
            sync_index[str(path)] = mtime
        except Exception as e:
            print(f"⚠️ 讀取失敗 {path}: {e}")
    return new_items

def update_metadata(hf_token: str, level: str):
    sync_index = load_sync_index()
    items = collect_metadata(level, sync_index)
    if not items:
        print(f"⏩ {level}: 無新變更，略過上傳")
        return

    items = sort_items(items, level)
    local_path = METADATA_CACHE_DIR / f"{level}_{METADATA_FILENAME}"
    write_jsonl(local_path, items)

    repo_map = {
        "segment": HF_REPO_SEGMENT,
        "episode": HF_REPO_EPISODE,
        "series": HF_REPO_SERIES,
    }
    upload_jsonl_to_hf(repo_map[level], local_path, hf_token)
    save_sync_index(sync_index)
    print(f"📝 {level}: 新增 {len(items)} 筆，已同步")

# === 與 main.py 相容的三個介面 ===
def update_segment_metadata(hf_token: str):
    update_metadata(hf_token, "segment")

def update_episode_metadata(hf_token: str):
    update_metadata(hf_token, "episode")

def update_series_metadata(hf_token: str):
    update_metadata(hf_token, "series")

# === 可單獨執行 ===
def main():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN", "")
    if not hf_token:
        raise RuntimeError("請先設定 HF_TOKEN")
    for level in ["segment", "episode", "series"]:
        update_metadata(hf_token, level)

if __name__ == "__main__":
    main()
