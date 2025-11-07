import os
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm, trange
from dotenv import load_dotenv
from datasets import load_dataset, Dataset
from huggingface_hub import HfApi, create_repo

import google.genai as genai
from moviepy import VideoFileClip, concatenate_videoclips

from segment_processor import generate_segment_queries
from episode_processor import generate_episode_queries
from series_processor import generate_series_queries

# ================== 基本設定 ==================
logging.basicConfig(level=logging.INFO)
load_dotenv()

# 支援單個或多個 API Key（用逗號分隔或 JSON 陣列）
GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").strip().split(',') if k.strip()]

HF_TOKEN = os.getenv("HF_TOKEN", "")

if not GEMINI_API_KEYS:
    raise RuntimeError("請先設定 GEMINI_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("請先設定 HF_TOKEN")

# API Key 輪換
current_key_index = 0

def get_next_api_key():
    """輪換使用 API Key 以避免速率限制"""
    global current_key_index
    key = GEMINI_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key

# Hugging Face 倉庫設定
HF_REPO_SEGMENT = "TakalaWang/anime-2024-winter-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-winter-episode-queries"
HF_REPO_SERIES = "TakalaWang/anime-2024-winter-series-queries"

# 快取目錄
CACHE_DIR = Path("./cache_gemini_video")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 資料集設定
TEST_DATASET = "JacobLinCool/anime-2024"
TEST_SPLIT = "winter"
SEGMENT_LENGTH = 60  # 片段長度（秒）
SEGMENT_OVERLAP = 5  # 片段重疊（秒）

def get_client():
    """獲取 Gemini client（使用輪換的 API key）"""
    return genai.Client(api_key=get_next_api_key())


# ================== Hugging Face 工具函數 ==================
def ensure_hf_repos():
    """確保 Hugging Face 倉庫存在"""
    create_repo(HF_REPO_SEGMENT, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    create_repo(HF_REPO_EPISODE, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    create_repo(HF_REPO_SERIES, token=HF_TOKEN, repo_type="dataset", exist_ok=True)


def upload_json_to_hf(repo_id: str, path: Path, repo_path: str):
    """上傳 JSON 檔案到 Hugging Face"""
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(path),
        repo_id=repo_id,
        path_in_repo=repo_path,
        repo_type="dataset",
    )


def upload_dataset_to_hf(repo_id: str, data: List[Dict[str, Any]]):
    """上傳數據列表到 HF dataset 格式，啟用 data viewer"""

    # 創建 Dataset 對象（直接從字典列表，保留嵌套結構）
    dataset = Dataset.from_list(data)

    # 上傳到 HF
    dataset.push_to_hub(repo_id, token=HF_TOKEN)

    print(f"📊 已上傳 dataset 到 HF: {repo_id} ({len(data)} 筆記錄)")


# ================== 數據集管理 ==================
def create_metadata_jsonl(repo_id: str, metadata_list: List[Dict[str, Any]], metadata_filename: str = "metadata.jsonl"):
    """創建 metadata.jsonl 文件並上傳到 HF dataset，啟用 data viewer"""
    # 確保所有記錄都有 file_name 字段
    processed_metadata = []
    for item in metadata_list:
        if "file_name" not in item:
            # 如果沒有 file_name，嘗試從其他字段推斷
            if "episode_name" in item:
                item["file_name"] = f"videos/{item.get('series_name', 'unknown')}/episode_{item['episode_name']}.mp4"
            elif "segment_index" in item:
                item["file_name"] = f"videos/segment_{item.get('episode_id', 'unknown')}_seg{item['segment_index']}.mp4"
            else:
                continue  # 跳過沒有文件名的記錄

        processed_metadata.append(item)

    # 寫入本地 metadata.jsonl 文件
    metadata_path = CACHE_DIR / metadata_filename
    with open(metadata_path, "w", encoding="utf-8") as f:
        for item in processed_metadata:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    # 上傳到 HF
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(metadata_path),
        repo_id=repo_id,
        path_in_repo=metadata_filename,
        repo_type="dataset",
    )

    print(f"📋 已創建並上傳 metadata.jsonl 到 {repo_id} ({len(processed_metadata)} 筆記錄)")
    return processed_metadata


def upload_video_to_hf(repo_id: str, video_path: Path, repo_path: str):
    """上傳影片檔案到 HuggingFace"""
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(video_path),
        repo_id=repo_id,
        path_in_repo=repo_path,
        repo_type="dataset",
    )
    print(f"📤 已上傳影片到 HF: {repo_path}")


# ================== 影片處理工具函數 ==================
def extract_video_segment(video_path: str, start_s: float, end_s: float, output_path: Path):
    """切割影片片段"""
    with VideoFileClip(video_path) as video:
        segment = video.subclipped(start_s, end_s)
        segment.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(output_path.parent / f"temp_{output_path.stem}_audio.m4a"),
            remove_temp=True,
            logger=None,
        )


def concatenate_videos(video_paths: List[str], output_path: Path):
    """合併多個影片"""
    clips = []
    for path in video_paths:
        clips.append(VideoFileClip(path))
    
    final_clip = concatenate_videoclips(clips, method="compose")
    final_clip.write_videofile(
        str(output_path),
        codec="libx264",
        audio_codec="aac",
        temp_audiofile=str(output_path.parent / f"temp_{output_path.stem}_audio.m4a"),
        remove_temp=True,
        logger=None,
    )
    
    # 關閉所有 clips
    for clip in clips:
        clip.close()
    final_clip.close()
    
    print(f"🔗 已合併影片: {output_path.name}")


def get_video_duration_from_path(path: str) -> float:
    """獲取影片長度（秒）"""
    with VideoFileClip(path) as clip:
        return clip.duration


# ================== API 工具函數 ==================
def call_with_retry(fn, *args, **kwargs):
    """帶重試機制的 API 呼叫"""
    sleep_sec = kwargs.pop("sleep_sec", 5)
    while True:
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate" in msg.lower():
                print(f"[rate limited] sleep {sleep_sec}s and retry ...")
                time.sleep(sleep_sec)
                continue
            print(f"[error] {e} -> sleep {sleep_sec}s and retry ...")
            time.sleep(sleep_sec)  
        


# ================== 資料處理函數 ==================
def load_and_group_dataset() -> Dict[str, List[Dict[str, Any]]]:
    """載入資料集並按系列分組"""
    print("載入資料集...")
    ds = load_dataset(TEST_DATASET, TEST_SPLIT, split="train")
    ds_raw = ds.with_format("arrow")

    # 依 series 分組
    series_groups: Dict[str, List[Dict[str, Any]]] = {}
    for i in trange(len(ds_raw), desc="group by series"):
        row = ds_raw[i:i+1]
        series_name = row["series_name"][0].as_py()
        episode_name = row["episode_name"][0].as_py()
        video_path = row["video"][0]["path"].as_py()
        duration = row["duration"][0].as_py() if "duration" in row.column_names else None
        release_date = row["release_date"][0].as_py() if "release_date" in row.column_names else None

        series_groups.setdefault(series_name, []).append({
            "episode_name": episode_name,
            "series_name": series_name,
            "video_path": video_path,
            "duration": duration,
            "release_date": release_date,
        })

    return series_groups


def process_segments_for_episode(episode_id: str, video_path: str, duration_s: float) -> List[Dict[str, Any]]:
    """處理單集的片段級別查詢生成"""
    segment_ranges = []
    start = 0.0
    while start < duration_s:
        end = min(start + SEGMENT_LENGTH, duration_s)
        if end - start < 5:
            break
        segment_ranges.append((start, end))
        if end >= duration_s:
            break
        start = start + SEGMENT_LENGTH - SEGMENT_OVERLAP

    segment_files = []
    for seg_idx, (s, e) in enumerate(tqdm(segment_ranges, desc=f"切割片段 {episode_id}", unit="seg")):
        segment_video_path = CACHE_DIR / f"segment_{episode_id}_seg{seg_idx}.mp4"
        if not segment_video_path.exists():
            extract_video_segment(video_path, s, e, segment_video_path)

        segment_files.append({
            "index": seg_idx,
            "path": segment_video_path,
        })

    # 處理每個片段
    seg_results = []
    for seg_info in segment_files:
        seg_idx = seg_info["index"]
        segment_path = seg_info["path"]

        cache_path = CACHE_DIR / f"segment_{episode_id}_seg{seg_idx}.json"
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
                seg_results.append(cached)
                continue

        client = get_client()

        data = call_with_retry(
            generate_segment_queries,
            client=client,
            video_path=str(segment_path),
            sleep_sec=5,
        )

        record = {
            "series_name": "",  # 稍後填充
            "episode_id": episode_id,
            "segment_index": seg_idx,
            "release_date": None,  # 稍後填充
            "queries": data,
        }

        # 儲存快取
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        seg_results.append(record)

        # 上傳片段影片到 HF
        segment_hf_path = f"videos/segment_{episode_id}_seg{seg_idx}.mp4"
        print(f"上傳片段 {seg_idx} 到 HuggingFace...")
        upload_video_to_hf(
            HF_REPO_SEGMENT,
            segment_path,
            segment_hf_path
        )

        # 添加 file_name 字段到 record
        record["file_name"] = segment_hf_path

    return seg_results


def process_episode_level(series_name: str, episode_id: str, video_path: str, duration_s: float, release_date: Any) -> Dict[str, Any]:
    """處理集數級別的查詢生成"""
    client = get_client()

    uploaded = client.files.upload(file=video_path)
    file_uri = uploaded.uri


    epi_result = call_with_retry(
        generate_episode_queries,
        client=client,
        file_uri=file_uri,
        sleep_sec=5,
    )

    # 上傳完整集數影片到 HF
    episode_video_hf_path = f"videos/{series_name}/episode_{episode_id}.mp4"
    print("  📤 上傳完整集數到 HuggingFace...")
    upload_video_to_hf(HF_REPO_EPISODE, Path(video_path), episode_video_hf_path)

    # 包裝 episode metadata 與模型回應
    episode_record = {
        "file_name": episode_video_hf_path,  # 添加 file_name 字段用於 data viewer
        "series_name": series_name,
        "episode_name": episode_id,
        "release_date": release_date,
        "duration": duration_s,
        "model_response": epi_result,
    }

    # 仍然保存單個 JSON 文件（向後兼容）
    epi_local = CACHE_DIR / f"episode_{episode_id}.json"
    with open(epi_local, "w", encoding="utf-8") as f:
        json.dump(episode_record, f, ensure_ascii=False, indent=2)
    upload_json_to_hf(HF_REPO_EPISODE, epi_local, f"episode_{episode_id}.json")

    return episode_record


def process_single_episode(series_name: str, episode_info: Dict[str, Any]) -> Tuple[str, str, float, Any, Dict[str, Any]]:
    """處理單個集數的所有層級，返回 (episode_id, video_path, duration, release_date, episode_record)"""
    episode_id = episode_info["episode_name"]
    video_path = episode_info["video_path"]

    duration_s = get_video_duration_from_path(video_path)

    print(f"\n{'='*60}")
    print(f"處理集數: {episode_id}")
    print(f"影片長度: {duration_s:.2f} 秒")
    print(f"{'='*60}\n")

    # ===== Segment level =====
    seg_results = process_segments_for_episode(episode_id, video_path, duration_s)

    # 填充 series_name 和 release_date
    for seg in seg_results:
        seg["series_name"] = series_name
        seg["release_date"] = episode_info.get("release_date")

    # 上傳 segment JSON 彙總
    seg_local = CACHE_DIR / f"segment_{episode_id}.json"
    with open(seg_local, "w", encoding="utf-8") as f:
        json.dump(seg_results, f, ensure_ascii=False, indent=2)
    upload_json_to_hf(HF_REPO_SEGMENT, seg_local, f"segment_{episode_id}.json")

    # ===== Episode level =====
    episode_record = process_episode_level(series_name, episode_id, video_path, duration_s, episode_info.get("release_date"))

    return episode_id, video_path, duration_s, episode_info.get("release_date"), episode_record


def process_series_level(series_name: str, processed_episodes: List[Tuple[str, str, float, Any]]) -> Dict[str, Any]:
    """處理系列級別的查詢生成"""
    print(f"\n📍 步驟 4: 處理整季資料: {series_name} ...")

    client = get_client()

    # 為 series 構建簡單的文字描述（目前可根據需要擴充）
    series_text_lines = [f"Series: {series_name}"]
    for ep_id, vp, dur, rd in processed_episodes:
        series_text_lines.append(f"Episode: {ep_id} | duration: {dur:.2f} s | path: {vp} | release_date: {rd}")
    series_text = "\n".join(series_text_lines)

    series_result = call_with_retry(
        generate_series_queries,
        client=client,
        series_text=series_text,
        sleep_sec=5,
    )

    # 合併並上傳整季影片
    print("  準備整季影片...")
    episode_video_paths = [vp for _, vp, _, _ in processed_episodes]
    series_video_path = CACHE_DIR / f"series_{series_name}.mp4"
    if not series_video_path.exists():
        print("  🔗 開始合併整季影片...")
        concatenate_videos(episode_video_paths, series_video_path)

    print("  📤 上傳整季影片到 HuggingFace...")
    upload_video_to_hf(
        HF_REPO_SERIES,
        series_video_path,
        f"videos/series_{series_name}.mp4"
    )

    # 建立 series metadata（不要存 episode_name，僅存 release_dates 與模型回應）
    release_dates = sorted({rd for (_eid, _vp, _dur, rd) in processed_episodes if rd is not None})
    series_record = {
        "series_name": series_name,
        "episode_count": len(processed_episodes),
        "release_dates": release_dates,
        "model_response": series_result,
    }

    series_local = CACHE_DIR / f"series_{series_name}.json"
    with open(series_local, "w", encoding="utf-8") as f:
        json.dump(series_record, f, ensure_ascii=False, indent=2)
    upload_json_to_hf(HF_REPO_SERIES, series_local, f"series_{series_name}.json")

    return series_record


def process_single_series(series_name: str, episodes: List[Dict[str, Any]]) -> Tuple[Tuple[str, List[Tuple[str, str, float, Any]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """處理單個系列的所有集數，返回 (result, episode_metadata, segment_metadata)"""
    print(f"開始處理系列: {series_name} (共 {len(episodes)} 集)")

    processed_episodes = []
    episode_metadata = []
    segment_metadata = []
    episode_video_paths = []  # 用於最後合併整季

    for episode_info in tqdm(episodes, desc=f"episodes - {series_name}", unit="ep"):
        episode_id, video_path, duration_s, release_date, episode_record = process_single_episode(series_name, episode_info)
        processed_episodes.append((episode_id, video_path, duration_s, release_date))
        episode_metadata.append(episode_record)
        episode_video_paths.append(video_path)

        # 收集 segment metadata
        seg_results = process_segments_for_episode(episode_id, video_path, duration_s)
        # 填充 series_name 和 release_date
        for seg in seg_results:
            seg["series_name"] = series_name
            seg["release_date"] = release_date
        segment_metadata.extend(seg_results)

    # 處理系列級別
    process_series_level(series_name, processed_episodes)

    return (series_name, processed_episodes), episode_metadata, segment_metadata


def process_all_series(series_groups: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Tuple[str, List[Tuple[str, str, float, Any]]]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """處理所有系列，返回 (processed_series, all_episode_metadata, all_segment_metadata)"""
    processed_series = []
    all_episode_metadata = []
    all_segment_metadata = []

    for series_name, episodes in series_groups.items():
        result, episode_metadata, segment_metadata = process_single_series(series_name, episodes)
        processed_series.append(result)
        all_episode_metadata.extend(episode_metadata)
        all_segment_metadata.extend(segment_metadata)

    return processed_series, all_episode_metadata, all_segment_metadata


# ================== 主程式 ==================
def main():
    """主程式入口"""
    ensure_hf_repos()
    
    print(f"已設定 {len(GEMINI_API_KEYS)} 個 Gemini API Key")

    # 載入並分組資料集
    series_groups = load_and_group_dataset()

    # 處理所有系列與 episodes
    processed_series, all_episode_metadata, all_segment_metadata = process_all_series(series_groups)

    # 創建 metadata.jsonl 文件以啟用 Dataset Viewer
    if all_episode_metadata:
        create_metadata_jsonl(HF_REPO_EPISODE, all_episode_metadata, "metadata.jsonl")

    if all_segment_metadata:
        create_metadata_jsonl(HF_REPO_SEGMENT, all_segment_metadata, "metadata.jsonl")

    # 完成總結
    total_series = len(processed_series)
    total_episodes = sum(len(eps) for _, eps in processed_series)

    print("\n" + "="*60)
    print("🎉 處理完成！")
    print(f"✅ 處理了 {total_series} 個系列，{total_episodes} 集")
    print(f"📊 Episode metadata: {len(all_episode_metadata)} 筆")
    print(f"📊 Segment metadata: {len(all_segment_metadata)} 筆")
    print("🔍 Dataset Viewer 現已啟用！")
    print("="*60)


if __name__ == "__main__":
    main()
