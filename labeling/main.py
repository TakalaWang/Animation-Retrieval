import os
import json
import time
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm, trange
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi, create_repo

import google.genai as genai
from moviepy import VideoFileClip, concatenate_videoclips



# 你自己的模組
from episode_processor import process_episode
from series_processor import process_series

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
    """輪換使用 API Key"""
    global current_key_index
    key = GEMINI_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key

HF_REPO_SEGMENT = "TakalaWang/anime-2024-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-episode-queries"
HF_REPO_SERIES = "TakalaWang/anime-2024-series-queries"

CACHE_DIR = Path("./cache_gemini_video")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEST_DATASET = "JacobLinCool/anime-2024"
TEST_SPLIT = "winter"
SEGMENT_LENGTH = 60
SEGMENT_OVERLAP = 5

def get_client():
    """獲取 Gemini client（使用輪換的 API key）"""
    return genai.Client(api_key=get_next_api_key())


# ================== HF 工具 ==================
def ensure_hf_repos():
    create_repo(HF_REPO_SEGMENT, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    create_repo(HF_REPO_EPISODE, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    create_repo(HF_REPO_SERIES, token=HF_TOKEN, repo_type="dataset", exist_ok=True)


def upload_json_to_hf(repo_id: str, path: Path, repo_path: str):
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(path),
        repo_id=repo_id,
        path_in_repo=repo_path,
        repo_type="dataset",
    )


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
            logger=None,  # 減少輸出
        )
    print(f"✂️  已切割片段: {output_path.name}")


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


# ================== 只有 Gemini 用的 retry ==================
def call_with_retry(fn, *args, **kwargs):
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
            # 不是 rate limit 也一樣等一下再試
            print(f"[error] {e} -> sleep {sleep_sec}s and retry ...")
            time.sleep(sleep_sec)


# ================== 本地影片長度 ==================
def get_video_duration_from_path(path: str) -> float:
    with VideoFileClip(path) as clip:
        return clip.duration  # 秒為單位 (float)

# ================== 上傳本地檔到 Gemini ==================
def upload_file_if_local(file_uri: str, client: genai.Client) -> str:
    """
    - http/https/s3/gs → 直接回傳
    - 本地 → 上傳、等 ACTIVE、回傳 uri
    """
    if re.match(r"^(https?://|s3://|gs://)", file_uri):
        return file_uri
    
    if file_uri.startswith("file://"):
        file_uri = file_uri[len("file://"):]
    
    p = Path(file_uri).expanduser()
    if not p.exists():
        raise FileNotFoundError(p)

    print(f"⬆️  上傳到 Gemini: {p.name}", flush=True)
    resp = client.files.upload(file=str(p))
    
    # 輪詢到 ACTIVE
    while resp.state.name == "PROCESSING":
        print("⏳ 處理中...", flush=True)
        time.sleep(10)
        resp = client.files.get(name=resp.name)

    if resp.state.name != "ACTIVE":
        raise RuntimeError(f"檔案處理失敗: {resp.state}")

    print(f"✅ 上傳完成: {resp.uri}", flush=True)
    return resp.uri


# ================== 主程式 ==================
def main():
    ensure_hf_repos()
    
    print(f"📝 已設定 {len(GEMINI_API_KEYS)} 個 Gemini API Key")

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

        series_groups.setdefault(series_name, []).append({
            "episode_name": episode_name,
            "series_name": series_name,
            "video_path": video_path,
            "duration": duration,
        })

    # 現在先處理第一個 series
    first_series = list(series_groups.keys())[0]
    episodes = series_groups[first_series]

    print(f"開始處理系列: {first_series} (共 {len(episodes)} 集)")
    print("🧪 測試模式：只處理第一集")

    # 只取第一集來測試
    episodes = episodes[:1]

    processed_episodes = []
    episode_video_paths = []  # 用於最後合併整季

    for idx, ex in enumerate(tqdm(episodes, desc="episodes", unit="ep"), 1):
        episode_id = ex["episode_name"] or f"{first_series}_ep{idx:02d}"
        video_path = ex["video_path"]

        # 長度
        duration_s = float(ex["duration"]) if ex.get("duration") else get_video_duration_from_path(video_path)

        print(f"\n{'='*60}")
        print(f"處理集數: {episode_id}")
        print(f"影片長度: {duration_s:.2f} 秒")
        print(f"{'='*60}\n")

        # ===== Segment level =====
        print("📍 步驟 1: 切割影片片段...")
        
        # 先在本地切割所有片段
        segment_files = []
        start = 0.0
        seg_idx = 0
        
        while start < duration_s:
            end = min(start + SEGMENT_LENGTH, duration_s)
            
            # 如果剩餘時間太短（小於 5 秒），就併入上一個片段或跳過
            if end - start < 5:
                break
            
            segment_video_path = CACHE_DIR / f"segment_{episode_id}_seg{seg_idx}.mp4"
            
            if not segment_video_path.exists():
                print(f"  ✂️  切割片段 {seg_idx}: {start:.1f}s - {end:.1f}s")
                extract_video_segment(video_path, start, end, segment_video_path)
            else:
                print(f"  📦 使用快取片段 {seg_idx}: {segment_video_path.name}")
            
            segment_files.append({
                "index": seg_idx,
                "start_s": start,
                "end_s": end,
                "path": segment_video_path,
            })
            
            # 如果這個片段已經到達影片結尾，結束循環
            if end >= duration_s:
                break
            
            # 計算下一個片段的起始位置（有重疊）
            start = start + SEGMENT_LENGTH - SEGMENT_OVERLAP
            seg_idx += 1

        print(f"\n📍 步驟 2: 上傳 {len(segment_files)} 個片段到 Gemini 並生成查詢...")
        
        # 處理每個片段：上傳到 Gemini -> 生成查詢 -> 上傳到 HF
        seg_results = []
        for seg_info in segment_files:
            seg_idx = seg_info["index"]
            segment_path = seg_info["path"]
            start_s = seg_info["start_s"]
            end_s = seg_info["end_s"]
            
            # 檢查快取
            cache_path = CACHE_DIR / f"segment_{episode_id}_seg{seg_idx}.json"
            if cache_path.exists():
                with open(cache_path, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                    seg_results.append(cached)
                    print(f"  📦 使用快取查詢: 片段 {seg_idx}")
                    continue
            
            # 上傳片段到 Gemini
            print(f"  ⬆️  上傳片段 {seg_idx} 到 Gemini...")
            client = get_client()
            segment_uri = upload_file_if_local(str(segment_path), client)
            
            # 生成查詢
            print(f"  🎬 生成查詢: 片段 {seg_idx}")
            from segment_processor import generate_segment_queries
            
            data = call_with_retry(
                generate_segment_queries,
                client=client,
                file_uri=segment_uri,
                sleep_sec=5,
            )
            
            record = {
                "episode_id": episode_id,
                "segment_index": seg_idx,
                "start_s": start_s,
                "end_s": end_s,
                "queries": data,
            }
            
            # 儲存快取
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(record, f, ensure_ascii=False, indent=2)
            
            seg_results.append(record)
            
            # 上傳片段影片到 HF
            print(f"  📤 上傳片段 {seg_idx} 到 HuggingFace...")
            upload_video_to_hf(
                HF_REPO_SEGMENT,
                segment_path,
                f"videos/{first_series}/segment_{episode_id}_seg{seg_idx}.mp4"
            )
        
        # 上傳 segment JSON 彙總
        seg_local = CACHE_DIR / f"segment_{episode_id}.json"
        with open(seg_local, "w", encoding="utf-8") as f:
            json.dump(seg_results, f, ensure_ascii=False, indent=2)
        upload_json_to_hf(HF_REPO_SEGMENT, seg_local, f"segment_{episode_id}.json")

        # ===== Episode level =====
        print(f"\n📍 步驟 3: 處理完整集數...")
        
        # 上傳完整影片到 Gemini
        print("  ⬆️  上傳完整集數到 Gemini...")
        client = get_client()
        uploaded_uri = upload_file_if_local(video_path, client)
        
        epi_result = process_episode(
            client=client,
            episode_id=episode_id,
            file_uri=uploaded_uri,
            cache_dir=CACHE_DIR,
            retry_fn=call_with_retry,
        )
        
        # 上傳完整集數影片到 HF
        episode_video_hf_path = f"videos/{first_series}/episode_{episode_id}.mp4"
        print("  📤 上傳完整集數到 HuggingFace...")
        upload_video_to_hf(HF_REPO_EPISODE, Path(video_path), episode_video_hf_path)
        
        # 上傳 episode JSON
        epi_local = CACHE_DIR / f"episode_{episode_id}.json"
        with open(epi_local, "w", encoding="utf-8") as f:
            json.dump(epi_result, f, ensure_ascii=False, indent=2)
        upload_json_to_hf(HF_REPO_EPISODE, epi_local, f"episode_{episode_id}.json")

        processed_episodes.append((episode_id, uploaded_uri, duration_s))
        episode_video_paths.append(video_path)

    # ===== Series level =====
    print(f"\n📍 步驟 4: 處理整季資料...")
    
    client = get_client()
    series_result = process_series(
        client=client,
        series_id=first_series,
        episodes=processed_episodes,
        cache_dir=CACHE_DIR,
        retry_fn=call_with_retry,
    )
    
    # 合併並上傳整季影片（測試模式下只有一集，所以直接複製）
    print("  � 準備整季影片...")
    series_video_path = CACHE_DIR / f"series_{first_series}.mp4"
    if not series_video_path.exists():
        if len(episode_video_paths) == 1:
            # 只有一集，直接複製
            import shutil
            shutil.copy2(episode_video_paths[0], series_video_path)
            print(f"  📋 已複製影片作為整季: {series_video_path.name}")
        else:
            # 多集，需要合併
            print("  🔗 開始合併整季影片...")
            concatenate_videos(episode_video_paths, series_video_path)
    
    print("  📤 上傳整季影片到 HuggingFace...")
    upload_video_to_hf(
        HF_REPO_SERIES,
        series_video_path,
        f"videos/series_{first_series}.mp4"
    )
    
    # 上傳 series JSON
    series_local = CACHE_DIR / f"series_{first_series}.json"
    with open(series_local, "w", encoding="utf-8") as f:
        json.dump(series_result, f, ensure_ascii=False, indent=2)
    upload_json_to_hf(HF_REPO_SERIES, series_local, f"series_{first_series}.json")

    print("\n" + "="*60)
    print("🎉 測試完成！")
    print(f"✅ 處理了 {len(episodes)} 集")
    print(f"✅ 生成了 {len(seg_results)} 個片段")
    print("="*60)


if __name__ == "__main__":
    main()
