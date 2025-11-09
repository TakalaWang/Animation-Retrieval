import os
import json
import time
import logging
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple

from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, Video
from huggingface_hub import HfApi, create_repo

import google.genai as genai
from moviepy import VideoFileClip


from segment_processor import generate_segment_queries, BlockedContentError
from episode_processor import generate_episode_queries
from series_processor import generate_series_queries
from update_metadata import update_segment_metadata, update_episode_metadata, update_series_metadata


# ================== 基本設定 ==================
logging.basicConfig(level=logging.INFO)
load_dotenv()

# 支援單個或多個 API Key（用逗號分隔或 JSON 陣列）
GEMINI_API_KEYS = [
    k.strip() for k in os.getenv("GEMINI_API_KEY", "").strip().split(",") if k.strip()
]

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
MAX_RETRIES = 5      # 最大重試次數
RETRY_SLEEP = 5    # 重試等待秒數


def get_client():
    """獲取 Gemini client（使用輪換的 API key）"""
    return genai.Client(api_key=get_next_api_key())


# ================== Hugging Face 工具函數 ==================
def ensure_hf_repos():
    """確保 Hugging Face 倉庫存在"""
    create_repo(HF_REPO_SEGMENT, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    create_repo(HF_REPO_EPISODE, token=HF_TOKEN, repo_type="dataset", exist_ok=True)
    create_repo(HF_REPO_SERIES, token=HF_TOKEN, repo_type="dataset", exist_ok=True)


def upload_dataset_to_hf(repo_id: str, data: List[Dict[str, Any]]):
    """上傳數據列表到 HF dataset 格式，啟用 data viewer"""

    # 創建 Dataset 對象（直接從字典列表，保留嵌套結構）
    dataset = Dataset.from_list(data)

    # 上傳到 HF
    dataset.push_to_hub(repo_id, token=HF_TOKEN)

    print(f"📊 已上傳 dataset 到 HF: {repo_id} ({len(data)} 筆記錄)")


# ================== 數據集管理 ==================
def create_metadata_jsonl(
    repo_id: str,
    metadata_list: List[Dict[str, Any]],
    metadata_filename: str = "metadata.jsonl",
):
    """創建 metadata.jsonl 文件並上傳到 HF dataset，啟用 data viewer"""
    # 處理每筆記錄，確保格式正確
    processed_metadata = []
    for item in metadata_list:
        # 創建新的記錄，包含所有字段
        record = {}
        
        # 確保有 file_name
        if "file_name" not in item:
            if "episode_name" in item:
                item["file_name"] = (
                    f"videos/{item.get('series_name', 'unknown')}/episode_{item['episode_name']}.mp4"
                )
            elif "segment_index" in item and "episode_id" in item:
                item["file_name"] = (
                    f"videos/segment_{item['episode_id']}_seg{item['segment_index']}.mp4"
                )
            else:
                print(f"⚠️  警告: 跳過沒有 file_name 的記錄: {item.keys()}")
                continue

        # 複製所有字段到新記錄
        for key, value in item.items():
            record[key] = value

        processed_metadata.append(record)

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

    print(
        f"📋 已創建並上傳 metadata.jsonl 到 {repo_id} ({len(processed_metadata)} 筆記錄)"
    )
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
def extract_video_segment(
    video_path: str, start_s: float, end_s: float, output_path: Path
):
    with VideoFileClip(video_path) as video:
        segment = video.subclipped(start_s, end_s)
        segment.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(
                output_path.parent / f"temp_{output_path.stem}_audio.m4a"
            ),
            remove_temp=True,
            logger=None,
        )


def down_video_fps(src_path: Path, dst_path: Path):
    """
    把已經 concat 好的原始大檔壓一個小版，專門給 Gemini 用
    - 降 FPS
    - 可選擇降解析度
    """
    dst_path = Path(dst_path)
    cmd = [
        "ffmpeg",
        "-y",                       # 自動覆寫
        "-i", str(src_path),        # 輸入影片
        "-vf", "fps=0.2",
        "-an",                      # 移除音訊
        "-c:v", "libx264",          # 使用 x264 編碼
        "-preset", "veryfast",      # 加快編碼速度（稍降壓縮效率）
        str(dst_path),
    ]
    subprocess.run(cmd, check=True)


def get_video_duration_from_path(path: str) -> float:
    """獲取影片長度（秒）"""
    with VideoFileClip(path) as clip:
        return clip.duration


# ================== API 工具函數 ==================
def upload_video_to_gemini(client: genai.Client, video_path: str) -> str:
    """
    上傳影片到 Gemini API 並等待處理完成
    
    Args:
        client: Gemini API 客戶端
        video_path: 影片檔案路徑
        
    Returns:
        file_uri: Gemini 處理完成的檔案 URI
    """
    original_path = Path(video_path)
    print(f"    📤 上傳影片: {original_path.name}")
    
    # 檢查檔名是否包含非 ASCII 字符（如中文）
    filename = original_path.name
    temp_file = None
    upload_path = video_path
    
    try:
        # 嘗試用 ASCII 編碼檔名，如果失敗則表示有非 ASCII 字符
        filename.encode('ascii')
    except UnicodeEncodeError:
        # 檔名包含中文或其他非 ASCII 字符，需要創建臨時副本
        
        # 使用檔案的後綴名和一個安全的 ASCII 名稱
        safe_name = f"temp_upload_{int(time.time() * 1000)}{original_path.suffix}"
        temp_file = Path(tempfile.gettempdir()) / safe_name
        
        # 複製檔案到臨時位置
        shutil.copy2(video_path, temp_file)
        upload_path = str(temp_file)
    
    try:
        # 每次都重新上傳，避免文件過期問題
        def do_upload():
            return client.files.upload(file=upload_path)
        uploaded = call_with_retry(do_upload)
        file_uri = uploaded.uri
        
        while uploaded.state.name == "PROCESSING":
            time.sleep(5)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name == "FAILED":
            raise ValueError(f"影片處理失敗: {uploaded.state.name}")
        
        print(f"    ✅ 完成")
        return file_uri
    
    finally:
        # 清理臨時檔案
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
                print(f"    🗑️  已刪除臨時檔案")
            except Exception as e:
                print(f"    ⚠️  無法刪除臨時檔案: {e}")


def call_with_retry(fn, *args, **kwargs):
    """執行 API 呼叫，失敗時自動更換 Gemini Key 並重試"""
    for _ in range(MAX_RETRIES):
        try:
            # 如果 fn 是一個無參數的閉包函數，直接調用
            if callable(fn) and not args and not kwargs:
                return fn()
            else:
                return fn(*args, **kwargs)
            
        except Exception as e:
            print(f"❌ [error] {type(e).__name__}: {e}")
            time.sleep(RETRY_SLEEP)
            continue

    raise RuntimeError(f"重試次數已達上限 ({MAX_RETRIES})，仍未成功。")


# ================== 資料處理函數 ==================
def load_and_group_dataset() -> Dict[str, List[Dict[str, Any]]]:
    print("載入資料集...")
    ds = load_dataset(TEST_DATASET, TEST_SPLIT, split="train")
    ds = ds.cast_column("video", Video(decode=False))

    series_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in tqdm(ds, desc="group by series"):
        series_name = row["series_name"]
        episode_id = row["episode_name"]
        video_path = row["video"]["path"]
        release_date = row.get("release_date")

        series_groups.setdefault(series_name, []).append(
            {
                "episode_id": episode_id,
                "series_name": series_name,
                "video_path": video_path,
                "release_date": release_date,
            }
        )

    return series_groups


def process_segments_for_episode(
    series_name: str,
    episode_id: str,
    video_path: str,
    duration_s: float,
    release_date: Any,
) -> List[Dict[str, Any]]:
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
    for seg_idx, (s, e) in enumerate(
        tqdm(segment_ranges, desc=f"切割片段 {episode_id}", unit="seg")
    ):
        # 使用 series_name 避免檔名衝突
        safe_series = series_name.replace(" ", "_").replace("/", "_")
        segment_video_path = CACHE_DIR / f"segment_{safe_series}_{episode_id}_seg{seg_idx}.mp4"
        if not segment_video_path.exists():
            extract_video_segment(video_path, s, e, segment_video_path)

        segment_files.append(
            {
                "index": seg_idx,
                "path": segment_video_path,
            }
        )

    seg_results = []
    for seg_info in segment_files:
        seg_idx = seg_info["index"]
        segment_path = seg_info["path"]

        safe_series = series_name.replace(" ", "_").replace("/", "_")
        cache_path = CACHE_DIR / f"segment_{safe_series}_{episode_id}_seg{seg_idx}.json"
        if cache_path.exists():
            # 就算有 cache，也幫它補上 series_name / release_date，避免舊檔是空的
            with open(cache_path, "r", encoding="utf-8") as f:
                cached = json.load(f)
            cached["series_name"] = series_name
            cached["release_date"] = release_date
            if "file_name" not in cached:
                cached["file_name"] = f"videos/segment_{episode_id}_seg{seg_idx}.mp4"
            seg_results.append(cached)
            # 回寫一次，讓檔案也變成新的
            with open(cache_path, "w", encoding="utf-8") as f:
                json.dump(cached, f, ensure_ascii=False, indent=2)
            continue

        # 使用 retry 包装上传和查询生成
        def process_segment():
            client = get_client()
            file_uri = upload_video_to_gemini(client, str(segment_path))
            return generate_segment_queries(client=client, file_uri=file_uri)
        
        try:
            data = call_with_retry(process_segment)
        except BlockedContentError as e:
            print(f"⚠️  片段 {seg_idx} 被阻止，跳過: {e}")
            # 創建一個空的查詢記錄
            data = {
                "visual_saliency": ["內容被阻止"] * 3,
                "character_emotion": ["內容被阻止"] * 3,
                "action_behavior": ["內容被阻止"] * 3,
                "dialogue": ["內容被阻止"] * 3,
                "symbolic_scene": ["內容被阻止"] * 3,
            }

        record = {
            "series_name": series_name,
            "episode_id": episode_id,
            "segment_index": seg_idx,
            "release_date": release_date,
            "file_name": f"videos/{safe_series}/segment_{safe_series}_{episode_id}_seg{seg_idx}.mp4",
            "query": data,  # 改名為 query
        }

        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)

        seg_results.append(record)

        # 上傳影片到 HF
        upload_video_to_hf(
            HF_REPO_SEGMENT,
            segment_path,
            record["file_name"]
        )

    return seg_results


def process_episode_level(
    series_name: str,
    episode_id: str,
    video_path: str,
    duration_s: float,
    release_date: Any,
) -> Dict[str, Any]:
    """處理集數級別的查詢生成"""
    
    # 使用 retry 包装上传和查询生成
    def process_episode():
        client = get_client()
        file_uri = upload_video_to_gemini(client, str(video_path))
        return generate_episode_queries(client=client, file_uri=file_uri)
    
    epi_result = call_with_retry(process_episode)

    # 上傳完整集數影片到 HF
    safe_series = series_name.replace(" ", "_").replace("/", "_")
    episode_video_hf_path = f"videos/{safe_series}/episode_{safe_series}_{episode_id}.mp4"
    print("  📤 上傳完整集數到 HuggingFace...")
    upload_video_to_hf(HF_REPO_EPISODE, Path(video_path), episode_video_hf_path)

    # 包裝 episode metadata 與模型回應
    episode_record = {
        "file_name": episode_video_hf_path,  # 添加 file_name 字段用於 data viewer
        "series_name": series_name,
        "episode_id": episode_id,  # 改名為 episode_id
        "release_date": release_date,
        "query": epi_result,  # 改名為 query
    }

    # 仍然保存單個 JSON 文件（向後兼容）
    epi_local = CACHE_DIR / f"episode_{safe_series}_{episode_id}.json"
    with open(epi_local, "w", encoding="utf-8") as f:
        json.dump(episode_record, f, ensure_ascii=False, indent=2)
    update_episode_metadata(HF_TOKEN)

    return episode_record


def process_single_episode(
    series_name: str, episode_info: Dict[str, Any]
) -> Tuple[str, str, float, Any, Dict[str, Any], List[Dict[str, Any]]]:
    episode_id = episode_info["episode_id"]
    video_path = episode_info["video_path"]
    release_date = episode_info.get("release_date")
    duration_s = get_video_duration_from_path(video_path)

    safe_series = series_name.replace(" ", "_").replace("/", "_")
    epi_local = CACHE_DIR / f"episode_{safe_series}_{episode_id}.json"
    seg_local = CACHE_DIR / f"segment_{safe_series}_{episode_id}.json"

    if epi_local.exists() and seg_local.exists():
        with open(epi_local, "r", encoding="utf-8") as f:
            episode_record = json.load(f)
        with open(seg_local, "r", encoding="utf-8") as f:
            seg_results = json.load(f)
        
        return (
            episode_id,
            video_path,
            duration_s,
            release_date,
            episode_record,
            seg_results,
        )

    print(f"\n{'='*60}")
    print(f"處理集數: {episode_id}")
    print(f"影片長度: {duration_s:.2f} 秒")
    print(f"{'='*60}\n")

    # ===== Segment level =====
    seg_results = process_segments_for_episode(
        series_name,
        episode_id,
        video_path,
        duration_s,
        release_date,
    )

    safe_series = series_name.replace(" ", "_").replace("/", "_")
    seg_local = CACHE_DIR / f"segment_{safe_series}_{episode_id}.json"
    with open(seg_local, "w", encoding="utf-8") as f:
        json.dump(seg_results, f, ensure_ascii=False, indent=2)
    update_segment_metadata(HF_TOKEN)

    # ===== Episode level =====
    episode_record = process_episode_level(
        series_name,
        episode_id,
        video_path,
        duration_s,
        release_date,
    )

    return (
        episode_id,
        video_path,
        duration_s,
        release_date,
        episode_record,
        seg_results,
    )



def process_series_level(
    series_name: str, processed_episodes: List[Tuple[str, str, float, Any]]
) -> Dict[str, Any]:
    """處理系列級別的查詢生成"""

    safe_series = series_name.replace(" ", "_").replace("/", "_")
    series_local = CACHE_DIR / f"series_{safe_series}.json"
    if series_local.exists():
        print(f"✅ series {series_name} 已有快取，略過重新生成")
        with open(series_local, "r", encoding="utf-8") as f:
            series_record = json.load(f)
        return series_record
    
    # 合併並上傳整季影片
    print("  準備整季影片...")
    episode_video_paths = [vp for _, vp, _, _ in processed_episodes]
    safe_series = series_name.replace(" ", "_").replace("/", "_")
    series_video_path = CACHE_DIR / f"series_{safe_series}.mp4"
    if not series_video_path.exists():
        print("  🔗 開始合併整季影片...")

        tmp_list = series_video_path.with_suffix(".txt")
        with open(tmp_list, "w", encoding="utf-8") as f:
            for p in episode_video_paths:
                f.write(f"file '{Path(p).absolute()}'\n")
        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(tmp_list),
            "-c", "copy",
            str(series_video_path),
        ]
        subprocess.run(cmd, check=True)
        tmp_list.unlink(missing_ok=True)

    low_fps_series_video_path = CACHE_DIR / f"series_{safe_series}_low_fps.mp4"
    if not low_fps_series_video_path.exists():
        down_video_fps(series_video_path, low_fps_series_video_path)

    # 使用 retry 包装上传和查询生成
    def process_series():
        client = get_client()
        file_uri = upload_video_to_gemini(client, str(low_fps_series_video_path))
        return generate_series_queries(client=client, file_uri=file_uri)
    
    series_result = call_with_retry(process_series)

    print("  📤 上傳整季影片到 HuggingFace...")
    upload_video_to_hf(
        HF_REPO_SERIES, series_video_path, f"videos/series_{safe_series}.mp4"
    )

    # 建立 series metadata（僅保留必要欄位）
    release_dates = sorted(
        {rd for (_eid, _vp, _dur, rd) in processed_episodes if rd is not None}
    )
    series_record = {
        "file_name": f"videos/series_{safe_series}.mp4",  # 添加 file_name 字段用於 data viewer
        "series_name": series_name,
        "release_date": release_dates[0] if release_dates else None,  # 使用首播日期
        "query": series_result,  # 改名為 query
    }

    with open(series_local, "w", encoding="utf-8") as f:
        json.dump(series_record, f, ensure_ascii=False, indent=2)
    update_series_metadata(HF_TOKEN)

    return series_record


def process_single_series(series_name: str, episodes: List[Dict[str, Any]]) -> Tuple[
    Tuple[str, List[Tuple[str, str, float, Any]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    """處理單個系列的所有集數，返回 (result, episode_metadata, segment_metadata, series_metadata)"""
    print(f"開始處理系列: {series_name} (共 {len(episodes)} 集)")

    processed_episodes = []
    episode_metadata = []
    segment_metadata = []

    for episode_info in tqdm(episodes, desc=f"episodes - {series_name}", unit="ep"):
        episode_id, video_path, duration_s, release_date, episode_record, seg_results = (
            process_single_episode(series_name, episode_info)
        )
        processed_episodes.append((episode_id, video_path, duration_s, release_date))
        episode_metadata.append(episode_record)
        segment_metadata.extend(seg_results)

    # 處理系列級別
    series_record = process_series_level(series_name, processed_episodes)

    return (
        (series_name, processed_episodes),
        episode_metadata,
        segment_metadata,
        series_record,
    )


def process_all_series(
    series_groups: Dict[str, List[Dict[str, Any]]],
) -> Tuple[
    List[Tuple[str, List[Tuple[str, str, float, Any]]]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """處理所有系列，返回 (processed_series, all_episode_metadata, all_segment_metadata, all_series_metadata)"""
    processed_series = []
    all_episode_metadata = []
    all_segment_metadata = []
    all_series_metadata = []

    for series_name, episodes in series_groups.items():
        result, episode_metadata, segment_metadata, series_metadata = (
            process_single_series(series_name, episodes)
        )
        processed_series.append(result)
        all_episode_metadata.extend(episode_metadata)
        all_segment_metadata.extend(segment_metadata)
        all_series_metadata.append(series_metadata)

    return (
        processed_series,
        all_episode_metadata,
        all_segment_metadata,
        all_series_metadata,
    )


# ================== 主程式 ==================
def main():
    """主程式入口"""
    ensure_hf_repos()

    print(f"已設定 {len(GEMINI_API_KEYS)} 個 Gemini API Key")

    # 載入並分組資料集
    series_groups = load_and_group_dataset()
    series_groups = {k: series_groups[k] for k in list(series_groups)[:5]}

    # 處理所有系列與 episodes
    (
        processed_series,
        all_episode_metadata,
        all_segment_metadata,
        all_series_metadata,
    ) = process_all_series(series_groups)

    # 創建 metadata.jsonl 文件以啟用 Dataset Viewer
    if all_episode_metadata:
        create_metadata_jsonl(HF_REPO_EPISODE, all_episode_metadata, "metadata.jsonl")

    if all_segment_metadata:
        create_metadata_jsonl(HF_REPO_SEGMENT, all_segment_metadata, "metadata.jsonl")

    if all_series_metadata:
        create_metadata_jsonl(HF_REPO_SERIES, all_series_metadata, "metadata.jsonl")

    # 完成總結
    total_series = len(processed_series)
    total_episodes = sum(len(eps) for _, eps in processed_series)

    print("\n" + "=" * 60)
    print("🎉 處理完成！")
    print(f"✅ 處理了 {total_series} 個系列，{total_episodes} 集")
    print(f"📊 Episode metadata: {len(all_episode_metadata)} 筆")
    print(f"📊 Segment metadata: {len(all_segment_metadata)} 筆")
    print("🔍 Dataset Viewer 現已啟用！")
    print("=" * 60)


if __name__ == "__main__":
    main()
