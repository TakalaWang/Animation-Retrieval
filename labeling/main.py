import os
import json
import time
import logging
import tempfile
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from datasets import load_dataset, Video
from huggingface_hub import HfApi, create_repo
import google.genai as genai
from moviepy import VideoFileClip

from segment_processor import generate_segment_queries, BlockedContentError
from episode_processor import generate_episode_queries
from series_processor import generate_series_queries
from update_metadata import (
    update_segment_metadata,
    update_episode_metadata,
    update_series_metadata,
)

# ========= 基本設定 =========
load_dotenv()
logging.basicConfig(level=logging.INFO)

GEMINI_KEYS = [k for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not GEMINI_KEYS:
    raise RuntimeError("need GEMINI_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("need HF_TOKEN")

HF_SEG = "TakalaWang/anime-2024-winter-segment-queries"
HF_EP  = "TakalaWang/anime-2024-winter-episode-queries"
HF_SER = "TakalaWang/anime-2024-winter-series-queries"

CACHE_ROOT = Path("cache_gemini_video"); CACHE_ROOT.mkdir(exist_ok=True)
VIDEO_ROOT = CACHE_ROOT / "videos"; VIDEO_ROOT.mkdir(exist_ok=True)
ERROR_LOG = CACHE_ROOT / "error_log.jsonl"

DATASET = "JacobLinCool/anime-2024"
SUBSET = "winter"
SEG_LEN = 60
SEG_OVERLAP = 5

# ========= 小工具 =========
_key_lock = threading.Lock()
_key_idx = 0

def safe_name(s: str) -> str:
    """把 series 名稱變成檔名安全的形式"""
    return s.replace(" ", "_").replace("/", "_").strip()

def make_client() -> genai.Client:
    """
    輪流拿一把 Gemini key
    無論成功或失敗，你每次呼叫這個都會拿到下一把
    """
    global _key_idx
    with _key_lock:
        key = GEMINI_KEYS[_key_idx]
        _key_idx = (_key_idx + 1) % len(GEMINI_KEYS)
        print(f"🔑 使用 Gemini key #{_key_idx}")
    return genai.Client(api_key=key)

def log_error(context: str, error: str):
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with ERROR_LOG.open("a", encoding="utf-8") as f:
        json.dump(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "context": context,
                "error": error,
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

# ======== 判斷要不要重試 ========
def _is_retryable_error(e: Exception) -> bool:
    s = str(e)
    retry_keys = [
        "503",  # Service Unavailable
        "429",  # Too Many Requests
        "UNAVAILABLE",
        "DEADLINE_EXCEEDED",
        "temporarily overloaded",
    ]
    return any(k in s for k in retry_keys)

def _is_fatal_error(e: Exception) -> bool:
    s = str(e)
    fatal_keys = [
        "PERMISSION_DENIED",  # 403，被停用
        "CONSUMER_SUSPENDED",
        "INVALID_ARGUMENT",   # 400
        "The request's total referenced files bytes are too large",
    ]
    return any(k in s for k in fatal_keys)

# ======== 通用重試器：每一輪都換 key ========
def retry(fn_factory, ctx: str, times: int = 5):
    """
    fn_factory: 一個接收 client 的函式，例如 lambda c: c.files.upload(...)
    每次重試都會重新建一個使用下一把 key 的 client
    成功、失敗都會「消耗」掉一把 key，達到平均分配
    """
    last = None
    for i in range(times):
        client = make_client()  # 這裡是關鍵：每一輪都換 client/換 key
        try:
            return fn_factory(client)
        except Exception as e:
            last = e

            if _is_fatal_error(e):
                logging.error(f"{ctx} fatal error: {e}")
                break

            if _is_retryable_error(e):
                # 第一次炸得很正常，給短一點
                wait = 2 if i == 0 else min(3 * (2 ** i), 10)
            else:
                wait = 5

            logging.warning(f"{ctx} 第 {i+1} 次失敗，{wait}s 後換下一把 key 再試：{e}")
            time.sleep(wait)

    log_error(ctx, str(last))
    return None

# ======== 上傳 ========
def upload_file_to_gemini(path: str) -> Optional[str]:
    """
    上傳檔案到 Gemini
    上傳本身也透過 retry，所以每次成功/失敗都會輪 key
    """
    p = Path(path)
    try:
        p.name.encode("ascii")
        up = str(p)
    except UnicodeEncodeError:
        tmp = Path(tempfile.gettempdir()) / f"tmp_{int(time.time()*1000)}{p.suffix}"
        shutil.copy2(p, tmp)
        up = str(tmp)

    # 用 retry，讓它自己換 client
    obj = retry(lambda c: c.files.upload(file=up), f"upload {path}")
    if not obj:
        return None

    # 等待處理完成：這裡也可以換 client 來 get
    while obj.state.name == "PROCESSING":
        time.sleep(5)
        client = make_client()
        obj = client.files.get(name=obj.name)

    if obj.state.name == "FAILED":
        log_error(f"gemini processing {path}", "state=FAILED")
        return None

    return obj.uri

# ========= episode 裡面用的 =========
def process_segments(series: str, ep: str, video_path: str, date: Any):
    s = safe_name(series)
    series_dir = VIDEO_ROOT / s
    series_dir.mkdir(parents=True, exist_ok=True)

    with VideoFileClip(video_path) as v:
        dur = v.duration

    start = 0
    idx = 0
    while start < dur - 5:
        end = min(start + SEG_LEN, dur)
        seg_mp4  = series_dir / f"segment_{s}_{ep}_seg{idx}.mp4"
        seg_json = series_dir / f"segment_{s}_{ep}_seg{idx}.json"
        hf_path  = f"videos/{s}/segment_{s}_{ep}_seg{idx}.mp4"

        if not seg_mp4.exists():
            with VideoFileClip(video_path) as v:
                v.subclipped(start, end).write_videofile(
                    str(seg_mp4),
                    codec="libx264",
                    audio_codec="aac",
                    logger=None,
                )

        if not seg_json.exists():
            file_uri = upload_file_to_gemini(str(seg_mp4))
            if not file_uri:
                log_error(f"segment upload {series} {ep} seg{idx}", "upload to gemini failed")
            else:
                # 這裡也用 retry，每一段都會平均使用不同 key
                def _call_segment(c):
                    return generate_segment_queries(client=c, file_uri=file_uri)

                q = retry(_call_segment, f"segment gen {series} {ep} seg{idx}")
                if q is not None:
                    seg_json.write_text(json.dumps({
                        "series_name": series,
                        "episode_id": ep,
                        "segment_index": idx,
                        "release_date": date,
                        "file_name": hf_path,
                        "query": q,
                    }, ensure_ascii=False, indent=2), encoding="utf-8")

        start += SEG_LEN - SEG_OVERLAP
        idx += 1

def process_episode(series: str, ep: str, video_path: str, date: Any):
    s = safe_name(series)
    series_dir = VIDEO_ROOT / s
    series_dir.mkdir(parents=True, exist_ok=True)

    ep_json = series_dir / f"episode_{s}_{ep}.json"
    ep_mp4  = series_dir / f"episode_{s}_{ep}.mp4"
    hf_path = f"videos/{s}/episode_{s}_{ep}.mp4"

    if not ep_mp4.exists():
        shutil.copy2(video_path, ep_mp4)

    if not ep_json.exists():
        file_uri = upload_file_to_gemini(str(ep_mp4))
        if not file_uri:
            log_error(f"episode upload {series} {ep}", "upload to gemini failed")
        else:
            def _call_episode(c):
                return generate_episode_queries(client=c, file_uri=file_uri)

            q = retry(_call_episode, f"episode {series} {ep}")
            if q is not None:
                ep_json.write_text(json.dumps({
                    "file_name": hf_path,
                    "series_name": series,
                    "episode_id": ep,
                    "release_date": date,
                    "query": q,
                }, ensure_ascii=False, indent=2), encoding="utf-8")

def run_one_episode(series: str, ep_info: Dict[str, Any]):
    ep_id = ep_info["episode_id"]
    video = ep_info["video_path"]
    date  = ep_info.get("release_date")
    process_segments(series, ep_id, video, date)
    process_episode(series, ep_id, video, date)

# ========= series 上傳 =========
def upload_one_series(series: str):
    api = HfApi(token=HF_TOKEN)
    s = safe_name(series)

    api.upload_large_folder(
        folder_path=str(CACHE_ROOT),
        repo_id=HF_SEG,
        repo_type="dataset",
        allow_patterns=[f"videos/{s}/segment_{s}_*.mp4"],
        commit_message=f"{series} segments batch",
    )

    api.upload_large_folder(
        folder_path=str(CACHE_ROOT),
        repo_id=HF_EP,
        repo_type="dataset",
        allow_patterns=[f"videos/{s}/episode_{s}_*.mp4"],
        commit_message=f"{series} episodes batch",
    )

    update_segment_metadata(HF_TOKEN)
    update_episode_metadata(HF_TOKEN)
    logging.info(f"✅ uploaded whole series {series}")

# ========= series-level =========
def process_series(series: str, eps: List[Dict[str, Any]]):
    s = safe_name(series)
    series_dir = VIDEO_ROOT / s
    series_dir.mkdir(parents=True, exist_ok=True)

    series_json = series_dir / f"series_{s}.json"
    if series_json.exists():
        return

    series_mp4 = series_dir / f"series_{s}.mp4"
    if not series_mp4.exists():
        txt = series_dir / f"series_{s}.txt"
        with txt.open("w") as f:
            for e in eps:
                f.write(f"file '{Path(e['video_path']).absolute()}'\n")
        subprocess.run(
            ["ffmpeg","-f","concat","-safe","0","-i",str(txt),"-c","copy",str(series_mp4)],
            check=True
        )
        txt.unlink()

    low = series_dir / f"series_{s}_low_fps.mp4"
    if not low.exists():
        subprocess.run([
            "ffmpeg","-y","-i",str(series_mp4),
            "-vf","fps=0.2","-an","-c:v","libx264","-crf","32","-preset","veryfast",
            str(low)
        ], check=True)

    file_uri = upload_file_to_gemini(str(low))
    if file_uri:
        def _call_series(c):
            return generate_series_queries(client=c, file_uri=file_uri)
        time.sleep(1)
        series_query = retry(_call_series, f"series {series}") or {"error": "gen failed"}
    else:
        log_error(f"series upload {series}", "upload to gemini failed")
        series_query = {"error": "upload failed"}

    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(series_mp4),
        repo_id=HF_SER,
        path_in_repo=f"videos/{s}/series_{s}.mp4",
        repo_type="dataset",
    )

    series_json.write_text(json.dumps({
        "file_name": f"videos/{s}/series_{s}.mp4",
        "series_name": series,
        "query": series_query,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    update_series_metadata(HF_TOKEN)

# ========= dataset =========
def load_and_group_dataset() -> Dict[str, List[Dict[str, Any]]]:
    ds = load_dataset(DATASET, SUBSET, split="train").cast_column("video", Video(decode=False))
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in ds:
        groups.setdefault(r["series_name"], []).append({
            "episode_id": r["episode_name"],
            "series_name": r["series_name"],
            "video_path": r["video"]["path"],
            "release_date": r.get("release_date"),
        })
    return groups

# ========= main =========
def main():
    # 確保 HF repo 存在
    for r in [HF_SEG, HF_EP, HF_SER]:
        create_repo(r, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

    groups = load_and_group_dataset()

    for series, eps in groups.items():
        logging.info(f"=== {series} ===")

        # 跑這個 series 的所有 episode
        for ep in eps:
            run_one_episode(series, ep)

        # 上傳這個 series 的 segment/episode
        upload_one_series(series)

        # 再做 series-level
        try:
            eps_sorted = sorted(eps, key=lambda e: float(e["episode_id"]))
        except Exception:
            eps_sorted = eps
        process_series(series, eps_sorted)

    logging.info("✅ all done")

if __name__ == "__main__":
    main()
