# simplified_main_autoupdate.py
import os, json, time, logging, shutil, tempfile, subprocess
from pathlib import Path
from typing import Any, Dict, List

from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset, Dataset, Video
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

# ================== 基本設定 ==================
logging.basicConfig(level=logging.INFO)
load_dotenv()

GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not GEMINI_API_KEYS:
    raise RuntimeError("請先設定 GEMINI_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("請先設定 HF_TOKEN")

current_key_index = 0
def get_next_api_key():
    global current_key_index
    key = GEMINI_API_KEYS[current_key_index]
    current_key_index = (current_key_index + 1) % len(GEMINI_API_KEYS)
    return key
def get_client(): return genai.Client(api_key=get_next_api_key())

# Hugging Face repositories
HF_REPO_SEGMENT = "TakalaWang/anime-2024-winter-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-winter-episode-queries"
HF_REPO_SERIES  = "TakalaWang/anime-2024-winter-series-queries"

CACHE_DIR = Path("./cache_gemini_video"); CACHE_DIR.mkdir(parents=True, exist_ok=True)

TEST_DATASET = "JacobLinCool/anime-2024"
TEST_SPLIT = "winter"
SEGMENT_LENGTH, SEGMENT_OVERLAP = 60, 5
MAX_RETRIES, RETRY_SLEEP = 5, 5

# ================== 共用工具 ==================
def call_with_retry(fn, *a, **kw):
    for _ in range(MAX_RETRIES):
        try:
            return fn(*a, **kw)
        except Exception as e:
            print(f"❌ [error] {type(e).__name__}: {e}")
            time.sleep(RETRY_SLEEP)
    raise RuntimeError("重試次數已達上限")

def down_video_fps(src: Path, dst: Path):
    subprocess.run([
        "ffmpeg","-y","-i",str(src),
        "-vf","fps=0.2","-an","-c:v","libx264","-preset","veryfast",str(dst)
    ], check=True)

def upload_video_to_gemini(client: genai.Client, video_path: str) -> str:
    path = Path(video_path)
    try:
        path.name.encode("ascii")
        upload_path = str(path)
    except UnicodeEncodeError:
        tmp = Path(tempfile.gettempdir()) / f"temp_{int(time.time()*1000)}{path.suffix}"
        shutil.copy2(video_path, tmp)
        upload_path = str(tmp)
    uploaded = call_with_retry(lambda: client.files.upload(file=upload_path))
    while uploaded.state.name == "PROCESSING":
        time.sleep(5); uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED": raise RuntimeError("影片處理失敗")
    return uploaded.uri

def upload_video_to_hf(repo: str, file: Path, repo_path: str):
    api = HfApi(token=HF_TOKEN)
    api.upload_file(
        path_or_fileobj=str(file),
        repo_id=repo,
        path_in_repo=repo_path,
        repo_type="dataset",
    )

# ================== 處理流程 ==================
def load_and_group_dataset() -> Dict[str, List[Dict[str, Any]]]:
    ds = load_dataset(TEST_DATASET, TEST_SPLIT, split="train").cast_column("video", Video(decode=False))
    groups = {}
    for r in ds:
        groups.setdefault(r["series_name"], []).append({
            "episode_id": r["episode_name"],
            "series_name": r["series_name"],
            "video_path": r["video"]["path"],
            "release_date": r.get("release_date")
        })
    return groups


def process_segments(series: str, episode: str, path: str, dur: float, date: Any):
    """產生 segment-level metadata，若有新資料則更新 HF"""
    updated = False
    results = []
    start = 0
    while start < dur - 5:
        end = min(start + SEGMENT_LENGTH, dur)
        safe = series.replace(" ", "_").replace("/", "_")
        seg_path = CACHE_DIR / f"segment_{safe}_{episode}_seg{len(results)}.mp4"
        cache_path = seg_path.with_suffix(".json")

        if not seg_path.exists():
            with VideoFileClip(path) as v:
                v.subclipped(start, end).write_videofile(str(seg_path), codec="libx264", audio_codec="aac", logger=None)

        # 如果已有 cache，跳過
        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            results.append(record)
            start += SEGMENT_LENGTH - SEGMENT_OVERLAP
            continue

        def gen():
            c = get_client()
            file_uri = upload_video_to_gemini(c, str(seg_path))
            return generate_segment_queries(client=c, file_uri=file_uri)
        try:
            data = call_with_retry(gen)
        except BlockedContentError:
            data = {k: ["內容被阻止"]*3 for k in ["visual_saliency","character_emotion","action_behavior","dialogue","symbolic_scene"]}

        record = {
            "series_name": series,
            "episode_id": episode,
            "segment_index": len(results),
            "release_date": date,
            "file_name": f"videos/{safe}/segment_{safe}_{episode}_seg{len(results)}.mp4",
            "query": data,
        }
        results.append(record)
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        upload_video_to_hf(HF_REPO_SEGMENT, seg_path, record["file_name"])
        updated = True
        start += SEGMENT_LENGTH - SEGMENT_OVERLAP

    if updated:
        update_segment_metadata(HF_TOKEN)
    return results


def process_episode(series: str, episode: str, path: str, date: Any):
    """產生 episode-level metadata，若有更新即更新 HF"""
    safe = series.replace(" ", "_").replace("/", "_")
    cache_path = CACHE_DIR / f"episode_{safe}_{episode}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def gen():
        c = get_client()
        file_uri = upload_video_to_gemini(c, str(path))
        return generate_episode_queries(client=c, file_uri=file_uri)
    query = call_with_retry(gen)

    hf_path = f"videos/{safe}/episode_{safe}_{episode}.mp4"
    upload_video_to_hf(HF_REPO_EPISODE, Path(path), hf_path)

    record = {"file_name": hf_path, "series_name": series, "episode_id": episode, "release_date": date, "query": query}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    update_episode_metadata(HF_TOKEN)
    return record


def process_series(series: str, episodes: List[Dict[str, Any]]):
    """產生 series-level metadata，若有更新即更新 HF"""
    safe = series.replace(" ", "_").replace("/", "_")
    cache_path = CACHE_DIR / f"series_{safe}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    series_video = CACHE_DIR / f"series_{safe}.mp4"
    if not series_video.exists():
        with open(series_video.with_suffix(".txt"), "w") as f:
            for e in episodes: f.write(f"file '{Path(e['video_path']).absolute()}'\n")
        subprocess.run(["ffmpeg","-f","concat","-safe","0","-i",str(series_video.with_suffix('.txt')),"-c","copy",str(series_video)], check=True)

    low = CACHE_DIR / f"series_{safe}_lowfps.mp4"
    if not low.exists(): down_video_fps(series_video, low)

    def gen():
        c = get_client()
        file_uri = upload_video_to_gemini(c, str(low))
        return generate_series_queries(client=c, file_uri=file_uri)
    query = call_with_retry(gen)
    upload_video_to_hf(HF_REPO_SERIES, series_video, f"videos/series_{safe}.mp4")

    date = sorted({e["release_date"] for e in episodes if e.get("release_date")})
    record = {"file_name": f"videos/series_{safe}.mp4","series_name": series,"release_date": date[0] if date else None,"query": query}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    update_series_metadata(HF_TOKEN)
    return record


# ================== 主程式 ==================
def main():
    print(f"🔑 使用 {len(GEMINI_API_KEYS)} 個 Gemini API Keys")
    for r in [HF_REPO_SEGMENT, HF_REPO_EPISODE, HF_REPO_SERIES]:
        create_repo(r, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

    groups = load_and_group_dataset()
    groups = dict(list(groups.items())[:5])  # 限制前 5 個系列

    for sname, eps in groups.items():
        print(f"🎬 系列: {sname}")
        for ep in eps:
            with VideoFileClip(ep["video_path"]) as v: dur = v.duration
            _ = process_segments(sname, ep["episode_id"], ep["video_path"], dur, ep.get("release_date"))
            _ = process_episode(sname, ep["episode_id"], ep["video_path"], ep.get("release_date"))
        _ = process_series(sname, eps)

    print("✅ 所有系列處理完成，metadata 皆已自動更新。")


if __name__ == "__main__":
    main()
