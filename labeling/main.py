import os, json, time, logging, shutil, tempfile, subprocess, threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
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

# ================== 基本設定 ==================
logging.basicConfig(level=logging.INFO)
load_dotenv()

GEMINI_API_KEYS = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
HF_TOKEN = os.getenv("HF_TOKEN", "")
if not GEMINI_API_KEYS:
    raise RuntimeError("請先設定 GEMINI_API_KEY")
if not HF_TOKEN:
    raise RuntimeError("請先設定 HF_TOKEN")

_api_key_lock = threading.Lock()
_current_key_index = 0

def get_next_api_key():
    global _current_key_index
    with _api_key_lock:
        key = GEMINI_API_KEYS[_current_key_index]
        _current_key_index = (_current_key_index + 1) % len(GEMINI_API_KEYS)
    return key

def get_client():
    return genai.Client(api_key=get_next_api_key())

# Hugging Face repositories
HF_REPO_SEGMENT = "TakalaWang/anime-2024-winter-segment-queries"
HF_REPO_EPISODE = "TakalaWang/anime-2024-winter-episode-queries"
HF_REPO_SERIES  = "TakalaWang/anime-2024-winter-series-queries"

CACHE_DIR = Path("./cache_gemini_video"); CACHE_DIR.mkdir(parents=True, exist_ok=True)
ERROR_LOG = Path("./error_log.jsonl")

TEST_DATASET = "JacobLinCool/anime-2024"
TEST_SPLIT = "winter"
SEGMENT_LENGTH, SEGMENT_OVERLAP = 60, 5

# retry 相關
MAX_RETRIES = 5
BASE_RETRY_SLEEP = 10  # sec
QUOTA_BACKOFF_MULTIPLIER = 10

# 執行緒數量，可依照你機器調
MAX_WORKERS = 4

# ================== 共用工具 ==================
def call_with_retry(fn, *a, **kw):
    """
    quota/429: 每次重試睡更久
    其他錯誤: 用固定的 BASE_RETRY_SLEEP
    """
    context = kw.pop("context", "未知任務")
    print(f"Processing {context}")
    last_exc = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            return fn(*a, **kw)
        except Exception as e:
            last_exc = e
            msg = str(e)
            # 粗略偵測 quota / 429 / RESOURCE_EXHAUSTED
            is_quota = ("429" in msg) or ("quota" in msg.lower()) or ("RESOURCE_EXHAUSTED" in msg)
            if is_quota:
                sleep_sec = BASE_RETRY_SLEEP * (QUOTA_BACKOFF_MULTIPLIER ** (attempt - 1))
                print(f"⚠️ quota 類錯誤，第 {attempt} 次，{sleep_sec}s 後再試：{e}")
            else:
                sleep_sec = BASE_RETRY_SLEEP
                print(f"❌ 非 quota 錯誤，第 {attempt} 次：{e}")

            time.sleep(sleep_sec)

    # 全部失敗才寫 log
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        json.dump(
            {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "context": context,
                "error": str(last_exc),
            },
            f,
            ensure_ascii=False,
        )
        f.write("\n")

    print(f"🚨 這筆一直失敗，請看 error_log.jsonl：{context}")
    return None

def down_video_fps(src: Path, dst: Path):
    subprocess.run([
        "ffmpeg","-y","-i",str(src),
        "-vf","fps=0.2","-an","-c:v","libx264","-crf","32","-preset","veryfast",str(dst)
    ], check=True)

def upload_video_to_gemini(client: genai.Client, video_path: str) -> Tuple[Optional[str], Optional[str]]:
    path = Path(video_path)
    try:
        path.name.encode("ascii")
        upload_path = str(path)
    except UnicodeEncodeError:
        tmp = Path(tempfile.gettempdir()) / f"temp_{int(time.time()*1000)}{path.suffix}"
        shutil.copy2(video_path, tmp)
        upload_path = str(tmp)
    uploaded = call_with_retry(lambda: client.files.upload(file=upload_path), context=f"upload {video_path}")
    if uploaded is None:
        return None, None
    while uploaded.state.name == "PROCESSING":
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)
    if uploaded.state.name == "FAILED":
        raise RuntimeError("影片處理失敗")
    return uploaded.uri, uploaded.name

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
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in ds:
        groups.setdefault(r["series_name"], []).append({
            "episode_id": r["episode_name"],
            "series_name": r["series_name"],
            "video_path": r["video"]["path"],
            "release_date": r.get("release_date")
        })
    return groups

def process_segments(series: str, episode: str, path: str, date: Any):
    updated = False
    results = []
    with VideoFileClip(path) as v:
        dur = v.duration
    start = 0
    while start < dur - 5:
        end = min(start + SEGMENT_LENGTH, dur)
        safe = series.replace(" ", "_").replace("/", "_")
        seg_idx = len(results)
        seg_path = CACHE_DIR / f"segment_{safe}_{episode}_seg{seg_idx}.mp4"
        cache_path = seg_path.with_suffix(".json")

        if cache_path.exists():
            with open(cache_path, "r", encoding="utf-8") as f:
                record = json.load(f)
            results.append(record)
            start += SEGMENT_LENGTH - SEGMENT_OVERLAP
            continue

        if not seg_path.exists():
            with VideoFileClip(path) as v:
                v.subclipped(start, end).write_videofile(str(seg_path), codec="libx264", audio_codec="aac", logger=None)

        def gen():
            c = get_client()
            file_uri, file_name = upload_video_to_gemini(c, str(seg_path))
            try:
                query = generate_segment_queries(client=c, file_uri=file_uri)
            finally:
                try:
                    c.files.delete(name=file_name)
                except Exception:
                    pass
            return query

        try:
            data = call_with_retry(gen, context=f"segment {series} {episode} seg{seg_idx}")
        except BlockedContentError:
            data = {k: ["內容被阻止"]*3 for k in ["visual_saliency","character_emotion","action_behavior","dialogue","symbolic_scene"]}

        record = {
            "series_name": series,
            "episode_id": episode,
            "segment_index": seg_idx,
            "release_date": date,
            "file_name": f"videos/{safe}/segment_{safe}_{episode}_seg{seg_idx}.mp4",
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
    safe = series.replace(" ", "_").replace("/", "_")
    cache_path = CACHE_DIR / f"episode_{safe}_{episode}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def gen():
        c = get_client()
        file_uri, file_name = upload_video_to_gemini(c, str(path))
        try:
            return generate_episode_queries(client=c, file_uri=file_uri)
        finally:
            try:
                c.files.delete(name=file_name)
            except Exception:
                pass

    query = call_with_retry(gen, context=f"episode {series} {episode}")

    hf_path = f"videos/{safe}/episode_{safe}_{episode}.mp4"
    upload_video_to_hf(HF_REPO_EPISODE, Path(path), hf_path)

    record = {
        "file_name": hf_path,
        "series_name": series,
        "episode_id": episode,
        "release_date": date,
        "query": query,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    update_episode_metadata(HF_TOKEN)
    return record

def process_series(series: str, episodes: List[Dict[str, Any]]):
    safe = series.replace(" ", "_").replace("/", "_")
    cache_path = CACHE_DIR / f"series_{safe}.json"
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    series_video = CACHE_DIR / f"series_{safe}.mp4"
    if not series_video.exists():
        with open(series_video.with_suffix(".txt"), "w") as f:
            for e in episodes:
                f.write(f"file '{Path(e['video_path']).absolute()}'\n")
        subprocess.run([
            "ffmpeg","-f","concat","-safe","0","-i",str(series_video.with_suffix('.txt')),
            "-c","copy",str(series_video)
        ], check=True)
        series_video.with_suffix(".txt").unlink()

    low = CACHE_DIR / f"series_{safe}_low_fps.mp4"
    if not low.exists():
        down_video_fps(series_video, low)

    def gen():
        c = get_client()
        file_uri, file_name = upload_video_to_gemini(c, str(low))
        try:
            return generate_series_queries(client=c, file_uri=file_uri)
        finally:
            try:
                c.files.delete(name=file_name)
            except Exception:
                pass

    query = call_with_retry(gen, context=f"series {series}")
    upload_video_to_hf(HF_REPO_SERIES, series_video, f"videos/series_{safe}.mp4")

    date = sorted({e["release_date"] for e in episodes if e.get("release_date")})
    record = {
        "file_name": f"videos/series_{safe}.mp4",
        "series_name": series,
        "release_date": date[0] if date else None,
        "query": query,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)

    update_series_metadata(HF_TOKEN)
    return record

# ================== 新增：一集包起來，用 call_with_retry 跑 ==================
def run_episode_with_retry(series_name: str, ep: Dict[str, Any]) -> bool:
    episode_id = ep["episode_id"]
    video_path = ep["video_path"]
    release_date = ep.get("release_date")

    def _do_one_episode():
        _ = process_segments(series_name, episode_id, video_path, release_date)
        _ = process_episode(series_name, episode_id, video_path, release_date)
        return True

    result = call_with_retry(
        _do_one_episode,
        context=f"episode {series_name} {episode_id}"
    )
    return bool(result)

# ================== 主程式 ==================
def main():
    print(f"🔑 使用 {len(GEMINI_API_KEYS)} 個 Gemini API Keys")
    for r in [HF_REPO_SEGMENT, HF_REPO_EPISODE, HF_REPO_SERIES]:
        create_repo(r, token=HF_TOKEN, repo_type="dataset", exist_ok=True)

    groups = load_and_group_dataset()

    for sname, eps in groups.items():
        print(f"🎬 系列: {sname}")

        futures = []
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            for ep in eps:
                futures.append(executor.submit(run_episode_with_retry, sname, ep))

            for f in as_completed(futures):
                f.result()
                    
        eps = sorted(eps, key=lambda e: float(e["episode_id"]))
        _ = process_series(sname, eps)

    print("✅ 所有系列處理完成，metadata 皆已自動更新。")


if __name__ == "__main__":
    main()
