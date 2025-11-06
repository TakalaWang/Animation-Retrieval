import os
import json
import argparse

import ffmpeg
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi
from google import genai
from google.genai import types



# 取得影片長度（用 ffmpeg-python，簡潔版）
def get_video_duration(video_path: str) -> float:
    probe = ffmpeg.probe(video_path)
    return float(probe["format"]["duration"])


# 產生一段描述
def generate_segment_description(client, model, file_uri, start_s, end_s):
    prompt = (
        "請針對這段動畫影片的片段產生一段描述，請使用一段話進行描述，"
        "描述要盡量清晰、明確且仔細，包含畫面中的人物、表情、動作、場景、氛圍與情節。"
    )
    parts = [
        types.Part(
            file_data=types.FileData(file_uri=file_uri),
            video_metadata=types.VideoMetadata(
                start_offset=f"{start_s:.1f}s",
                end_offset=f"{end_s:.1f}s",
                fps=1,
            ),
        ),
        types.Part.from_text(text=prompt),
    ]
    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=parts)],
    )
    return resp.text.strip()


# 用 structured output 產 10 個 query
def generate_queries_from_description(client, model, description):
    if not description:
        return []

    schema = types.GenerateContentSchema(
        type="object",
        properties={
            "queries": types.GenerateContentSchema(
                type="array",
                items=types.GenerateContentSchema(type="string"),
            )
        },
        required=["queries"],
    )

    prompt = (
        "請根據以下動畫片段的描述，產生 10 句使用者可能會搜尋的句子：\n"
        "要求：\n"
        "1. 全部用繁體中文。\n"
        "2. 一行一句，不要加序號。\n"
        "3. 請適度隱藏部分資訊（時間人物地點）甚至可以可以含有刻意（時間人物地點）甚至可以可以含有刻意。\n"
        "4. 語氣自然像在搜尋。\n\n"
        f"描述如下：\n{description}\n"
    )

    resp = client.models.generate_content(
        model=model,
        contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=schema,
        ),
    )
    parsed = resp.parsed
    return [q for q in parsed["queries"] if q.strip()]


# 處理一筆 HF 的影片資料
def process_one_item(client, model, item, segment_len=60):
    video_path = item["video"]
    uploaded = client.files.upload(file=video_path)
    file_uri = uploaded.uri

    duration = get_video_duration(video_path)

    results = []
    t = 0.0
    segment_id = 0
    while t < duration:
        start_s = t
        end_s = min(t + segment_len, duration)

        desc = generate_segment_description(client, model, file_uri, start_s, end_s)
        queries = generate_queries_from_description(client, model, desc)

        record = dict(item)
        record.update(
            {
                "segment_id": segment_id,
                "segment_start": start_s,
                "segment_end": end_s,
                "description": desc,
                "queries": queries,
            }
        )
        results.append(record)

        segment_id += 1
        t += segment_len

    return results


def upload_to_hf(output_path: str, repo_id: str, token: str):
    api = HfApi()
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, token=token)
    api.upload_file(
        path_or_fileobj=output_path,
        path_in_repo=os.path.basename(output_path),
        repo_id=repo_id,
        repo_type="dataset",
    )


def main():
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("--segment-length", type=int, default=60)
    parser.add_argument("--output", type=str, default="data.json")
    parser.add_argument("--model", type=str, default="models/gemini-2.5-pro")
    parser.add_argument(
        "--upload",
        action="store_true",
        help="upload to TakalaWang/Anime-2024-winter-query",
    )
    args = parser.parse_args()

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # 從 HF 讀你指定的 dataset
    dataset = load_dataset("JacobLinCool/anime-2024", "winter", split="train", token=os.environ.get("HF_API_TOKEN"))

    all_records = []
    for item in dataset[:10]:
        segments = process_one_item(
            client, args.model, item, segment_len=args.segment_length
        )
        all_records.extend(segments)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_records, f, indent=2, ensure_ascii=False)

    print(f"done, wrote {len(all_records)} records to {args.output}")

    
    upload_to_hf(args.output, "TakalaWang/Anime-2024-winter-query", os.environ.get("HF_API_TOKEN"))
    print("uploaded to huggingface")


if __name__ == "__main__":
    main()
