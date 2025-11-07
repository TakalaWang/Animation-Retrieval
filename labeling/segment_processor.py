"""Segment Level 處理模組：處理動畫片段的查詢生成"""

import json
from pathlib import Path
from typing import Any, Dict, List

import google.genai as genai
from google.genai import types


# ================== Schema 定義 ==================

SEGMENT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "visual_saliency": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "描述此片段最顯眼的畫面、顏色、光線、鏡頭或特效，例如「背景突然變紅」「鏡頭快速拉近」。"
        },
        "character_emotion": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "描述片段中角色的表情或情緒反應，例如「他忍著眼淚說話」「她露出放心的笑」。"
        },
        "action_behavior": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "描述片段中清楚可見的動作或行為，例如「他揮拳」「她轉身離開」「兩人同時衝上去」。"
        },
        "dialogue": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "此片段中觀眾最可能記住的台詞、喊話或旁白，例如「我不會再逃避了」「拜託你相信我」。"
        },
        "symbolic_scene": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "具有象徵或轉折意味的畫面描述，例如「花瓣被風吹走」「畫面轉黑」「夕陽照在他身上」。"
        }
    },
    "required": [
        "visual_saliency",
        "character_emotion",
        "action_behavior",
        "dialogue",
        "symbolic_scene"
    ]
}

PROMPT = """你將獲得一段影片的資訊，請你根據該片段的內容，
模擬「人類憑記憶想搜尋影片片段」時會說出的自然中文查詢句。

這些查詢要貼近真實觀看經驗，包含視覺、情緒與動作的細節，
請根據以下五個面向撰寫：

1. 【視覺顯著物】  
   - 描述畫面中最引人注意的視覺特徵，例如光線變化、色彩、構圖、鏡頭切換、特效等。  

2. 【角色與情緒】  
   - 聚焦角色的表情、姿態、心理狀態或情緒反應。  

3. 【動作／行為】  
   - 描述角色的明顯動作、互動或物理行為。  

4. 【語言內容（台詞）】  
   - 擷取或重述具有情感或意義的台詞、喊話、旁白。  

5. 【關鍵章節或象徵性場面】  
   - 描述片段中具有象徵意義或情緒轉折的畫面。  

請為每一個面向生成 3 個自然中文查詢句，句子要像真實觀眾在回憶影片時會說的話，
不要只是摘要或重述劇情，請盡量具體、生動、有感覺。
"""


# ================== Gemini API 呼叫 ==================

def generate_segment_queries(
    client: genai.Client,
    file_uri: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """使用 Gemini 生成片段級別的查詢語句"""
    resp = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=file_uri),
                    video_metadata=types.VideoMetadata(fps=5)
                ),
                types.Part(text=PROMPT),
            ]
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": SEGMENT_SCHEMA,
        }
    )
    return json.loads(resp.text)


# ================== 處理邏輯 ==================

def process_segments(
    client: genai.Client,
    episode_id: str,
    file_uri: str,
    cache_dir: Path,
    
    retry_fn=None,
) -> List[Dict[str, Any]]:
    """處理整集影片的所有片段"""
    results: List[Dict[str, Any]] = []
    start = 0.0
    seg_idx = 0
    

    cache_path = cache_dir / f"segment_{episode_id}_seg{seg_idx}.json"
    
    # 檢查快取
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            results.append(cached)
    else:
        
        # 呼叫 Gemini API
        if retry_fn:
            data = retry_fn(
                generate_segment_queries,
                client=client,
                file_uri=file_uri,
                sleep_sec=5,
            )
        else:
            data = generate_segment_queries(
                client=client,
                file_uri=file_uri,
            )
        
        record = {
            "episode_id": episode_id,
            "segment_index": seg_idx,
            "queries": data,
        }
        
        # 儲存快取
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        
        results.append(record)

    
    return results
