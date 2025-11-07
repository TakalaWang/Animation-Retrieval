"""Series Level 處理模組：處理整季/整部動畫的查詢生成"""

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import google.genai as genai
from google.genai import types


# ================== Schema 定義 ==================

SERIES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "narrative_arc": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "整部作品的故事流程或劇情弧線，例如「主角從弱小一路成長」「一開始大家分散後面組成團隊」「最終要面對真正的黑幕」。"
        },
        "character_appearance": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "主要角色的外觀特徵、服裝或標誌性元素，例如「白髮戴紅圍巾的男生」「總是穿藍色洋裝的女主」「戴著面具的反派」。"
        },
        "character_development": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "角色在整部作品中的成長與人際變化，例如「他從不信任任何人到願意交給夥伴」「兩人從敵對變成戀人」「隊長學會向夥伴道歉」。"
        },
        "theme": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "作品想傳達的主題與寓意，例如「這部在講友情與犧牲」「核心是面對過去」「強調團隊合作的重要性」。"
        },
        "visual_emotional_impression": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "整體畫風、色調與情緒給人的感覺，例如「整體色調溫暖柔和」「戰鬥場面很華麗」「結尾超催淚」。"
        }
    },
    "required": [
        "narrative_arc",
        "character_appearance",
        "character_development",
        "theme",
        "visual_emotional_impression"
    ]
}

PROMPT = """你將獲得一部完整作品的資訊（包含劇情與角色資料），
請你模擬「觀眾回憶整部作品」時會提出的自然中文查詢句。

請依下列五個方向生成查詢，每個面向提供 3 句：

1. 【故事整體流程】  
   - 概述整體故事弧線，例如從起點到結局的發展過程。  
   - 模擬語句：「那部是他從孤獨變成領導者的故事」、「整體講成長與救贖」。

2. 【角色人物外貌】  
   - 描述主要角色的外觀特徵、服裝、造型、或標誌性元素。  
   - 模擬語句：「那個白髮少年戴著紅色圍巾」、「女主總是穿著藍裙子」。

3. 【角色成長曲線與關係變化】  
   - 描述角色在整個故事中的心理或關係轉變。  
   - 模擬語句：「他從討厭夥伴到願意為大家犧牲」、「兩人從敵對變戀人」。

4. 【主題與寓意】  
   - 提煉出作品核心思想或哲學主題。  
   - 模擬語句：「這部動畫講友情與犧牲」、「作品在探討時間輪迴與後悔」。

5. 【視覺／情緒印象】  
   - 描述整體畫風與情緒色調，像觀眾印象中的感覺。  
   - 模擬語句：「整體色調很溫柔」、「戰鬥場面特效超華麗」、「結尾超感人」。

請生成真實、自然、貼近觀眾記憶的查詢句，
讓語氣像在和朋友聊天時描述「那部作品」的印象。
"""


# ================== Gemini API 呼叫 ==================

def generate_series_queries(
    client: genai.Client,
    series_text: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """使用 Gemini 生成整季/整部級別的查詢語句"""

    resp = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(text=series_text),
                types.Part(text=PROMPT),
            ]
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": SERIES_SCHEMA,
        }
    )
    return json.loads(resp.text)


# ================== 處理邏輯 ==================

def process_series(
    client: genai.Client,
    series_id: str,
    episodes: List[Tuple[str, str, float]],
    cache_dir: Path,
    retry_fn=None,
) -> Dict[str, Any]:
    """處理整季/整部動畫的查詢生成"""
    cache_path = cache_dir / f"series_{series_id}.json"
    
    # 檢查快取
    if cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
            print("  📦 使用快取")
            return cached
    
    print("  🎭 生成查詢...")
    
    # 建立系列摘要
    lines = [f"這是一部名為 {series_id} 的動畫，包含以下集數："]
    for ep_id, file_uri, dur in episodes:
        lines.append(f"- 集數 {ep_id} ，影片來源 {file_uri} ，長度約 {int(dur)} 秒。")
    series_text = "\n".join(lines)
    
    # 呼叫 Gemini API
    if retry_fn:
        data = retry_fn(
            generate_series_queries,
            client=client,
            series_text=series_text,
            sleep_sec=5,
        )
    else:
        data = generate_series_queries(
            client=client,
            series_text=series_text,
        )
    
    record = {
        "series_id": series_id,
        "queries": data,
        "episodes": [
            {"episode_id": ep_id, "file_uri": file_uri, "duration_s": dur}
            for (ep_id, file_uri, dur) in episodes
        ],
    }
    
    # 儲存快取
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=2)
    
    return record
