"""
Episode Level 處理模組：處理單集動畫的查詢生成

此模組負責使用 Gemini API 為單集動畫生成自然語言查詢語句，
模擬觀眾憑記憶搜尋這一集時會說出的話。

主要功能：
- 定義集數級別的查詢生成 schema
- 提供生成查詢的提示詞
- 呼叫 Gemini API 進行內容分析
"""

import json
from typing import Any, Dict

import google.genai as genai
from google.genai import types


# ================== Schema 定義 ==================

EPISODE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "main_plot": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "這一集的核心劇情主軸或主題事件，例如「這集在講他參加比賽」「這集揭開了女主的身世」。"
        },
        "turning_point": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "本集中明顯的劇情或情緒轉折，例如「他被迫說出真相」「比賽突然失敗」「敵人變成盟友」。"
        },
        "relationship_change": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "角色之間的關係變化，例如「這一集他們吵架又和好」「學長終於認同他」「兩人開始合作」。"
        },
        "episode_mood": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "整集的情緒或氛圍，例如「整集都很感人」「這集超級緊張」「看完會覺得很溫暖」。"
        },
        "notable_scene": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": "這一集中最有畫面感、最容易被觀眾記住的場景，例如「最後大家在夕陽下舉杯」「雨中道別」「屋頂上對決」。"
        }
    },
    "required": [
        "main_plot",
        "turning_point",
        "relationship_change",
        "episode_mood",
        "notable_scene"
    ]
}

# 提示詞：指導 Gemini 生成自然語言查詢
PROMPT = """你將獲得一集影片的資訊，請你根據整集的內容，
模擬「觀眾憑記憶搜尋這一集」時會說出的自然中文查詢句。

請根據以下五個任務方向生成內容，每個面向提供 3 句查詢：

1. 【劇情主軸】  
   - 用一兩句話表達這集的核心故事或主題事件。  

2. 【轉折點】  
   - 描述這集中最明顯的情節或情緒轉變，例如揭露秘密、失敗、和好等。  

3. 【角色關係變化】  
   - 說明角色之間的關係如何變化，情感、立場或信任上的轉折。  

4. 【主題或情緒氛圍】  
   - 用形容詞描述整集給人的氣氛或節奏，例如溫暖、感人、緊張。  

5. 【特定場景描述】  
   - 描述這集中最具代表性的場景，例如畫面構圖、情感高潮或收尾片段。  
   
請讓這些查詢句聽起來像觀眾在用模糊印象尋找影片，
自然口語、具象、生動，不要僅僅重述內容。
"""


# ================== Gemini API 呼叫 ==================

def generate_episode_queries(
    client: genai.Client,
    file_uri: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    使用 Gemini 生成單集級別的查詢語句

    Args:
        client: Gemini API 客戶端
        file_uri: 上傳到 Gemini 的檔案 URI
        model_name: 使用的模型名稱

    Returns:
        包含查詢語句的字典
    """
    resp = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=file_uri),
                    video_metadata=types.VideoMetadata(fps=2)
                ),
                types.Part(text=PROMPT),
            ]
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": EPISODE_SCHEMA,
        }
    )
    return json.loads(resp.text)
