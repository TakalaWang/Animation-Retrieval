"""
Segment Level 處理模組：處理動畫片段的查詢生成

此模組負責使用 Gemini API 為動畫片段生成自然語言查詢語句，
模擬人類憑記憶想搜尋影片片段時會說出的話。

主要功能：
- 定義片段級別的查詢生成 schema
- 提供生成查詢的提示詞
- 呼叫 Gemini API 進行內容分析
"""

import json
from typing import Any, Dict

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

# 提示詞：指導 Gemini 生成自然語言查詢
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
    video_path: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    使用 Gemini 生成片段級別的查詢語句

    Args:
        client: Gemini API 客戶端
        video_path: 本地影片檔案路徑
        model_name: 使用的模型名稱

    Returns:
        包含查詢語句的字典
    """
    file = client.files.upload(file=video_path)
    resp = client.models.generate_content(
        model=model_name,
        contents=types.Content(
            parts=[
                types.Part(
                    file_data=types.FileData(file_uri=file.uri),
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

