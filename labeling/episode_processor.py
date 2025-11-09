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
            "description": (
                "以觀眾回憶這一集時可能會說的方式，描述本集中實際發生、可辨識的主要事件或內容焦點。"
                "內容必須能從本集畫面或台詞取得，不得加入其他集或未出現的世界觀。"
            )
        },
        "turning_point": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述本集中可以明顯感受到的劇情或情緒轉變，例如局勢反轉、氣氛改變、角色立場出現差異。"
                "轉變必須能從本集實際內容觀察到，不得推測未出現的陰謀或背景設定。"
            )
        },
        "relationship_change": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述本集中角色之間實際出現的互動變化，如態度轉柔、發生衝突、重建信任或開始合作。"
                "必須以本集可見的對話與行為為依據，不得引用本集未呈現的感情線。"
            )
        },
        "episode_mood": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述本集整體可感受到的氣氛或節奏，例如偏日常、緊張、溫馨或情緒沉重。"
                "描述需能對應到本集實際內容，不得脫離畫面或台詞自行定義。"
            )
        },
        "notable_scene": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述本集中最容易被觀眾記住、可用來辨識是哪一集的具體畫面或收尾場景。"
                "必須是本集實際出現的場景，不得生成系列層級或想像中的畫面。"
            )
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

PROMPT = """你會取得「同一集動畫影片」的內容，你的任務是：幫這一整集產生觀眾之後想找回這一集時，會在搜尋框輸入的中文句子。

⚠️一定要遵守的規則：
1. 只能根據這一集裡實際能看到或聽到的內容來撰寫，不得引用其他集、OVA、漫畫版或未出現的世界觀。
2. 若影片中沒有明確提到地點、國家或學校名稱，請不要自行命名，可用「看起來像○○的地方」這種描述方式。
3. 句子要自然、口語化，像觀眾在回憶這一集的畫面，而不是在撰寫劇情摘要。
4. 每一句都要能單獨作為搜尋用語。
5. 若某事件只是可以推測但影片中未明說，請使用「好像」「看起來」等語氣降低確定性。
6. 若你不確定某件事情是否真的在這一集出現，請不要寫進去。

請依照下面五個面向，各寫 3 句自然的中文搜尋語句。總共要 15 句。

1.【main_plot 劇情主軸】  
描述這一集中最主要、最能代表這一集的事件或內容焦點。  

2.【turning_point 轉折點】  
指出這集中劇情或情緒明顯轉變的地方，例如氣氛、立場或結果的變化。  

3.【relationship_change 角色關係變化】  
說明這一集中角色之間的互動關係出現了哪些具體變化，如吵架、和解、合作或信任的建立。  

4.【episode_mood 整集氛圍】  
描述整集的整體氣氛或節奏，例如輕鬆、緊張、溫暖、沉重、高潮迭起等。  

5.【notable_scene 易記場景】  
描述這一集中最容易讓觀眾記住的畫面、收尾場景或象徵性時刻，必須是影片中實際出現的內容。  
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
                    video_metadata=types.VideoMetadata(fps=0.5)
                ),
                types.Part(text=PROMPT),
            ]
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": EPISODE_SCHEMA,
        }
    )
    
    # 嘗試多種方式獲取響應內容
    if resp.text:
        return json.loads(resp.text)
    elif hasattr(resp, 'candidates') and resp.candidates:
        candidate = resp.candidates[0]
        if hasattr(candidate, 'content') and candidate.content:
            if hasattr(candidate.content, 'parts') and candidate.content.parts:
                text = candidate.content.parts[0].text
                if text:
                    return json.loads(text)
    
    raise ValueError("Gemini API 返回空響應")
