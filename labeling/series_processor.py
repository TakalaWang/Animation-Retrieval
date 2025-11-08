"""
Series Level 處理模組：處理整季/整部動畫的查詢生成

此模組負責使用 Gemini API 為整季動畫生成自然語言查詢語句，
模擬觀眾回憶整部作品時會提出的查詢。

主要功能：
- 定義系列級別的查詢生成 schema
- 提供生成查詢的提示詞
- 呼叫 Gemini API 進行內容分析
"""

import json
from typing import Any, Dict

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
            "description": (
                "用觀眾要找『這整部/整季』時會說的話，描述作品從開頭到結尾的大方向，"
                "必須只根據輸入的系列資料，不要新增沒出現的劇情或角色。"
            )
        },
        "characters": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "讓觀眾可以用外觀來找這部作品的描述，或是角色性格特徵，或行為"
                "僅可使用資料中真的出現的角色特徵。"
            )
        },
        "character_development": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述角色在整季/整部裡真的有表現出來的成長或關係變化"
                "不可虛構沒有在資料裡的長線感情線。"
            )
        },
        "theme": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "把這部作品觀眾會這樣形容的主題說法寫出來，"
                "要能當成搜尋句用，而不是學術摘要。"
            )
        },
        "visual_emotional_impression": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "整部看下來的視覺風格或情緒印象"
                "必須和輸入內容一致。"
            )
        }
    },
    "required": [
        "narrative_arc",
        "characters",
        "character_development",
        "theme",
        "visual_emotional_impression"
    ]
}

# 提示詞：指導 Gemini 生成自然語言查詢
PROMPT = """你會拿到一段「關於某一季動畫的完整影片」，這就是你唯一可以依據的資料來源。

你的任務是：幫這一季生成「觀眾在回憶這部作品時，會輸入到搜尋框裡的自然中文句子」。

⚠️一定要遵守的規則：
1. 只能根據我給你的這份系列描述文字來寫，不要補充描述裡沒有提到的世界觀、角色、支線或續作內容。
2. 句子要像觀眾說的話，而不是官方簡介或評論文。
3. 每一句都要能單獨被當成搜尋用的自然語言句子。
4. 可以用模糊但很像人說的句式。
5. 不要具體寫出你沒有看到的地名、學校名、國家名；若資料裡有，就可以寫。

請依照下面五個面向，各寫 3 句自然的中文搜尋語句。總共要 15 句。

1.【narrative_arc 整體劇情弧線】
- 要能讓人一聽就知道是這一季/這一部的主線走向。
- 要能夠描述這部作品從開頭到結尾的大方向。

2.【characters 角色辨識】
- 幫觀眾用角色外型記住這部作品。
- 專注在角色的外觀特徵或是性格等方面。

3.【character_development 角色成長與關係】
- 說明從第一集到最後角色的變化，限於描述文字裡真的寫過的。
- 強調角色之間的關係變化或個人成長。

4.【theme 主題與寓意】
- 把這部作品看起來在講的東西變成搜尋句。
- 要能反映出作品的核心主題或訊息。

5.【visual_emotional_impression 視覺/情緒印象】
- 寫整部給人的美術+氣氛。
- 描述整體的視覺風格或情緒印象。
"""


# ================== Gemini API 呼叫 ==================

def generate_series_queries(
    client: genai.Client,
    file_uri: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    使用 Gemini 生成整季/整部級別的查詢語句

    Args:
        client: Gemini API 客戶端
        file_uri: 上傳到 Gemini 的整季影片 URI
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
                    video_metadata=types.VideoMetadata(fps=1)
                ),
                types.Part(text=PROMPT),
            ]
        ),
        config={
            "response_mime_type": "application/json",
            "response_schema": SERIES_SCHEMA,
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

