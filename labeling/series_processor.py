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
                "例如「一開始大家是各自行動後面才組成牌手小隊的那一季」"
                "必須只根據輸入的系列資料，不要新增沒出現的劇情或角色。"
            )
        },
        "character_appearance": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "讓觀眾可以用外觀來找這部作品的描述，例如「有一個老是戴墨鏡拿撲克牌的金髮男」"
                "或「有個穿大禮服的女主管」，僅可使用資料中真的出現的角色特徵。"
            )
        },
        "character_development": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述角色在整季/整部裡真的有表現出來的成長或關係變化，"
                "例如「一開始他不想進組織最後卻願意為大家出手的那部」"
                "不可虛構沒有在資料裡的長線感情線。"
            )
        },
        "theme": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "把這部作品觀眾會這樣形容的主題說法寫出來，像是「這部其實在講信任」"
                "「這季一直在講權力爭奪」，要能當成搜尋句用，而不是學術摘要。"
            )
        },
        "visual_emotional_impression": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "整部看下來的視覺風格或情緒印象，例如「整季都很華麗像撲克牌舞台」"
                "「打鬥都是夜景+火焰的很帥那部」，必須和輸入內容一致。"
            )
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

# 提示詞：指導 Gemini 生成自然語言查詢
PROMPT = """你會拿到一段「關於某一季動畫的完整影片」，這就是你唯一可以依據的資料來源。

你的任務是：幫這一季生成「觀眾在回憶這部作品時，會輸入到搜尋框裡的自然中文句子」。

⚠️一定要遵守的規則：
1. 只能根據我給你的這份系列描述文字來寫，不要補充描述裡沒有提到的世界觀、角色、支線或續作內容。
2. 句子要像觀眾說的話，而不是官方簡介或評論文。
3. 每一句都要能單獨被當成搜尋用的 query。
4. 可以用「那部一直在…的」「那一季大家都在搶撲克牌能力的」這種模糊但很像人說的句式。
5. 不要具體寫出你沒有看到的地名、學校名、國家名；若資料裡有，就可以寫。

請依照下面五個面向，每個面向各寫 3 句，共 15 句，並用我指示的 JSON 格式輸出。

1.【narrative_arc 整體劇情弧線】
- 要能讓人一聽就知道是這一季/這一部的主線走向。
- 範例語氣：「我要找那部一開始大家只是拿牌後來才發現牌背後有超能力組織的那部」。

2.【character_appearance 角色外觀辨識】
- 幫觀眾用角色外型記住這部作品。
- 範例語氣：「有個戴墨鏡金髮、出場都拿著牌的那個男的那部」。

3.【character_development 角色成長與關係】
- 說明從第一集到最後角色的變化，限於描述文字裡真的寫過的。
- 範例語氣：「我要找那部男主一開始不想捲入牌的事件後來變成核心成員的那部」。

4.【theme 主題與寓意】
- 把這部作品看起來在講的東西變成搜尋句。
- 範例語氣：「那部一直在講『力量要被誰掌控』的」。

5.【visual_emotional_impression 視覺/情緒印象】
- 寫整部給人的美術+氣氛。
- 範例語氣：「整季都很夜景+火焰效果很帥的那部」。
"""


# ================== Gemini API 呼叫 ==================

def generate_series_queries(
    client: genai.Client,
    series_text: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    使用 Gemini 生成整季/整部級別的查詢語句

    Args:
        client: Gemini API 客戶端
        series_text: 系列描述文字
        model_name: 使用的模型名稱

    Returns:
        包含查詢語句的字典
    """
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

