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
                "用『觀眾想找這一集』會說的話，描述這一集中真正有出現的核心事件，"
                "例如「那一集一開始他們就去參加比賽，後面才發現被陷害的那集」。"
                "必須是這一集中確實出現的情節，不可加入其他集或沒看見的世界觀。"
            )
        },
        "turning_point": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述這一集中可以清楚感覺到的劇情或情緒轉折，例如「本來以為他會輸結果反轉了」"
                "或「說出祕密之後氣氛變得很沉」。"
                "轉折必須能從這一集的畫面或台詞推得出來，不要寫沒出現的陰謀或背景設定。"
            )
        },
        "relationship_change": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述在這一集裡『真的有表現出來』的人物關係變化，例如吵架、和好、開始合作、"
                "終於承認對方。要以這一集能觀察到的互動為主，不要引用其他集沒有看到的感情線。"
            )
        },
        "episode_mood": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述這一集整體的情緒基調或觀看體感，例如「這集幾乎都在打」「這集比較溫馨校園感」"
                "「這集後面突然變得很沉重」。要能當成搜尋句使用。"
            )
        },
        "notable_scene": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "這一集中最容易被記住、可以用來指認是哪一集的畫面，例如「最後大家在屋頂看夜景的那集」"
                "「開頭就爆炸那一集」"
                "必須確實出現在這一集裡，不要生成系列級別或想像中的場景。"
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

⚠️一定要遵守的原則：
1. 只能根據這一集裡真的看得到或聽得到的內容來寫，不要引用其他集、OVA、漫畫版或你想像的世界觀。
2. 若影片中沒有說明地點、國家、學校名稱，就不要自己取名字；可以用「看起來像○○的地方」描述。
3. 句子要像觀眾在回憶：「我記得有一集是……那一集是哪一集？」而不是像在寫劇情大綱。
4. 每一句都要能單獨被拿去搜尋。
5. 如果情節只是推得出來而不是明說，請用「好像」「看起來」這種語氣。

請依照下面五個面向，每個面向產生 3 句，共 15 句，輸出成 JSON：

1.【main_plot 劇情主軸】
- 描述這一集中最主要、最能代表這一集的事情。
- 範例語氣：「我要找他們去參加比賽但被對手陷害的那一集」、「有一集是他被叫回總部受審的」。

2.【turning_point 轉折點】
- 寫這集中氣氛或走向有明顯變化的地方。
- 範例語氣：「本來大家都以為可以贏結果消息傳來全都崩掉的那一段」。

3.【relationship_change 角色關係變化】
- 寫這一集裡角色之間吵架、和好、開始相信、終於說開的情況。
- 範例語氣：「我要找他跟隊長終於說出實情，所以隊長才認同他的那一集」。

4.【episode_mood 整集氛圍】
- 把這一集看起來是溫馨、沉重、熱血、情報量很大之類的感受寫出來，當成搜尋句。
- 範例語氣：「那集一開始很日常後面突然超級打鬥的」。

5.【notable_scene 易記場景】
- 寫這一集最能被拿來認的畫面（開頭就爆炸、結尾在雨中道別、最後全部人在屋頂）。
- 只能寫這一集裡真實出現的畫面。

如果你不確定某件事情有沒有在這一集出現，就不要寫進去。

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
