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
import time
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
            "description": (
                "描述影片中最明顯、最容易記住的『實際畫面特徵』，"
                "只能根據影片中真的可見的內容，不可加入腦補或未出現的場景。"
            ),
        },
        "character_emotion": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述角色『在此片段中』可觀察到的情緒或表情，例如："
                "若影片中沒有明確表情，不要亂猜。"
            ),
        },
        "action_behavior": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述片段中角色實際做出的動作"
                "必須是影片中清楚可見的行為，不可假設或延伸到其他劇情。"
            ),
        },
        "dialogue": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "影片中可聽到或觀察到的語句或喊話。"
                "若片段無台詞，請生成與畫面相符的常見呼喊方式，但不得虛構對話。"
            ),
        },
        "symbolic_scene": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 3,
            "maxItems": 3,
            "description": (
                "描述片段中看起來有轉折或象徵感的畫面，或是動畫經典畫面"
                "角色背影、鏡頭停留等。"
                "必須確實出現在影片中，不得創造新的事件或象徵意象。"
            ),
        },
    },
    "required": [
        "visual_saliency",
        "character_emotion",
        "action_behavior",
        "dialogue",
        "symbolic_scene",
    ],
}


# 提示詞：指導 Gemini 生成自然語言查詢
PROMPT = """你會看到「一小段實際動畫影片片段」，你的任務是：幫這一小段動畫片段產生「觀眾事後想找回這段影片時會在搜尋框打的中文句子」。

⚠️一定要遵守的規則：
1. 只能描述影片中「真的看得到或聽得到」的內容。不要幫影片腦補沒出現的背景、世界觀、城市名稱、人物關係、回憶、前後集內容。
2. 如果場景、地點、人物名字在影片裡沒有明說，就不要自己取名。
3. 每一句都要寫成自然語言回想影片片段的說法，要具體、有畫面、有可見動作或台詞。
4. 可以用觀察到的外型來指人，不要寫出影片中沒有的稱呼。
5. 不要出現這一小段影片裡沒有的事件，如果沒看到就不能寫。
6. 若你不確定物品的功能，可以用「像是…的東西」這種不確定語氣，但仍然必須是影片中真的看到的東西。

請依照下面五個面向，各寫 3 句自然的中文搜尋語句。總共要 15 句。

1.【視覺顯著物 visual_saliency】
- 描述畫面最明顯、最容易記住的影像，例如光很亮的地方、火焰、發光的物件、鏡頭特寫、動態特效。
- 句子要能讓人一聽就知道是找畫面，不是說故事。

2.【角色情緒 character_emotion】
- 描述你從角色的表情、語氣、姿勢推得出的情緒，例如很生氣、很自信、很沉重、快哭了。
- 如果情緒只是看起來像，就說「看起來很○○」。

3.【動作行為 action_behavior】
- 描述你真的看到的動作（拿槍、把牌變成武器、往前衝、轉頭、揮手、跳開）。
- 不要寫影片裡沒出現的打鬥或怪物。

4.【語句／台詞 dialogue】
- 如果影片裡有說話，請寫出觀眾可能記得的那句話或接近的說法。
- 如果只聽得出大概意思，也可以寫成「他好像說了『…』」。
- 沒有台詞就寫這類型片段常見的呼喊方式，但要跟畫面對得上。

5.【象徵性畫面 symbolic_scene】
- 描述看起來像轉折、像伏筆、像重要登場的畫面，但仍須來自你真的看到的影像。
- 不要幻想出不存在的神秘儀式或多人場景。

如果你不確定某件事情有沒有在這一片段出現，就不要寫進去。
"""


# ================== Gemini API 呼叫 ==================

def generate_segment_queries(
    client: genai.Client,
    file_uri: str,
    model_name: str = "models/gemini-2.5-flash",
) -> Dict[str, Any]:
    """
    使用 Gemini 生成片段級別的查詢語句

    Args:
        client: Gemini API 客戶端
        file_uri: 影片檔案 URI
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

