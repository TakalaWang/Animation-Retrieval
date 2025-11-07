# Animation Labeling 模組重構

## 📁 檔案結構

```
labeling/
├── main.py                    # 主程式入口
├── segment_processor.py       # Segment Level 處理
├── episode_processor.py       # Episode Level 處理
├── series_processor.py        # Series Level 處理
└── cache_gemini_video/        # 快取資料夾
```

## 🎯 模組說明

### 1. `segment_processor.py` - 片段級處理
負責處理動畫的短片段（預設 60 秒）：
- **Schema**: `SEGMENT_SCHEMA` - 定義片段查詢的結構
- **函數**: 
  - `generate_segment_queries()` - 呼叫 Gemini API 生成片段查詢
  - `process_segments()` - 處理整集影片的所有片段
- **查詢類型**: 
  - 視覺突出 (visual_saliency)
  - 角色情緒 (character_emotion)
  - 動作行為 (action_behavior)
  - 對話台詞 (dialogue)
  - 象徵場景 (symbolic_scene)

### 2. `episode_processor.py` - 單集級處理
負責處理整集動畫：
- **Schema**: `EPISODE_SCHEMA` - 定義單集查詢的結構
- **函數**:
  - `generate_episode_queries()` - 呼叫 Gemini API 生成單集查詢
  - `process_episode()` - 處理單集動畫
- **查詢類型**:
  - 主要劇情 (main_plot)
  - 轉折點 (turning_point)
  - 關係變化 (relationship_change)
  - 集數氛圍 (episode_mood)
  - 重要場景 (notable_scene)

### 3. `series_processor.py` - 整季級處理
負責處理整季/整部動畫：
- **Schema**: `SERIES_SCHEMA` - 定義整季查詢的結構
- **函數**:
  - `generate_series_queries()` - 呼叫 Gemini API 生成整季查詢
  - `process_series()` - 處理整季動畫
- **查詢類型**:
  - 敘事弧線 (narrative_arc)
  - 角色外觀 (character_appearance)
  - 角色發展 (character_development)
  - 主題 (theme)
  - 視覺情感印象 (visual_emotional_impression)

### 4. `main.py` - 主程式
整合所有處理模組，負責：
- 載入資料集
- 管理環境變數和設定
- 呼叫各個處理模組
- 上傳結果到 HuggingFace

## 🚀 使用方式

### 1. 設定環境變數
建立 `.env` 檔案：
```bash
GEMINI_API_KEY=your_gemini_api_key
HF_TOKEN=your_huggingface_token
```

### 2. 修改設定
在 `main.py` 中調整：
```python
# HuggingFace Repos 設定
HF_REPO_SEGMENT = "yourname/anime-2024-segment-queries"
HF_REPO_EPISODE = "yourname/anime-2024-episode-queries"
HF_REPO_SERIES = "yourname/anime-2024-series-queries"

# 測試設定
TEST_DATASET = "JacobLinCool/anime-2024"
TEST_SPLIT = "winter"  # spring, summer, fall, winter
SEGMENT_LENGTH = 60    # 每個片段的長度（秒）
SEGMENT_OVERLAP = 5    # 片段之間的重疊時間（秒）
```

### 3. 執行程式
```bash
cd labeling
python main.py
```

## ✨ 特點

1. **模組化設計**: 每個處理層級獨立成一個檔案，易於維護和測試
2. **快取機制**: 自動快取 API 結果，避免重複呼叫
3. **錯誤重試**: 自動處理 API rate limit 和其他錯誤
4. **簡化測試**: 預設只處理一集動畫，快速驗證流程

## 📝 與舊版差異

### 舊版 (使用 `schema.py`)
```
labeling/
├── main.py           # 包含所有邏輯
└── schema.py         # 只有 schema 定義
```

### 新版 (模組化)
```
labeling/
├── main.py                # 主程式（簡化）
├── segment_processor.py   # Segment 完整邏輯
├── episode_processor.py   # Episode 完整邏輯
└── series_processor.py    # Series 完整邏輯
```

**優點**：
- ✅ 程式碼更清晰，每個檔案職責單一
- ✅ 容易測試單一處理層級
- ✅ 可以獨立重用各個處理模組
- ✅ Schema 定義和處理邏輯放在一起，更直觀

## 🔧 開發建議

如果要修改某個層級的查詢邏輯：
1. 直接編輯對應的 processor 檔案
2. 修改 Schema 定義或處理函數
3. 不需要改動其他檔案

如果要測試單一層級：
```python
from segment_processor import process_segments
from episode_processor import process_episode
from series_processor import process_series

# 單獨測試某個處理器
```
