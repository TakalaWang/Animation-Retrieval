---
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: series_name
    dtype: string
  - name: episode_count
    dtype: int64
  - name: release_dates
    sequence: string
  - name: model_response
    struct:
    - name: narrative_arc
      sequence: string
    - name: characters
      sequence: string
    - name: character_development
      sequence: string
    - name: theme
      sequence: string
    - name: visual_emotional_impression
      sequence: string
configs:
- config_name: default
  data_files:
  - split: train
    path: "videos/**/*.mp4"
---

# Anime 2024 Winter - Series Queries

這個數據集包含 2024 年冬季動畫的系列級別查詢語句。

## 數據集結構

- **file_name**: 影片文件路徑
- **series_name**: 動畫系列名稱
- **episode_count**: 集數數量
- **release_dates**: 發布日期列表
- **model_response**: 模型生成的查詢語句
  - narrative_arc: 整體劇情弧線
  - characters: 角色辨識
  - character_development: 角色成長與關係
  - theme: 主題與寓意
  - visual_emotional_impression: 視覺/情緒印象

