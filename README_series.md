---
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: series_name
    dtype: string
  - name: release_date
    dtype: string
  - name: query
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
- config_name: winter
  data_files:
    - videos/**/*.mp4
    - metadata.jsonl
---

# Anime 2024 Winter - Series Queries

這個數據集包含 2024 年冬季動畫的系列級別查詢語句。

## 數據集結構

- **file_name**: 影片文件路徑（用於定位整季影片位置）
- **series_name**: 動畫系列名稱
- **release_date**: 首播日期
- **query**: 模型生成的查詢語句
  - narrative_arc: 整體劇情弧線
  - characters: 角色辨識
  - character_development: 角色成長與關係
  - theme: 主題與寓意
  - visual_emotional_impression: 視覺/情緒印象

