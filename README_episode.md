---
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: series_name
    dtype: string
  - name: episode_id
    dtype: string
  - name: release_date
    dtype: string
  - name: query
    struct:
    - name: main_plot
      sequence: string
    - name: turning_point
      sequence: string
    - name: relationship_change
      sequence: string
    - name: episode_mood
      sequence: string
    - name: notable_scene
      sequence: string
configs:
- config_name: winter
  data_files:
    - videos/**/*.mp4
    - metadata.jsonl
---

# Anime 2024 Winter - Episode Queries

這個數據集包含 2024 年冬季動畫的集數級別查詢語句。

## 數據集結構

- **file_name**: 影片文件路徑（用於定位集數影片位置）
- **series_name**: 動畫系列名稱
- **episode_id**: 集數 ID
- **release_date**: 發布日期
- **query**: 模型生成的查詢語句
  - main_plot: 主要劇情
  - turning_point: 轉折點
  - relationship_change: 關係變化
  - episode_mood: 集數氛圍
  - notable_scene: 易記場景
