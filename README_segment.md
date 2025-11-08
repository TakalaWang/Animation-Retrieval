---
dataset_info:
  features:
  - name: file_name
    dtype: string
  - name: series_name
    dtype: string
  - name: episode_id
    dtype: string
  - name: segment_index
    dtype: int64
  - name: release_date
    dtype: string
  - name: query
    struct:
    - name: visual_saliency
      sequence: string
    - name: character_emotion
      sequence: string
    - name: action_behavior
      sequence: string
    - name: dialogue
      sequence: string
    - name: symbolic_scene
      sequence: string
configs:
- config_name: winter
  data_files:
    - videos/**/*.mp4
    - metadata.jsonl
---

# Anime 2024 Winter - Segment Queries

這個數據集包含 2024 年冬季動畫的片段級別查詢語句。

## 數據集結構

- **file_name**: 影片文件路徑（用於定位影片片段位置）
- **series_name**: 動畫系列名稱
- **episode_id**: 集數 ID
- **segment_index**: 片段索引
- **release_date**: 發布日期
- **query**: 查詢語句集合
  - visual_saliency: 視覺顯著物
  - character_emotion: 角色情緒
  - action_behavior: 動作行為
  - dialogue: 對話台詞
  - symbolic_scene: 象徵性畫面
