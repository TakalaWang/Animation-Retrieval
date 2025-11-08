#!/bin/bash

echo "=== Animation Retrieval 處理進度監控 ==="
echo "PID: $(pgrep -f 'labeling/main.py' | head -1)"
echo ""

while true; do
    clear
    echo "=== 最新進度 ($(date '+%Y-%m-%d %H:%M:%S')) ==="
    echo ""
    
    # 顯示最新的重要信息
    tail -n 100 output.log | grep -E "(處理系列|處理集數|episodes|阻止|完成|✅|⚠️|❌)" | tail -n 20
    
    echo ""
    echo "=== 統計信息 ==="
    echo "已處理片段數: $(grep -c '✅ 完成' output.log)"
    echo "被阻止片段數: $(grep -c '被阻止，跳過' output.log)"
    echo "錯誤數: $(grep -c '❌ \[error\]' output.log)"
    
    echo ""
    echo "按 Ctrl+C 停止監控"
    sleep 10
done
