#!/bin/bash
# process_eval_folders.sh

set -e  # 遇到错误立即退出

# 配置文件路径
EXPERIMENT_NAME=$1
EVAL_RESULTS_DIR="eval_results/${EXPERIMENT_NAME}"
MODEL_PATH=${2:-"/default/path/to/model"}

echo "========================================="
echo "开始批量处理 eval_results 下的文件夹"
echo "========================================="

# 检查目录是否存在
if [ ! -d "$EVAL_RESULTS_DIR" ]; then
    echo "错误: 目录 $EVAL_RESULTS_DIR 不存在"
    exit 1
fi

# 获取所有子目录（仅一级）
folders=($(find "$EVAL_RESULTS_DIR" -maxdepth 1 -type d ! -path "$EVAL_RESULTS_DIR"))

if [ ${#folders[@]} -eq 0 ]; then
    echo "警告: $EVAL_RESULTS_DIR 下没有找到任何子文件夹"
else
    echo "找到 ${#folders[@]} 个子文件夹:"
    printf '%s\n' "${folders[@]}"
    echo ""
    
    # 遍历每个文件夹并处理
    for folder in "${folders[@]}"; do
        echo "处理文件夹: $folder"
        
        # 运行Python脚本处理单个文件夹
        python length_calculation.py \
            --model "$MODEL_PATH" \
            --dir "$folder" \
            || echo "处理 $folder 时发生错误，继续下一个..."
        
        echo "-----------------------------------------"
    done
    
    echo "所有子文件夹处理完成!"
    echo ""
fi

# 最后处理整个eval_results目录
echo "现在处理整个 $EVAL_RESULTS_DIR 目录..."
python length_summary.py --dir "$EVAL_RESULTS_DIR"

echo "========================================="
echo "批量处理全部完成!"
echo "========================================="