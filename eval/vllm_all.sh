#!/bin/bash
# set -e

export VLLM_USE_V1=0

# ================= 参数定义 =================
# $1: 测试数据文件路径 (例如: aime)
# $2: Repeat 次数
# $3: Concurrency 并发数

PROJECT_NAME='ALP'
EXPERIMENT_NAME='alp_disprm_1.5B_4k_1e-7'
ENVIRONMENT='cky_alp'

CHECKPOINT_ROOT="../checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}"
OUTPUT_ROOT="./eval_results/${EXPERIMENT_NAME}"
DATA="../deepscaler/data/test/${1}.json"
REPEAT=${2:-64}
CONCURRENCY=${3:-64}

# 定义所有可用端口和设备
ALL_PORTS=(8010 8011 8012 8013 8014 8015 8016 8017)
DEVICES=(0 1 2 3 4 5 6 7)  # 根据实际GPU数量修改

# 根据设备数量自动选择端口
NUM_DEVICES=${#DEVICES[@]}
PORTS=("${ALL_PORTS[@]:0:$NUM_DEVICES}")

echo "Using $NUM_DEVICES GPUs with ports: ${PORTS[*]}"

# 创建总输出目录
if [ ! -d "$OUTPUT_ROOT" ]; then
    mkdir -p "$OUTPUT_ROOT"
fi

# 定义汇总文件,用于记录所有 step 的结果
SUMMARY_FILE="$OUTPUT_ROOT/all_checkpoints_summary.csv"
echo "Step,Path,Pass@1,Pass@K" > "$SUMMARY_FILE"

# ================= 查找并排序 Checkpoints =================
echo "Looking for checkpoints in $CHECKPOINT_ROOT..."
CHECKPOINT_DIRS=$(find "$CHECKPOINT_ROOT" -maxdepth 1 -type d -name "global_step_*" | sort -V)

if [ -z "$CHECKPOINT_DIRS" ]; then
    echo "No global_step directories found in $CHECKPOINT_ROOT"
    exit 1
fi

# ================= 开始循环测试 =================
for STEP_DIR in $CHECKPOINT_DIRS; do
    # 1. 构造具体的模型路径和输出路径
    STEP_NAME=$(basename "$STEP_DIR")
    MODEL_PATH="$STEP_DIR/actor/huggingface"
    CURRENT_OUTPUT_DIR="$OUTPUT_ROOT/$STEP_NAME"

    # 检查模型路径是否存在
    if [ ! -d "$MODEL_PATH" ]; then
        echo "Warning: Model path not found: $MODEL_PATH. Skipping..."
        continue
    fi

    # 检查是否已经跑过
    if [ -f "$CURRENT_OUTPUT_DIR/results_summary.json" ]; then
        echo "Result for $STEP_NAME already exists. Skipping..."
        # 把已有结果追加到 summary 文件
        P1=$(python -c "import json; print(json.load(open('$CURRENT_OUTPUT_DIR/results_summary.json'))['pass@1'])" 2>/dev/null || echo "N/A")
        PK=$(python -c "import json; print(json.load(open('$CURRENT_OUTPUT_DIR/results_summary.json'))['pass@$REPEAT'])" 2>/dev/null || echo "N/A")
        echo "$STEP_NAME,$MODEL_PATH,$P1,$PK" >> "$SUMMARY_FILE"
        continue
    fi

    echo "========================================================"
    echo "Starting evaluation for: $STEP_NAME"
    echo "Model Path: $MODEL_PATH"
    echo "Output Dir: $CURRENT_OUTPUT_DIR"
    echo "========================================================"

    # 2. 启动 vLLM 服务
    for i in "${!DEVICES[@]}"; do
        PORT="${PORTS[$i]}"
        DEVICE="${DEVICES[$i]}"
        
        LOG_FILE="${OUTPUT_ROOT}/vllm_server_${STEP_NAME}_gpu${DEVICE}.log"

        CUDA_VISIBLE_DEVICES=$DEVICE vllm serve $MODEL_PATH \
        --max_model_len 32768 \
        --enforce-eager \
        --gpu-memory-utilization 0.93 \
        --port $PORT > "$LOG_FILE" 2>&1 &
    done

    # 等待服务启动
    echo "Waiting 60s for vLLM to start..."
    sleep 60
    
    # 创建当前 step 的输出目录
    mkdir -p $CURRENT_OUTPUT_DIR

    # 3. 运行评测脚本
    # 将端口数组转换为逗号分隔的字符串
    PORTS_STR=$(IFS=,; echo "${PORTS[*]}")
    
    python vllm_reason.py \
        --model $MODEL_PATH \
        --file $DATA \
        --ports $PORTS_STR \
        --repeat $REPEAT \
        --concurrency $CONCURRENCY \
        --output_dir $CURRENT_OUTPUT_DIR

    # 4. 结果汇总
    if [ -f "$CURRENT_OUTPUT_DIR/results_summary.json" ]; then
        P1=$(python -c "import json; print(json.load(open('$CURRENT_OUTPUT_DIR/results_summary.json'))['pass@1'])" 2>/dev/null || echo "N/A")
        PK=$(python -c "import json; print(json.load(open('$CURRENT_OUTPUT_DIR/results_summary.json'))['pass@$REPEAT'])" 2>/dev/null || echo "N/A")
        
        echo "$STEP_NAME,$MODEL_PATH,$P1,$PK" >> "$SUMMARY_FILE"
        echo "Recorded results for $STEP_NAME: Pass@1=$P1, Pass@$REPEAT=$PK"
    else
        echo "$STEP_NAME,$MODEL_PATH,FAILED,FAILED" >> "$SUMMARY_FILE"
        echo "Error: Result file not generated for $STEP_NAME"
    fi

    # 5. 清理环境
    echo "Cleaning up vLLM services..."
    ps -ef | grep ${ENVIRONMENT} | grep -v grep | awk '{print $2}' | xargs kill -9
    sleep 1m
    ps -ef | grep ${ENVIRONMENT} | grep -v grep | awk '{print $2}' | xargs kill -9
    sleep 1m
    ps -ef | grep ${ENVIRONMENT} | grep -v grep | awk '{print $2}' | xargs kill -9
    echo "Cleanup completed. Waiting for ports to be released..."
    sleep 1m

done

echo "========================================================"
echo "All checkpoints evaluated!"
echo "Summary saved to $SUMMARY_FILE"
echo "========================================================"
