export VLLM_USE_V1=0

PROJECT_NAME='ALP'
EXPERIMENT_NAME='alp_disprm_1.5B_4k_1e-7'
GLOBAL_STEP='global_step_10'
ENVIRONMENT='cky_alp'

MODEL_PATH="../checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}/${GLOBAL_STEP}/actor/huggingface"
OUTPUT_DIR="./eval_results/${EXPERIMENT_NAME}/${GLOBAL_STEP}"
DATA="../deepscaler/data/test/${1}.json"
REPEAT=${2:-64}
CONCURRENCY=${3:-64}

# 定义所有可用端口和设备
ALL_PORTS=(8010 8011 8012 8013 8014 8015 8016 8017)
DEVICES=(0 1)  # 根据实际GPU数量修改

# 根据设备数量自动选择端口
NUM_DEVICES=${#DEVICES[@]}
PORTS=("${ALL_PORTS[@]:0:$NUM_DEVICES}")
PORTS_STR=$(IFS=,; echo "${PORTS[*]}")
echo "Using $NUM_DEVICES GPUs with ports: ${PORTS[*]}"

# 检查模型路径是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "Warning: Model path not found: $MODEL_PATH. Skipping..."
    exit 1
fi

# 创建总输出目录
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# 检查是否已存在结果 (防止重复跑)
if [ -f "$OUTPUT_DIR/results_summary.json" ]; then
    echo "Warning: Result already exists in $OUTPUT_DIR/results_summary.json"
    echo "If you want to re-run, please delete the folder or comment out this check."
    exit 0 # 如果不想覆盖，取消注释这行
fi

for i in "${!DEVICES[@]}"; do
    PORT="${PORTS[$i]}"
    DEVICE="${DEVICES[$i]}"
    LOG_FILE="${OUTPUT_DIR}/vllm_server_gpu${DEVICE}.log"

    CUDA_VISIBLE_DEVICES=$DEVICE vllm serve $MODEL_PATH \
    --max_model_len 32768 \
    --enforce-eager \
    --gpu-memory-utilization 0.93 \
    --port $PORT > "$LOG_FILE" 2>&1 &
done

# 等待服务启动
echo "Waiting 60s for vLLM to start..."
sleep 60

# 运行评测脚本
python vllm_reason.py \
    --model $MODEL_PATH \
    --file $DATA \
    --ports $PORTS_STR \
    --repeat $REPEAT \
    --concurrency $CONCURRENCY \
    --output_dir $OUTPUT_DIR

echo "Cleaning up vLLM services..."
ps -ef | grep ${ENVIRONMENT} | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 60
ps -ef | grep ${ENVIRONMENT} | grep -v grep | awk '{print $2}' | xargs kill -9
sleep 30
ps -ef | grep ${ENVIRONMENT} | grep -v grep | awk '{print $2}' | xargs kill -9
echo "Cleanup completed. Waiting for ports to be released..."
sleep 30