#!/bin/bash
#SBATCH --job-name=sb_train_job
#SBATCH --partition=general          
#SBATCH --nodes=1                   
#SBATCH --ntasks=1                   
#SBATCH --gres=gpu:a100:8               
#SBATCH --cpus-per-task=16          
#SBATCH --mem=256G                  
#SBATCH --time=12:00:00            
#SBATCH --output=${LOG_DIR}/slurm_logs/verl_grpo/%x_%j.out      
#SBATCH --error=${LOG_DIR}/slurm_logs/verl_grpo/%x_%j.err  
set -x

# Configuration through environment variables
# Set these variables before running:
export PROJECT_HOME="Process_Verification"
export LOG_DIR="/path/to/logs"
#export WANDB_API_KEY="65b60732e2f7c060c315c11549a69110ea53e184"
export DATASET_DIR="deepscaler/data"

# ----------------------------------------
# To change the reward function hyperparameters, please change the alpha and beta in the following:
# /ShorterBetter/verl/verl/workers/reward_manager/naive.py line 244
# By default, alpha=2.0, beta=0.001
# ----------------------------------------

# ----------------------------------------
# The training process will print out the output lengths and correct counts for each batch.
# You can use check_acc_len.py to plot the accuracy and output length trends.
# ----------------------------------------


# Warning: Export VLLM_ATTENTION_BACKEND on every machine before starting Ray cluster.
# vLLM without XFORMERS will results in CUDA errors.
export PYTHONPATH="${PROJECT_HOME}:$PYTHONPATH"
export VLLM_ATTENTION_BACKEND=XFORMERS
# ----------------------------------------

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL_PATH="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# Default model path if not specified
MODEL_PATH=${MODEL_PATH:-"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"}
PROJECT_NAME='ALP'
EXPERIMENT_NAME='alp_disprm_1.5B_16k_1e-7'

# Train over a single node, 8 A100-80GB GPUs.
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATASET_DIR}/train_filtered.parquet \
    data.val_files=${DATASET_DIR}/aime.parquet \
    data.train_batch_size=128 \
    data.val_batch_size=512 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    +actor_rollout_ref.rollout.val_temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.n=8 \
    +actor_rollout_ref.rollout.n_val=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    reward_model.reward_manager=batch \
    custom_reward_function.path=verl/prm/discriminative_prm/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    +custom_reward_function.kwargs.beta=1e-7 \
    +custom_reward_function.kwargs.prm_threshold=0.8 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    ++trainer.val_before_train=False \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=-1 \
    trainer.default_hdfs_dir=null \
    actor_rollout_ref.actor.checkpoint.save_contents=['hf_model'] \
    trainer.total_epochs=1 "${@:1}" \
    trainer.total_training_steps=100 > $ROOT_DIR/checkpoints/${project_name}/${experiment_name}.log

# Notes on the training cofig:

# 1. data.train_batch_size=128; can be change to 64

# 2. data.train_files=${DATASET_DIR}/train_filtered.parquet; Here _filtered means the datapoints are filtered by the data.max_prompt_length=1500

# 3. data.val_files=${DATASET_DIR}/aime.parquet; But we don't evaluate during training, this is just a placeholder.

# 4. actor_rollout_ref.rollout.n=8; can be changed to 4 but the length reduction process will be slower.

# 5. actor_rollout_ref.actor.optim.lr=1e-6 is recommended.