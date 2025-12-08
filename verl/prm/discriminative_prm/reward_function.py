# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
判别式 PRM 奖励函数

奖励公式:
    r(y, q) = 1[answer(y) = y*] - β·N·max(p_solved(q), K^(-1))
    
其中:
    - 1[answer(y) = y*]: 最终答案是否正确 (0 或 1)
    - β: 全局长度惩罚系数
    - N: 响应的 token 数量
    - p_solved(q) = (1/K) Σ 1[answer(y^(k)) = y*]: 步骤准确率
    - K: 步骤总数
    
特殊情况处理:
    - 如果没有 </think> 标签或没有步骤：
        * 仍然计算 accuracy（outcome reward）
        * p_solved = 0（因为没有步骤可评估）
        * 长度惩罚系数使用 K^(-1)，当 K=0 时，视为极大值（使用一个大的默认值）
    
使用方法:
    reward_model.reward_manager=batch \
    custom_reward_function.path=prm/discriminative_prm/reward_function.py \
    custom_reward_function.name=compute_score_batch \
    custom_reward_function.kwargs.beta=1e-7 \
    custom_reward_function.kwargs.prm_threshold=0.8
"""

import os
import re
import json
import requests
from time import sleep
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple
from pathlib import Path

from transformers import AutoTokenizer

from verl.utils.reward_score.math import last_boxed_only_string, remove_boxed, is_equiv

# ============================================================================
# 全局配置
# ============================================================================

# PRM API 配置
PRM_API_URL = "http://localhost:8034/pooling"  # 修改为你的 PRM API 地址
PRM_MODEL_NAME = "/mnt/luoyingfeng/model_card/Qwen2.5-Math-PRM-7B"

# API 调用配置
MAX_RETRIES = 5
BASE_DELAY = 2
MAX_WORKERS = 8
TIMEOUT = 500

# 初始化 tokenizer
tokenizer = AutoTokenizer.from_pretrained(PRM_MODEL_NAME, trust_remote_code=True)


# ============================================================================
# 响应解析函数
# ============================================================================

def extract_content_after_think(solution_str):
    """
    提取 </think> 标签之后的内容
    
    Args:
        solution_str: 完整的 solution 字符串
    
    Returns:
        content: </think> 之后的内容（如果没有标签，返回 None）
        steps: 分割后的步骤列表
    """
    # 查找 </think> 标签
    match = re.search(r'</think>', solution_str, re.IGNORECASE)
    
    if not match:
        # 如果没有 </think> 标签，返回 None
        return None, []
    
    # 提取 </think> 之后的内容
    content = solution_str[match.end():].strip()
    
    # 按 \n\n 分割成步骤
    steps = [s.strip() for s in content.split('\n\n') if s.strip()]
    
    return content, steps


def verify_correctness(solution_str, ground_truth):
    """
    验证最终答案的正确性
    
    Args:
        solution_str: 解答字符串
        ground_truth: 标准答案
    
    Returns:
        是否正确 (bool)
    """
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            return is_equiv(answer, ground_truth)
    except Exception as e:
        print(f"Error in verify_correctness: {e}")
    
    return False


# ============================================================================
# PRM API 调用函数
# ============================================================================

def call_prm_api(question, response_steps, system_prompt=None):
    """
    调用 PRM API 获取步骤级分数
    
    Args:
        question: 问题文本
        response_steps: 响应步骤列表（纯文本）
        system_prompt: 系统提示（可选）
    
    Returns:
        step_scores: 步骤分数列表 [float, ...]
    """
    if system_prompt is None:
        system_prompt = "Please reason step by step, and put your final answer within \\boxed{}."
    
    # 构建消息格式（使用 <extra_0> 作为步骤分隔符）
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
        {"role": "assistant", "content": "<extra_0>".join(response_steps) + "<extra_0>"},
    ]
    
    # 构建 API 请求
    prompt = {
        "model": PRM_MODEL_NAME,
        "messages": messages
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            # 发送 HTTP 请求
            headers = {"User-Agent": "Test Client"}
            response = requests.post(PRM_API_URL, headers=headers, json=prompt, timeout=TIMEOUT)
            response.raise_for_status()
            
            # 解析响应
            output_json = response.json()
            output_data = output_json['data'][0]['data']
            
            # 提取步骤分数（取第二列，即正类概率）
            step_scores = [item[1] for item in output_data]
            
            return step_scores
            
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                print(f"PRM API call failed (attempt {attempt + 1}/{MAX_RETRIES}): {e}")
                delay = BASE_DELAY * (2 ** attempt)
                sleep(delay)
            else:
                print(f"PRM API call failed after {MAX_RETRIES} attempts: {e}")
                # 返回默认分数（0.5 表示不确定）
                return [0.5] * len(response_steps)
    
    return [0.5] * len(response_steps)


# ============================================================================
# 奖励计算函数
# ============================================================================

def compute_reward(
    accuracy,
    step_scores,
    num_tokens,
    prm_threshold=0.8,
    beta=1e-7,
    no_steps_penalty_weight=100.0
):
    """
    根据论文公式计算奖励
    
    公式: r(y, q) = accuracy - β·N·max(p_solved(q), K^(-1))
    
    Args:
        accuracy: 最终答案是否正确 (0 或 1)
        step_scores: PRM 步骤分数列表（如果没有步骤，则为空列表）
        num_tokens: 响应的 token 数量 (N)
        prm_threshold: PRM 阈值（判定步骤正确的分数）
        beta: 全局长度惩罚系数
        no_steps_penalty_weight: 当 K=0 时，K^(-1) 的替代值（较大值，表示重惩罚）
    
    Returns:
        reward: 奖励值
        metrics: 指标字典
    """
    K = len(step_scores)  # 步骤总数
    
    if K == 0:
        # 如果没有步骤
        # p_solved = 0（没有步骤可评估）
        p_solved = 0.0
        step_correctness = []
        
        # 长度惩罚系数使用一个较大的值（模拟 K^(-1) → ∞）
        # 但不能太大，以免完全抵消 accuracy
        length_coefficient = no_steps_penalty_weight
        
    else:
        # 正常情况：有步骤
        # 计算步骤准确率: p_solved = (1/K) Σ 1[score > threshold]
        step_correctness = [1 if score > prm_threshold else 0 for score in step_scores]
        p_solved = sum(step_correctness) / K
        
        # 长度惩罚系数: max(p_solved, K^(-1))
        length_coefficient = max(p_solved, 1.0 / K)
    
    # 奖励公式: r(y, q) = accuracy - β·N·max(p_solved, K^(-1))
    reward = float(accuracy) - beta * num_tokens * length_coefficient
    
    # 收集指标
    metrics = {
        "accuracy": accuracy,
        "num_tokens": num_tokens,
        "num_steps": K,
        "p_solved": p_solved,
        "length_coefficient": length_coefficient,
        "step_scores": step_scores,
        "step_correctness": step_correctness,
        "reward": reward,
        "has_steps": K > 0
    }
    
    return reward, metrics


# ============================================================================
# 单个样本计算函数
# ============================================================================

def compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs):
    """
    计算单个样本的奖励分数
    
    Args:
        data_source: 数据源
        solution_str: 解答字符串
        ground_truth: 标准答案
        extra_info: 额外信息（包含 question, split 等）
        **kwargs: 其他参数（beta, prm_threshold 等）
    
    Returns:
        reward_score: 奖励分数（float 或 dict）
    """
    split = extra_info.get("split", "train")
    
    # 如果是测试集，使用默认评分
    if split == "test":
        from verl.utils.reward_score import default_compute_score
        return default_compute_score(data_source, solution_str, ground_truth, extra_info)
    
    # 提取参数
    beta = kwargs.get('beta', 1e-7)
    prm_threshold = kwargs.get('prm_threshold', 0.8)
    no_steps_penalty_weight = kwargs.get('no_steps_penalty_weight', 100.0)
    
    # 步骤 1: 验证最终答案正确性（始终计算）
    accuracy = 1.0 if verify_correctness(solution_str, ground_truth) else 0.0
    
    # 步骤 2: 提取 </think> 之后的内容
    content, steps = extract_content_after_think(solution_str)
    
    # 步骤 3: 计算 token 数量
    num_tokens = len(solution_str.split())  # 粗略估算
    
    # 步骤 4: 根据是否有步骤，选择不同的处理路径
    if content is None or len(steps) == 0:
        # 情况 A: 没有 </think> 标签或没有步骤
        # 但仍然给予 accuracy reward，只是长度惩罚更重
        print(f"Warning: No valid steps found, using accuracy with heavy length penalty")
        
        reward, metrics = compute_reward(
            accuracy=accuracy,
            step_scores=[],  # 空列表，表示没有步骤
            num_tokens=num_tokens,
            prm_threshold=prm_threshold,
            beta=beta,
            no_steps_penalty_weight=no_steps_penalty_weight
        )
    else:
        # 情况 B: 有步骤，正常调用 PRM API
        question = extra_info.get("question", "")
        step_scores = call_prm_api(question, steps)
        
        reward, metrics = compute_reward(
            accuracy=accuracy,
            step_scores=step_scores,
            num_tokens=num_tokens,
            prm_threshold=prm_threshold,
            beta=beta,
            no_steps_penalty_weight=no_steps_penalty_weight
        )
    
    return reward


# ============================================================================
# 批量计算函数（主入口）
# ============================================================================

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos, data=None, **kwargs):
    """
    批量计算奖励分数（使用线程池并行）
    
    Args:
        data_sources: 数据源列表
        solution_strs: 解答字符串列表
        ground_truths: 标准答案列表
        extra_infos: 额外信息列表
        data: DataProto 对象（可选，用于获取实际 token 数量）
        **kwargs: 其他参数
            - beta: 长度惩罚系数（默认 1e-7）
            - prm_threshold: PRM 阈值（默认 0.8）
            - max_workers: 最大工作线程数（默认 8）
            - no_steps_penalty_weight: 无步骤时的惩罚权重（默认 100.0）
    
    Returns:
        results: 奖励分数列表
    """
    # 提取参数
    beta = kwargs.get('beta', 1e-7)
    prm_threshold = kwargs.get('prm_threshold', 0.8)
    max_workers = kwargs.get('max_workers', MAX_WORKERS)
    no_steps_penalty_weight = kwargs.get('no_steps_penalty_weight', 100.0)
    
    print(f"\n{'='*80}")
    print(f"PRM Reward Calculation")
    print(f"  Total samples: {len(data_sources)}")
    print(f"  Beta: {beta}")
    print(f"  PRM threshold: {prm_threshold}")
    print(f"  Max workers: {max_workers}")
    print(f"  No-steps penalty weight: {no_steps_penalty_weight}")
    print(f"{'='*80}\n")
    
    # ========================================================================
    # 步骤 1: 预处理 - 提取步骤和验证正确性
    # ========================================================================
    
    preprocessed_data = []
    
    for idx, (data_source, solution_str, ground_truth, extra_info) in enumerate(
        zip(data_sources, solution_strs, ground_truths, extra_infos, strict=True)
    ):
        split = extra_info.get("split", "train")
        
        # 测试集使用默认评分
        if split == "test":
            from verl.utils.reward_score import default_compute_score
            result = default_compute_score(data_source, solution_str, ground_truth, extra_info)
            preprocessed_data.append({
                "index": idx,
                "is_test": True,
                "result": result
            })
            continue
        
        # 验证最终答案正确性（始终计算）
        accuracy = 1.0 if verify_correctness(solution_str, ground_truth) else 0.0
        
        # 提取 </think> 之后的内容
        content, steps = extract_content_after_think(solution_str)
        
        # 计算 token 数量
        if data is not None:
            # 从 DataProto 获取实际 token 数量
            prompt_length = data.batch['prompts'][idx].shape[-1]
            num_tokens = data.batch['attention_mask'][idx][prompt_length:].sum().item()
        else:
            # 粗略估算
            num_tokens = len(solution_str.split())
        
        # 保存预处理结果
        preprocessed_data.append({
            "index": idx,
            "is_test": False,
            "has_steps": content is not None and len(steps) > 0,
            "question": extra_info.get("question", ""),
            "steps": steps if content is not None else [],
            "accuracy": accuracy,
            "num_tokens": num_tokens,
            "solution_str": solution_str,
            "ground_truth": ground_truth
        })
    
    # ========================================================================
    # 步骤 2: 批量调用 PRM API（仅对有步骤的样本）
    # ========================================================================
    
    def call_prm_for_sample(item):
        """为单个样本调用 PRM"""
        if item["is_test"] or not item.get("has_steps", False):
            # 测试集或没有步骤，不调用 PRM
            return item
        
        question = item["question"]
        steps = item["steps"]
        
        # 调用 PRM API
        step_scores = call_prm_api(question, steps)
        item["step_scores"] = step_scores
        
        return item
    
    print(f"Calling PRM API for samples with steps...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        preprocessed_data = list(executor.map(call_prm_for_sample, preprocessed_data))
    
    print(f"PRM API calls completed.\n")
    
    # ========================================================================
    # 步骤 3: 计算每个样本的奖励
    # ========================================================================
    
    results = []
    
    for item in preprocessed_data:
        if item["is_test"]:
            # 测试集样本
            results.append(item["result"])
        else:
            # 训练样本：计算奖励
            step_scores = item.get("step_scores", [])
            
            reward, metrics = compute_reward(
                accuracy=item["accuracy"],
                step_scores=step_scores,
                num_tokens=item["num_tokens"],
                prm_threshold=prm_threshold,
                beta=beta,
                no_steps_penalty_weight=no_steps_penalty_weight
            )
            
            results.append(reward)
            
            # 打印前几个样本的详细信息
            if item["index"] < 3:
                print(f"\n[Sample {item['index']}]")
                print(f"  Has steps: {metrics['has_steps']}")
                print(f"  Accuracy: {metrics['accuracy']}")
                print(f"  Num tokens: {metrics['num_tokens']}")
                print(f"  Num steps: {metrics['num_steps']}")
                print(f"  p_solved: {metrics['p_solved']:.3f}")
                print(f"  Length coefficient: {metrics['length_coefficient']:.3f}")
                print(f"  Reward: {metrics['reward']:.4f}")
        
        # 保存日志
        log_dict = {
            "steps": item.get("steps", []),
            "step_scores": item.get("step_scores", []),
            "correctness": item.get("accuracy", 0.0),
            "num_tokens": item.get("num_tokens"),
            "reward": results[-1],
            "has_steps": item.get("has_steps", False)
        }
        log_dict["p_solved"] = sum(1 for x in log_dict["step_scores"] if x > prm_threshold) / len(log_dict["step_scores"]) if len(log_dict["step_scores"]) != 0 else 0
        
        file_path = "verl/prm/discriminative_prm/output_log.jsonl"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_dict, ensure_ascii=False) + '\n')

    print(f"\n{'='*80}")
    print(f"Reward calculation completed for {len(results)} samples")
    print(f"{'='*80}\n")
    
    return results