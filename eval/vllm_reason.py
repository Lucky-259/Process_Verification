import sys
import json
import argparse
import threading
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from deepscaler.rewards.math_reward import deepscaler_reward_fn
import traceback

suffix_1 = (
    " Let's think step by step and output the final answer within \\boxed{}."  # original suffix
)
suffix_2 = (
    "\nLet's reason step by step. Enclose the reasoning process within <think>...</think>, then summarize it and present the final answer within \\boxed{} — for example: <think>reasoning process here</think> \\boxed{answer here}."
)
suffix_3 = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}." # DeepSeek
)

def request_model(client, prompt, seed, model):
    """封装一次推理请求，便于在线程池中调用"""
    def single_request():
        response = client.chat.completions.create(
            model=model,
            messages=[
                # {"content": "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think></think> and <answer></answer> tags, respectively, i.e., <think> reasoning process here</think><answer> answer here</answer>. The output of the assistant should be within 4000 tokens.", "role": "system"},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "add_generation_prompt": True,
                "seed": seed,
            },
            # max_tokens=32768,
            temperature=0.6,
            top_p=0.95,
            timeout=100000000,
        )
        return response
    response = single_request()
    model_response = response.choices[0].message.content.strip()

    return model_response

def main(seed):
    parser = argparse.ArgumentParser(description="vLLM AIME 推断并统计 pass@1 与 pass@16")
    parser.add_argument('--model', type=str, required=True, help="vLLM 模型名称")
    parser.add_argument('--file', type=str, required=True, help="测试集 JSON 文件路径")
    parser.add_argument('--ports', type=str, required=True, help="vLLM server 端口号列表，多个端口用逗号分隔")
    parser.add_argument('--repeat', type=int, required=True, help="每个问题重复的次数（例如16）")
    parser.add_argument('--concurrency', type=int, required=True, help="每个端口最大并发请求数 X")
    parser.add_argument('--output_dir', type=str, required=True, help="输出文件夹")
    args = parser.parse_args()
    suffix = suffix_3

    # 构造多个 vLLM 客户端，每个对应一个端口，并用 Semaphore 限制并发数
    ports = args.ports.split(',')
    client_list = []
    for port in ports:
        openai_api_key = "EMPTY"
        openai_api_base = f"http://localhost:{port}/v1"
        client = OpenAI(api_key=openai_api_key, base_url=openai_api_base)
        sem = threading.Semaphore(args.concurrency)
        client_list.append((client, sem))
    total_clients = len(client_list)
    total_concurrency = total_clients * args.concurrency
    print(f"共构造 {total_clients} 个客户端，总并发数为 {total_concurrency}")

    # 读取原始 AIME 测试数据（列表形式），每个元素包含 "problem" 和 "answer"
    with open(args.file, "r", encoding="utf-8") as f:
        original_data = json.load(f)
    if not original_data:
        print("测试集为空。")
        return

    # 构造任务列表，每个任务对应一次请求（同一问题重复 args.repeat 次）
    tasks = []   # 每个元素：(question_index, prompt, ground_truth, repetition_index)
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        if q_idx == 0:
            print(f"问题样例：{prompt}，答案样例：{ground_truth}")
        for rep in range(args.repeat):
            tasks.append((q_idx, prompt, ground_truth, rep))

    # 用于记录每个问题的详细结果（用于保存详细文件）
    detailed_results = {}
    for q_idx, item in enumerate(original_data):
        prompt = item["problem"] + suffix
        prompt = prompt.strip()
        ground_truth = item["answer"]
        detailed_results[q_idx] = {
            "prompt": prompt,
            "ground_truth": ground_truth,
            "responses": [None] * args.repeat,       # 模型的 args.repeat 条回复
            "correct_flags": [False] * args.repeat     # 每条回复是否正确
        }

    total_responses = len(tasks)
    correct_count = 0  # 用于统计所有单次请求中正确的数量（pass@1）

    # 使用全局线程池进行并发请求，总并发数为 M*X
    executor = ThreadPoolExecutor(max_workers=total_concurrency)
    future_list = []  # 存储 (question_index, repetition_index, prompt, ground_truth, future)
    global_index = 0  # 用于轮询分配客户端

    def request_task(client, semaphore, prompt, seed, model):
        with semaphore:
            return request_model(client, prompt, seed, model)

    for task in tasks:
        q_idx, prompt, ground_truth, rep = task
        client, sem = client_list[global_index % total_clients]
        global_index += 1
        fut = executor.submit(request_task, client, sem, prompt, seed, args.model)
        future_list.append((q_idx, rep, prompt, ground_truth, fut))

    # 等待所有任务完成并处理结果
    for q_idx, rep, prompt, ground_truth, fut in future_list:
        try_count = 0
        while try_count <= 10:
            try:
                model_response = fut.result()
                break
            except Exception as e:
                try_count += 1
                error_message = traceback.format_exc()
                print("Error, retrying", error_message, flush=True)
                model_response = f"Error, {error_message}"
        processed_response = model_response.replace("\n", "")
        try:
            is_correct = deepscaler_reward_fn(model_response, ground_truth)
        except Exception as e:
            is_correct = False

        # 更新详细结果（按问题分组）
        detailed_results[q_idx]["responses"][rep] = processed_response
        detailed_results[q_idx]["correct_flags"][rep] = is_correct

        if is_correct:
            correct_count += 1

    # 计算 pass@1：所有单次回复中正确比例
    pass_at_1 = correct_count / total_responses

    # 计算 pass@16：对于每个问题，只要16次回复中至少有一次正确，则该题算对
    pass16_correct = 0
    for q_idx, result in detailed_results.items():
        question_correct = any(result["correct_flags"])
        result["is_correct"] = question_correct
        # 计算该问题的正确率（16次回复中正确的比例）
        result["question_accuracy"] = sum(result["correct_flags"]) / args.repeat
        if question_correct:
            pass16_correct += 1
    total_questions = len(original_data)
    pass_at_16 = pass16_correct / total_questions

    # 将统计结果保存到文件1（例如 results_summary.json）
    summary_results = {
        "file": args.file,
        "model": args.model,
        "pass@1": pass_at_1,
        f"pass@{args.repeat}": pass_at_16,
        "total_responses": total_responses,
        "total_questions": total_questions
    }
    with open(args.output_dir + "/results_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, ensure_ascii=False, indent=2)

    # 将详细结果保存到文件2（例如 results_details.json）
    # 每个问题包含 prompt、真实答案、该题正确率、是否正确以及模型的 args.repeat 条回复
    detailed_results_list = []
    for q_idx in sorted(detailed_results.keys()):
        detailed_results_list.append(detailed_results[q_idx])
    with open(args.output_dir + "/results_details.json", "w", encoding="utf-8") as f:
        json.dump(detailed_results_list, f, ensure_ascii=False, indent=2)

    print("######### Deepscaler 统计结果:")
    print(f"pass@1 = {pass_at_1:.4f}, pass@{args.repeat} = {pass_at_16:.4f}")
    print(f"统计信息已保存至 {args.output_dir}/results_summary.json 与 {args.output_dir}/results_details.json")

if __name__ == "__main__":
    main(None)
