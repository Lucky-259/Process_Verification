import pandas as pd
import json

def convert_dataset(input_path, output_path):
    """转换整个 parquet 数据集"""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    
    if "problem" in df.columns:
        # 将 problem 列转换为 prompt 消息数组
        df["prompt"] = df["problem"].apply(
            lambda x: [{"content": x, "role": "user"}] if pd.notnull(x) else []
        )
        df = df.drop(columns=["problem"])
    if "answer" in df.columns:
        df["reward_model"] = df["answer"].apply(
            lambda x: {"ground_truth": x, "style": "rule"} if pd.notnull(x) else {}
        )
        df = df.drop(columns=["answer"])
    if "problem_number" in df.columns:
        df["extra_info"] = df["problem_number"].apply(
            lambda x: {"index": x, "split": "test"}
        )
        df = df.drop(columns=["problem_number"])
    
    df["data_source"] = "aime"
    
    # 保存
    df.to_parquet(output_path, index=False)
    return df

# 使用
convert_dataset("/mnt/luoyingfeng/changkaiyan/Process_Verification/data_preprocess/test/aime.json", "/mnt/luoyingfeng/changkaiyan/Process_Verification/data_preprocess/test/aime.parquet")