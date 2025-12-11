import pandas as pd
import json
import os

def json_to_parquet(config):
    """
    通用转换函数：根据配置字典将不同格式的 JSON 转换为标准 Schema 的 Parquet。
    """
    
    # 1. 解包配置
    json_path = config['io']['input_json']
    parquet_path = config['io']['output_parquet']
    mapping = config['mapping']      # 字段映射关系
    static = config['static_info']   # 静态字段值 (data_source, ability 等)
    sys_prompt = config.get('system_prompt') # 可选的 system prompt
    n_preview = config.get('n_preview', 3)

    # 2. 检查输入文件
    if not os.path.exists(json_path):
        print(f"错误: 找不到文件 {json_path}")
        return

    # 3. 读取 JSON 数据
    print(f"正在读取文件: {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"JSON 读取失败: {e}")
        return

    # 4. 预览原始数据 (打印第一条，方便核对 key 是否正确)
    print("-" * 50)
    print(f"原始数据样本 (前 1 条):")
    print(json.dumps(raw_data[:1], indent=2, ensure_ascii=False))
    
    processed_rows = []
    
    # 5. 核心处理循环
    for idx, item in enumerate(raw_data):
        # --- A. 动态提取内容 ---
        # 根据 mapping 配置的 key 从 item 中取值，如果找不到则给空字符串
        user_content = item.get(mapping['user_content_key'], "")
        ground_truth = item.get(mapping['ground_truth_key'], "")
        
        # --- B. 构建 prompt 列表 (处理可选的 System Prompt) ---
        prompt_structure = []
        
        # 只有在配置了 system_prompt 且不为空时才添加
        if sys_prompt and sys_prompt.strip():
            prompt_structure.append({
                "role": "system",
                "content": sys_prompt
            })
            
        # 必须添加的 user 部分
        prompt_structure.append({
            "role": "user",
            "content": user_content
        })

        # --- C. 构建 reward_model (嵌套结构) ---
        reward_model_structure = {
            "style": "rule",
            "ground_truth": ground_truth
        }

        # --- D. 构建 extra_info (嵌套结构) ---
        extra_info_structure = {
            "split": static.get('split', 'default'),
            "index": idx
        }

        # --- E. 组装整行 ---
        row = {
            "data_source": static.get('data_source', ''),
            "prompt": prompt_structure,
            "ability": static.get('ability', 'general'),
            "reward_model": reward_model_structure,
            "extra_info": extra_info_structure
        }
        
        processed_rows.append(row)

    # 6. 生成 DataFrame
    df = pd.DataFrame(processed_rows)

    # 7. 预览处理结果
    print("-" * 50)
    print(f"转换后数据预览 (前 {n_preview} 条):")
    # 临时调整显示设置以便看清嵌套结构
    with pd.option_context('display.max_colwidth', None, 'display.max_columns', None):
        print(df.head(n_preview))
    print("-" * 50)

    # 8. 保存文件
    try:
        df.to_parquet(parquet_path, engine='pyarrow', index=False)
        print(f"处理完成。")
        print(f"保存路径: {parquet_path}")
        print(f"总条目数: {len(df)}")
    except Exception as e:
        print(f"Parquet 保存失败: {e}")

# ================= 自动化配置区域 =================
# 你只需要修改这个字典中的内容即可适配不同的数据集

# 配置示例 1：处理 AIME 数据 (有 System Prompt)
config_aime = {
    "io": {
        "input_json": "/mnt/luoyingfeng/changkaiyan/Process_Verification/deepscaler/data/test/aime.json",
        "output_parquet": "/mnt/luoyingfeng/changkaiyan/Process_Verification/deepscaler/data/aime_verl_test.parquet"
    },
    "mapping": {
        # 原始 JSON 中对应 "问题" 的 Key 是什么？
        "user_content_key": "problem",  
        # 原始 JSON 中对应 "答案/Solution" 的 Key 是什么？
        "ground_truth_key": "answer"
    },
    "static_info": {
        "data_source": "aime",
        "ability": "math",
        "split": "test"
    },
    # 如果不需要 System Prompt，将此处设为 None 或空字符串 ""
    "system_prompt": "Please reason step by step, and put your final answer within \\boxed{}.", # None
    "n_preview": 2
}


# ================= 主程序入口 =================
if __name__ == "__main__":
    
    json_to_parquet(config_aime)
    
    # 如果你有其他数据集，解除下面的注释并运行
    # json_to_parquet(config_gsm8k)