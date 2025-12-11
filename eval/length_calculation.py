import json
import os
from tqdm import tqdm
import argparse
from transformers import AutoTokenizer

def calculate_mean_response_length(input_file, output_file, summary_file, model_path):
    """
    读取JSON文件，计算每个元素的responses数组的平均字符串长度，
    并保存处理结果
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出JSON文件路径（包含problem和mean_output_length）
        summary_file: 汇总文件路径（包含所有元素的平均长度统计）
    """
    
    # 1. 读取输入JSON文件
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
        return
    except json.JSONDecodeError:
        print(f"错误：文件 {input_file} 不是有效的JSON格式")
        return
    
    # 检查数据是否是数组格式
    if not isinstance(data, list):
        print("错误：输入JSON文件应该是一个数组格式")
        return
    
    # 2. 处理每个元素
    processed_data = []
    all_mean_lengths = []
    
    for i, item in tqdm(enumerate(data), total=len(data), desc="processing..."):
        # 检查item是否是字典类型
        if not isinstance(item, dict):
            print(f"警告：索引 {i} 的元素不是字典类型，已跳过")
            continue
        
        # 检查是否存在responses键
        if 'responses' not in item:
            print(f"警告：索引 {i} 的元素没有responses键，已跳过")
            continue
        
        responses = item.get('responses', [])
        
        # 检查responses是否是列表
        if not isinstance(responses, list):
            print(f"警告：索引 {i} 的responses不是列表类型，已跳过")
            continue
        
        # 计算平均字符串长度
        if responses:  # 如果列表不为空
            mean_length = 0
            tokenizer = AutoTokenizer.from_pretrained(model_path, fix_mistral_regex=True)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            encoded = tokenizer(responses, padding=False, truncation=False, return_tensors=None)
            for input_ids in encoded['input_ids']:
                mean_length += len(input_ids)

            mean_length /= len(responses)
        else:
            mean_length = 0  # 空列表的平均长度为0
        
        # 保存处理后的数据
        processed_item = {}
        
        # 保留prompt键（如果存在）
        if 'prompt' in item:
            processed_item['prompt'] = item['prompt']
        else:
            processed_item['prompt'] = f"item_{i}"  # 如果没有prompt键，使用默认值
        
        # 添加平均长度
        processed_item['mean_output_length'] = mean_length
        
        processed_data.append(processed_item)
        all_mean_lengths.append(mean_length)
    
    # 3. 保存处理后的JSON文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)
        print(f"已保存处理后的数据到: {output_file}")
    except Exception as e:
        print(f"保存文件时出错: {e}")
        return
    
    # 4. 计算所有元素的平均长度并保存统计信息
    if all_mean_lengths:
        overall_mean = sum(all_mean_lengths) / len(all_mean_lengths)
        
        summary = {
            "total_questions": len(processed_data),
            "overall_mean_length": overall_mean,
            "length_distribution": {
                "0-1000": len([l for l in all_mean_lengths if 0 <= l < 1000]),
                "1000-5000": len([l for l in all_mean_lengths if 1000 <= l < 5000]),
                "5000-10000": len([l for l in all_mean_lengths if 5000 <= l < 10000]),
                "10000+": len([l for l in all_mean_lengths if l >= 10000])
            }
        }
        
        try:
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            print(f"已保存统计信息到: {summary_file}")
        except Exception as e:
            print(f"保存统计文件时出错: {e}")
    else:
        print("警告：没有有效的元素可以计算统计信息")

# 使用示例
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="统计平均长度")
    parser.add_argument('--model', type=str, required=True, help="模型路径")
    parser.add_argument('--dir', type=str, required=True, help="输入json文件路径")
    args = parser.parse_args()

    # 文件路径配置
    model_path = args.model
    input_file = args.dir + "/results_details.json"  # 输入JSON文件路径
    output_file = args.dir + "/length_details.json" # 输出JSON文件路径
    summary_file = args.dir + "/length_summary.json" # 统计信息文件路径
    
    # 执行处理
    calculate_mean_response_length(input_file, output_file, summary_file, model_path)
    
    # 显示处理结果
    print("\n处理完成！")
    print(f"输入文件: {input_file}")
    print(f"输出文件: {output_file}")
    print(f"统计文件: {summary_file}")
    
    # 显示输出文件内容
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            output_data = json.load(f)
        print(f"\n输出文件内容（前3个元素）:")
        for i, item in enumerate(output_data[:3]):
            print(f"  元素 {i}: {item}")
    except:
        pass