import json
import os
import csv
import re
import argparse

def extract_steps_and_lengths(base_dir):
    """
    读取base_dir下所有子文件夹中的length_summary.json文件
    提取step（文件夹名）和overall_mean_length
    
    Args:
        base_dir: 基础目录路径
    """
    
    results = []
    
    # 遍历base_dir下的所有子文件夹
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        
        # 确保是文件夹
        if os.path.isdir(folder_path):
            json_file = os.path.join(folder_path, "length_summary.json")
            
            # 检查文件是否存在
            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 获取overall_mean_length
                    if 'overall_mean_length' in data:
                        length = data['overall_mean_length']
                        results.append((folder_name, length))
                        print(f"√ {folder_name}: {length}")
                    else:
                        print(f"× {folder_name}: 未找到overall_mean_length")
                        
                except Exception as e:
                    print(f"× {folder_name}: 读取失败 - {e}")
    
    return results

def save_to_csv(results, output_file):
    """
    将结果保存为CSV文件
    """
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # 写入标题
        writer.writerow(['step', 'length'])
        
        # 写入数据
        for step, length in results:
            writer.writerow([step, length])
    
    print(f"\n结果已保存到: {output_file}")

# 主程序
if __name__ == "__main__":
    # 设置目录路径
    parser = argparse.ArgumentParser(description="统计平均长度")
    parser.add_argument('--dir', type=str, required=True, help="输入json文件路径")
    args = parser.parse_args()
    base_dir = args.dir
    output_path = os.path.join(base_dir, "all_checkpoints_length_summary.csv")
    
    # 提取数据
    results = extract_steps_and_lengths(base_dir)
    
    if results:
        # 保存为CSV
        save_to_csv(results, output_path)
        
        # 显示统计数据
        print(f"\n总计: {len(results)} 个文件夹")
        lengths = [length for _, length in results]
        print(f"平均长度: {sum(lengths)/len(lengths):.4f}")
    else:
        print("未找到任何有效数据")