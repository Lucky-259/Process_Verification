import json

def calculate_empty_steps_ratio(file_path):
    """
    计算JSONL文件中steps键为空列表的元素占总元素的比例
    
    Args:
        file_path (str): JSONL文件路径
    
    Returns:
        float: 空steps元素的比例
    """
    total_elements = 0
    empty_steps_elements = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_number, line in enumerate(file, 1):
                # 跳过空行
                if not line.strip():
                    continue
                    
                try:
                    data = json.loads(line)
                    total_elements += 1
                    
                    # 检查steps键是否存在且为空列表
                    if isinstance(data.get('steps'), list) and len(data.get('steps')) == 0:
                        empty_steps_elements += 1
                        
                except json.JSONDecodeError as e:
                    print(f"警告: 第{line_number}行JSON解析错误: {e}")
                    continue
                    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 不存在")
        return None
    except Exception as e:
        print(f"错误: 读取文件时发生错误: {e}")
        return None
    
    if total_elements == 0:
        print("警告: 文件中没有有效的JSON元素")
        return 0.0
    
    ratio = empty_steps_elements / total_elements
    return total_elements, empty_steps_elements, ratio

def main():
    # 文件路径
    file_path = "verl/prm/discriminative_prm/output_log.jsonl"  # 替换为你的文件路径
    
    # 计算比例
    total, empty_steps, ratio = calculate_empty_steps_ratio(file_path)
        
    print("=" * 50)
    print("统计结果:")
    print(f"总元素数量: {total}")
    print(f"steps为空列表的元素数量: {empty_steps}")
    print(f"空steps元素比例: {ratio:.4f} ({ratio:.2%})")
    print("=" * 50)

if __name__ == "__main__":
    main()