# import pandas as pd
# import json
# import numpy as np

# # 读取 Parquet 文件
# df = pd.read_parquet('/mnt/luoyingfeng/changkaiyan/Process_Verification/data_preprocess/test/aime_1.parquet')

# # 获取第一条数据
# first_row = df.iloc[0]

# # 将第一条数据转换为字典
# first_row_dict = first_row.to_dict()

# # 处理字典中的 ndarray 或 Series 类型，转换为 list
# def convert_ndarray(obj):
#     if isinstance(obj, (np.ndarray, pd.Series)): 
#         return obj.tolist()  # 如果是 ndarray 或 Series 转换为列表
#     elif isinstance(obj, list): 
#         return [convert_ndarray(item) for item in obj]  # 递归处理列表中的 ndarray
#     elif isinstance(obj, (bytes, bytearray)):
#         return obj.decode()  # 如果是字节流，转换为字符串
#     return obj

# # 对字典中的值进行转换
# first_row_dict = {key: convert_ndarray(value) for key, value in first_row_dict.items()}

# # 将字典保存为 JSON 文件
# with open('/mnt/luoyingfeng/changkaiyan/Process_Verification/data_preprocess/first_sample/aime_1.json', 'w') as f:
#     json.dump(first_row_dict, f, indent=4)


import pandas as pd
import os

def inspect_and_convert_parquet(
    parquet_path, 
    json_path, 
    preview_n=3,   # 在终端打印查看前 N 条
    save_m='all'   # 保存多少条？可以是数字 (e.g., 100) 或 'all' (全部)
):
    """
    读取 Parquet，统计数据量，预览数据，并按需导出为 JSON。
    """
    
    # 1. 检查文件是否存在
    if not os.path.exists(parquet_path):
        print(f"错误：找不到文件 {parquet_path}")
        return

    print(f"正在读取: {parquet_path} ...")
    
    # 2. 读取 Parquet 文件
    df = pd.read_parquet(parquet_path)
    
    # 3. 统计并输出基础信息
    total_rows = len(df)
    print("-" * 50)
    print(f"读取成功！")
    print(f"数据总行数: {total_rows}")
    print(f"数据列名: {list(df.columns)}")
    print("-" * 50)

    # 4. 终端预览前 N 条 (只是为了看，不影响保存)
    if preview_n > 0:
        print(f"预览前 {preview_n} 条数据:")
        # pd.set_option 防止内容过长被折叠
        with pd.option_context('display.max_columns', None, 'display.max_colwidth', 100):
            print(df.head(preview_n))
        print("-" * 50)

    # 5. 确定要保存的数据切片
    if save_m == 'all':
        save_df = df
        save_msg = f"所有数据 ({total_rows} 条)"
    elif isinstance(save_m, int) and save_m > 0:
        # 如果请求保存的数量大于总数，就保存全部
        actual_save_count = min(save_m, total_rows)
        save_df = df.head(actual_save_count)
        save_msg = f"前 {actual_save_count} 条数据"
    else:
        print("save_m 参数错误，未保存任何数据。")
        return

    # 6. 保存为 JSON
    # orient='records' 会输出 [{col:val}, {col:val}] 的标准列表格式
    # force_ascii=False 保证中文正常显示
    save_df.to_json(
        json_path, 
        orient='records', 
        indent=4, 
        force_ascii=False
    )
    
    print(f"已将 {save_msg} 保存至:\n   -> {json_path}")


# ================= 配置区域 =================

# 输入文件路径
input_file = '/mnt/luoyingfeng/changkaiyan/Process_Verification/deepscaler_sb/data/aime.parquet'

# 输出文件路径
output_file = '/mnt/luoyingfeng/changkaiyan/Process_Verification/deepscaler/data/aime_myy.json'

# --- 执行转换 ---
# 示例 1：保存所有数据
# inspect_and_convert_parquet(input_file, output_file, preview_n=3, save_m='all')

# 示例 2：只保存前 50 条用于测试，并预览前 2 条
inspect_and_convert_parquet(input_file, output_file, preview_n=2, save_m=50)