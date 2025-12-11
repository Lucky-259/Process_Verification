import re
import matplotlib.pyplot as plt
import numpy as np
import argparse

def extract_losses(log_file_path):
    """
    从.log文件中提取所有actor/pg_loss:后面的浮点数
    
    Args:
        log_file_path: .log文件的路径
    
    Returns:
        list: 包含所有匹配的loss值的列表
    """
    losses = []
    
    # 匹配 actor/pg_loss: 后面的浮点数
    # 正则表达式解释: actor/pg_loss:\s* 匹配"actor/pg_loss:"和可能的空白字符
    # ([-+]?\d*\.\d+|\d+) 匹配浮点数或整数
    pattern = r'actor/pg_loss:\s*([-+]?\d*\.\d+|\d+)'
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                match = re.search(pattern, line)
                if match:
                    # 将匹配到的字符串转换为浮点数
                    loss_value = float(match.group(1))
                    losses.append(loss_value)
        
        print(f"成功提取了 {len(losses)} 个loss值")
        if losses:
            print(f"第一个loss值: {losses[0]}")
            print(f"最后一个loss值: {losses[-1]}")
            
    except FileNotFoundError:
        print(f"错误: 找不到文件 {log_file_path}")
    except Exception as e:
        print(f"读取文件时发生错误: {e}")
    
    return losses

def plot_loss_curve(losses, save_path=None):
    """
    绘制loss曲线图
    
    Args:
        losses: loss值列表
        save_path: 图片保存路径（可选）
    """
    if not losses:
        print("没有数据可绘制图表")
        return
    
    # 创建横坐标（step数） - 最小单位是1
    steps = list(range(len(losses)))
    
    # 设置图表大小
    plt.figure(figsize=(12, 6))
    
    # 绘制折线图 - 使用橙色 (#FF9800) 或 'tab:orange'
    plt.plot(steps, losses, color='#FF9800', linewidth=2, label='actor/pg_loss')
    
    # 添加标题和标签
    plt.title('Training Loss (actor/pg_loss) over Steps', fontsize=16, fontweight='bold')
    plt.xlabel('Training Step', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    
    # 添加网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 添加图例
    plt.legend()
    
    # 确保x轴的最小单位是1
    # 设置x轴为整数刻度
    if len(steps) > 0:
        # 设置x轴范围，从0开始
        plt.xlim(left=0, right=len(steps)-1)
        
        # 如果步数较少，显示所有刻度
        if len(steps) <= 20:
            plt.xticks(steps)
        else:
            # 步数较多时，选择合理的刻度间隔
            # 确保刻度间隔至少为1
            max_ticks = 20  # 最多显示20个刻度
            interval = max(1, len(steps) // max_ticks)
            # 生成刻度位置
            tick_positions = steps[::interval]
            # 确保包含最后一个step
            if steps[-1] not in tick_positions:
                tick_positions = list(tick_positions) + [steps[-1]]
            plt.xticks(tick_positions)
    
    # 设置y轴范围，稍微扩展以显示完整曲线
    if len(losses) > 0:
        y_min = min(losses)
        y_max = max(losses)
        y_range = y_max - y_min
        # 添加5%的边距
        margin = y_range * 0.05 if y_range > 0 else 1
        plt.ylim(bottom=y_min - margin, top=y_max + margin)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    # 显示图表
    plt.show()
    
    # 打印统计信息
    print("\n===== Loss统计信息 =====")
    print(f"总步数: {len(losses)}")
    print(f"最小值: {min(losses):.6f} (step: {np.argmin(losses)})")
    print(f"最大值: {max(losses):.6f} (step: {np.argmax(losses)})")
    print(f"平均值: {np.mean(losses):.6f}")
    print(f"中位数: {np.median(losses):.6f}")


# 主程序
if __name__ == "__main__":
    # 设置你的.log文件路径
    parser = argparse.ArgumentParser(description="统计loss")
    parser.add_argument('--log', type=str, required=True, help="log路径")
    args = parser.parse_args()

    log_file_path = args.log
    
    # 提取loss值
    losses = extract_losses(log_file_path)
    print(losses)
    
    if losses:
        # 绘制图表
        plot_loss_curve(losses, save_path="data_preprocess_scripts/loss_curve.png")
        
        # 可选：将loss值保存到文件
        #with open("loss_values.txt", "w") as f:
        #    for i, loss in enumerate(losses):
        #        f.write(f"Step {i}: {loss}\n")
        #print("Loss值已保存到 loss_values.txt")