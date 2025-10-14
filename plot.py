import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_excel_files(directory):
    """
    读取指定目录下的所有 Excel 文件，绘制每个文件的数据折线图，并将图表保存为 PNG 文件，文件名与原 Excel 文件相同。

    参数:
    directory (str): 存放 Excel 文件的目录路径
    """
    # 获取目录下的所有 Excel 文件

    excel_files = [f for f in os.listdir(directory) if f.endswith('.xlsx')]

    # 遍历每个 Excel 文件
    for file in excel_files:
        # 构建完整的文件路径
        file_path = os.path.join(directory, file)
        
        # 读取 Excel 文件
        df = pd.read_excel(file_path, engine='openpyxl')
        
        # 确保数据有三列
        if df.shape[1] != 3:
            print(f"{file} error")
            continue
        
        # 创建一个图形
        plt.figure(figsize=(15, 6))

        # 绘制三条折线
        for i in range(3):
            plt.plot(df.index, df.iloc[:, i], label=df.columns[i])

        # 添加标题和标签
        plt.title(f'{file}')
        plt.xlabel('rnd')
        plt.ylabel('y')

        # 添加图例
        plt.legend()

        # 保存图表为 PNG 文件，文件名与原 Excel 文件相同
        save_path = os.path.join(directory, f'{os.path.splitext(file)[0]}.png')
        plt.savefig(save_path)

        # 清空当前图形，准备绘制下一个文件的图表
        plt.close()

    print("所有文件处理完成！")

if __name__ == '__main__':
    plot_excel_files('./FedResultv6/v6fedresult012all')