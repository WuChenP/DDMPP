import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 设置中文字体，解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_excel_data(file_path):
    """读取数据（处理复合表头）"""
    try:
        df = pd.read_excel(file_path, header=1)  # 从第2行开始读数据
        df.columns = ['温度', '盐种类', '盐浓度（mM）', '聚合物浓度(mM)', '剪切速率', '应力', '[n]']
        print(f"数据形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取失败: {e}")
        raise


def preprocess_corr(df, include_categorical=False):
    """预处理：处理缺失值，可选是否编码分类特征（盐种类）"""
    df_processed = df.copy()

    # 1. 填充缺失值：数值列均值，类别列众数
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

    cat_cols = df_processed.select_dtypes(include=object).columns
    for col in cat_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # 2. 处理分类特征（盐种类）：
    if include_categorical and '盐种类' in cat_cols:
        # 独热编码（drop_first=True避免多重共线性）
        df_processed = pd.get_dummies(df_processed, columns=['盐种类'], drop_first=True)

    # 3. 仅保留数值特征（包括编码后的分类特征）
    numeric_features = df_processed.select_dtypes(include=np.number).columns
    return df_processed[numeric_features]


def plot_correlation_heatmap(df, title="特征相关性热图"):
    """绘制相关性热图（红-蓝配色，带数值标注）"""
    # 计算皮尔逊相关系数
    corr = df.corr()

    # 绘制热图
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,  # 显示相关系数数值
        cmap="RdBu_r",  # 红（正相关）→ 蓝（负相关）配色，与参考图一致
        vmin=-1, vmax=1,  # 颜色范围：-1（蓝）到1（红）
        fmt=".2f",  # 数值保留2位小数
        linewidths=0.5,  # 格子线宽度
        annot_kws={"size": 10}  # 注释字体大小
    )
    plt.title(title, fontsize=14)
    plt.xticks(rotation=45, ha="right")  # 旋转X轴标签，避免重叠
    plt.yticks(rotation=0)  # Y轴标签不旋转
    plt.tight_layout()  # 自动调整布局
    plt.savefig('retu.png', dpi=300)
    plt.show()


def main():
    file_path = '../data/original experimental data.xlsx'  # 替换为实际路径
    df = read_excel_data(file_path)

    # 场景1：仅分析数值特征（排除盐种类，因为是分类变量）
    df_numeric = preprocess_corr(df, include_categorical=False)
    # 场景2：编码盐种类后分析（需打开下面一行）
    # df_numeric = preprocess_corr(df, include_categorical=True)

    plot_correlation_heatmap(df_numeric, title="特征相关性热图（数值特征）")


if __name__ == "__main__":
    main()