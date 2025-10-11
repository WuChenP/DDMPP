import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_excel_data(file_path):
    """读取Excel数据（处理复合表头）"""
    try:
        df = pd.read_excel(file_path, header=1)
        df.columns = ['温度', '盐种类', '盐浓度（mM）', '聚合物浓度(mM)', '剪切速率', '应力', '[n]']
        print(f"数据读取成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取失败: {e}")
        raise FileNotFoundError("检查文件路径是否正确")


def preprocess_data(df):
    """预处理：标签编码盐种类特征（原始特征无需额外归一化，仅处理分类特征）"""
    df_processed = df.copy()

    # 填充缺失值
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())
    cat_cols = df_processed.select_dtypes(include=object).columns
    for col in cat_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # 特征和目标分离
    X = df_processed[['温度', '盐种类', '盐浓度（mM）', '聚合物浓度(mM)', '剪切速率', '应力']].copy()
    y = df_processed['[n]'].copy()

    # 对盐种类进行标签编码
    le = LabelEncoder()
    X['盐种类'] = le.fit_transform(X['盐种类'])
    print(f"盐种类编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X, y


def plot_feature_importance(model, X):
    """可视化特征重要性（百分比形式）"""
    importance = model.feature_importances_
    total = sum(importance)
    importance_percent = (importance / total) * 100
    feature_names = X.columns

    # 特征名映射
    name_map = {
        '温度': '温度',
        '盐种类': '盐种类',
        '盐浓度（mM）': '盐浓度 (mM)',
        '聚合物浓度(mM)': '聚合物浓度 (mM)',
        '剪切速率': '剪切速率',
        '应力': '应力'
    }
    display_names = [name_map.get(col, col) for col in feature_names]

    # 排序并打印
    indices = np.argsort(importance_percent)[::-1]
    sorted_importance = importance_percent[indices]
    sorted_features = [display_names[i] for i in indices]
    print("\n=== 随机森林特征重要性百分比 ===")
    for name, imp in zip(sorted_features, sorted_importance):
        print(f"{name}: {imp:.1f}%")

    # 绘图
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(importance_percent)), sorted_importance, align='center')
    plt.yticks(range(len(importance_percent)), sorted_features)
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%', va='center', ha='left')
    plt.xlabel('特征重要性 (%)')
    plt.title('随机森林特征重要性')
    plt.gca().invert_yaxis()
    plt.xlim(0, max(importance_percent) * 1.2)
    plt.tight_layout()
    plt.savefig('rf_feature_importance_percent.png', dpi=300)
    plt.show()


def train_random_forest(X, y):
    """训练随机森林模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 模型性能评估
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_score = r2_score(y_train, train_pred)
    test_score = r2_score(y_test, test_pred)
    print(f"模型性能: 训练集R²={train_score:.3f}, 测试集R²={test_score:.3f}")

    return model, X_test


def normalize_shap_values(shap_values, target_range=(-1, 1)):
    """
    归一化SHAP值到目标范围（默认[-1,1]）
    :param shap_values: 原始SHAP值数组（形状：[样本数, 特征数]）
    :param target_range: 目标范围，默认(-1,1)，可改为(0,1)等
    :return: 归一化后的SHAP值数组
    """
    # 对所有SHAP值（全局）计算最大最小值（保证缩放一致性）
    shap_min = np.min(shap_values)
    shap_max = np.max(shap_values)

    # 避免除以0（若所有SHAP值相同，直接返回目标范围最小值）
    if shap_max == shap_min:
        return np.full_like(shap_values, target_range[0])

    # 线性归一化公式：y = (x - min) / (max - min) * (target_max - target_min) + target_min
    normalized_shap = (shap_values - shap_min) / (shap_max - shap_min) * \
                      (target_range[1] - target_range[0]) + target_range[0]


    print(f"\nSHAP值归一化完成：")
    print(f"原始SHAP值范围：[{shap_min:.4f}, {shap_max:.4f}]")
    print(f"归一化后SHAP值范围：[{np.min(normalized_shap):.1f}, {np.max(normalized_shap):.1f}]")
    return normalized_shap


def plot_shap_beeswarm(model, X, normalize_shap=True, target_range=(-1, 1)):
    """
    生成SHAP蜂拥图（支持SHAP值归一化）
    :param model: 训练好的模型（如随机森林）
    :param X: 输入特征数据（用于匹配特征名，无需归一化）
    :param normalize_shap: 是否归一化SHAP值，默认True
    :param target_range: SHAP值归一化目标范围，默认[-1,1]
    """
    # 1. 计算原始SHAP值
    explainer = shap.TreeExplainer(model)
    print(f"\n计算所有样本的SHAP值，样本数: {X.shape[0]}, 特征数: {X.shape[1]}")
    shap_values = explainer.shap_values(X)

    # 处理回归问题的SHAP值格式（TreeExplainer返回list，取第一个元素）
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    print(f"原始SHAP值形状: {shap_values.shape}")

    # 2. 归一化SHAP值（核心步骤）
    if normalize_shap:
        shap_values = normalize_shap_values(shap_values, target_range=target_range)

    # 定义固定的特征显示顺序（按需要调整）
    fixed_feature_order = [
        '剪切速率',
        '聚合物浓度(mM)',
        '盐浓度（mM）',
        '应力',
        '盐种类',
        '温度'
    ]

    # 特征名映射
    feature_names_map = {
        '温度': '温度',
        '盐种类': '盐种类',
        '盐浓度（mM）': '盐浓度 (mM)',
        '聚合物浓度(mM)': '聚合物浓度 (mM)',
        '剪切速率': '剪切速率',
        '应力': '应力'
    }
    # 转换为显示名称
    fixed_display_names = [feature_names_map[name] for name in fixed_feature_order]

    # 获取固定顺序对应的索引（将SHAP值和特征按固定顺序重新排列）
    feature_indices = [X.columns.get_loc(name) for name in fixed_feature_order]
    shap_values_sorted = shap_values[:, feature_indices]  # 按固定顺序重新排列SHAP值
    X_sorted = X[fixed_feature_order]  # 按固定顺序重新排列特征

    # 绘制蜂群图（使用重新排列后的SHAP值和特征）
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values_sorted,  # 按固定顺序排列的SHAP值
        X_sorted,  # 按固定顺序排列的特征
        feature_names=fixed_display_names,  # 固定的显示名称
        plot_type="dot",
        color=plt.get_cmap('coolwarm'),
        alpha=0.7,
        show=False,
        sort=False,  # 关键：禁用默认排序
    )

    # 优化图表标签（明确标注归一化后的范围）
    plt.gca().set_xlabel('SHAP value(impact on model output)', fontsize=12)
    plt.tight_layout()
    plt.savefig('rf_shap_normalized.png', dpi=300)  # 保存归一化后的图
    plt.show()


def main():
    file_path = '../data/original experimental data.xlsx'  # 替换为实际路径
    df = read_excel_data(file_path)
    X, y = preprocess_data(df)

    # 输出预处理后的特征
    print("\n预处理后的特征（前5行）:")
    print(X.head())

    # 训练模型
    model, X_test = train_random_forest(X, y)

    # 生成特征重要性图
    plot_feature_importance(model, X)

    # 生成归一化后的SHAP蜂拥图（可调整target_range，如改为(0,1)）
    plot_shap_beeswarm(model, X, normalize_shap=True, target_range=(-1, 1))


if __name__ == "__main__":
    main()