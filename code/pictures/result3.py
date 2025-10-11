import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_excel_data(file_path):
    """读取Excel数据（处理复合表头）"""
    try:
        df = pd.read_excel(file_path,sheet_name='结果3', header=1)
        # 确保列名正确
        df.columns = ['redox', 'azo', 'ybc', 'C', 'T', 'AMPS', 'fe', 'R-AMPS']
        print(f"数据读取成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"读取失败: {e}")
        raise FileNotFoundError("检查文件路径是否正确")


def preprocess_data(df):
    """预处理：标签编码盐种类特征"""
    df_processed = df.copy()

    # 填充缺失值
    numeric_cols = df_processed.select_dtypes(include=np.number).columns
    df_processed[numeric_cols] = df_processed[numeric_cols].fillna(df_processed[numeric_cols].mean())

    cat_cols = df_processed.select_dtypes(include=object).columns
    for col in cat_cols:
        df_processed[col] = df_processed[col].fillna(df_processed[col].mode()[0])

    # 特征和目标分离
    X = df_processed['redox', 'azo', 'ybc', 'C', 'T', 'AMPS', 'fe']
    y = df_processed['R-AMPS']

    # 对盐种类进行标签编码（保持为单一特征）
    le = LabelEncoder()
    X['盐种类'] = le.fit_transform(X['盐种类'])
    print(f"盐种类编码映射: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return X, y


def plot_feature_importance(model, X):
    """可视化特征重要性（百分比形式）"""
    importance = model.feature_importances_

    # 转换为百分比
    total = sum(importance)
    importance_percent = (importance / total) * 100

    feature_names = X.columns

    # 创建特征名映射
    name_map = {
        '温度': '温度',
        '盐种类': '盐种类',
        '盐浓度（mM）': '盐浓度 (mM)',
        '聚合物浓度(mM)': '聚合物浓度 (mM)',
        '剪切速率': '剪切速率',
        '应力': '应力'
    }
    display_names = [name_map.get(col, col) for col in feature_names]

    # 排序特征重要性
    indices = np.argsort(importance_percent)[::-1]
    sorted_importance = importance_percent[indices]
    sorted_features = [display_names[i] for i in indices]

    # 打印数值
    print("\n=== 随机森林特征重要性百分比 ===")
    for name, imp in zip(sorted_features, sorted_importance):
        print(f"{name}: {imp:.1f}%")

    # 可视化
    plt.figure(figsize=(10, 6))
    bars = plt.barh(range(len(importance_percent)), sorted_importance, align='center')
    plt.yticks(range(len(importance_percent)), sorted_features)

    # 在条形上添加数值标签
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.5, bar.get_y() + bar.get_height() / 2,
                 f'{width:.1f}%',
                 va='center', ha='left')

    plt.xlabel('特征重要性 (%)')
    plt.title('随机森林特征重要性')
    plt.gca().invert_yaxis()  # 最重要的在顶部
    plt.xlim(0, max(importance_percent) * 1.2)  # 留出空间显示标签
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
        n_jobs=-1  # 使用所有CPU核心
    )

    model.fit(X_train, y_train)

    # 输出模型性能
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    train_score = r2_score(y_train, train_pred)
    test_score = r2_score(y_test, test_pred)
    print(f"模型性能: 训练集R²={train_score:.3f}, 测试集R²={test_score:.3f}")

    return model, X_test


def plot_shap_beeswarm(model, X):
    """生成SHAP蜂拥图（固定左侧特征顺序）"""
    # 计算SHAP值
    explainer = shap.TreeExplainer(model)
    print(f"计算所有样本的SHAP值，样本数: {X.shape[0]}")
    shap_values = explainer.shap_values(X)
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    print(f"SHAP值形状: {np.array(shap_values).shape}")

    # 定义固定的特征显示顺序（按需要调整）
    fixed_feature_order = [
        '剪切速率',
        '聚合物浓度(mM)',
        '盐浓度（mM）',
        '应力',
        '盐种类',
        '温度'
    ]

    # 特征名映射（保持显示名称友好）
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
        sort=False # 关键：禁用默认排序
    )

    # 优化坐标轴和布局
    plt.gca().set_xlabel('SHAP value(impact on model output)', fontsize=12)
    plt.tight_layout()
    plt.savefig('rf_shap.png', dpi=300)
    plt.show()


def main():
    file_path = '../data/All data is used for ML.xlsx'  # 替换为实际路径
    df = read_excel_data(file_path)
    X, y = preprocess_data(df)

    # 输出预处理后的特征信息
    print("\n预处理后的特征:")
    print(X.head())

    # 训练模型
    model, X_test = train_random_forest(X, y)

    # 生成特征重要性图
    plot_feature_importance(model, X)

    # 使用所有数据生成SHAP图
    plot_shap_beeswarm(model, X)


if __name__ == "__main__":
    main()