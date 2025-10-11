import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from catboost import CatBoostRegressor
from sklearn.preprocessing import LabelEncoder

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def read_excel_data(file_path):
    """读取Excel数据（处理复合表头）"""
    try:
        df = pd.read_excel(file_path, header=1)
        # 确保列名正确
        df.columns = ['温度', '盐种类', '盐浓度（mM）', '聚合物浓度(mM)', '剪切速率', '应力', '[n]']
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
    X = df_processed[['温度', '盐种类', '盐浓度（mM）', '聚合物浓度(mM)', '剪切速率', '应力']]
    y = df_processed['[n]']

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
    print("\n=== CatBoost特征重要性百分比 ===")
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
    plt.title('CatBoost特征重要性')
    plt.gca().invert_yaxis()  # 最重要的在顶部
    plt.xlim(0, max(importance_percent) * 1.2)  # 留出空间显示标签
    plt.tight_layout()
    plt.savefig('cb_feature_importance_percent.png', dpi=300)
    plt.show()


def train_catboost(X, y):
    """训练CatBoost模型"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 指定类别特征索引（盐种类）
    cat_features = [1]  # 盐种类在特征矩阵中的位置

    model = CatBoostRegressor(
        cat_features=cat_features,
        random_state=42,
        verbose=0,
        iterations=1000,
        learning_rate=0.05,
        depth=6
    )
    model.fit(X_train, y_train)

    # 输出模型性能
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"模型性能: 训练集R²={train_score:.3f}, 测试集R²={test_score:.3f}")

    return model, X_test


def plot_shap_beeswarm(model, X):
    """生成SHAP蜂拥图（使用所有数据）"""
    # 创建解释器
    explainer = shap.TreeExplainer(model)

    print(f"计算所有样本的SHAP值，样本数: {X.shape[0]}")

    # 计算所有样本的SHAP值
    shap_values = explainer.shap_values(X)

    # 创建自定义特征名称（优化显示）
    feature_names = {
        '温度': '温度',
        '盐种类': '盐种类',
        '盐浓度（mM）': '盐浓度 (mM)',
        '聚合物浓度(mM)': '聚合物浓度 (mM)',
        '剪切速率': '剪切速率',
        '应力': '应力'
    }

    # 绘制蜂群图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values,
        X,
        feature_names=[feature_names.get(col, col) for col in X.columns],
        plot_type="dot",  # 使用点图避免重叠
        color=plt.get_cmap('coolwarm'),  # 蓝-红渐变
        alpha=0.7,  # 透明度
        show=False,
        plot_size=None,
    )

    # 优化坐标轴和布局
    plt.gca().set_xlabel('SHAP值 (对模型输出的影响)', fontsize=12)
    plt.tight_layout()
    plt.savefig('catboost_shap_plot.png', dpi=300)
    plt.show()


def main():
    file_path = '../data/original experimental data.xlsx'  # 替换为实际路径
    df = read_excel_data(file_path)
    X, y = preprocess_data(df)

    # 输出预处理后的特征信息
    print("\n预处理后的特征:")
    print(X.head())

    # 训练模型
    model, X_test = train_catboost(X, y)

    # 生成特征重要性图
    plot_feature_importance(model, X)

    # 使用所有数据生成SHAP图
    plot_shap_beeswarm(model, X)

if __name__ == "__main__":
    main()