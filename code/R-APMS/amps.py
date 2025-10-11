from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import cross_val_score, KFold
import numpy as np
from sklearn.svm import SVR
import processing
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor  # sklearn风格接口
import os
from sklearn.metrics import mean_squared_error


# -------------------------- 1. 核心辅助函数--------------------------
def get_feature_importance(model, model_name, feature_names):
    """计算特征重要性，适配不同模型类型"""
    try:
        if model_name == 'SVR':
            return np.abs(model.coef_[0]) if hasattr(model, 'coef_') else None
        elif model_name == 'Linear Regression':
            return np.abs(model.coef_)
        elif model_name == 'XGB' and isinstance(model, XGBRegressor):
            return model.feature_importances_
        elif hasattr(model, 'feature_importances_'):
            return model.feature_importances_
    except Exception as e:
        print(f"  {model_name} 特征重要性计算失败: {str(e)}")
        return None


def get_sample_size_range(X_train):
    """生成训练样本数量序列（从40开始，避免小样本无效训练）"""
    total_samples = len(X_train)
    sample_sizes = list(range(40, total_samples + 1, 5))
    return sample_sizes if sample_sizes else [total_samples]


# -------------------------- 2. 主训练流程 --------------------------
if __name__ == "__main__":
    # 配置参数
    random_state = 42
    data_path = '../data/All data is used for ML.xlsx'
    split_data_save_dir = './split_data'  # 训练/测试集保存目录
    model_save_dir = './model'
    correct_columns = ['redox', 'azo', 'ybc', 'C', 'T', 'AMPS', 'fe', 'R-AMPS']

    # 创建模型保存目录
    os.makedirs(model_save_dir, exist_ok=True)
    print(f" 模型将保存至: {model_save_dir}")

    # 创建拆分数据保存目录
    os.makedirs(split_data_save_dir, exist_ok=True)
    print(f" 拆分后的训练/测试集将保存至: {split_data_save_dir}")

    # -------------------------- 步骤1：读取并验证数据 --------------------------
    try:
        data = pd.read_excel(data_path, sheet_name='结果3', skiprows=[0], names=correct_columns)
        print(f"\n 数据读取成功，共 {data.shape[0]} 条样本，{data.shape[1]} 列")
        print("数据前5行预览:")
        print(data[correct_columns].head(5))
    except Exception as e:
        print(f" 数据读取失败: {str(e)}")
        exit()

    # -------------------------- 步骤2：先拆分训练集/测试集 --------------------------
    try:
        train_data, test_data = processing.split_data(
            data, train_size=0.8, random_state=random_state
        )
        print(f"\n 数据拆分完成:")
        print(f"  - 训练集: {train_data.shape[0]} 条样本")
        print(f"  - 测试集: {test_data.shape[0]} 条样本")
        # -------------------------- 保存拆分后的训练集和测试集 --------------------------
        # 保存为CSV
        train_csv_path = f"{split_data_save_dir}/train_data_randomstate_{random_state}.csv"
        test_csv_path = f"{split_data_save_dir}/test_data_randomstate_{random_state}.csv"
        train_data.to_csv(train_csv_path, index=True, encoding="utf-8")
        test_data.to_csv(test_csv_path, index=True, encoding="utf-8")

        print(f"  训练集已保存至: \n      - {train_csv_path}")
        print(f"  测试集已保存至: \n      - {test_csv_path}")
    except Exception as e:
        print(f" 数据拆分失败: {str(e)}")
        exit()

    # -------------------------- 步骤3：目标变量处理（仅对数变换） --------------------------

    # 直接使用原始目标变量，无对数变换
    y_train = train_data['R-AMPS'].copy()
    y_test = test_data['R-AMPS'].copy()

    print(f"\n 目标变量处理完成（使用原始R-AMPS值）")

    # -------------------------- 步骤4：特征处理（MinMaxScaler） --------------------------
    X_train = train_data.drop('R-AMPS', axis=1)
    X_test = test_data.drop('R-AMPS', axis=1)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print(f" 特征缩放完成（MinMaxScaler）")

    # -------------------------- 步骤5：特征选择 --------------------------
    try:
        X_train_selected, X_test_selected = processing.select_features(
            X_train_scaled, y_train, X_test_scaled, n_features=7
        )
        print(f" 特征选择完成，保留 {X_train_selected.shape[1]} 个特征")
    except Exception as e:
        print(f" 特征选择失败: {str(e)}")
        exit()

    # -------------------------- 步骤6：格式统一 --------------------------
    X_train_selected = pd.DataFrame(
        X_train_selected, columns=X_train.columns, index=train_data.index
    )
    X_test_selected = pd.DataFrame(
        X_test_selected, columns=X_train.columns, index=test_data.index
    )
    # 目标变量名称改为原始名称
    y_train = pd.Series(y_train, name='R-AMPS', index=train_data.index)
    y_test = pd.Series(y_test, name='R-AMPS', index=test_data.index)

    # -------------------------- 步骤7：保存训练统计量 --------------------------
    training_stats = {
        'feature_names': X_train.columns.tolist(),
        'scaler_min': scaler.data_min_.tolist(),
        'scaler_max': scaler.data_max_.tolist(),
        # 'target_epsilon': 1e-9
    }
    joblib.dump(training_stats, f"{model_save_dir}/training_stats.pkl")
    print(f"\n 训练统计量保存至: {model_save_dir}/training_stats.pkl")

    # -------------------------- 步骤8：定义模型 --------------------------
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=50, random_state=random_state),
        'Decision Tree': DecisionTreeRegressor(random_state=random_state),
        'GBT': GradientBoostingRegressor(n_estimators=100, random_state=random_state),
        'XGB': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=random_state),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Linear Regression': LinearRegression(n_jobs=-1)
    }
    print(f"\n 待评估模型: {list(models.keys())}")

    model_performance = {}

    # -------------------------- 步骤9：训练与评估模型 --------------------------
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f" 正在评估模型: {model_name}")
        print(f"{'='*70}")

        sample_sizes = get_sample_size_range(X_train_selected)
        print(f"  样本量序列: {sample_sizes}（共 {len(sample_sizes)} 个点）")

        train_r2_list = []
        val_r2_list = []
        kf = KFold(n_splits=5, shuffle=True, random_state=random_state)

        # 按样本量递增训练
        for sample_size in sample_sizes:
            # 用np.random.seed替代random_state参数，确保每次运行结果一致
            np.random.seed(random_state + sample_size)  # +sample_size避免不同样本量随机结果重复
            sample_indices = np.random.choice(
                len(X_train_selected), size=sample_size, replace=False  # 移除random_state参数
            )
            X_sample = X_train_selected.iloc[sample_indices]
            y_sample = y_train.iloc[sample_indices]

            # 训练与评估
            model.fit(X_sample, y_sample)
            train_r2 = round(model.score(X_sample, y_sample), 4)
            val_r2 = round(np.mean(cross_val_score(model, X_sample, y_sample, cv=kf, scoring='r2')), 4)

            train_r2_list.append(train_r2)
            val_r2_list.append(val_r2)

            print(f"  样本量={sample_size:4d} | 训练R²={train_r2:6.4f} | 验证R²={val_r2:6.4f}")

        # 全量数据交叉验证
        full_cv_r2 = cross_val_score(model, X_train_selected, y_train, cv=kf, scoring='r2')
        full_cv_mean = round(np.mean(full_cv_r2), 4)
        full_cv_std = round(np.std(full_cv_r2), 4)
        print(f"\n  全量数据5折CV:")
        print(f"    R²分数: {np.round(full_cv_r2, 4)}")
        print(f"    平均R²: {full_cv_mean} (±{full_cv_std})")

        # 测试集评估
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        test_r2 = round(model.score(X_test_selected, y_test), 4)
        test_mse = round(mean_squared_error(y_test, y_pred), 4)
        print(f"  测试集性能:")
        print(f"    R²: {test_r2} | MSE: {test_mse}")

        # 特征重要性
        feature_importance = get_feature_importance(model, model_name, X_train.columns.tolist())
        if feature_importance is not None:
            # 归一化线性回归/SVR的特征重要性
            if model_name in ['Linear Regression', 'SVR']:
                feature_importance = feature_importance / np.sum(feature_importance)
            # 打印全部特征
            importance_df = pd.DataFrame({
                'Feature': X_train.columns,
                'Importance': np.round(feature_importance, 4)
            }).sort_values('Importance', ascending=False)
            print(f"\n  所有特征重要性（按重要性排序）:")
            print(importance_df.to_string(index=False))  # 打印全部特征，不限制数量


        # 保存性能
        model_performance[model_name] = {
            'model': model,
            'cv_mean_r2': full_cv_mean,
            'cv_std_r2': full_cv_std,
            'test_r2': test_r2,
            'test_mse': test_mse,
            'feature_importance': feature_importance
        }

    # -------------------------- 步骤10：保存模型（按R²排序） --------------------------
    sorted_models = sorted(
        model_performance.items(), key=lambda x: x[1]['test_r2'], reverse=True
    )

    print(f"\n{'='*70}")
    print(f" 模型性能排名（按测试集R²）")
    print(f"{'='*70}")

    for rank, (model_name, info) in enumerate(sorted_models, 1):
        model_path = f"{model_save_dir}/{rank}_{model_name.replace(' ', '_').lower()}_model.pkl"
        joblib.dump(info['model'], model_path)

        print(f"  第{rank}名: {model_name}")
        print(f"    - 测试集R²: {info['test_r2']} | MSE: {info['test_mse']}")
        print(f"    - 交叉验证R²: {info['cv_mean_r2']} (±{info['cv_std_r2']})")
        print(f"    - 模型保存至: {model_path}")
        print()

    # -------------------------- 步骤11：保存特征信息 --------------------------
    feature_info = {
        'feature_names': X_train.columns.tolist(),
        'scaler': scaler
    }
    joblib.dump(feature_info, f"{model_save_dir}/feature_info.pkl")
    print(f" 特征信息保存至: {model_save_dir}/feature_info.pkl")

    # -------------------------- 步骤12：打印汇总表 --------------------------
    print(f"\n{'='*90}")
    print(f" 模型性能汇总表")
    print(f"{'='*90}")
    summary_data = []
    for rank, (model_name, info) in enumerate(sorted_models, 1):
        summary_data.append({
            '排名': rank,
            '模型': model_name,
            'CV平均R²': info['cv_mean_r2'],
            'CV标准差': info['cv_std_r2'],
            '测试集R²': info['test_r2'],
            '测试集MSE': info['test_mse']
        })
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print(f"{'='*90}")

    print(f"\n 所有模型训练完成！")