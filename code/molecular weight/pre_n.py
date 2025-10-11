import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from glob import glob  # 用于获取所有模型文件路径


def read_pred_data(file_path, expected_columns):
    """读取待预测数据并验证列名"""
    try:
        df = pd.read_excel(file_path, skiprows=[0], names=expected_columns)
        print(f" 待预测数据读取成功，共 {df.shape[0]} 条样本，{df.shape[1]} 列")
        print("数据前3行预览：")
        print(df.head(3))

        required_features = expected_columns[:-1]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f" 缺失必要特征：{missing_features}")
            return None
        return df
    except Exception as e:
        print(f" 读取待预测数据失败：{str(e)}")
        return None


def load_training_assets(stats_path, feature_info_path):
    """加载训练时的scaler、特征名和目标变量参数"""
    try:
        training_stats = joblib.load(stats_path)
        target_epsilon = training_stats['target_epsilon']
        feature_names = training_stats['feature_names']

        feature_info = joblib.load(feature_info_path)
        scaler = feature_info['scaler']

        print(f"\n 训练资产加载成功：")
        print(f"  - 特征列：{feature_names}")
        print(f"  - 目标变量对数偏移量：{target_epsilon}")
        return scaler, feature_names, target_epsilon
    except Exception as e:
        print(f" 加载训练资产失败：{str(e)}")
        return None, None, None


def preprocess_pred_data(df, feature_names, scaler):
    """预处理待预测数据（与训练时一致）"""
    df_processed = df.copy()
    X = df_processed[feature_names].copy()

    # 处理缺失值
    for col in X.columns:
        if X[col].isnull().any():
            fill_val = X[col].mean()
            X[col] = X[col].fillna(fill_val)
            print(f"  特征 {col} 存在缺失值，用均值 {fill_val:.4f} 填充")

    # 特征缩放
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=df_processed.index)
    print(f"\n 特征缩放完成，缩放后前5行特征：")
    print(X_scaled_df.head(5))

    return X_scaled_df


def get_all_model_paths(model_dir):
    """获取model目录下所有模型文件的路径"""
    model_paths = glob(os.path.join(model_dir, "*.pkl"))
    # 过滤非模型文件（只保留含"_model.pkl"的文件）
    model_paths = [p for p in model_paths if "_model.pkl" in p]

    if not model_paths:
        print(f" 在 {model_dir} 目录下未找到模型文件")
        return None

    print(f"\n 找到 {len(model_paths)} 个模型文件：")
    for i, path in enumerate(model_paths, 1):
        model_name = os.path.basename(path).split("_model.pkl")[0].split("_", 1)[1]  # 提取模型名（如"svr"）
        print(f"  {i}. {model_name} -> {path}")
    return model_paths


def predict_with_model(model_path, X_scaled, target_epsilon):
    """使用单个模型进行预测并返回结果"""
    try:
        model = joblib.load(model_path)
        model_name = os.path.basename(model_path).split("_model.pkl")[0].split("_", 1)[1].upper()  # 格式化模型名（如"SVR"）

        # 预测并反变换
        y_pred_log = model.predict(X_scaled)
        y_pred = np.exp(y_pred_log) - target_epsilon
        y_pred = np.maximum(y_pred, 0)  # 确保非负

        print(f"  {model_name} 预测完成，前5个预测值：{np.round(y_pred[:5], 2)}")
        return model_name, y_pred
    except Exception as e:
        print(f"   模型 {os.path.basename(model_path)} 预测失败：{str(e)}")
        return None, None


def evaluate_and_save_all(df, all_predictions, output_path):
    """合并所有模型的预测结果，计算误差并保存"""
    result_df = df.copy()

    # 添加所有模型的预测列
    for model_name, y_pred in all_predictions.items():
        result_df[f'预测[n]_{model_name}'] = np.round(y_pred, 2)

        # 计算每个模型的误差
        if '[n]' in result_df.columns:
            result_df[f'绝对误差_{model_name}'] = np.round(result_df['[n]'] - result_df[f'预测[n]_{model_name}'], 2)
            result_df[f'相对误差(%)_{model_name}'] = np.round(
                (result_df[f'绝对误差_{model_name}'] / result_df['[n]']) * 100, 2
            )

    # 计算每个模型的评估指标
    if '[n]' in result_df.columns and not result_df['[n]'].isnull().all():
        valid_mask = result_df['[n]'] > 0
        y_true_valid = result_df.loc[valid_mask, '[n]']

        print(f"\n{'=' * 70}")
        print(f" 各模型预测评估指标（原始尺度）")
        print(f"{'=' * 70}")
        metrics_summary = []

        for model_name in all_predictions.keys():
            y_pred_valid = result_df.loc[valid_mask, f'预测[n]_{model_name}']

            # 计算指标
            r2 = round(r2_score(y_true_valid, y_pred_valid), 4)
            mae = round(mean_absolute_error(y_true_valid, y_pred_valid), 2)
            rmse = round(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)), 2)

            metrics_summary.append({
                '模型': model_name,
                'R²': r2,
                'MAE': mae,
                'RMSE': rmse
            })

            # 打印单个模型指标
            print(f"{model_name}:")
            print(f"  R²: {r2} | MAE: {mae} | RMSE: {rmse}")

        # 打印指标汇总表
        print(f"\n{'=' * 50}")
        print("模型预测性能汇总表：")
        print(pd.DataFrame(metrics_summary).to_string(index=False))
        print(f"{'=' * 50}")

    # 保存合并结果
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n 所有模型预测结果已保存至：{output_path}")
    print(f"结果包含 {len(all_predictions)} 个模型的预测值及误差分析")
    return result_df


def main():
    # 配置参数
    config = {
        'pred_data_path': '../data/All data is used for ML.xlsx',  # 待预测数据路径
        'model_dir': './model',  # 模型存放目录
        'stats_path': './model/training_stats.pkl',
        'feature_info_path': './model/feature_info.pkl',
        'output_path': '分子量预测结果汇总.csv'  # 合并结果保存路径
    }

    # 训练时的列名（7特征+1目标）
    expected_columns = ['redox', 'azo', 'ybc', 'C', 'T', 'AMPS', 'fe', '[n]']

    # 1. 读取待预测数据
    df_pred = read_pred_data(config['pred_data_path'], expected_columns)
    if df_pred is None:
        return

    # 2. 加载训练资产（scaler、特征名等）
    scaler, feature_names, target_epsilon = load_training_assets(
        config['stats_path'], config['feature_info_path']
    )
    if scaler is None or feature_names is None:
        return

    # 3. 预处理数据（所有模型共用同一套预处理后的特征）
    X_scaled = preprocess_pred_data(df_pred, feature_names, scaler)

    # 4. 获取所有模型路径
    model_paths = get_all_model_paths(config['model_dir'])
    if not model_paths:
        return

    # 5. 逐个模型预测
    all_predictions = {}  # 存储{模型名: 预测结果}
    print(f"\n{'=' * 50}")
    print(f"开始使用所有模型预测...")
    print(f"{'=' * 50}")

    for path in model_paths:
        model_name, y_pred = predict_with_model(path, X_scaled, target_epsilon)
        if model_name and y_pred is not None:
            all_predictions[model_name] = y_pred

    if not all_predictions:
        print(" 所有模型预测失败，无法生成结果")
        return

    # 6. 合并结果并保存
    evaluate_and_save_all(df_pred, all_predictions, config['output_path'])

    print(f"\n 所有模型预测流程完成！")


if __name__ == "__main__":
    main()
