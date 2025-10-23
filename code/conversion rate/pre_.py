import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import os
from glob import glob  # Used to get all model file paths


def read_pred_data(file_path, expected_columns):
    """Read prediction data and validate column names"""
    try:
        df = pd.read_excel(file_path, sheet_name='结果2', skiprows=[0], names=expected_columns)
        print(f" Prediction data loaded successfully, total {df.shape[0]} samples, {df.shape[1]} columns")
        print("First 3 rows preview:")
        print(df.head(3))

        required_features = expected_columns[:-1]
        missing_features = [f for f in required_features if f not in df.columns]
        if missing_features:
            print(f" Missing required features: {missing_features}")
            return None
        return df
    except Exception as e:
        print(f" Failed to read prediction data: {str(e)}")
        return None


def load_training_assets(stats_path, feature_info_path):
    """Load scaler and feature names from training (remove target_epsilon as no log transformation was applied)"""
    try:
        training_stats = joblib.load(stats_path)
        feature_names = training_stats['feature_names']  # Load only feature names

        feature_info = joblib.load(feature_info_path)
        scaler = feature_info['scaler']

        print(f"\n Training assets loaded successfully:")
        print(f"  - Feature columns: {feature_names}")
        return scaler, feature_names
    except Exception as e:
        print(f" Failed to load training assets: {str(e)}")
        return None, None


def preprocess_pred_data(df, feature_names, scaler):
    """Preprocess prediction data (consistent with training)"""
    df_processed = df.copy()
    X = df_processed[feature_names].copy()

    # Handle missing values
    for col in X.columns:
        if X[col].isnull().any():
            fill_val = X[col].mean()
            X[col] = X[col].fillna(fill_val)
            print(f"  Feature {col} contains missing values, filled with mean {fill_val:.4f}")

    # Feature scaling
    X_scaled = scaler.transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names, index=df_processed.index)
    print(f"\n Feature scaling completed, first 5 rows after scaling:")
    print(X_scaled_df.head(5))

    return X_scaled_df


def get_all_model_paths(model_dir):
    """Get paths of all model files in the model directory"""
    model_paths = glob(os.path.join(model_dir, "*.pkl"))
    model_paths = [p for p in model_paths if "_model.pkl" in p]  # Filter non-model files

    if not model_paths:
        print(f" No model files found in {model_dir} directory")
        return None

    print(f"\n Found {len(model_paths)} model files:")
    for i, path in enumerate(model_paths, 1):
        model_name = os.path.basename(path).split("_model.pkl")[0].split("_", 1)[1]
        print(f"  {i}. {model_name} -> {path}")
    return model_paths


def predict_with_model(model_path, X_scaled):
    """Predict using a single model (no log inverse transformation, return original predictions directly)"""
    try:
        model = joblib.load(model_path)
        model_name = os.path.basename(model_path).split("_model.pkl")[0].split("_", 1)[1].upper()

        # Predict directly
        y_pred = model.predict(X_scaled)
        # Ensure predictions are non-negative based on conversion（%）business meaning
        y_pred = np.maximum(y_pred, 0)

        print(f"  {model_name} prediction completed, first 5 predictions: {np.round(y_pred[:5], 2)}")
        return model_name, y_pred
    except Exception as e:
        print(f"   Prediction failed for model {os.path.basename(model_path)}: {str(e)}")
        return None, None


def evaluate_and_save_all(df, all_predictions, output_path):
    """Merge predictions from all models, calculate errors and save"""
    result_df = df.copy()
    target_col = 'conversion（%）'  # Target column name

    # Add prediction columns for all models
    for model_name, y_pred in all_predictions.items():
        pred_col = f'predicted{target_col}_{model_name}'
        result_df[pred_col] = np.round(y_pred, 2)

        # Calculate errors for each model
        if target_col in result_df.columns:
            abs_err_col = f'absolute_error_{model_name}'
            rel_err_col = f'relative_error(%)_{model_name}'
            # Handle cases where target value is 0 (avoid division by zero)
            zero_mask = result_df[target_col] == 0
            result_df[abs_err_col] = np.round(result_df[target_col] - result_df[pred_col], 2)
            result_df.loc[~zero_mask, rel_err_col] = np.round(
                (result_df.loc[~zero_mask, abs_err_col] / result_df.loc[~zero_mask, target_col]) * 100, 2
            )
            result_df.loc[zero_mask, rel_err_col] = np.nan  # Set relative error to NaN for zero target values

    # Calculate evaluation metrics for each model
    if target_col in result_df.columns and not result_df[target_col].isnull().all():
        valid_mask = result_df[target_col] > 0  # Filter zero values to avoid interference
        if valid_mask.sum() == 0:
            print(f"\n⚠️  No valid {target_col} samples (all values ≤ 0), unable to calculate evaluation metrics")
        else:
            y_true_valid = result_df.loc[valid_mask, target_col]

            print(f"\n{'=' * 70}")
            print(f" Prediction evaluation metrics for all models (original scale, based on {valid_mask.sum()} valid samples)")
            print(f"{'=' * 70}")
            metrics_summary = []

            for model_name in all_predictions.keys():
                pred_col = f'predicted{target_col}_{model_name}'
                y_pred_valid = result_df.loc[valid_mask, pred_col]

                # Calculate metrics
                r2 = round(r2_score(y_true_valid, y_pred_valid), 4)
                mae = round(mean_absolute_error(y_true_valid, y_pred_valid), 2)
                rmse = round(np.sqrt(mean_squared_error(y_true_valid, y_pred_valid)), 2)

                metrics_summary.append({
                    'Model': model_name,
                    'R²': r2,
                    'MAE': mae,
                    'RMSE': rmse
                })

                # Print individual model metrics
                print(f"{model_name}:")
                print(f"  R²: {r2} | MAE: {mae} | RMSE: {rmse}")

            # Print metrics summary table
            print(f"\n{'=' * 50}")
            print("Model prediction performance summary:")
            print(pd.DataFrame(metrics_summary).to_string(index=False))
            print(f"{'=' * 50}")

    # Save merged results
    result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n All model prediction results saved to: {output_path}")
    print(f"Results include predictions and error analysis from {len(all_predictions)} models")
    return result_df


def main():
    # Configuration parameters
    config = {
        'pred_data_path': '../data/All data is used for ML.xlsx',  # Path to prediction data
        'model_dir': './model',  # Training model directory for conversion（%）
        'stats_path': './model/training_stats.pkl',     # Training statistics (without target_epsilon)
        'feature_info_path': './model/feature_info.pkl',# Feature information (including scaler)
        'output_path': 'Predicted results of conversion（%）.csv'          # Merged results save path
    }

    # Column names from training (7 features + 1 target: conversion（%）)
    expected_columns = ['redox', 'azo', 'ybc', 'C', 'T', 'AMPS', 'fe', 'conversion（%）']

    # 1. Read prediction data
    df_pred = read_pred_data(config['pred_data_path'], expected_columns)
    if df_pred is None:
        return

    # 2. Load training assets
    scaler, feature_names = load_training_assets(
        config['stats_path'], config['feature_info_path']
    )
    if scaler is None or feature_names is None:
        return

    # 3. Preprocess data
    X_scaled = preprocess_pred_data(df_pred, feature_names, scaler)

    # 4. Get all model paths
    model_paths = get_all_model_paths(config['model_dir'])
    if not model_paths:
        return

    # 5. Predict with each model
    all_predictions = {}
    print(f"\n{'=' * 50}")
    print(f"Starting prediction for {expected_columns[-1]} using all models...")
    print(f"{'=' * 50}")

    for path in model_paths:
        model_name, y_pred = predict_with_model(path, X_scaled)  # No longer pass target_epsilon
        if model_name and y_pred is not None:
            all_predictions[model_name] = y_pred

    if not all_predictions:
        print(" All models failed to predict, unable to generate results")
        return

    # 6. Merge results and save
    evaluate_and_save_all(df_pred, all_predictions, config['output_path'])

    print(f"\n All model prediction processes completed!")


if __name__ == "__main__":
    main()