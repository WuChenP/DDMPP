import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeRegressor
import joblib
from sklearn.metrics import mean_squared_error, r2_score

# 1. Load data and preprocess
def load_and_preprocess_data(filepath):
    """
    Load an Excel file and perform basic preprocessing: set column names and remove the first row of data.

    Parameters:
        filepath (str): Path to the data file.

    Returns:
        pd.DataFrame: Preprocessed data.
    """
    data = pd.read_excel(filepath)
    data.columns = data.iloc[0]  # Set the first row as column names
    data = data.drop(0)  # Remove the first row of data
    return data


# 2. Data type conversion
def convert_data_types(data):
    """
    Convert data types to float64.

    Parameters:
        data (pd.DataFrame): Input data.

    Returns:
        pd.DataFrame: Data after type conversion.
    """
    return data.astype('float64')


# 3. Apply log transformation to specified columns
def apply_log_transformation(data, columns, epsilon=1e-5):
    """
    Apply log10 transformation to specified columns to avoid issues with log(0).

    Parameters:
        data (pd.DataFrame): Input data.
        columns (list): List of column names to undergo log transformation.
        epsilon (float): A small value to prevent log(0), default is 1e-5.

    Returns:
        pd.DataFrame: Data after transformation.
    """
    data[columns] = np.abs(np.log10(data[columns] + epsilon))
    return data


# 4. Save data
def save_data_to_excel(data, output_filepath):
    """
    Save data as an Excel file.

    Parameters:
        data (pd.DataFrame): Input data.
        output_filepath (str): Output file path.
    """
    data.to_excel(output_filepath, index=False)


# 5. Perform PCA and generate combined features
def apply_pca(data, columns, n_components=1):
    """
    Perform PCA dimensionality reduction on specified columns and add the results as new combined features to the data.

    Parameters:
        data (pd.DataFrame): Input data.
        columns (list): List of column names for PCA.
        n_components (int): Number of dimensions for PCA reduction, default is 1.

    Returns:
        pd.DataFrame: Data with added PCA combined features.
    """
    pca = PCA(n_components=n_components)
    combined_feature = pca.fit_transform(data[columns])
    return combined_feature


# 6. Normalization function
def normalize_columns(data, columns_to_normalize):
    """
    Perform Min-Max normalization on specified columns and update the normalized data back to the original data.

    Parameters:
        data (pd.DataFrame): Input data.
        columns_to_normalize (list): List of column names to normalize.

    Returns:
        pd.DataFrame: Normalized data.
    """
    scaler = MinMaxScaler()
    data_to_normalize = data[columns_to_normalize]

    # Perform normalization
    normalized_data = scaler.fit_transform(data_to_normalize)

    # Update the original data with normalized results
    data[columns_to_normalize] = pd.DataFrame(normalized_data, columns=columns_to_normalize, index=data.index)

    return data


# 7. Dataset splitting function
def split_data(data, train_size=150, random_state=None):
    """
    Split the dataset into training set and test set.

    Parameters:
        data (pd.DataFrame): Input data.
        train_size (int): Number of samples in the training set, default is 150.
        random_state (int): Random seed for reproducible results.

    Returns:
        pd.DataFrame, pd.DataFrame: Training set and test set.
    """
    train_data, test_data = train_test_split(data, train_size=train_size, random_state=random_state)
    return train_data, test_data


# # 8. Feature and label extraction function
# def extract_features_and_labels(data, label_column_index=-2):
#     """
#     Extract features and labels from the dataset.
#
#     Parameters:
#         data (pd.DataFrame): Input data.
#         label_column_index (int): Index of the label column, default is the second last column.
#
#     Returns:
#         pd.DataFrame, pd.Series: Features and labels.
#     """
#     X = data.iloc[:, :-2]  # Extract features
#     y = data.iloc[:, label_column_index]  # Extract labels
#     return X, y

# 9. Feature scaling function
def scale_features(X_train, X_test):
    """
    Scale the features.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        X_test (pd.DataFrame): Test feature data.

    Returns:
        np.ndarray, np.ndarray: Scaled training features and test features.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled


# 10. Recursive Feature Elimination function
def select_features(X_train, y_train, X_test, n_features=7):
    """
    Perform feature selection using Recursive Feature Elimination (RFE).

    Parameters:
        X_train (np.ndarray): Training feature data.
        y_train (pd.Series): Training label data.
        X_test (np.ndarray): Test feature data.
        n_features (int): Number of features to select, default is 7.

    Returns:
        np.ndarray, np.ndarray: Selected training features and test features.
    """
    estimator = DecisionTreeRegressor()
    selector = RFE(estimator, n_features_to_select=n_features, step=1)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    return X_train_selected, X_test_selected


# 11. Model training and evaluation function
def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Train the model and evaluate its performance.

    Parameters:
        model: Model to be trained.
        X_train (np.ndarray): Training features.
        y_train (pd.Series): Training labels.
        X_test (np.ndarray): Test features.
        y_test (pd.Series): Test labels.

    Returns:
        float, float: Mean Squared Error (MSE) and R^2 score.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


# 12. Model saving function
def save_model(model, model_name, save_path):
    """
    Save the trained model.

    Parameters:
        model: Trained model.
        model_name (str): Name of the model.
        save_path (str): Save path.
    """
    joblib.dump(model, save_path + f'{model_name}_model.pkl')
    print(f'{model_name} model saved to {save_path + f"{model_name}_model.pkl"}')

# 13. Remove specific values from specified columns
def remove_rows_by_value(data, column_name, value):
    """
    Remove rows from the DataFrame where the specified column equals a specific value.

    Parameters:
        data (pd.DataFrame): Input dataset.
        column_name (str): Name of the column to check.
        value: Specific value in the column for which rows are to be deleted.

    Returns:
        pd.DataFrame: New dataset after removing specified rows.
    """
    # Check if the column exists in the data
    if column_name not in data.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the data.")

    # Remove rows where the specified column has the specific value
    filtered_data = data[data[column_name] != value]
    return filtered_data

