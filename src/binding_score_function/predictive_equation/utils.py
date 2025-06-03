from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame = None):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler



def top_k_accuracy(y_true, y_pred, k=25):

    if len(y_true) < k:
        k = len(y_true)

    true_top_k_indices = set(np.argsort(y_true)[-k:])
    
    pred_top_k_indices = set(np.argsort(y_pred)[-k:])
    
    overlap = len(true_top_k_indices & pred_top_k_indices)
    
    return overlap / k