from sklearn.preprocessing import StandardScaler
import pandas as pd

def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame = None):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler