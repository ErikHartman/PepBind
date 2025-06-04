from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, f1_score, roc_auc_score
from scipy.stats import spearmanr, kendalltau, pearsonr

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

def get_regression_results(y_true, y_pred, prefix=""):

    if np.var(y_pred) == 0 or len(np.unique(y_pred)) == 1:
        pearson_r = np.nan
        spearman_r = np.nan
        kendall_tau = np.nan
    else:
        pearson_r = pearsonr(y_true, y_pred)[0]
        spearman_r = spearmanr(y_true, y_pred)[0]
        kendall_tau = kendalltau(y_true, y_pred)[0]

    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    top_k_accuracy_true = top_k_accuracy(y_true, y_pred, k=25)
    
    return {
        f'{prefix}rmse': rmse,
        f'{prefix}r2': r2,
        f'{prefix}mae': mae,
        f'{prefix}pearson_r': pearson_r,
        f'{prefix}spearman_r': spearman_r,
        f'{prefix}kendall_tau': kendall_tau,
        f'{prefix}top_k_accuracy_true': top_k_accuracy_true
    }


def get_classification_results(y_true, y_pred, y_proba, prefix=""):
    """
    Calculate comprehensive classification metrics.
    
    Args:
        y_true: True binary labels
        y_pred: Predicted binary labels  
        y_proba: Predicted probabilities for positive class
        prefix: Prefix for metric names (e.g., "train_", "val_")
    
    Returns:
        Dictionary of classification metrics
    """
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='binary')
    auc = roc_auc_score(y_true, y_proba)
    
    return {
        f'{prefix}acc': acc,
        f'{prefix}f1': f1,
        f'{prefix}auc': auc
    }


def create_model_comparison(results_dict: dict) -> pd.DataFrame:

    all_metrics = set()
    for result in results_dict.values():
        all_metrics.update(result.keys())

    excluded_keys = {'model', 'scaler', 'train_pred', 'val_pred', 'coefficients', 
                     'feature_importance', 'all_equations', 'top_equations', 'best_expr'}

    numeric_metrics = sorted([key for key in all_metrics 
                             if key not in excluded_keys 
                             and any(isinstance(results_dict[model].get(key), (int, float, np.number)) 
                                   for model in results_dict.keys() 
                                   if key in results_dict[model])])
    

    comparison = {"Model": list(results_dict.keys())}

    for metric in numeric_metrics:
        comparison[metric] = [result.get(metric, np.nan) for result in results_dict.values()]
    
    comparison_df = pd.DataFrame(comparison)
    
    return comparison_df
