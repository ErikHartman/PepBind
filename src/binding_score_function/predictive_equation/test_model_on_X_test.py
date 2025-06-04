#!/usr/bin/env python3
"""
Test model on X_test data using symbolic regression equations.

This script:
1. Loads all test data (X and y for both regression and classification)
2. Scales the data using regression scaling parameters
3. Provides a function to test custom symbolic equations
4. Generates predictions and visualizations
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, Tuple, Optional
import sympy as sp
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set plotting style
sns.set_context("paper")
color_palette = {
    "real": "#2C8C99",
    "shuffled": "#E88873", 
    "random": "#F46036",
    "prediction": "#124E78"
}

# =============================================================================
# DATA LOADING AND SCALING FUNCTIONS
# =============================================================================

def load_test_data(base_path="/srv/data1/general/immunopeptides_data/"):
    """Load all test datasets."""
    scores_path = os.path.join(base_path, "outputs/binding_score_function_prod/4_processed_scores/")
    
    print("Loading test data...")
    
    # Load real test data (regression)
    X_real_test_df = pd.read_csv(os.path.join(scores_path, "real_X_test.csv"))
    y_real_test_df = pd.read_csv(os.path.join(scores_path, "real_y_test.csv"))
    
    # Load decoy test data (classification)
    X_shuffle_test_df = pd.read_csv(os.path.join(scores_path, "shuffle_X_test.csv"))
    X_random_test_df = pd.read_csv(os.path.join(scores_path, "random_X_test.csv"))
    
    # Remove any 'Unnamed:_0' columns
    for df in [X_real_test_df, X_shuffle_test_df, X_random_test_df]:
        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
    
    # Set complex_filename as index
    X_real_test = X_real_test_df.set_index("complex_filename")
    X_shuffle_test = X_shuffle_test_df.set_index("complex_filename")
    X_random_test = X_random_test_df.set_index("complex_filename")
    
    # Extract pKd values
    y_real_test = y_real_test_df["pKd"].values
    
    print(f"Loaded test data:")
    print(f"  Real: {X_real_test.shape} samples")
    print(f"  Shuffled: {X_shuffle_test.shape} samples") 
    print(f"  Random: {X_random_test.shape} samples")
    
    return X_real_test, y_real_test, X_shuffle_test, X_random_test

def load_scaling_params(scaling_type="regression"):
    """Load scaling parameters from training."""
    scaling_path = f"/home/er8813ha/immunopeptides/plots/{scaling_type}/scaling_params.csv"
    return pd.read_csv(scaling_path, index_col=0)

def scale_data(X, scaling_params):
    """Scale data using provided scaling parameters."""
    feature_means = scaling_params['mean']
    feature_stds = scaling_params['std']
    return (X - feature_means) / feature_stds


def calculate_regression_metrics(y_true, y_pred):
    """Calculate regression performance metrics."""
    # Handle any NaN or infinite predictions
    valid_mask = np.isfinite(y_pred) & np.isfinite(y_true)
    
    if not valid_mask.any():
        return {
            'rmse': np.nan, 'mae': np.nan, 'r2': np.nan,
            'pearson_r': np.nan, 'spearman_r': np.nan,
            'valid_predictions': 0, 'total_predictions': len(y_pred)
        }
    
    y_true_valid = y_true[valid_mask]
    y_pred_valid = y_pred[valid_mask]
    
    rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
    mae = mean_absolute_error(y_true_valid, y_pred_valid)
    r2 = r2_score(y_true_valid, y_pred_valid)
    
    # Correlation metrics
    pearson_r, _ = pearsonr(y_true_valid, y_pred_valid)
    spearman_r, _ = spearmanr(y_true_valid, y_pred_valid)
    
    return {
        'rmse': rmse, 'mae': mae, 'r2': r2,
        'pearson_r': pearson_r, 'spearman_r': spearman_r,
        'valid_predictions': len(y_pred_valid), 'total_predictions': len(y_pred)
    }

def print_metrics(metrics):
    """Print regression metrics."""
    print(f"\nRegression Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE: {metrics['mae']:.4f}")  
    print(f"  R²: {metrics['r2']:.4f}")
    print(f"  Pearson r: {metrics['pearson_r']:.4f}")
    print(f"  Spearman r: {metrics['spearman_r']:.4f}")
    print(f"  Valid predictions: {metrics['valid_predictions']}/{metrics['total_predictions']}")


def save_predictions(results, y_true, output_dir="/home/er8813ha/immunopeptides/plots/test"):
    """Save predictions to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save real data predictions
    real_df = pd.DataFrame({
        'complex_filename': results['real']['data'].index.tolist(),
        'y_true': y_true,
        'y_pred': results['real']['predictions']
    })
    
    real_path = os.path.join(output_dir, "test_predictions.csv")
    real_df.to_csv(real_path, index=False)
    print(f"Real test predictions saved to: {real_path}")
    
    # Save decoy predictions
    decoy_data = []
    
    if results['shuffled'] is not None:
        for name, pred in zip(results['shuffled']['data'].index, results['shuffled']['predictions']):
            decoy_data.append({'complex_filename': name, 'data_type': 'shuffled', 'prediction': pred})
    
    if results['random'] is not None:
        for name, pred in zip(results['random']['data'].index, results['random']['predictions']):
            decoy_data.append({'complex_filename': name, 'data_type': 'random', 'prediction': pred})
    
    if decoy_data:
        decoy_df = pd.DataFrame(decoy_data)
        decoy_path = os.path.join(output_dir, "decoy_predictions.csv")
        decoy_df.to_csv(decoy_path, index=False)
        print(f"Decoy predictions saved to: {decoy_path}")



def equation_string_to_pandas(equation_string, X):
    """
    Convert equation string to pandas operations.
    
   
    """

    
    # Parse the equation string
    expr = sp.sympify(equation_string)
    
    # Get all symbols (variables) from the equation
    symbols = list(expr.free_symbols)
    
    # Create a mapping from symbol names to DataFrame columns
    substitutions = {}
    for symbol in symbols:
        col_name = str(symbol)
        if col_name in X.columns:
            substitutions[symbol] = X[col_name].values
        else:
            raise ValueError(f"Column '{col_name}' not found in DataFrame. Available columns: {list(X.columns)}")
    
    # Convert sympy expression to numpy function
    # Create a lambda function that can handle numpy arrays
    func = sp.lambdify(symbols, expr, modules=['numpy'])
    
    # Apply the function
    if len(symbols) == 1:
        # Single variable case
        result = func(list(substitutions.values())[0])
    else:
        # Multiple variables case
        result = func(*substitutions.values())
    
    return result

def scoring_func_regression(X, equation_string):
    """Apply regression equation from string to DataFrame."""
    return equation_string_to_pandas(equation_string, X)

def scoring_func_classification(X, equation_string):
    """Apply classification equation from string to DataFrame."""
    scores = equation_string_to_pandas(equation_string, X)
    return scores


def main():

    regression_equation_string = "0.39*(0.74*interface_dG - 1)**2 + 6.3"
    classification_equation_string = "0.48 - 0.35*peptide_pae"

    X_real, y_real, X_shuffle, X_random = load_test_data()
    
    scaling_params_reg = load_scaling_params("regression")
    scaling_params_class = load_scaling_params("classification")
    

    X_real_scaled_predictions_reg = scoring_func_regression(scale_data(X_real, scaling_params_reg), regression_equation_string)
    X_shuffle_scaled_predictions_reg = scoring_func_regression(scale_data(X_shuffle, scaling_params_reg), regression_equation_string)
    X_random_scaled_predictions_reg = scoring_func_regression(scale_data(X_random, scaling_params_reg), regression_equation_string)
    X_real_scaled_predictions_class = scoring_func_classification(scale_data(X_real, scaling_params_class), classification_equation_string)
    X_shuffle_scaled_predictions_class = scoring_func_classification(scale_data(X_shuffle, scaling_params_class), classification_equation_string)
    X_random_scaled_predictions_class = scoring_func_classification(scale_data(X_random, scaling_params_class), classification_equation_string)

    # regression plot
    plt.figure(figsize=(4, 4))
    sns.regplot(x=y_real, y=X_real_scaled_predictions_reg)
    plt.xlabel("True pKd")
    plt.ylabel("Predicted pKd")
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/regression_scatter_plot.png", dpi=300)
    
    plt.figure(figsize=(4, 4))
    sns.regplot(x=y_real, y=X_real_scaled_predictions_class)
    plt.xlabel("True pKd")
    plt.ylabel("Predicted pKd")
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/classification_scatter_plot.png", dpi=300)

    # histogram of classification predictions
    plt.figure(figsize=(4, 4))
    sns.kdeplot(X_real_scaled_predictions_class, fill=True, color=color_palette['real'])
    sns.kdeplot(X_shuffle_scaled_predictions_class, fill=True, color=color_palette['shuffled'])
    sns.kdeplot(X_random_scaled_predictions_class, fill=True, color=color_palette['random'])
    plt.xlabel("Predicted pKd")
    plt.ylabel("Density")
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/classification_histogram.png", dpi=300)

    # histogram of regression predictions
    plt.figure(figsize=(4, 4))
    sns.kdeplot(X_real_scaled_predictions_reg, fill=True, color=color_palette['real'])
    sns.kdeplot(X_shuffle_scaled_predictions_reg, fill=True, color=color_palette['shuffled'])
    sns.kdeplot(X_random_scaled_predictions_reg, fill=True, color=color_palette['random'])
    plt.xlabel("Predicted pKd")
    plt.ylabel("Density")
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/regression_histogram.png", dpi=300)


if __name__ == "__main__":
    main()