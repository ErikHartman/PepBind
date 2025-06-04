import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp

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

def load_data(data="test", base_path="/srv/data1/general/immunopeptides_data/"):

    assert data in {"train", "val", "test", "all"}, "data must be one of 'train', 'val', 'test', or 'all'"
    scores_path = os.path.join(base_path, "outputs/binding_score_function_prod/4_processed_scores/")
    
    def load_split(split):
        X_real = pd.read_csv(os.path.join(scores_path, f"real_X_{split}.csv")).set_index("complex_filename")
        y_real = pd.read_csv(os.path.join(scores_path, f"real_y_{split}.csv"))["pKd"]
        X_shuffle = pd.read_csv(os.path.join(scores_path, f"shuffle_X_{split}.csv")).set_index("complex_filename")
        X_random = pd.read_csv(os.path.join(scores_path, f"random_X_{split}.csv")).set_index("complex_filename")

        for df in [X_real, X_shuffle, X_random]:
            df.drop(columns=[col for col in df.columns if col.startswith("Unnamed:")], inplace=True, errors='ignore')

        return X_real, y_real, X_shuffle, X_random

    if data == "all":
        X_real_all, y_real_all, X_shuffle_all, X_random_all = [], [], [], []
        for split in ["train", "val", "test"]:
            X_r, y_r, X_s, X_rand = load_split(split)
            X_real_all.append(X_r)
            y_real_all.append(y_r)
            X_shuffle_all.append(X_s)
            X_random_all.append(X_rand)
        
        X_real = pd.concat(X_real_all)
        y_real = pd.concat(y_real_all).values
        X_shuffle = pd.concat(X_shuffle_all)
        X_random = pd.concat(X_random_all)
    else:
        X_real, y_real, X_shuffle, X_random = load_split(data)
        y_real = y_real.values

    print(f"Loaded '{data}' data:")
    print(f"  Real: {X_real.shape} samples")
    print(f"  Shuffled: {X_shuffle.shape} samples")
    print(f"  Random: {X_random.shape} samples")

    return X_real, y_real, X_shuffle, X_random



def load_scaling_params(scaling_type="regression"):
    """Load scaling parameters from training."""
    scaling_path = f"/home/er8813ha/immunopeptides/plots/{scaling_type}/scaling_params.csv"
    return pd.read_csv(scaling_path, index_col=0)

def scale_data(X, scaling_params):
    """Scale data using provided scaling parameters."""
    feature_means = scaling_params['mean']
    feature_stds = scaling_params['std']
    return (X - feature_means) / feature_stds


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


    os.makedirs("/home/er8813ha/immunopeptides/plots/test", exist_ok=True)

    regression_equation_string = "0.39*(0.74*interface_dG - 1)**2 + 6.3"
    classification_equation_string = "0.48 - 0.35*peptide_pae"

    both_equation_string = f"({regression_equation_string}) + ({classification_equation_string})"
    print(both_equation_string)

    X_real, y_real, X_shuffle, X_random = load_data('test')
    
    scaling_params_reg = load_scaling_params("regression")
    scaling_params_class = load_scaling_params("classification")
    

    X_real_scaled_predictions_reg = scoring_func_regression(scale_data(X_real, scaling_params_reg), regression_equation_string)
    X_shuffle_scaled_predictions_reg = scoring_func_regression(scale_data(X_shuffle, scaling_params_reg), regression_equation_string)
    X_random_scaled_predictions_reg = scoring_func_regression(scale_data(X_random, scaling_params_reg), regression_equation_string)
    X_real_scaled_predictions_class = scoring_func_classification(scale_data(X_real, scaling_params_class), classification_equation_string)
    X_shuffle_scaled_predictions_class = scoring_func_classification(scale_data(X_shuffle, scaling_params_class), classification_equation_string)
    X_random_scaled_predictions_class = scoring_func_classification(scale_data(X_random, scaling_params_class), classification_equation_string)

    X_real_scaled_predictions_both = scoring_func_regression(scale_data(X_real, scaling_params_reg), both_equation_string)
    X_shuffle_scaled_predictions_both = scoring_func_regression(scale_data(X_shuffle, scaling_params_reg), both_equation_string)
    X_random_scaled_predictions_both = scoring_func_regression(scale_data(X_random, scaling_params_reg), both_equation_string)

    # regression plot
    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    sns.regplot(x=y_real, y=X_real_scaled_predictions_reg, ax=axs[0])
    axs[0].set_xlabel("True pKd")
    axs[0].set_ylabel("Predicted pKd")

    axs[0].annotate(
        f"r = {np.corrcoef(y_real, X_real_scaled_predictions_reg)[0, 1]:.2f}",
        xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10
    )

    sns.histplot(X_real_scaled_predictions_reg, kde=True, color=color_palette['real'], ax=axs[1], label='Real')
    sns.histplot(X_shuffle_scaled_predictions_reg, kde=True, color=color_palette['shuffled'], ax=axs[1], label='Shuffled')
    sns.histplot(X_random_scaled_predictions_reg, kde=True, color=color_palette['random'], ax=axs[1], label='Random')
    axs[1].set_xlabel("Predicted pKd")
    axs[1].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/regression_test.png", dpi=300)
    
    # classification plot
    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    sns.regplot(x=y_real, y=X_real_scaled_predictions_class, ax=axs[0])
    axs[0].set_xlabel("True pKd")
    axs[0].set_ylabel("Predicted pKd")

    sns.histplot(X_real_scaled_predictions_class, kde=True, color=color_palette['real'], ax=axs[1], label='Real')
    sns.histplot(X_shuffle_scaled_predictions_class, kde=True, color=color_palette['shuffled'], ax=axs[1], label='Shuffled')
    sns.histplot(X_random_scaled_predictions_class, kde=True, color=color_palette['random'], ax=axs[1], label='Random')
    axs[1].set_xlabel("Predicted pKd")
    axs[1].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/classification_test.png", dpi=300)


    # histogram of both predictions
    fig, axs = plt.subplots(1,2, figsize=(8, 4))
    sns.regplot(x=y_real, y=X_real_scaled_predictions_both, ax=axs[0])
    axs[0].set_xlabel("True pKd")
    axs[0].set_ylabel("Predicted pKd")
    axs[0].annotate(
        f"r = {np.corrcoef(y_real, X_real_scaled_predictions_both)[0, 1]:.2f}",
        xy=(0.05, 0.95), xycoords='axes fraction', fontsize=10
    )

    sns.histplot(X_real_scaled_predictions_both, kde=True, color=color_palette['real'], ax=axs[1], label='Real')
    sns.histplot(X_shuffle_scaled_predictions_both, kde=True, color=color_palette['shuffled'], ax=axs[1], label='Shuffled')
    sns.histplot(X_random_scaled_predictions_both, kde=True, color=color_palette['random'], ax=axs[1], label='Random')
    axs[1].set_xlabel("Predicted pKd")
    axs[1].set_ylabel("Density")
    plt.tight_layout()
    plt.savefig("/home/er8813ha/immunopeptides/plots/test/both_test.png", dpi=300)
    

if __name__ == "__main__":
    main()