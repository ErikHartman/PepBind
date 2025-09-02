import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from sklearn.metrics import roc_curve, auc
import sympy as sp

# Set plotting style
sns.set_context("paper")
color_palette = {
    "real": "#2C8C99",
    "shuffled": "#E88873", 
    "random": "#F46036",
    "prediction": "#124E78",
    "real_test": "#2C8C99", 
    "real_all": "#124E78" 
}


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

def scale_data(X, scaling_params, scaler_type="standard"):
    """
    Scale data using provided scaling parameters.
    
    """
    if scaler_type == "standard":
        feature_means = scaling_params['mean']
        feature_stds = scaling_params['std']
        return (X - feature_means) / feature_stds
    
    elif scaler_type == "minmax":
        feature_mins = scaling_params['min']
        feature_maxs =  scaling_params['max']
        return (X - feature_mins) / (feature_maxs - feature_mins)
    
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}. Use 'standard' or 'minmax'.")


def save_predictions(results, y_true, output_dir="/home/er8813ha/immunopeptides/plots/test"):
    """Save predictions to CSV files."""
    os.makedirs(output_dir, exist_ok=True)
    real_df = pd.DataFrame({
        'complex_filename': results['real']['data'].index.tolist(),
        'y_true': y_true,
        'y_pred': results['real']['predictions']
    })
    
    real_path = os.path.join(output_dir, "test_predictions.csv")
    real_df.to_csv(real_path, index=False)
    print(f"Real test predictions saved to: {real_path}")
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
    expr = sp.sympify(equation_string)
    symbols = list(expr.free_symbols)
    
    substitutions = {}
    for symbol in symbols:
        col_name = str(symbol)
        if col_name in X.columns:
            substitutions[symbol] = X[col_name].values
        else:
            raise ValueError(f"Column '{col_name}' not found in DataFrame. Available columns: {list(X.columns)}")
    
    func = sp.lambdify(symbols, expr, modules=['numpy'])
    result = func(*substitutions.values())
    
    return result

def scoring_func_regression(X, equation_string):
    """Apply regression equation from string to DataFrame."""
    return equation_string_to_pandas(equation_string, X)

def scoring_func_classification(X, equation_string):
    """Apply classification equation from string to DataFrame."""
    scores = equation_string_to_pandas(equation_string, X)
    return scores

def combined_scoring_function(X, reg_equation, class_equation, 
                             scaling_params_reg=None, scaling_params_class=None,
                             scaler_type="standard"):
    """
    Apply combined scoring using both regression and classification models with proper scaling.
    """
    # Scale data appropriately for each model
    if scaling_params_reg is not None:
        X_reg_scaled = scale_data(X, scaling_params_reg, scaler_type=scaler_type)
    else:
        X_reg_scaled = X 
        
    if scaling_params_class is not None:
        X_class_scaled = scale_data(X, scaling_params_class, scaler_type=scaler_type)
    else:
        X_class_scaled = X
    
    reg_predictions = scoring_func_regression(X_reg_scaled, reg_equation)
    class_predictions = scoring_func_classification(X_class_scaled, class_equation)
    

    combined_scores = reg_predictions * class_predictions
    
    return combined_scores


def main():
    """Run the analysis pipeline with the specified configuration."""
    # Create output directory
    output_dir = "/home/er8813ha/immunopeptides/plots/test"
    os.makedirs(output_dir, exist_ok=True)

    # Define equation strings for both models
    regression_equation_string = "-rosetta_score - (distance_score + 3.0185924)*(interface_dG - 2.4250882)"
    classification_equation_string = "-iptm*(peptide_pae - 1.2075188)"

    # Load test data
    data_split = 'test'
    X_real, y_real, X_shuffle, X_random = load_data(data_split)
    
    # Load scaling parameters for each model type
    scaling_params_reg = load_scaling_params("regression")
    scaling_params_class = load_scaling_params("classification")
    
    # Set scaling type
    scaler_type = "minmax"  # "minmax" or "standard"

    # Regression predictions with proper scaling
    X_real_scaled_predictions_reg = scoring_func_regression(
        scale_data(X_real, scaling_params_reg, scaler_type=scaler_type), 
        regression_equation_string
    )
    X_shuffle_scaled_predictions_reg = scoring_func_regression(
        scale_data(X_shuffle, scaling_params_reg, scaler_type=scaler_type), 
        regression_equation_string
    )
    X_random_scaled_predictions_reg = scoring_func_regression(
        scale_data(X_random, scaling_params_reg, scaler_type=scaler_type), 
        regression_equation_string
    )
    
    # Classification predictions with proper scaling
    X_real_scaled_predictions_class = scoring_func_classification(
        scale_data(X_real, scaling_params_class, scaler_type=scaler_type), 
        classification_equation_string
    )
    X_shuffle_scaled_predictions_class = scoring_func_classification(
        scale_data(X_shuffle, scaling_params_class, scaler_type=scaler_type), 
        classification_equation_string
    )
    X_random_scaled_predictions_class = scoring_func_classification(
        scale_data(X_random, scaling_params_class, scaler_type=scaler_type), 
        classification_equation_string
    )
    
    # Apply the combined scoring function with proper scaling for each component
    X_real_scaled_predictions_both = combined_scoring_function(
        X_real, 
        regression_equation_string, 
        classification_equation_string,
        scaling_params_reg=scaling_params_reg,
        scaling_params_class=scaling_params_class,
        scaler_type=scaler_type
    )
    
    X_shuffle_scaled_predictions_both = combined_scoring_function(
        X_shuffle, 
        regression_equation_string, 
        classification_equation_string,
        scaling_params_reg=scaling_params_reg,
        scaling_params_class=scaling_params_class,
        scaler_type=scaler_type
    )
    
    X_random_scaled_predictions_both = combined_scoring_function(
        X_random, 
        regression_equation_string, 
        classification_equation_string,
        scaling_params_reg=scaling_params_reg,
        scaling_params_class=scaling_params_class,
        scaler_type=scaler_type
    )

    fig, axs = plt.subplots(1, 3, figsize=(9,2))
    
    sns.regplot(x=y_real, y=X_real_scaled_predictions_reg, ax=axs[0], scatter_kws={'s':5}, line_kws={"color": "#E88873"})
    axs[0].set_xlabel("True pKd")
    axs[0].set_ylabel("Predicted pKd")
    axs[0].annotate(
        f"r = {np.corrcoef(y_real, X_real_scaled_predictions_reg)[0, 1]:.2f}",
        xy=(0.05, 0.9), xycoords='axes fraction'
    )
    
    hist_data_reg = pd.DataFrame({
        'Score': np.concatenate([X_real_scaled_predictions_reg, X_shuffle_scaled_predictions_reg, X_random_scaled_predictions_reg]),
        'Type': np.concatenate([
            np.repeat('Real', len(X_real_scaled_predictions_reg)),
            np.repeat('Shuffled', len(X_shuffle_scaled_predictions_reg)),
            np.repeat('Random', len(X_random_scaled_predictions_reg))
        ])
    })
    
    sns.histplot(
        data=hist_data_reg,
        x='Score',
        hue='Type',
        kde=True,
        bins=10,
        palette={'Real': color_palette['real'], 'Shuffled': color_palette['shuffled'], 'Random': color_palette['random']},
        ax=axs[2]
    )
    axs[2].set_xlabel("Predicted pKd")
    axs[2].set_ylabel("Density")
    axs[2].legend(frameon=False)
    
    y_combined_all_decoys = np.concatenate([
        np.ones(len(X_real_scaled_predictions_reg)),
        np.zeros(len(X_shuffle_scaled_predictions_reg) + len(X_random_scaled_predictions_reg))
    ])
    scores_combined_all_decoys = np.concatenate([
        X_real_scaled_predictions_reg, 
        X_shuffle_scaled_predictions_reg,
        X_random_scaled_predictions_reg
    ])
    fpr_vs_all_decoys, tpr_vs_all_decoys, _ = roc_curve(y_combined_all_decoys, scores_combined_all_decoys)
    roc_auc_vs_all_decoys = auc(fpr_vs_all_decoys, tpr_vs_all_decoys)
    
    axs[1].plot(fpr_vs_all_decoys, tpr_vs_all_decoys,  
              label=f'AUC = {roc_auc_vs_all_decoys:.2f}')

    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"/home/er8813ha/immunopeptides/plots/test/regression_{data_split}.svg", dpi=300)
    
    fig, axs = plt.subplots(1, 3, figsize=(9,2))
    
    sns.regplot(x=y_real, y=X_real_scaled_predictions_class, ax=axs[0], scatter_kws={'s':5}, line_kws={"color": "#E88873"})
    axs[0].set_xlabel("True pKd")
    axs[0].set_ylabel("Predicted score")
    axs[0].annotate(
        f"r = {np.corrcoef(y_real, X_real_scaled_predictions_class)[0, 1]:.2f}",
        xy=(0.05, 0.9), xycoords='axes fraction'
    )


    hist_data_class = pd.DataFrame({
    'Score': np.concatenate([X_real_scaled_predictions_class, X_shuffle_scaled_predictions_class, X_random_scaled_predictions_class]),
    'Type': np.concatenate([
        np.repeat('Real', len(X_real_scaled_predictions_class)),
        np.repeat('Shuffled', len(X_shuffle_scaled_predictions_class)),
        np.repeat('Random', len(X_random_scaled_predictions_class))
        ])
    })

    # Plot histogram with a single call
    sns.histplot(
        data=hist_data_class,
        x='Score',
        hue='Type',
        kde=True,
        palette={'Real': color_palette['real'], 'Shuffled': color_palette['shuffled'], 'Random': color_palette['random']},
        ax=axs[2]
    )
    

    y_combined_all_decoys = np.concatenate([
        np.ones(len(X_real_scaled_predictions_class)),
        np.zeros(len(X_shuffle_scaled_predictions_class) + len(X_random_scaled_predictions_class))
    ])
    scores_combined_all_decoys = np.concatenate([
        X_real_scaled_predictions_class, 
        X_shuffle_scaled_predictions_class,
        X_random_scaled_predictions_class
    ])
    fpr_vs_all_decoys, tpr_vs_all_decoys, _ = roc_curve(y_combined_all_decoys, scores_combined_all_decoys)
    roc_auc_vs_all_decoys = auc(fpr_vs_all_decoys, tpr_vs_all_decoys)
    
    # Plot ROC curves - only Real vs All Decoys
    axs[1].plot(fpr_vs_all_decoys, tpr_vs_all_decoys,  
              label=f'AUC = {roc_auc_vs_all_decoys:.2f}')

    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(f"/home/er8813ha/immunopeptides/plots/test/classification_{data_split}.svg", dpi=300)



    fig, axs = plt.subplots(1, 3, figsize=(9,2))
    

    sns.regplot(x=y_real, y=X_real_scaled_predictions_both, ax=axs[0], scatter_kws={'s':5}, line_kws={"color": "#E88873"})

    axs[0].annotate(
        
        f"r = {np.corrcoef(y_real, X_real_scaled_predictions_both)[0, 1]:.2f}",
        xy=(0.05, 0.9), xycoords='axes fraction'
    )
    axs[0].set_xlabel("True pKd")
    axs[0].set_ylabel("Combined Score")
    
    hist_data_both = pd.DataFrame({
    'Score': np.concatenate([X_real_scaled_predictions_both, X_shuffle_scaled_predictions_both, X_random_scaled_predictions_both]),
    'Type': np.concatenate([
        np.repeat('Real', len(X_real_scaled_predictions_both)),
        np.repeat('Shuffled', len(X_shuffle_scaled_predictions_both)),
        np.repeat('Random', len(X_random_scaled_predictions_both))
        ])
    })

    # Plot histogram with a single call
    sns.histplot(
        data=hist_data_both,
        x='Score',
        hue='Type',
        kde=True,
        bins=10,
        palette={'Real': color_palette['real'], 'Shuffled': color_palette['shuffled'], 'Random': color_palette['random']},
        ax=axs[2]
    )
        
    axs[2].set_xlabel("Combined Score")
    axs[2].set_ylabel("Density")
    axs[2].legend(loc='upper right', frameon=False)
    

    y_combined_all_decoys = np.concatenate([
        np.ones(len(X_real_scaled_predictions_both)),
        np.zeros(len(X_shuffle_scaled_predictions_both) + len(X_random_scaled_predictions_both))
    ])
    scores_combined_all_decoys = np.concatenate([
        X_real_scaled_predictions_both, 
        X_shuffle_scaled_predictions_both,
        X_random_scaled_predictions_both
    ])
    fpr_vs_all_decoys, tpr_vs_all_decoys, _ = roc_curve(y_combined_all_decoys, scores_combined_all_decoys)
    roc_auc_vs_all_decoys = auc(fpr_vs_all_decoys, tpr_vs_all_decoys)
    
    # Plot ROC curves - only Real vs All Decoys
    axs[1].plot(fpr_vs_all_decoys, tpr_vs_all_decoys, 
              label=f'AUC = {roc_auc_vs_all_decoys:.2f}')
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    # Save the figure with the combination method in the filename
    plt.savefig(f"/home/er8813ha/immunopeptides/plots/test/combined_{data_split}.svg", dpi=300)

    # save all predictions and real values to csv
    results = {
        'real': {
            'data': X_real,
            'predictions': X_real_scaled_predictions_both
        },
        'shuffled': {
            'data': X_shuffle,
            'predictions': X_shuffle_scaled_predictions_both
        },
        'random': {
            'data': X_random,
            'predictions': X_random_scaled_predictions_both
        }
    }
    save_predictions(results, y_real, output_dir=output_dir)

    from bopep.scoring.scores_to_objective import ScoresToObjective, benchmark_objective

    # build the raw-scores dict in the shape your benchmark_objective expects
    raw_scores = {}
    for idx, row in X_real.iterrows():
        raw_scores[idx] = {
            "rosetta_score":    row["rosetta_score"],
            "interface_dG":     row["interface_dG"],
            "distance_score":   row["distance_score"],
            "iptm":             row["iptm"],
            "peptide_pae":      row["peptide_pae"],
            "in_binding_site":  True
        }

    # compute benchmarked objectives
    objective = ScoresToObjective()
    bench_results = objective.create_objective(raw_scores, benchmark_objective)

    # extract in the same order as X_real.index
    bench_preds = np.array([bench_results[name] for name in X_real.index])

    # quick sanity check: print the first five from each
    print("First 5 combined_scoring_function preds:", X_real_scaled_predictions_both[:5])
    print("First 5 benchmark_objective preds:    ", bench_preds[:5])

    # optionally compute correlation to see how well they agree
    corr = np.corrcoef(X_real_scaled_predictions_both, bench_preds)[0,1]
    print(f"Pearson r between pipelines: {corr:.3f}")
    



if __name__ == "__main__":
    main()