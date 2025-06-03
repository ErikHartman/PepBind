import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import pandas as pd
import os
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr, spearmanr

color_palette = {
    "symbolic": "#124E78", 
    "lasso": "#B388EB",    
    "rf": "#57A773",        
    "svr": "#2C8C99",   
    "logreg": "#B388EB", 
    "svc": "#2C8C99",     

    "random": "#F46036",
    "shuffle": "#E88873",  
    "real": "#2C8C99",  

    "train": "#2C8C99", 
    "val": "#3943B7", 

}
    
sns.set_context("paper")

def plot_classification_metrics(y_true, y_pred, y_proba, y_train=None, train_proba=None, model_name="Model", output_path=None):
    """
    Plot confusion matrix and ROC curve for classification results.
    If train data is provided, both train and val ROC curves will be shown.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    val_auc = roc_auc_score(y_true, y_proba)
    
    # Plot val ROC curve
    ax2.plot(fpr, tpr, color=color_palette["val"], linestyle='--',
             label=f'Val ROC (AUC = {val_auc:.3f})')
    
    # Plot train ROC curve if provided
    if y_train is not None and train_proba is not None:
        fpr_train, tpr_train, _ = roc_curve(y_train, train_proba)
        train_auc = roc_auc_score(y_train, train_proba)
        ax2.plot(fpr_train, tpr_train, color=color_palette["train"],
                 label=f'Train ROC (AUC = {train_auc:.3f})')
    
    ax2.set_xlim([-0.05, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_regression_scatter(
    y_train, train_preds, 
    y_val=None, val_preds=None, 
    model_name="Model", 
    output_path=None
):
    """
    Creates and saves a regplot (scatter with regression line) of actual vs. predicted,
    with a subplot showing the residuals (deviation from perfect prediction).
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # First subplot: Actual vs Predicted with regression line
    # Train data with regplot
    sns.regplot(
        x=y_train, 
        y=train_preds, 
        scatter_kws={'alpha': 0.5, 's': 5, 'color': color_palette["train"]}, 
        line_kws={'color': color_palette["train"]},
        label='Train',
        ax=ax1
    )
    
    if y_val is not None and val_preds is not None:
        # Val data with regplot
        sns.regplot(
            x=y_val, 
            y=val_preds, 
            scatter_kws={'alpha': 0.5, 's': 10, 'color': color_palette["val"]}, 
            line_kws={'color': color_palette["val"]},
            label='Val',
            ax=ax1
        )
        y_all = np.concatenate([y_train, y_val])
    else:
        y_all = y_train
    
    # Perfect prediction line (diagonal)
    ax1.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--', alpha=0.5, label='Perfect prediction')
    
    ax1.set_xlabel('Actual pKd')
    ax1.set_ylabel('Predicted pKd')
    ax1.legend(frameon=False)
    
    # Second subplot: Residuals (actual - predicted)
    train_residuals = y_train - train_preds
    
    # Plot training residuals
    sns.scatterplot(
        x=y_train, 
        y=train_residuals, 
        s=10, 
        color=color_palette["train"],
        label='Train',
        ax=ax2
    )
    
    # Add a horizontal line at y=0 (perfect prediction)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # If val data is available, add val residuals
    if y_val is not None and val_preds is not None:
        val_residuals = y_val - val_preds
        sns.scatterplot(
            x=y_val, 
            y=val_residuals, 
            s=10, 
            color=color_palette["val"],
            label='Val',
            ax=ax2
        )
    
    # Calculate and annotate RMSE for train and val
    train_rmse = np.sqrt(np.mean(train_residuals**2))
    rmse_text = f"Train RMSE: {train_rmse:.3f}"
    
    if y_val is not None and val_preds is not None:
        val_rmse = np.sqrt(np.mean((y_val - val_preds)**2))
        rmse_text += f"\nVal RMSE: {val_rmse:.3f}"
    
    # Add RMSE text to the residual plot
    ax2.text(
        0.05, 0.95, rmse_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax2.set_xlabel('Actual pKd')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.legend(frameon=False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importances(feature_df, model_name="Model", output_path=None):
    """
    Creates and saves a barplot of feature importances or coefficients.
    """
    plt.figure(figsize=(4,4))
    feature_df = feature_df.sort_values(by=feature_df.columns[1], ascending=False).head(10)
    plt.barh(feature_df['Feature'], feature_df.iloc[:, 1])
    
    if 'Importance' in feature_df.columns:
        plt.xlabel('Importance')
    else:
        plt.xlabel('Coefficient')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_model_comparison(comparison_df, roc_data=None, output_path=None):
    model_names = comparison_df['Model'].values
    metrics = [col for col in comparison_df.columns if col != 'Model']
    
    # If it's classification metrics
    if 'Train Accuracy' in metrics:
        metric_list = ['Val Accuracy','Val AUC']
        metric_list = [m for m in metric_list if m in metrics]
        
        # Determine if we should add ROC curve subplot
        add_roc_subplot = roc_data is not None
        
        n_subplots = len(metric_list) + (1 if add_roc_subplot else 0)
        fig, axs = plt.subplots(1, n_subplots, figsize=(3 * n_subplots, 3), squeeze=False)
        axs = axs.flatten()  # easier to iterate
        
        # Plot bar charts for each metric
        for i, metric in enumerate(metric_list):
            colors = [color_palette.get(x.lower(), 'gray') for x in model_names]
            ax = axs[i]
            ax.bar(model_names, comparison_df[metric].values, color=colors)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)

            if metric == 'Val AUC':
                # Set y-axis limits for AUC
                ax.set_ylim([0.8, 1])
            if metric == 'Val Accuracy':
                # Set y-axis limits for accuracy
                ax.set_ylim([0.6, 1])
        
        # Plot ROC curves in the last subplot if provided
        if add_roc_subplot:
            ax = axs[-1]
            for model in model_names:
                if model in roc_data:
                    fpr, tpr, roc_auc = roc_data[model]
                    ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})', 
                            color=color_palette.get(model.lower(), 'gray'))
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.legend(loc='lower right', frameon=False)
        
        plt.tight_layout()
    
    # If it's regression metrics
    elif 'Train RMSE' in metrics:
        regression_metrics = []
        for m in ['Val MAE', 'Val R²']:
            if m in metrics:
                regression_metrics.append(m)
        n_subplots = len(regression_metrics)
        fig, axs = plt.subplots(1, n_subplots, figsize=(3 * n_subplots, 3), squeeze=False)
        axs = axs.flatten()
        for i, metric in enumerate(regression_metrics):
            colors = [color_palette.get(x.lower(), 'gray') for x in model_names]
            ax = axs[i]
            ax.bar(model_names, comparison_df[metric].values, color=colors)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.set_ylim([min(comparison_df[metric].values) * 0.9, max(comparison_df[metric].values) * 1.1])
        plt.tight_layout()
        
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_symbolic_complexity_tradeoff(
    equations_df: pd.DataFrame,
    metric: str = 'val_rmse',
    model_type: str = 'regression',
    top_n: int = 50,
    output_path: str = None,
    show_equations: bool = True,
    max_equations: int = 5,
    lower_is_better: bool = False,
):
    """
    Plot the tradeoff between model complexity and accuracy for symbolic models.
    For regression: lower metric (RMSE) is better
    For classification: higher metric (AUC/accuracy) is better
    """
    if len(equations_df) == 0:
        print("No equations to plot.")
        return
    
    # In case we have more equations than we want to plot
    if top_n and len(equations_df) > top_n:
        if model_type == 'regression':
            # For regression, sort by metric ascending (lower RMSE is better)
            df = equations_df.sort_values(by=metric).head(top_n)
        else:
            # For classification, sort by metric descending (higher AUC is better)
            df = equations_df.sort_values(by=metric, ascending=False).head(top_n)
    else:
        df = equations_df.copy()
    
    # Drop rows with NaN in the metric column
    df = df.dropna(subset=[metric])
    
    if len(df) == 0:
        print(f"No valid equations with metric {metric}.")
        return
        
    # Get the corresponding train metric
    train_metric = metric.replace('val_', 'train_')
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 7))
    
    # Create gridspec to have plots on top (larger) and equation legend below
    gs = fig.add_gridspec(2, 2, height_ratios=[1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax_legend = fig.add_subplot(gs[1, :])
    
    # Plot complexity vs val metric
    scatter_val = ax1.scatter(
        df['complexity'], 
        df[metric], 
        s=100,
        color=color_palette["val"],
    )
    
    ax1.set_xlabel('Complexity')
    val_label = metric.replace('_', ' ').title()
    ax1.set_ylabel(val_label)
    
    # Plot complexity vs train metric if available
    if train_metric in df.columns:
        scatter_train = ax2.scatter(
            df['complexity'], 
            df[train_metric], 
            s=100, 
            color=color_palette["train"],
        )
        
        ax2.set_xlabel('Complexity')

    if lower_is_better:
        # For regression, lower metric is better (e.g., RMSE)
        is_pareto = np.ones(len(df), dtype=bool)
        for i, (c1, m1) in enumerate(zip(df['complexity'], df[metric])):
            for c2, m2 in zip(df['complexity'], df[metric]):
                if (c2 < c1 and m2 <= m1) or (c2 <= c1 and m2 < m1):
                    is_pareto[i] = False
                    break
    else:
        # For classification, higher metric is better (e.g., AUC)
        is_pareto = np.ones(len(df), dtype=bool)
        for i, (c1, m1) in enumerate(zip(df['complexity'], df[metric])):
            for c2, m2 in zip(df['complexity'], df[metric]):
                if (c2 < c1 and m2 >= m1) or (c2 <= c1 and m2 > m1):
                    is_pareto[i] = False
                    break
                    
    # Highlight pareto front points
    pareto_points = df[is_pareto]
    pareto_scatter = ax1.scatter(
        pareto_points['complexity'], 
        pareto_points[metric], 
        edgecolor='orange', 
        facecolor='none', 
        s=200, 
        linewidth=2,
        label='Pareto front'
    )
    
    # Add numbered annotations on pareto points, with legend below plot
    if show_equations and len(pareto_points) > 0:
        # If there are too many Pareto points, only annotate the top few by score
        if len(pareto_points) > max_equations:
            pareto_points = pareto_points.sort_values(by='score', ascending=False).head(max_equations)
        
        # Turn off axis in the legend subplot
        ax_legend.axis('off')
        
        # Add numbered annotations to plot and corresponding equations to legend
        for i, (idx, row) in enumerate(pareto_points.iterrows(), 1):
            # Add number to the point
            ax1.annotate(
                str(i),
                xy=(row['complexity'], row[metric]),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='circle', fc='white', ec='red'),
                fontsize=10,
            )
            
            # Get full equation text and add to legend
            eq_str = row['equation']
            eq_metric = row[metric]
            
            # Format based on model type
            if model_type == 'regression':
                metric_str = f"{val_label}: {eq_metric:.4f}"
            else:
                metric_str = f"{val_label}: {eq_metric:.4f}"
                
            # Add to legend with equation complexity 
            legend_text = f"{i}. Complexity: {row['complexity']}, {metric_str}\n    {eq_str}"
            
            # Calculate vertical position for equation in legend (bottom to top)
            y_pos = 0.8 - (i-1) * (0.8 / max(max_equations, len(pareto_points)))
            
            # Add equation text to legend area
            ax_legend.text(0.05, y_pos, legend_text, 
                         va='center', fontsize=10, wrap=True, 
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))
    
    plt.tight_layout()

    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_probability_histograms(y_true, y_proba, model_name, output_path=None):
    """
    Plot histograms of predicted probabilities for each class.
    """
    plt.figure(figsize=(3,3))
    
    # Get probabilities for each class
    pos_probs = y_proba[y_true == 1]
    neg_probs = y_proba[y_true == 0]
    
    # Plot histograms
    sns.histplot(
        x=neg_probs, bins=30, alpha=0.8, kde=True, color=color_palette["random"], 
        label=f'Decoys'
    )
    sns.histplot(
        x=pos_probs, bins=30, alpha=0.8, kde=True, color=color_palette["real"], 
        label=f'Real'
    )
    
    # Add vertical line at decision threshold (0.5)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision threshold')
    
    plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.legend(frameon=False)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()

def plot_predictions_by_data_type(predictions_df, output_dir):
    """
    Create plots comparing model predictions on different data types (real, shuffle, random).
    """
    # Add binary labels: 1 for real, 0 for non-real (shuffle or random)
    predictions_df['binary_label'] = predictions_df['data_type'].apply(lambda x: 1 if x == 'Real' else 0)
    
    fig = plt.figure(figsize=(8, 3))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    
    # Stripplot in top-left position
    ax1 = fig.add_subplot(gs[:, 0])
    sns.stripplot(
        data=predictions_df, 
        x="model", 
        y="prediction", 
        hue="data_type",
        palette={"Real": color_palette["real"], "Shuffled": color_palette["shuffle"], "Random": color_palette["random"]},
        dodge=True,
        ax=ax1
    )
    
    ax1.set_ylim([3, 10]) # type: ignore
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Predicted pKd")
    ax1.legend(frameon=False)
    

    sns.boxplot(
        data=predictions_df, 
        x="model", 
        y="prediction", 
        hue="data_type",
        palette={"Real": color_palette["real"], "Shuffled": color_palette["shuffle"], "Random": color_palette["random"]},
        fill=False,
        ax=ax1, legend=False
    )

    
    ax3 = fig.add_subplot(gs[:, 1])
    
    model_names = predictions_df['model'].unique()
    for model_name in model_names:
        model_data = predictions_df[predictions_df['model'] == model_name]
        
        if len(model_data) > 0 and len(model_data['binary_label'].unique()) > 1:
            # Use prediction as score (higher pKd indicates more likely to be real)
            fpr, tpr, _ = roc_curve(model_data['binary_label'], model_data['prediction'])
            roc_auc = roc_auc_score(model_data['binary_label'], model_data['prediction'])
            
            ax3.plot(
                fpr, tpr, 
                label=f'{model_name} (AUC = {roc_auc:.3f})',
                color= color_palette.get(model_name.lower(), 'gray')
            )
    
    ax3.set_xlim([0.0, 1.0]) # type: ignore
    ax3.set_ylim([0.0, 1.05]) # type: ignore
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_shuffle_random_analysis.png"), dpi=300)
    plt.close()
    
    return {
        "combined_plot": os.path.join(output_dir, "real_vs_shuffle_random_analysis.png")
    }

def plot_pkd_probability_correlation(
    pkd_values, 
    probabilities, 
    model_names, 
    output_path=None,
    pkd_values_train=None,
    probabilities_train=None
):
    """
    Plot correlation between pKd values and classification probabilities for real samples.
    """
    if len(model_names) == 0:
        print("No models to plot correlations for.")
        return
    
    # Calculate number of rows and columns for subplots
    n_models = len(model_names)
    n_cols = n_models
    n_rows = 1
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        if model_name in probabilities:
            ax = axes[i]
            
            # Plot val data
            sns.regplot(
                x=pkd_values, 
                y=probabilities[model_name], 
                ax=ax,
                color=color_palette.get(model_name.lower(), 'blue'),
            )
            
            pearson_r, _ = pearsonr(pkd_values, probabilities[model_name])
            
            correlation_text = (
                f"Pearson r: {pearson_r:.3f}\n"
            )
            
            ax.text(
                0.05, 0.95, correlation_text,
                transform=ax.transAxes,
                verticalalignment='top',

            )
            
            ax.set_xlabel('pKd value')
            ax.set_ylabel(f'{model_name} probability')

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    """
    Command-line interface to regenerate plots from saved results.
    """
    parser = argparse.ArgumentParser(description="Generate plots from saved results")
    parser.add_argument("--mode", choices=["classification", "regression", "regression_and_classification"], required=True,
                       help="Mode: classification or regression")
    parser.add_argument("--input-dir", default=None, 
                       help="Directory containing saved results (default: './plots/{mode}')")
    parser.add_argument("--output-dir", default=None,
                       help="Directory to save plots (default: same as input-dir)")
    parser.add_argument("--symbolic-only", action="store_true",
                       help="Only regenerate symbolic model plots")

    args = parser.parse_args()
    
    # Set up directories
    if args.input_dir is None:
        args.input_dir = f"./plots/{args.mode}"
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print(f"Reading results from: {args.input_dir}")
    print(f"Saving plots to: {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    comparison_path = os.path.join(args.input_dir, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        print(f"Loaded model comparison from {comparison_path}")
    else:
        print(f"Warning: Model comparison file not found at {comparison_path}")
        comparison_df = None
    
    if args.mode == "classification":
        train_preds_path = os.path.join(args.input_dir, "train_predictions.csv")
        val_preds_path = os.path.join(args.input_dir, "val_predictions.csv")
        train_proba_path = os.path.join(args.input_dir, "train_probabilities.csv")
        val_proba_path = os.path.join(args.input_dir, "val_probabilities.csv")
        
        if all(os.path.exists(p) for p in [train_preds_path, val_preds_path, train_proba_path, val_proba_path]):
            train_preds = pd.read_csv(train_preds_path)
            val_preds = pd.read_csv(val_preds_path)
            train_proba = pd.read_csv(train_proba_path)
            val_proba = pd.read_csv(val_proba_path)
            print("Loaded prediction data for classification")
            
            # Calculate ROC curves for model comparison
            if comparison_df is not None:
                roc_data = {}
                for model in ["LogReg", "RF", "SVC", "Symbolic"]:
                    if f"{model.lower()}_proba" in val_proba.columns:
                        fpr, tpr, _ = roc_curve(val_proba["y_val"], val_proba[f"{model.lower()}_proba"])
                        roc_auc = roc_auc_score(val_proba["y_val"], val_proba[f"{model.lower()}_proba"])
                        roc_data[model] = (fpr, tpr, roc_auc)
                
                # Plot model comparison
                plot_model_comparison(
                    comparison_df, 
                    roc_data=roc_data,
                    output_path=os.path.join(args.output_dir, "model_comparison.png")
                )
                print("Generated model comparison plot")
            
            # Generate individual model plots
            if not args.symbolic_only:
                for model in ["logreg", "rf", "svc"]:
                    if f"{model}_proba" in val_proba.columns:
                        # Classification metrics plot
                        plot_classification_metrics(
                            val_preds["y_val"],
                            val_preds[f"{model}_pred"],
                            val_proba[f"{model}_proba"],
                            train_preds["y_train"],
                            train_proba[f"{model}_proba"],
                            model_name=model.upper(),
                            output_path=os.path.join(args.output_dir, f"{model}_metrics.png")
                        )
                        
                        # Probability histograms
                        plot_probability_histograms(
                            val_preds["y_val"],
                            val_proba[f"{model}_proba"],
                            model_name=model.upper(),
                            output_path=os.path.join(args.output_dir, f"{model}_proba_hist.png")
                        )
                print("Generated standard model plots")
            
            # Generate correlation plots between pKd and probabilities if available
            pkd_val_path = os.path.join(args.input_dir, "real_val_pkd.csv")
            pkd_train_path = os.path.join(args.input_dir, "real_train_pkd.csv")
            
            if os.path.exists(pkd_val_path):
                pkd_val_df = pd.read_csv(pkd_val_path)
                
                # Check for train data
                pkd_train_df = None
                if os.path.exists(pkd_train_path):
                    pkd_train_df = pd.read_csv(pkd_train_path)
                
                if not pkd_val_df.empty and "pKd" in pkd_val_df.columns:
                    print("Generating pKd-probability correlation plots")
                    
                    # Get model names
                    model_names = []
                    val_proba_dict = {}
                    train_proba_dict = {}
                    
                    for model in ["logreg", "rf", "svc", "symbolic"]:
                        if f"{model}_proba" in val_proba.columns:
                            # Filter to only real val samples
                            real_val_indices = val_proba["y_val"] == 1
                            if real_val_indices.sum() > 0:
                                model_upper = model.upper()
                                model_names.append(model_upper)
                                val_proba_dict[model_upper] = val_proba.loc[real_val_indices, f"{model}_proba"].values
                                
                                # Get train probabilities if available
                                if pkd_train_df is not None and not pkd_train_df.empty and "pKd" in pkd_train_df.columns:
                                    real_train_indices = train_proba["y_train"] == 1
                                    if real_train_indices.sum() > 0:
                                        train_proba_dict[model_upper] = train_proba.loc[real_train_indices, f"{model}_proba"].values
                    
                    if model_names:
                        plot_pkd_probability_correlation(
                            pkd_val_df["pKd"].values,
                            val_proba_dict,
                            model_names,
                            output_path=os.path.join(args.output_dir, "pkd_probability_correlation.png"),
                            pkd_values_train=pkd_train_df["pKd"].values if pkd_train_df is not None else None,
                            probabilities_train=train_proba_dict if train_proba_dict else None
                        )
                        print("Generated pKd-probability correlation plot")
            
            # Generate symbolic model plots
            if "symbolic_proba" in val_proba.columns:
                # Classification metrics plot
                plot_classification_metrics(
                    val_preds["y_val"],
                    val_preds["symbolic_pred"],
                    val_proba["symbolic_proba"],
                    train_preds["y_train"],
                    train_proba["symbolic_proba"],
                    model_name="Symbolic",
                    output_path=os.path.join(args.output_dir, "symbolic_metrics.png")
                )
                
                # Probability histograms
                plot_probability_histograms(
                    val_preds["y_val"],
                    val_proba["symbolic_proba"],
                    model_name="Symbolic",
                    output_path=os.path.join(args.output_dir, "symbolic_proba_hist.png")
                )
                
                # Complexity tradeoff plot (if data available)
                symbolic_eqs_path = os.path.join(args.input_dir, "symbolic_classification_all_equations.csv")
                if os.path.exists(symbolic_eqs_path):
                    equations_df = pd.read_csv(symbolic_eqs_path)
                    plot_symbolic_complexity_tradeoff(
                        equations_df,
                        metric='val_auc',
                        model_type='classification',
                        output_path=os.path.join(args.output_dir, "symbolic_classification_complexity_tradeoff.png"),
                        lower_is_better=False
                    )
                    print("Generated symbolic complexity tradeoff plot")
                else:
                    print(f"Warning: Symbolic equations file not found at {symbolic_eqs_path}")
                    
                print("Generated symbolic model plots")
                
        else:
            print("Error: Missing prediction data files")
            
    elif args.mode == "regression" or args.mode == "regression_and_classification":
        # Load predictions
        train_preds_path = os.path.join(args.input_dir, "train_predictions.csv")
        val_preds_path = os.path.join(args.input_dir, "val_predictions.csv")
        
        if all(os.path.exists(p) for p in [train_preds_path, val_preds_path]):
            train_preds = pd.read_csv(train_preds_path)
            val_preds = pd.read_csv(val_preds_path)
            print("Loaded prediction data for regression")
            
            # Plot model comparison
            if comparison_df is not None:
                plot_model_comparison(
                    comparison_df,
                    output_path=os.path.join(args.output_dir, "model_comparison.png")
                )
                print("Generated model comparison plot")
            
            # Generate individual model plots
            if not args.symbolic_only:
                for model in ["lasso", "rf", "svr"]:
                    if f"{model}_pred" in val_preds.columns:
                        plot_regression_scatter(
                            train_preds["y_train"],
                            train_preds[f"{model}_pred"],
                            val_preds["y_val"],
                            val_preds[f"{model}_pred"],
                            model_name=model.upper(),
                            output_path=os.path.join(args.output_dir, f"{model}_scatter.png")
                        )
                print("Generated standard model plots")
            
            # Generate symbolic model plots
            if "symbolic_pred" in val_preds.columns:
                plot_regression_scatter(
                    train_preds["y_train"],
                    train_preds["symbolic_pred"],
                    val_preds["y_val"],
                    val_preds["symbolic_pred"],
                    model_name="Symbolic",
                    output_path=os.path.join(args.output_dir, "symbolic_scatter.png")
                )
                
                # Complexity tradeoff plot (if data available)
                symbolic_eqs_path = os.path.join(args.input_dir, "symbolic_regression_all_equations.csv")
                if os.path.exists(symbolic_eqs_path):
                    equations_df = pd.read_csv(symbolic_eqs_path)
                    plot_symbolic_complexity_tradeoff(
                        equations_df,
                        metric='val_r2',
                        model_type='regression',
                        output_path=os.path.join(args.output_dir, "symbolic_regression_complexity_tradeoff.png"),
                        lower_is_better=False
                    )
                    print("Generated symbolic complexity tradeoff plot")
                else:
                    print(f"Warning: Symbolic equations file not found at {symbolic_eqs_path}")
                    
                print("Generated symbolic model plots")
            
            # Generate random/shuffle data analysis
            shuffle_random_path = os.path.join(args.input_dir, "shuffle_random_predictions.csv")
            if os.path.exists(shuffle_random_path):
                predictions_df = pd.read_csv(shuffle_random_path)
                plot_predictions_by_data_type(predictions_df, args.output_dir)
                print("Generated real vs. shuffle/random data analysis plots")
            else:
                print(f"Warning: Shuffled/random predictions file not found at {shuffle_random_path}")
        else:
            print("Error: Missing prediction data files")
    
    print("Plot generation complete")

if __name__ == "__main__":
    main()
