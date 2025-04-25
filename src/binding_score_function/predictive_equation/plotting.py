import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import pandas as pd
import os
import argparse
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import pearsonr

color_palette = {
    "symbolic": "#124E78", 
    "lasso": "#B388EB",    
    "rf": "#57A773",        
    "svr": "#2C8C99",   
    "logreg": "#B388EB", 
    "svc": "#2C8C99",     

    "random": "#F46036",
    "shuffled": "#E88873",  
    "real": "#2C8C99",  

    "train": "#2C8C99", 
    "test": "#3943B7", 

}
    
sns.set_context("paper")

def plot_classification_metrics(y_true, y_pred, y_proba, y_train=None, train_proba=None, model_name="Model", output_path=None):
    """
    Plot confusion matrix and ROC curve for classification results.
    If train data is provided, both train and test ROC curves will be shown.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8,4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title(f'{model_name} Confusion Matrix')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    test_auc = roc_auc_score(y_true, y_proba)
    
    # Plot test ROC curve
    ax2.plot(fpr, tpr, color=color_palette["test"], linestyle='--', lw=2, 
             label=f'Test ROC (AUC = {test_auc:.3f})')
    
    # Plot train ROC curve if provided
    if y_train is not None and train_proba is not None:
        fpr_train, tpr_train, _ = roc_curve(y_train, train_proba)
        train_auc = roc_auc_score(y_train, train_proba)
        ax2.plot(fpr_train, tpr_train, color=color_palette["train"], lw=2,
                 label=f'Train ROC (AUC = {train_auc:.3f})')
    
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name} ROC Curve')
    ax2.legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

def plot_regression_scatter(
    y_train, train_preds, 
    y_test=None, test_preds=None, 
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
    
    if y_test is not None and test_preds is not None:
        # Test data with regplot
        sns.regplot(
            x=y_test, 
            y=test_preds, 
            scatter_kws={'alpha': 0.5, 's': 10, 'color': color_palette["test"]}, 
            line_kws={'color': color_palette["test"]},
            label='Test',
            ax=ax1
        )
        y_all = np.concatenate([y_train, y_test])
    else:
        y_all = y_train
    
    # Perfect prediction line (diagonal)
    ax1.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--', alpha=0.5, label='Perfect prediction')
    
    ax1.set_xlabel('Actual pKd')
    ax1.set_ylabel('Predicted pKd')
    ax1.set_title(f'{model_name}: actual vs. predicted')
    ax1.legend(frameon=False)
    
    # Second subplot: Residuals (actual - predicted)
    train_residuals = y_train - train_preds
    
    # Plot training residuals
    sns.scatterplot(
        x=y_train, 
        y=train_residuals, 
        alpha=0.5, 
        s=5, 
        color=color_palette["train"],
        label='Train',
        ax=ax2
    )
    
    # Add a horizontal line at y=0 (perfect prediction)
    ax2.axhline(y=0, color='k', linestyle='--', alpha=0.5)
    
    # If test data is available, add test residuals
    if y_test is not None and test_preds is not None:
        test_residuals = y_test - test_preds
        sns.scatterplot(
            x=y_test, 
            y=test_residuals, 
            alpha=0.5, 
            s=10, 
            color=color_palette["test"],
            label='Test',
            ax=ax2
        )
    
    # Calculate and annotate RMSE for train and test
    train_rmse = np.sqrt(np.mean(train_residuals**2))
    rmse_text = f"Train RMSE: {train_rmse:.3f}"
    
    if y_test is not None and test_preds is not None:
        test_rmse = np.sqrt(np.mean((y_test - test_preds)**2))
        rmse_text += f"\nTest RMSE: {test_rmse:.3f}"
    
    # Add RMSE text to the residual plot
    ax2.text(
        0.05, 0.95, rmse_text,
        transform=ax2.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    ax2.set_xlabel('Actual pKd')
    ax2.set_ylabel('Residuals (Actual - Predicted)')
    ax2.set_title(f'{model_name}: prediction residuals')
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
        metric_list = ['Test Accuracy','Test AUC']
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
            ax.set_title(metric)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot ROC curves in the last subplot if provided
        if add_roc_subplot:
            ax = axs[-1]
            for model in model_names:
                if model in roc_data:
                    fpr, tpr, roc_auc = roc_data[model]
                    ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})', 
                            color=color_palette.get(model.lower(), 'gray'))
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc='lower right', frameon=False)
        
        plt.tight_layout()
    
    # If it's regression metrics
    elif 'Train RMSE' in metrics:
        regression_metrics = []
        for m in ['Test MAE', 'Test R²']:
            if m in metrics:
                regression_metrics.append(m)
        n_subplots = len(regression_metrics)
        fig, axs = plt.subplots(1, n_subplots, figsize=(3 * n_subplots, 3), squeeze=False)
        axs = axs.flatten()
        for i, metric in enumerate(regression_metrics):
            colors = [color_palette.get(x.lower(), 'gray') for x in model_names]
            ax = axs[i]
            ax.bar(model_names, comparison_df[metric].values, color=colors)
            ax.set_title(metric)
            ax.set_xlabel('Model')
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
    metric: str = 'test_rmse',
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
    train_metric = metric.replace('test_', 'train_')
    
    # Create figure with two subplots
    fig = plt.figure(figsize=(10, 7))
    
    # Create gridspec to have plots on top (larger) and equation legend below
    gs = fig.add_gridspec(2, 2, height_ratios=[2,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax_legend = fig.add_subplot(gs[1, :])
    
    # Plot complexity vs test metric
    scatter_test = ax1.scatter(
        df['complexity'], 
        df[metric], 
        s=100,
        color=color_palette["test"],
    )
    
    ax1.set_xlabel('Complexity')
    test_label = metric.replace('_', ' ').title()
    ax1.set_ylabel(test_label)
    ax1.set_title(f'Complexity vs. {test_label} (Test)')
    
    # Plot complexity vs train metric if available
    if train_metric in df.columns:
        scatter_train = ax2.scatter(
            df['complexity'], 
            df[train_metric], 
            s=100, 
            color=color_palette["train"],
        )
        
        ax2.set_xlabel('Complexity')
        train_label = train_metric.replace('_', ' ').title()
        ax2.set_title(f'Complexity vs. {train_label} (Train)')
        
        # Add horizontal line connecting the same equation in both plots
        for _, row in df.iterrows():
            if pd.notna(row[train_metric]) and pd.notna(row[metric]):
                x = [row['complexity'], row['complexity']]
                y_test = row[metric]
                y_train = row[train_metric]
                
                # Draw a light gray line connecting test to train for same equation
                plt.plot(
                    [row['complexity'], row['complexity'] + 0.001], 
                    [y_test, y_train],
                    color='lightgray', alpha=0.3, linestyle='-', linewidth=0.5
                )

    if lower_is_better:
        # For regression, lower metric is better (e.g., RMSE)
        is_pareto = np.ones(len(df), dtype=bool)
        for i, (c1, m1) in enumerate(zip(df['complexity'], df[metric])):
            # An equation is not on Pareto front if there exists another
            # with both lower complexity AND lower (=better) metric
            for c2, m2 in zip(df['complexity'], df[metric]):
                if (c2 < c1 and m2 <= m1) or (c2 <= c1 and m2 < m1):
                    is_pareto[i] = False
                    break
    else:
        # For classification, higher metric is better (e.g., AUC)
        is_pareto = np.ones(len(df), dtype=bool)
        for i, (c1, m1) in enumerate(zip(df['complexity'], df[metric])):
            # An equation is not on Pareto front if there exists another
            # with both lower complexity AND higher (=better) metric
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
                metric_str = f"{test_label}: {eq_metric:.4f}"
            else:
                metric_str = f"{test_label}: {eq_metric:.4f}"
                
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
    plt.figure(figsize=(8, 4))
    
    # Get probabilities for each class
    pos_probs = y_proba[y_true == 1]
    neg_probs = y_proba[y_true == 0]
    
    # Plot histograms
    plt.hist(
        neg_probs, bins=20, alpha=0.5, color=color_palette["random"], 
        label=f'Negative class (n={len(neg_probs)})', density=True
    )
    plt.hist(
        pos_probs, bins=20, alpha=0.5, color=color_palette["real"], 
        label=f'Positive class (n={len(pos_probs)})', density=True
    )
    
    # Add vertical line at decision threshold (0.5)
    plt.axvline(x=0.5, color='black', linestyle='--', alpha=0.7, label='Decision threshold')
    
    plt.xlabel('Predicted probability')
    plt.ylabel('Density')
    plt.title(f'{model_name}: Probability Distributions')
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
    Create plots comparing model predictions on different data types (real, shuffled, random).
    """
    # Add binary labels: 1 for real, 0 for non-real (shuffled or random)
    predictions_df['binary_label'] = predictions_df['data_type'].apply(lambda x: 1 if x == 'Real' else 0)
    
    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[2, 1])
    
    # Stripplot in top-left position
    ax1 = fig.add_subplot(gs[0, 0])
    sns.stripplot(
        data=predictions_df, 
        x="model", 
        y="prediction", 
        hue="data_type",
        palette={"Real": color_palette["real"], "Shuffled": color_palette["shuffled"], "Random": color_palette["random"]},
        dodge=True,
        ax=ax1
    )
    
    ax1.set_ylim([3, 10])
    ax1.set_title("Model Predictions by Data Type")
    ax1.set_xlabel("Model")
    ax1.set_ylabel("Predicted pKd")
    ax1.legend(frameon=False)
    
    ax2 = fig.add_subplot(gs[1, 0])
    sns.boxplot(
        data=predictions_df, 
        x="model", 
        y="prediction", 
        hue="data_type",
        palette={"Real": color_palette["real"], "Shuffled": color_palette["shuffled"], "Random": color_palette["random"]},
        ax=ax2
    )
    
    ax2.set_ylim([3, 10])
    ax2.set_title("Distribution of Predictions by Data Type")
    ax2.set_xlabel("Model")
    ax2.set_ylabel("Predicted pKd")
    ax2.legend(frameon=False)
    
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
    
    ax3.set_xlim([0.0, 1.0])
    ax3.set_ylim([0.0, 1.05])
    ax3.set_xlabel('False Positive Rate')
    ax3.set_ylabel('True Positive Rate')
    ax3.set_title('Real vs decoy data')
    ax3.legend(loc='lower right', frameon=False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "real_vs_shuffled_random_analysis.png"), dpi=300)
    plt.close()
    
    return {
        "combined_plot": os.path.join(output_dir, "real_vs_shuffled_random_analysis.png")
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
    
    Parameters:
    -----------
    pkd_values : numpy.ndarray
        The pKd values for real test samples
    probabilities : dict of numpy.ndarray
        Dictionary mapping model names to their predicted probabilities for test samples
    model_names : list
        List of model names to include in the plot
    output_path : str, optional
        Path to save the plot
    pkd_values_train : numpy.ndarray, optional
        The pKd values for real training samples
    probabilities_train : dict of numpy.ndarray, optional
        Dictionary mapping model names to their predicted probabilities for train samples
    """
    if len(model_names) == 0:
        print("No models to plot correlations for.")
        return
    
    # Calculate number of rows and columns for subplots
    n_models = len(model_names)
    n_cols = min(2, n_models)
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    
    # Make axes iterable even if there's only one subplot
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, model_name in enumerate(model_names):
        if model_name in probabilities:
            ax = axes[i]
            
            # Plot test data
            sns.scatterplot(
                x=pkd_values, 
                y=probabilities[model_name], 
                ax=ax,
                alpha=0.7,
                color=color_palette.get("test", 'blue'),
                label='Test',
                s=40,
                marker='o'
            )
            
            # Plot train data if provided
            has_train_data = (pkd_values_train is not None and 
                             probabilities_train is not None and
                             model_name in probabilities_train and
                             len(pkd_values_train) == len(probabilities_train[model_name]))
            
            if has_train_data:
                sns.scatterplot(
                    x=pkd_values_train, 
                    y=probabilities_train[model_name], 
                    ax=ax,
                    alpha=0.5,
                    color=color_palette.get("train", 'green'),
                    label='Train',
                    s=25,
                    marker='x'
                )
                
                # Calculate combined correlation if both train and test data exist
                combined_pkd = np.concatenate([pkd_values, pkd_values_train])
                combined_probs = np.concatenate([probabilities[model_name], probabilities_train[model_name]])
                
                # Add regression line for combined data
                sns.regplot(
                    x=combined_pkd, 
                    y=combined_probs, 
                    ax=ax,
                    scatter=False,
                    color='red',
                    line_kws={'linestyle':'-'}
                )
                
                # Calculate correlation coefficients for combined data
                pearson_r, _ = pearsonr(combined_pkd, combined_probs)
                
                # Add correlation info to plot (combined)
                correlation_text = (
                    f"Combined Pearson r: {pearson_r:.3f}\n"
                )
            else:
                # Add regression line for just test data
                sns.regplot(
                    x=pkd_values, 
                    y=probabilities[model_name], 
                    ax=ax,
                    scatter=False,
                    color='red'
                )
                
                # Calculate correlation coefficients for test data only
                pearson_r, _ = pearsonr(pkd_values, probabilities[model_name])
                
                # Add correlation info to plot (test only)
                correlation_text = (
                    f"Test Pearson r: {pearson_r:.3f}\n"
                )
            
            # Add separate correlations for train and test if both exist
            if has_train_data:
                # Calculate test-only correlations
                test_pearson_r, _ = pearsonr(pkd_values, probabilities[model_name])
            
                # Calculate train-only correlations
                train_pearson_r, _ = pearsonr(pkd_values_train, probabilities_train[model_name])
                
                # Add detailed correlation info
                correlation_text += (
                    f"\n\nTest-only Pearson r: {test_pearson_r:.3f}\n"
                    f"Train-only Pearson r: {train_pearson_r:.3f}"
                )
            
            # Display the correlation text
            ax.text(
                0.05, 0.95, correlation_text,
                transform=ax.transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
            )
            
            ax.set_xlabel('pKd value')
            ax.set_ylabel(f'{model_name} probability')
            ax.set_title(f'{model_name}: pKd vs Probability Correlation')
            ax.legend()

    # Hide any unused subplots
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
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Common functionality: Load model comparison data
    comparison_path = os.path.join(args.input_dir, "model_comparison.csv")
    if os.path.exists(comparison_path):
        comparison_df = pd.read_csv(comparison_path)
        print(f"Loaded model comparison from {comparison_path}")
    else:
        print(f"Warning: Model comparison file not found at {comparison_path}")
        comparison_df = None
    
    if args.mode == "classification":
        # Load predictions and probabilities
        train_preds_path = os.path.join(args.input_dir, "train_predictions.csv")
        test_preds_path = os.path.join(args.input_dir, "test_predictions.csv")
        train_proba_path = os.path.join(args.input_dir, "train_probabilities.csv")
        test_proba_path = os.path.join(args.input_dir, "test_probabilities.csv")
        
        if all(os.path.exists(p) for p in [train_preds_path, test_preds_path, train_proba_path, test_proba_path]):
            train_preds = pd.read_csv(train_preds_path)
            test_preds = pd.read_csv(test_preds_path)
            train_proba = pd.read_csv(train_proba_path)
            test_proba = pd.read_csv(test_proba_path)
            print("Loaded prediction data for classification")
            
            # Calculate ROC curves for model comparison
            if comparison_df is not None:
                roc_data = {}
                for model in ["LogReg", "RF", "SVC", "Symbolic"]:
                    if f"{model.lower()}_proba" in test_proba.columns:
                        fpr, tpr, _ = roc_curve(test_proba["y_test"], test_proba[f"{model.lower()}_proba"])
                        roc_auc = roc_auc_score(test_proba["y_test"], test_proba[f"{model.lower()}_proba"])
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
                    if f"{model}_proba" in test_proba.columns:
                        # Classification metrics plot
                        plot_classification_metrics(
                            test_preds["y_test"],
                            test_preds[f"{model}_pred"],
                            test_proba[f"{model}_proba"],
                            train_preds["y_train"],
                            train_proba[f"{model}_proba"],
                            model_name=model.upper(),
                            output_path=os.path.join(args.output_dir, f"{model}_metrics.png")
                        )
                        
                        # Probability histograms
                        plot_probability_histograms(
                            test_preds["y_test"],
                            test_proba[f"{model}_proba"],
                            model_name=model.upper(),
                            output_path=os.path.join(args.output_dir, f"{model}_proba_hist.png")
                        )
                print("Generated standard model plots")
            
            # Generate correlation plots between pKd and probabilities if available
            pkd_test_path = os.path.join(args.input_dir, "real_test_pkd.csv")
            pkd_train_path = os.path.join(args.input_dir, "real_train_pkd.csv")
            
            if os.path.exists(pkd_test_path):
                pkd_test_df = pd.read_csv(pkd_test_path)
                
                # Check for train data
                pkd_train_df = None
                if os.path.exists(pkd_train_path):
                    pkd_train_df = pd.read_csv(pkd_train_path)
                
                if not pkd_test_df.empty and "pKd" in pkd_test_df.columns:
                    print("Generating pKd-probability correlation plots")
                    
                    # Get model names
                    model_names = []
                    test_proba_dict = {}
                    train_proba_dict = {}
                    
                    for model in ["logreg", "rf", "svc", "symbolic"]:
                        if f"{model}_proba" in test_proba.columns:
                            # Filter to only real test samples
                            real_test_indices = test_proba["y_test"] == 1
                            if real_test_indices.sum() > 0:
                                model_upper = model.upper()
                                model_names.append(model_upper)
                                test_proba_dict[model_upper] = test_proba.loc[real_test_indices, f"{model}_proba"].values
                                
                                # Get train probabilities if available
                                if pkd_train_df is not None and not pkd_train_df.empty and "pKd" in pkd_train_df.columns:
                                    real_train_indices = train_proba["y_train"] == 1
                                    if real_train_indices.sum() > 0:
                                        train_proba_dict[model_upper] = train_proba.loc[real_train_indices, f"{model}_proba"].values
                    
                    if model_names:
                        plot_pkd_probability_correlation(
                            pkd_test_df["pKd"].values,
                            test_proba_dict,
                            model_names,
                            output_path=os.path.join(args.output_dir, "pkd_probability_correlation.png"),
                            pkd_values_train=pkd_train_df["pKd"].values if pkd_train_df is not None else None,
                            probabilities_train=train_proba_dict if train_proba_dict else None
                        )
                        print("Generated pKd-probability correlation plot")
            
            # Generate symbolic model plots
            if "symbolic_proba" in test_proba.columns:
                # Classification metrics plot
                plot_classification_metrics(
                    test_preds["y_test"],
                    test_preds["symbolic_pred"],
                    test_proba["symbolic_proba"],
                    train_preds["y_train"],
                    train_proba["symbolic_proba"],
                    model_name="Symbolic",
                    output_path=os.path.join(args.output_dir, "symbolic_metrics.png")
                )
                
                # Probability histograms
                plot_probability_histograms(
                    test_preds["y_test"],
                    test_proba["symbolic_proba"],
                    model_name="Symbolic",
                    output_path=os.path.join(args.output_dir, "symbolic_proba_hist.png")
                )
                
                # Complexity tradeoff plot (if data available)
                symbolic_eqs_path = os.path.join(args.input_dir, "symbolic_classification_all_equations.csv")
                if os.path.exists(symbolic_eqs_path):
                    equations_df = pd.read_csv(symbolic_eqs_path)
                    plot_symbolic_complexity_tradeoff(
                        equations_df,
                        metric='test_auc',
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
        test_preds_path = os.path.join(args.input_dir, "test_predictions.csv")
        
        if all(os.path.exists(p) for p in [train_preds_path, test_preds_path]):
            train_preds = pd.read_csv(train_preds_path)
            test_preds = pd.read_csv(test_preds_path)
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
                    if f"{model}_pred" in test_preds.columns:
                        plot_regression_scatter(
                            train_preds["y_train"],
                            train_preds[f"{model}_pred"],
                            test_preds["y_test"],
                            test_preds[f"{model}_pred"],
                            model_name=model.upper(),
                            output_path=os.path.join(args.output_dir, f"{model}_scatter.png")
                        )
                print("Generated standard model plots")
            
            # Generate symbolic model plots
            if "symbolic_pred" in test_preds.columns:
                plot_regression_scatter(
                    train_preds["y_train"],
                    train_preds["symbolic_pred"],
                    test_preds["y_test"],
                    test_preds["symbolic_pred"],
                    model_name="Symbolic",
                    output_path=os.path.join(args.output_dir, "symbolic_scatter.png")
                )
                
                # Complexity tradeoff plot (if data available)
                symbolic_eqs_path = os.path.join(args.input_dir, "symbolic_regression_all_equations.csv")
                if os.path.exists(symbolic_eqs_path):
                    equations_df = pd.read_csv(symbolic_eqs_path)
                    plot_symbolic_complexity_tradeoff(
                        equations_df,
                        metric='test_r2',
                        model_type='regression',
                        output_path=os.path.join(args.output_dir, "symbolic_regression_complexity_tradeoff.png"),
                        lower_is_better=False
                    )
                    print("Generated symbolic complexity tradeoff plot")
                else:
                    print(f"Warning: Symbolic equations file not found at {symbolic_eqs_path}")
                    
                print("Generated symbolic model plots")
            
            # Generate random/shuffled data analysis
            shuffled_random_path = os.path.join(args.input_dir, "shuffled_random_predictions.csv")
            if os.path.exists(shuffled_random_path):
                predictions_df = pd.read_csv(shuffled_random_path)
                plot_predictions_by_data_type(predictions_df, args.output_dir)
                print("Generated real vs. shuffled/random data analysis plots")
            else:
                print(f"Warning: Shuffled/random predictions file not found at {shuffled_random_path}")
        else:
            print("Error: Missing prediction data files")
    
    print("Plot generation complete")

if __name__ == "__main__":
    main()
