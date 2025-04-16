# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve, roc_auc_score
import seaborn as sns
import pandas as pd
    
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
    ax2.plot(fpr, tpr, color='red', linestyle='--', lw=2, 
             label=f'Test ROC (AUC = {test_auc:.3f})')
    
    # Plot train ROC curve if provided
    if y_train is not None and train_proba is not None:
        fpr_train, tpr_train, _ = roc_curve(y_train, train_proba)
        train_auc = roc_auc_score(y_train, train_proba)
        ax2.plot(fpr_train, tpr_train, color='blue', lw=2,
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
    Creates and saves a regplot (scatter with regression line) of actual vs. predicted.
    """
    plt.figure(figsize=(4,4))
    
    # Train data with regplot
    sns.regplot(
        x=y_train, 
        y=train_preds, 
        scatter_kws={'alpha': 0.5, 's': 5, 'color': 'blue'}, 
        line_kws={'color': 'blue'},
        label='Train'
    )
    
    if y_test is not None and test_preds is not None:
        # Test data with regplot
        sns.regplot(
            x=y_test, 
            y=test_preds, 
            scatter_kws={'alpha': 0.5, 's': 10, 'color': 'red'}, 
            line_kws={'color': 'red', 'linestyle': '--'},
            label='Test'
        )
        y_all = np.concatenate([y_train, y_test])
    else:
        y_all = y_train
    
    # Perfect prediction line (diagonal)
    plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--', alpha=0.5, label='Perfect prediction')
    
    plt.xlabel('Actual pKd')
    plt.ylabel('Predicted pKd')
    plt.title(f'{model_name}: Actual vs. predicted')
    plt.legend(frameon=False)
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
        metric_list = ['Test Accuracy', 'Test F1', 'Test AUC']
        metric_list = [m for m in metric_list if m in metrics]
        
        # Determine if we should add ROC curve subplot
        add_roc_subplot = roc_data is not None
        
        n_subplots = len(metric_list) + (1 if add_roc_subplot else 0)
        fig, axs = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 3), squeeze=False)
        axs = axs.flatten()  # easier to iterate
        
        # Plot bar charts for each metric
        for i, metric in enumerate(metric_list):
            ax = axs[i]
            ax.bar(model_names, comparison_df[metric].values, color='skyblue')
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
                    ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})')
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
        for m in ['Test RMSE', 'Test MAE', 'Test R²']:
            if m in metrics:
                regression_metrics.append(m)
        n_subplots = len(regression_metrics)
        fig, axs = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 3), squeeze=False)
        axs = axs.flatten()
        for i, metric in enumerate(regression_metrics):
            ax = axs[i]
            ax.bar(model_names, comparison_df[metric].values)
            ax.set_title(metric)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
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
    fig = plt.figure(figsize=(15, 9))
    
    # Create gridspec to have plots on top (larger) and equation legend below
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1], sharey=ax1)
    ax_legend = fig.add_subplot(gs[1, :])
    
    # Plot complexity vs test metric
    scatter_test = ax1.scatter(
        df['complexity'], 
        df[metric], 
        s=100
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
            color='orange'
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
        
        # Title for the equation legend
        ax_legend.text(0.5, 0.9, 'Top Equations', 
                      ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Add numbered annotations to plot and corresponding equations to legend
        for i, (idx, row) in enumerate(pareto_points.iterrows(), 1):
            # Add number to the point
            ax1.annotate(
                str(i),
                xy=(row['complexity'], row[metric]),
                xytext=(5, 5),
                textcoords='offset points',
                bbox=dict(boxstyle='circle', fc='white', ec='red'),
                fontsize=9,
                fontweight='bold'
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
                         va='center', fontsize=9, wrap=True, 
                         bbox=dict(boxstyle='round', fc='lightyellow', alpha=0.5))
    
    # If there's a big gap between train and test metrics, add a note
    if train_metric in df.columns:
        avg_test = df[metric].mean()
        avg_train = df[train_metric].mean()
        
        # For regression (RMSE), lower is better
        if 'rmse' in metric.lower() and avg_train < avg_test:
            gap_ratio = avg_test / avg_train if avg_train > 0 else 0
            if gap_ratio > 1.5:  # If test error is 50% higher than train
                plt.figtext(0.5, 0.01, 
                    f"Potential overfitting: Avg train {train_metric}={avg_train:.3f}, " +
                    f"Avg test {metric}={avg_test:.3f} ({gap_ratio:.1f}x difference)",
                    ha="center", bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
        
        # For classification metrics (AUC, Accuracy), higher is better
        elif ('auc' in metric.lower() or 'acc' in metric.lower()) and avg_train > avg_test:
            gap_ratio = avg_train / avg_test if avg_test > 0 else 0
            if gap_ratio > 1.2:  # If train accuracy is 20% higher than test
                plt.figtext(0.5, 0.01, 
                    f"Potential overfitting: Avg train {train_metric}={avg_train:.3f}, " +
                    f"Avg test {metric}={avg_test:.3f} ({gap_ratio:.1f}x difference)",
                    ha="center", bbox={"facecolor":"orange", "alpha":0.2, "pad":5})
    
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.15)  # Make room for the overfitting note
    
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
        neg_probs, bins=20, alpha=0.5, color='red', 
        label=f'Negative class (n={len(neg_probs)})', density=True
    )
    plt.hist(
        pos_probs, bins=20, alpha=0.5, color='blue', 
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