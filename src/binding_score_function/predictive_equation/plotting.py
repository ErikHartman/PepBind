# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve
import seaborn as sns
import pandas as pd
    
sns.set_context("paper")

def plot_classification_metrics(y_true, y_pred, y_proba, model_name, output_path=None):
    """
    Plot confusion matrix and ROC curve for classification results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1)
    ax1.set_xlabel('Predicted labels')
    ax1.set_ylabel('True labels')
    ax1.set_title(f'{model_name} Confusion Matrix')
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    ax2.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
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
    
    plt.title(f'{model_name}: Feature Importance')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_model_comparison(comparison_df, roc_data=None, output_path=None):
    model_names = comparison_df['Model'].values
    metrics = [col for col in comparison_df.columns if col != 'Model']
    

    if 'Train Accuracy' in metrics:
        metric_list = []
        for m in ['Test Accuracy', 'Test F1' 'Test AUC']:
            if m in metrics:
                metric_list.append(m)
        add_roc_subplot = roc_data is not None
        
        n_subplots = len(metric_list) + (1 if add_roc_subplot else 0)
        fig, axs = plt.subplots(1, n_subplots, figsize=(4 * n_subplots, 4), squeeze=False)
        axs = axs.flatten()
        
        # Plot bar charts for each metric.
        for i, metric in enumerate(metric_list):
            ax = axs[i]
            ax.bar(model_names, comparison_df[metric].values)
            ax.set_title(metric)
            ax.set_xlabel('Model')
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
        
        # Plot ROC curves in the last subplot if provided.
        if add_roc_subplot:
            ax = axs[-1]
            for model in model_names:
                if model in roc_data:
                    fpr, tpr, roc_auc = roc_data[model]
                    ax.plot(fpr, tpr, lw=2, label=f'{model} (AUC = {roc_auc:.3f})')
            ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curves')
            ax.legend(loc='lower right', fontsize='x-small', frameon=False)
        
        plt.tight_layout()

    # For regression metrics
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
    lower_is_better:bool=False,
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
    

    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(
        df['complexity'], 
        df[metric], 
        s=100
    )
    
    # Set axis labels
    ax.set_xlabel('Complexity')
    y_label = metric.replace('_', ' ').title()
    ax.set_ylabel(y_label)

    # Highlight Pareto front - the equations that are not dominated by others
    # Dominated means there's another equation with both better metric AND lower complexity
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
    ax.scatter(
        pareto_points['complexity'], 
        pareto_points[metric], 
        edgecolor='orange', 
        facecolor='none', 
        s=200, 
        linewidth=2,
        label='Pareto front'
    )
    
    # Add equation annotations for Pareto front points (or top few points by score)
    if show_equations:
        # If there are too many Pareto points, only annotate the top few by score
        if len(pareto_points) > max_equations:
            pareto_points = pareto_points.sort_values(by='score', ascending=False).head(max_equations)
        
        for i, row in pareto_points.iterrows():
            # Shorten equation if it's too long
            eq_str = row['equation']
            if len(eq_str) > 40:
                eq_str = eq_str[:36] + ' ...'
                
            ax.annotate(
                eq_str,
                xy=(row['complexity'], row[metric]),
                xytext=(10, 10),
                textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.5', fc='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0')
            )
    
    ax.set_title(f'Complexity vs. {y_label}')
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()