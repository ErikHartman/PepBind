# plotting.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import auc, confusion_matrix, roc_curve
import seaborn as sns

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
    ax2.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title(f'{model_name} ROC Curve')
    ax2.legend(loc='lower right')
    
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
    Creates and saves a scatter plot of actual vs. predicted.
    """
    plt.figure(figsize=(4,4))
    plt.scatter(y_train, train_preds, alpha=0.5, label='Train', s=5)
    
    if y_test is not None and test_preds is not None:
        plt.scatter(y_test, test_preds, alpha=0.5, label='Test', s=10)
        y_all = np.concatenate([y_train, y_test])
    else:
        y_all = y_train
    
    plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--')
    plt.xlabel('Actual pKd')
    plt.ylabel('Predicted pKd')
    plt.title(f'{model_name}: Actual vs. Predicted')
    plt.legend()
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()

def plot_feature_importances(feature_df, model_name="Model", output_path=None):
    """
    Creates and saves a barplot of feature importances or coefficients.
    """
    plt.figure(figsize=(6,5))
    # feature_df may have ['Feature', 'Importance'] or ['Feature', 'Coefficient']
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

def plot_model_comparison(comparison_df, output_path=None):
    """
    Creates a bar plot or grouped bar plot for comparing multiple models.
    """
    model_names = comparison_df['Model'].values
    x = np.arange(len(model_names))
    width = 0.2
    
    # Check which metrics are available in the dataframe
    metrics = [col for col in comparison_df.columns if col != 'Model']
    
    # If it's classification metrics
    if 'Train Accuracy' in metrics:
        plt.figure(figsize=(10,6))
        
        # Plot available metrics
        metrics_to_plot = ['Train Accuracy', 'Test Accuracy', 'Test F1', 'Test AUC']
        available_metrics = [m for m in metrics_to_plot if m in metrics]
        
        for i, metric in enumerate(available_metrics):
            plt.bar(x + (i - len(available_metrics)/2 + 0.5) * width, 
                   comparison_df[metric], width, label=metric)
        
        plt.xticks(x, model_names, rotation=45)
        plt.xlabel('Model')
        plt.ylabel('Score')
        plt.title('Classification Model Performance')
        plt.legend(loc='lower right')
        plt.tight_layout()
    
    # If it's regression metrics
    elif 'Train RMSE' in metrics:
        plt.figure(figsize=(8,5))
        plt.bar(x - width*1.5, comparison_df['Train RMSE'], width, label='Train RMSE')
        plt.bar(x - width/2, comparison_df['Test RMSE'], width, label='Test RMSE')
        plt.bar(x + width/2, comparison_df['Test MAE'], width, label='Test MAE')
        plt.bar(x + width*1.5, comparison_df['Test R²'], width, label='Test R²')
        
        plt.xticks(x, model_names, rotation=45)
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.title('Regression Model Performance')
        plt.legend()
        plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()
