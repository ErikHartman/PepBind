from sklearn.linear_model import Lasso, LassoCV
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.gridspec as gridspec
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")
scores_path = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/2_scored/benchmark_scores.csv")

scores_df = pd.read_csv(scores_path)
affinity_df = pd.read_csv(os.path.join(DATA_DIR, "databases/benchmark_data/new_run/1_preprocessed/pdbs.csv"))

scores_df['PDB code'] = scores_df['pdb_file'].str.replace('.pdb', '', regex=False)
scores_df = scores_df.merge(affinity_df, on='PDB code', how='inner')
scores_df = scores_df[~scores_df['Binding data'].str.contains('IC50', na=False)]
scores_df['Binding data'] = pd.to_numeric(scores_df['Binding data'].str.extract(r'=(\d+\.?\d*)')[0], errors='coerce')

print(scores_df.columns.tolist())

def preprocess_data(df):
    target_column = "Binding data"
    non_feature_cols = [col for col in ['pdb_file', 'peptide_sequence', target_column] if col in df.columns]
    feature_cols = [col for col in df.columns if col not in non_feature_cols 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    y = df[target_column].values
    X = df[feature_cols]
    
    if X.isnull().any().any() or np.isnan(y).any():
        y_series = pd.Series(y, index=X.index)
        complete_indices = ~X.isnull().any(axis=1) & ~y_series.isna()
        X = X.loc[complete_indices]
        y = y_series.loc[complete_indices].values
    
    return X, y, target_column

def train_lasso_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    alphas = np.logspace(-6, 2, 100)
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000, alphas=alphas)
    lasso_cv.fit(X_train, y_train)
    
    best_alpha = lasso_cv.alpha_
    lasso = Lasso(alpha=best_alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)
    
    return {
        'model': lasso,
        'best_alpha': best_alpha,
        'feature_names': X.columns,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

def create_feature_importance_plot(results, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    
    coefs = results['model'].coef_
    feature_names = results['feature_names']
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': coefs})
    
    coef_df['Abs_Coefficient'] = np.abs(coef_df['Coefficient'])
    coef_df = coef_df.sort_values('Abs_Coefficient', ascending=False)
    coef_df = coef_df[coef_df['Coefficient'] != 0]
    
    colors = ['red' if c < 0 else 'blue' for c in coef_df['Coefficient']]
    
    ax.bar(range(len(coef_df)), coef_df['Coefficient'], color=colors)
    ax.set_xticks(range(len(coef_df)))
    ax.set_xticklabels(coef_df['Feature'], rotation=90)
    ax.set_title('Feature Importance (Lasso Coefficients)')
    ax.set_ylabel('Coefficient Value')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    ax.bar(0, 0, color='blue', label='Positive Impact')
    ax.bar(0, 0, color='red', label='Negative Impact')
    ax.legend()
    
    return ax

def create_prediction_plot(results, target_name, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))
    
    y_actual = np.concatenate([results['y_train'], results['y_test']])
    y_predicted = np.concatenate([results['y_pred_train'], results['y_pred_test']])
    
    ax.scatter(y_actual, y_predicted, alpha=0.5)
    
    min_val = min(np.min(y_actual), np.min(y_predicted))
    max_val = max(np.max(y_actual), np.max(y_predicted))
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2)
    ax.set_xlabel(f'Actual {target_name}')
    ax.set_ylabel(f'Predicted {target_name}')
    ax.set_title(f'Actual vs. Predicted {target_name}')
    
    ax.annotate(f"Train R²: {results['train_r2']:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
    ax.annotate(f"Test R²: {results['test_r2']:.3f}", xy=(0.05, 0.90), xycoords='axes fraction')
    
    return ax

def create_residual_plot(results, target_name, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 6))
    
    y_actual = np.concatenate([results['y_train'], results['y_test']])
    y_predicted = np.concatenate([results['y_pred_train'], results['y_pred_test']])
    residuals = y_actual - y_predicted
    
    ax.scatter(y_predicted, residuals, alpha=0.5)
    ax.axhline(y=0, color='red', linestyle='--', lw=2)
    
    ax.set_xlabel(f'Predicted {target_name}')
    ax.set_ylabel('Residuals')
    ax.set_title('Residual Plot')
    
    ax.annotate(f"Train RMSE: {results['train_rmse']:.3f}", xy=(0.05, 0.95), xycoords='axes fraction')
    ax.annotate(f"Test RMSE: {results['test_rmse']:.3f}", xy=(0.05, 0.90), xycoords='axes fraction')
    
    return ax

def create_correlation_heatmap(X, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))
    
    corr_matrix = X.corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    
    sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', vmin=-1, vmax=1, 
                center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title('Feature Correlation Heatmap')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    return ax

def create_multi_plot(results, X, target_name):
    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1.2, 1])
    
    create_feature_importance_plot(results, ax=plt.subplot(gs[0, 0]))
    create_prediction_plot(results, target_name, ax=plt.subplot(gs[0, 1]))
    create_residual_plot(results, target_name, ax=plt.subplot(gs[1, 0]))
    create_correlation_heatmap(X, ax=plt.subplot(gs[1, 1]))
    
    plt.tight_layout()
    return fig

def main():
    print(f"Data loaded. Shape: {scores_df.shape}")
    print(f"Columns: {scores_df.columns.tolist()}")
    
    X, y, target_name = preprocess_data(scores_df)
    print(f"Features shape: {X.shape}, Target shape: {y.shape}")
    
    results = train_lasso_model(X, y)
    print(f"Best alpha: {results['best_alpha']:.6f}")
    print(f"Train R²: {results['train_r2']:.3f}, Test R²: {results['test_r2']:.3f}")
    
    fig = create_multi_plot(results, X, target_name)
    
    output_dir = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/analysis")
    os.makedirs(output_dir, exist_ok=True)
    fig_path = os.path.join(output_dir, "lasso_feature_importance.png")
    fig.savefig(fig_path, dpi=300, bbox_inches='tight')
    
    print(f"Analysis complete. Plot saved to {fig_path}")
    
    coefs = results['model'].coef_
    feature_names = results['feature_names']
    feature_importance = sorted(zip(feature_names, coefs), key=lambda x: -abs(x[1]))
    
    print("\nTop 10 most important features:")
    for i, (feature, coef) in enumerate(feature_importance[:10]):
        if coef != 0:  
            print(f"{i+1}. {feature}: {coef:.6f}")

if __name__ == "__main__":
    main()
