from adalasso import AdaptiveLasso
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import dotenv

dotenv.load_dotenv()
DATA_DIR = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")
scores_path = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/2_scored/benchmark_scores.csv")

scores_df = pd.read_csv(scores_path)
affinity_df = pd.read_csv(os.path.join(DATA_DIR, "databases/benchmark_data/new_run/0_unprocessed/pdbs.csv"))

def preprocess_data(scores_df, affinity_df):
    scores_df['PDB code'] = scores_df['pdb_file'].str.replace('.pdb', '', regex=False)
    scores_df = scores_df.merge(affinity_df, on='PDB code', how='inner')
    scores_df = scores_df[~scores_df['Binding data'].str.contains('IC50', na=False)]

    binding_data = scores_df['Binding data'].str.extract(r'([=<>]?)(\d+\.?\d*)([a-zA-Z]*)')
    binding_data[1] = pd.to_numeric(binding_data[1], errors='coerce')
    unit_conversion = {'pM': 1e-12, 'nM': 1e-9, 'uM': 1e-6, 'mM': 1e-3, 'M': 1, 'fM': 1e-15}
    scores_df['Binding data'] = binding_data[1] * binding_data[2].map(unit_conversion)

    target_column = "Binding data"
    non_feature_cols = ['pdb_file', 'peptide_sequence', 'Release year', target_column]
    feature_cols = scores_df.select_dtypes(include=[np.number]).columns.difference(non_feature_cols).tolist()

    y = -np.log10(scores_df[target_column].values)
    X = scores_df[feature_cols]
    if X.isnull().any().any() or np.isnan(y).any():
        complete_indices = ~X.isnull().any(axis=1) & ~pd.Series(y, index=X.index).isna()
        X = X.loc[complete_indices]
        y = y[complete_indices]

    return X, y

def train_lasso_model(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    

    lasso = AdaptiveLasso(alpha=0.1, gamma=1.0)
    lasso.fit(X_train, y_train)
    
    
    y_pred_train = lasso.predict(X_train)
    y_pred_test = lasso.predict(X_test)
    
    return {
        'model': lasso,
        'feature_names': X.columns,
        'X_train': X_train, 'X_test': X_test,
        'y_train': y_train, 'y_test': y_test,
        'y_pred_train': y_pred_train, 'y_pred_test': y_pred_test,
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
        'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
    }

def main():
    X, y = preprocess_data(scores_df, affinity_df)
    model_results = train_lasso_model(X, y)
    
    print(f"Train R^2: {model_results['train_r2']}")
    print(f"Test R^2: {model_results['test_r2']}")
    print(f"Train RMSE: {model_results['train_rmse']}")
    print(f"Test RMSE: {model_results['test_rmse']}")
    
    feature_importances = pd.Series(model_results['model'].coef_, index=model_results['feature_names'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Lasso Regression Analysis', fontsize=16)

    sns.barplot(x=feature_importances.values, y=feature_importances.index, ax=axes[0, 0])
    axes[0, 0].set_title('Feature Importances')
    axes[0, 0].set_xlabel('Coefficient Value')
    axes[0, 0].set_ylabel('Feature')

    sns.scatterplot(x=model_results['y_test'], y=model_results['y_pred_test'], ax=axes[0, 1])
    axes[0, 1].plot([model_results['y_test'].min(), model_results['y_test'].max()],
                    [model_results['y_test'].min(), model_results['y_test'].max()], 'k--')
    axes[0, 1].set_title('True vs Predicted Values')
    axes[0, 1].set_xlabel('True Values')
    axes[0, 1].set_ylabel('Predicted Values')

    sns.histplot(model_results['y_test'] - model_results['y_pred_test'], bins=30, kde=True, ax=axes[1, 0])
    axes[1, 0].set_title('Residuals Distribution')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')

    corr_matrix = pd.DataFrame(model_results['X_train'], columns=model_results['feature_names']).corr()
    sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=axes[1, 1])
    axes[1, 1].set_title('Feature Correlation Matrix')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_plot_path = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/2_scored/lasso_analysis.png")
    plt.savefig(output_plot_path)
    print(f"Plot saved to {output_plot_path}")

if __name__ == "__main__":
    main()
