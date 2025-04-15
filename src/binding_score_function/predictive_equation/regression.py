# regression.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List
import seaborn as sns
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from pysr import PySRRegressor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame = None):
    """
    Scale features using StandardScaler.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def train_lasso(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    n_alphas: int = 100,
    output_dir: str = None
) -> Dict:
    """
    Train a Lasso regression model with automatic alpha selection.
    """
    logger.info("Training Lasso regression model...")
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    alphas = np.logspace(-6, 2, n_alphas)
    lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train)
    
    best_alpha = lasso_cv.alpha_
    logger.info(f"Best alpha value: {best_alpha:.6f}")
    
    lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    feature_names = X_train.columns
    coef_df = pd.DataFrame({'Feature': feature_names, 'Coefficient': lasso.coef_})
    coef_df = coef_df[coef_df['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False)
    
    train_pred = lasso.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': lasso,
        'scaler': scaler,
        'best_alpha': best_alpha,
        'coefficients': coef_df,
        'train_pred': train_pred,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'feature_importance': pd.Series(lasso.coef_, index=feature_names),
    }
    
    if X_test is not None and y_test is not None:
        test_pred = lasso.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        results.update({
            'test_pred': test_pred,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        
        logger.info(f"Lasso Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
                    f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        coef_df.to_csv(os.path.join(output_dir, 'lasso_coefficients.csv'), index=False)
        
        plt.figure(figsize=(6, 5))
        plt.barh(coef_df['Feature'].values, coef_df['Coefficient'].values)
        plt.xlabel('Coefficient Value')
        plt.title('Lasso Regression Coefficients')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lasso_coefficients.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(4,4))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Train', s=5)
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange', s=10)
        y_all = np.concatenate([y_train, y_test]) if y_test is not None else y_train
        plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('Lasso: Actual vs. Predicted')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lasso_prediction.png'), dpi=300)
        plt.close()
    
    return results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
    output_dir: str = None
) -> Dict:
    """
    Train Random Forest Regressor with GridSearchCV.
    """
    logger.info("Training Random Forest regressor...")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    logger.info(f"RF best parameters: {grid_search.best_params_}")
    
    train_pred = best_rf.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    feature_names = X_train.columns
    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    
    results = {
        'model': best_rf,
        'best_params': grid_search.best_params_,
        'feature_importance': feature_importance,
        'train_pred': train_pred,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
    }
    
    if X_test is not None and y_test is not None:
        test_pred = best_rf.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        results.update({
            'test_pred': test_pred,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        logger.info(f"RF Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
                    f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        feature_importance.to_csv(os.path.join(output_dir, 'rf_feature_importance.csv'), index=False)

        plt.figure(figsize=(6,5))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Random Forest Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), dpi=300)
        plt.close()
        
        plt.figure(figsize=(4,4))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Train', s=5)
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange', s=10)
        y_all = np.concatenate([y_train, y_test]) if y_test is not None else y_train
        plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('Random Forest: Actual vs. Predicted')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_prediction.png'), dpi=300)
        plt.close()
    
    return results


def train_svr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
    output_dir: str = None
) -> Dict:
    """
    Train SVR with GridSearchCV.
    """
    logger.info("Training SVR model...")
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf']
        }
    
    svr = SVR()
    grid_search = GridSearchCV(
        svr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_svr = grid_search.best_estimator_
    logger.info(f"SVR best params: {grid_search.best_params_}")

    train_pred = best_svr.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': best_svr,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'train_pred': train_pred,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
    }
    
    if X_test is not None and y_test is not None:
        test_pred = best_svr.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        results.update({
            'test_pred': test_pred,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        logger.info(f"SVR Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
                    f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

        plt.figure(figsize=(4,4))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Train', s=5)
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange', s=10)
        y_all = np.concatenate([y_train, y_test]) if y_test is not None else y_train
        plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('SVR: Actual vs. Predicted')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'svr_prediction.png'), dpi=300)
        plt.close()
    
    return results


def perform_symbolic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    niterations: int = 200,
    populations: int = 50,
    population_size: int = 100,
    maxsize: int = 40,
    maxdepth: int = 10,
    binary_operators: List[str] = None,
    unary_operators: List[str] = None,
    output_dir: str = None
) -> Dict:
    """
    Perform symbolic regression to find mathematical expressions.
    """
    logger.info("Performing Symbolic Regression...")

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["square", "log", "sqrt"]

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    model = PySRRegressor(
        model_selection="best",
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=populations,
        population_size=population_size,
        maxsize=maxsize,
        maxdepth=maxdepth,
        verbosity=1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train, variable_names=list(X_train.columns))
    
    best_expr = model.sympy()
    logger.info(f"Best expression: {best_expr}")
    
    train_pred = model.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': model,
        'scaler': scaler,
        'best_expr': str(best_expr),
        'train_pred': train_pred,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
    }
    
    if X_test is not None and y_test is not None:
        test_pred = model.predict(X_test_scaled)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        results.update({
            'test_pred': test_pred,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        logger.info(f"Symbolic Reg Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
                    f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}")

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, 'symbolic_expressions.txt'), 'w') as f:
            f.write(f"Best expression: {best_expr}\n")
            f.write("All equations:\n")
            for i, eq in enumerate(model.equations_):
                f.write(f"Eq {i}: {eq}\n")
        
        model.equations_.to_csv(os.path.join(output_dir, 'symbolic_hall_of_fame.csv'), index=False)
        
        plt.figure(figsize=(4,4))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Train', s=5)
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange', s=10)
        y_all = np.concatenate([y_train, y_test]) if y_test is not None else y_train
        plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('Symbolic Regression: Actual vs. Predicted')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'symbolic_prediction.png'), dpi=300)
        plt.close()
    
    return results


def compare_models(results_dict: Dict, output_dir: str = None) -> pd.DataFrame:
    """
    Compare multiple regression model results in a DataFrame.
    """
    model_names = list(results_dict.keys())
    
    comparison = {
        'Model': [],
        'Train RMSE': [],
        'Train R²': [],
        'Test RMSE': [],
        'Test R²': [],
        'Test MAE': []
    }
    
    for model_name, result in results_dict.items():
        comparison['Model'].append(model_name)
        comparison['Train RMSE'].append(result.get('train_rmse', np.nan))
        comparison['Train R²'].append(result.get('train_r2', np.nan))
        comparison['Test RMSE'].append(result.get('test_rmse', np.nan))
        comparison['Test R²'].append(result.get('test_r2', np.nan))
        comparison['Test MAE'].append(result.get('test_mae', np.nan))
    
    comparison_df = pd.DataFrame(comparison)
    logger.info("\nRegression Model Comparison:\n" + comparison_df.to_string(index=False))
    
    if output_dir:
        comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(model_names))
        width = 0.2
        
        plt.bar(x - width*1.5, comparison_df['Train RMSE'], width, label='Train RMSE')
        plt.bar(x - width/2, comparison_df['Test RMSE'], width, label='Test RMSE')
        plt.bar(x + width/2, comparison_df['Test MAE'], width, label='Test MAE')
        plt.bar(x + width*1.5, comparison_df['Test R²'], width, label='Test R²')
        
        plt.xticks(x, model_names)
        plt.xlabel('Model')
        plt.ylabel('Metric Value')
        plt.title('Regression Model Performance Comparison')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
    
    return comparison_df
