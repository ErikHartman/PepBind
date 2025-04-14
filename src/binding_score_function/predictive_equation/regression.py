import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from pysr import PySRRegressor
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def scale_data(X_train: pd.DataFrame, X_test: pd.DataFrame = None) -> Tuple:
    """
    Scale features using StandardScaler.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training data features
    X_test : pd.DataFrame, optional
        Test data features
        
    Returns:
    --------
    Tuple containing scaled data and scaler object
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
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target values
    X_test : pd.DataFrame, optional
        Test features
    y_test : np.ndarray, optional
        Test target values
    cv : int
        Cross-validation folds
    n_alphas : int
        Number of alpha values to test
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    Dict containing model, coefficients, performance metrics and predictions
    """
    logger.info("Training Lasso regression model...")
    
    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Find optimal alpha using cross-validation
    alphas = np.logspace(-6, 2, n_alphas)
    lasso_cv = LassoCV(alphas=alphas, cv=cv, random_state=42, max_iter=10000)
    lasso_cv.fit(X_train_scaled, y_train)
    
    best_alpha = lasso_cv.alpha_
    logger.info(f"Best alpha value: {best_alpha:.6f}")
    
    # Train final model with best alpha
    lasso = Lasso(alpha=best_alpha, random_state=42, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    # Get non-zero coefficients
    feature_names = X_train.columns
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lasso.coef_
    })
    coef_df = coef_df[coef_df['Coefficient'] != 0].sort_values(by='Coefficient', key=abs, ascending=False)
    
    # Evaluate performance
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
        
        logger.info(f"Lasso Performance: Train RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
        logger.info(f"Lasso Performance: Test RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        

        
        # Save coefficients
        coef_df.to_csv(os.path.join(output_dir, 'lasso_coefficients.csv'), index=False)
        
        # Save feature importance plot
        plt.figure(figsize=(12, 8))
        sorted_idx = coef_df['Coefficient'].abs().argsort()[::-1]
        plt.barh(coef_df['Feature'].values, coef_df['Coefficient'].values, color='steelblue')
        plt.xlabel('Coefficient Value')
        plt.title('Lasso Regression Coefficients')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'lasso_coefficients.png'), dpi=300)
        plt.close()
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Training')
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange')
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('Lasso: Actual vs Predicted')
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
    Train a Random Forest regression model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target values
    X_test : pd.DataFrame, optional
        Test features
    y_test : np.ndarray, optional
        Test target values
    cv : int
        Cross-validation folds
    param_grid : Dict, optional
        Grid of hyperparameters to search
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    Dict containing model, performance metrics and feature importance
    """
    logger.info("Training Random Forest model...")
    
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    
    # Hyperparameter tuning
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train, y_train)
    
    best_rf = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate performance
    train_pred = best_rf.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    # Get feature importance
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
        
        logger.info(f"Random Forest Performance: Train RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
        logger.info(f"Random Forest Performance: Test RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        

        
        # Save feature importance
        feature_importance.to_csv(os.path.join(output_dir, 'rf_feature_importance.csv'), index=False)
        
        # Plot feature importance
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20), palette='viridis')
        plt.title('Random Forest Feature Importance (Top 20)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), dpi=300)
        plt.close()
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Training')
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange')
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('Random Forest: Actual vs Predicted')
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
    Train a Support Vector Regression model with hyperparameter tuning.
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target values
    X_test : pd.DataFrame, optional
        Test features
    y_test : np.ndarray, optional
        Test target values
    cv : int
        Cross-validation folds
    param_grid : Dict, optional
        Grid of hyperparameters to search
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    Dict containing model, performance metrics and other results
    """
    logger.info("Training SVR model...")
    
    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf']
        }
    
    # Hyperparameter tuning
    svr = SVR()
    grid_search = GridSearchCV(
        svr, param_grid, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_svr = grid_search.best_estimator_
    logger.info(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate performance
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
        
        logger.info(f"SVR Performance: Train RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
        logger.info(f"SVR Performance: Test RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Training')
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange')
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('SVR: Actual vs Predicted')
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
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : np.ndarray
        Training target values
    X_test : pd.DataFrame, optional
        Test features
    y_test : np.ndarray, optional
        Test target values
    niterations : int
        Number of iterations
    populations : int
        Number of populations
    population_size : int
        Size of each population
    maxsize : int
        Maximum expression size
    maxdepth : int
        Maximum expression depth
    binary_operators : List[str], optional
        Binary operators to use
    unary_operators : List[str], optional
        Unary operators to use
    output_dir : str, optional
        Directory to save results
        
    Returns:
    --------
    Dict containing model and performance metrics
    """
    logger.info("Performing Symbolic Regression...")

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["square", "log", "sqrt"]

    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Create and train the model
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
    )
    model.fit(X_train_scaled, y_train, variable_names=list(X_train.columns))
    
    # Get best expression
    best_expr = model.sympy()
    best_lambda = model.raw_sympy()
    logger.info(f"Best expression: {best_expr}")
    
    # Evaluate performance
    train_pred = model.predict(X_train_scaled)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    results = {
        'model': model,
        'scaler': scaler,
        'best_expr': str(best_expr),
        'best_lambda': str(best_lambda),
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
        
        logger.info(f"Symbolic Regression Performance: Train RMSE = {train_rmse:.4f}, R² = {train_r2:.4f}")
        logger.info(f"Symbolic Regression Performance: Test RMSE = {test_rmse:.4f}, R² = {test_r2:.4f}, MAE = {test_mae:.4f}")
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save expressions
        with open(os.path.join(output_dir, 'symbolic_expressions.txt'), 'w') as f:
            f.write(f"Best expression: {best_expr}\n")
            f.write(f"Best lambda: {best_lambda}\n")
            f.write(f"\nAll equations:\n")
            for i, eq in enumerate(model.equations_):
                f.write(f"Eq {i}: {eq['sympy_expr']} (score: {eq['score']:.6f})\n")
        
        # Save hall of fame dataframe
        model.equations_.to_csv(os.path.join(output_dir, 'symbolic_hall_of_fame.csv'), index=False)
        
        # Plot actual vs predicted
        plt.figure(figsize=(8, 8))
        plt.scatter(y_train, train_pred, alpha=0.5, label='Training')
        if X_test is not None:
            plt.scatter(y_test, test_pred, alpha=0.5, label='Test', color='orange')
        plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], 'k--')
        plt.xlabel('Actual pKd')
        plt.ylabel('Predicted pKd')
        plt.title('Symbolic Regression: Actual vs Predicted')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'symbolic_prediction.png'), dpi=300)
        plt.close()
    
    return results


def compare_models(results_dict: Dict, output_dir: str = None) -> pd.DataFrame:
    """
    Compare multiple model results.
    
    Parameters:
    -----------
    results_dict : Dict
        Dictionary of model results
    output_dir : str, optional
        Directory to save comparison results
        
    Returns:
    --------
    pd.DataFrame with model comparisons
    """
    model_names = list(results_dict.keys())
    
    # Create comparison DataFrame
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
        comparison['Train RMSE'].append(result['train_rmse'])
        comparison['Train R²'].append(result['train_r2'])
        
        if 'test_rmse' in result:
            comparison['Test RMSE'].append(result['test_rmse'])
            comparison['Test R²'].append(result['test_r2'])
            comparison['Test MAE'].append(result['test_mae'])
        else:
            comparison['Test RMSE'].append(np.nan)
            comparison['Test R²'].append(np.nan)
            comparison['Test MAE'].append(np.nan)
    
    comparison_df = pd.DataFrame(comparison)
    
    # Print comparison
    logger.info("\nModel Comparison:")
    logger.info(comparison_df.to_string(index=False))
    
    if output_dir:
        # Save comparison to CSV
        comparison_df.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
        
        # Create comparison plot
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(model_names))
        width = 0.2
        
        plt.bar(x - width*1.5, comparison_df['Train RMSE'], width, label='Train RMSE')
        plt.bar(x - width/2, comparison_df['Test RMSE'], width, label='Test RMSE')
        plt.bar(x + width/2, comparison_df['Test MAE'], width, label='Test MAE')
        plt.bar(x + width*1.5, comparison_df['Test R²'], width, label='Test R²')
        
        plt.xlabel('Model')
        plt.ylabel('Value')
        plt.title('Model Performance Comparison')
        plt.xticks(x, model_names)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300)
        plt.close()
    
    return comparison_df


if __name__ == "__main__":
    # Set paths
    base_path = "/srv/data1/general/immunopeptides_data/"
    scores_path = os.path.join(base_path, "outputs/binding_score_function/4_processed_scores/")
    output_dir = "./plots/predictive_equation"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load data
    X_train = pd.read_csv(os.path.join(scores_path, "real_X_train.csv")).set_index("complex_filename") # real amino acid sequences
    y_train = pd.read_csv(os.path.join(scores_path, "real_y_train.csv"))["pKd"].values

    shuffled_X_train = pd.read_csv(os.path.join(scores_path, "shuffled_X_train.csv")).set_index("complex_filename") # shuffled amino acid sequences
    random_X_train = pd.read_csv(os.path.join(scores_path, "random_X_train.csv")).set_index("complex_filename") # random amino acid sequences

    """
    2 tasks:
    1. Classification model that separates real docking results from shuffled and random ones
    2. Regression model that predicts the binding affinity (pKd) of the real docking results

    Classification between X_train and shuffled_X_train+random_X_train
    RF, SVC, logistic lasso and Symbolic Regression
    I think shuffled_X_train is slightly more difficult than random_X_train

    Regression on X_train:
    Lasso, RF, SVR and Symbolic Regression

    I want to do cross validation for models so that we evaluate it on the entire training set
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X_train, y_train, test_size=0.1, random_state=42
    )
    
    logger.info(f"Data loaded: X_train shape={X_train.shape}, X_test shape={X_test.shape}")
    
    # Train models
    lasso_results = train_lasso(
        X_train, y_train, X_test, y_test, 
        output_dir=os.path.join(output_dir, "lasso")
    )
    
    rf_results = train_random_forest(
        X_train, y_train, X_test, y_test,
        output_dir=os.path.join(output_dir, "random_forest")
    )
    
    svr_results = train_svr(
        X_train, y_train, X_test, y_test,
        output_dir=os.path.join(output_dir, "svr")
    )
    
    symbolic_results = perform_symbolic_regression(
        X_train, y_train, X_test, y_test,
        niterations=100,
        output_dir=os.path.join(output_dir, "symbolic")
    )
    
    # Compare models
    model_results = {
        'Lasso': lasso_results,
        'Random Forest': rf_results,
        'SVR': svr_results,
        'Symbolic Regression': symbolic_results
    }
    
    comparison = compare_models(model_results, output_dir)
    
    # Print best model
    best_model = comparison.iloc[comparison['Test R²'].idxmax()]
    logger.info(f"\nBest model: {best_model['Model']} with Test R² = {best_model['Test R²']:.4f}")
    
    # If symbolic regression was best, show the expression
    if best_model['Model'] == 'Symbolic Regression':
        logger.info(f"Best expression: {symbolic_results['best_expr']}")