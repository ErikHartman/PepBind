import numpy as np
import pandas as pd
from typing import Dict, List
import logging

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import Lasso, LassoCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from pysr import PySRRegressor

from utils import scale_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lasso(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    n_alphas: int = 100,
) -> Dict:
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
    # Keep only non-zero coefficients
    coef_df = coef_df[coef_df['Coefficient'] != 0].sort_values(
        by='Coefficient', key=abs, ascending=False
    )
    
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
        
        logger.info(
            f"Lasso Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}"
        )
    
    return results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
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
        logger.info(
            f"RF Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}"
        )
    
    return results


def train_svr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
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
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
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
        logger.info(
            f"SVR Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}"
        )
    
    return results


def perform_symbolic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    niterations: int = 200,
    populations: int = 50,
    population_size: int = 100,
    binary_operators: List[str] = None,
    unary_operators: List[str] = None,
    model_selection:str="accuracy",
    select_k_features: int = None,
) -> Dict:
    logger.info("Performing Symbolic Regression...")

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["square", "log", "sqrt"]

    model = PySRRegressor(
        model_selection=model_selection,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=populations,
        population_size=population_size,
        select_k_features=select_k_features,
        verbosity=0
    )
    

    model.fit(X_train, y_train)
    
    # Get equations dataframe
    equations = model.equations_.reset_index().rename(columns={"index": "eq_index"})
    equations = equations.sort_values(by="loss", ascending=True).reset_index(drop=True)
    
    # Get the best expression (according to loss)
    best_expr = model.sympy(equations.iloc[0]['eq_index'])
    logger.info(f"Best symbolic expression found: {best_expr}")

    # Predict with best equation (lowest loss)
    train_pred = model.predict(X_train.values, index=equations.iloc[0]['eq_index'])
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    # Get all equations and their metrics
    all_eqs = []
    for i, row in equations.iterrows():
        eq_index = row['eq_index']
        eq_str = str(model.sympy(eq_index))
        complexity = row['complexity']
        loss = row['loss']
        score = row['score']
        
        # Calculate predictions for this specific equation
        try:
            eq_pred_train = model.predict(X_train.values, index=eq_index)
            
            # Safeguard against NaNs or Infs
            valid_mask_train = np.isfinite(eq_pred_train)
            if not valid_mask_train.all():
                logger.warning(f"Equation {i}: {valid_mask_train.sum()}/{len(valid_mask_train)} valid predictions on train")
                # Use only valid predictions for metrics
                train_rmse_eq = np.sqrt(mean_squared_error(y_train[valid_mask_train], eq_pred_train[valid_mask_train])) if valid_mask_train.any() else np.nan
                train_r2_eq = r2_score(y_train[valid_mask_train], eq_pred_train[valid_mask_train]) if valid_mask_train.any() else np.nan
            else:
                train_rmse_eq = np.sqrt(mean_squared_error(y_train, eq_pred_train))
                train_r2_eq = r2_score(y_train, eq_pred_train)
            
            # Test metrics if test data provided
            test_rmse_eq, test_r2_eq, test_mae_eq = np.nan, np.nan, np.nan
            if X_test is not None and y_test is not None:
                eq_pred_test = model.predict(X_test.values, index=eq_index)
                
                # Safeguard against NaNs in test predictions
                valid_mask_test = np.isfinite(eq_pred_test)
                if not valid_mask_test.all():
                    logger.warning(f"Equation {i}: {valid_mask_test.sum()}/{len(valid_mask_test)} valid predictions on test")
                    if valid_mask_test.any():
                        test_rmse_eq = np.sqrt(mean_squared_error(y_test[valid_mask_test], eq_pred_test[valid_mask_test]))
                        test_r2_eq = r2_score(y_test[valid_mask_test], eq_pred_test[valid_mask_test])
                        test_mae_eq = mean_absolute_error(y_test[valid_mask_test], eq_pred_test[valid_mask_test])
                else:
                    test_rmse_eq = np.sqrt(mean_squared_error(y_test, eq_pred_test))
                    test_r2_eq = r2_score(y_test, eq_pred_test)
                    test_mae_eq = mean_absolute_error(y_test, eq_pred_test)
        
        except Exception as e:
            logger.warning(f"Error calculating metrics for equation {i}: {str(e)}")
            train_rmse_eq, train_r2_eq = np.nan, np.nan
            test_rmse_eq, test_r2_eq, test_mae_eq = np.nan, np.nan, np.nan
            
        all_eqs.append({
            'equation': eq_str,
            'complexity': complexity,
            'loss': loss,
            'score': score,
            'train_rmse': train_rmse_eq,
            'train_r2': train_r2_eq,
            'test_rmse': test_rmse_eq,
            'test_r2': test_r2_eq,
            'test_mae': test_mae_eq,
            'equation_index': eq_index
        })
    
    # Calculate metrics for best model (safeguarded against NaNs)
    valid_mask_train = np.isfinite(train_pred)
    if not valid_mask_train.all():
        logger.warning(f"Best model: {valid_mask_train.sum()}/{len(valid_mask_train)} valid predictions on train")
        train_rmse = np.sqrt(mean_squared_error(y_train[valid_mask_train], train_pred[valid_mask_train])) if valid_mask_train.any() else np.nan
        train_r2 = r2_score(y_train[valid_mask_train], train_pred[valid_mask_train]) if valid_mask_train.any() else np.nan
    
    # Prepare results dict
    results = {
        'model': model,
        'best_expr': str(best_expr),
        'train_pred': train_pred,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'all_equations': pd.DataFrame(all_eqs),
    }
    
    if X_test is not None and y_test is not None:
        test_pred = model.predict(X_test.values, index=equations.iloc[0]['eq_index'])
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_r2 = r2_score(y_test, test_pred)
        test_mae = mean_absolute_error(y_test, test_pred)
        
        # Safeguard against NaNs in test predictions for best model
        valid_mask_test = np.isfinite(test_pred)
        if not valid_mask_test.all():
            logger.warning(f"Best model: {valid_mask_test.sum()}/{len(valid_mask_test)} valid predictions on test")
            if valid_mask_test.any():
                test_rmse = np.sqrt(mean_squared_error(y_test[valid_mask_test], test_pred[valid_mask_test]))
                test_r2 = r2_score(y_test[valid_mask_test], test_pred[valid_mask_test])
                test_mae = mean_absolute_error(y_test[valid_mask_test], test_pred[valid_mask_test])
            else:
                test_rmse, test_r2, test_mae = np.nan, np.nan, np.nan
        
        results.update({
            'test_pred': test_pred,
            'test_rmse': test_rmse,
            'test_r2': test_r2,
            'test_mae': test_mae
        })
        logger.info(
            f"Symbolic Reg Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Test RMSE={test_rmse:.4f}, R²={test_r2:.4f}, MAE={test_mae:.4f}"
        )

    return results

