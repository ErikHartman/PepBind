import os
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
import sympy

from utils import scale_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_lasso(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    cv: int = 5,
    n_alphas: int = 100,
) -> Dict:
    logger.info("Training Lasso regression model...")
    
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    
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
    
    if X_val is not None and y_val is not None:
        val_pred = lasso.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        results.update({
            'val_pred': val_pred,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        })
        
        logger.info(
            f"Lasso Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Val RMSE={val_rmse:.4f}, R²={val_r2:.4f}, MAE={val_mae:.4f}"
        )
    
    return results


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
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
    
    if X_val is not None and y_val is not None:
        val_pred = best_rf.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        results.update({
            'val_pred': val_pred,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        })
        logger.info(
            f"RF Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Val RMSE={val_rmse:.4f}, R²={val_r2:.4f}, MAE={val_mae:.4f}"
        )
    
    return results


def train_svr(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train SVR with GridSearchCV.
    """
    logger.info("Training SVR model...")
    
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    
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
    
    if X_val is not None and y_val is not None:
        val_pred = best_svr.predict(X_val_scaled)
        val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        val_r2 = r2_score(y_val, val_pred)
        val_mae = mean_absolute_error(y_val, val_pred)
        
        results.update({
            'val_pred': val_pred,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        })
        logger.info(
            f"SVR Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Val RMSE={val_rmse:.4f}, R²={val_r2:.4f}, MAE={val_mae:.4f}"
        )
    
    return results


def perform_symbolic_regression(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame = None,
    y_val: np.ndarray = None,
    niterations: int = 200,
    populations: int = 50,
    population_size: int = 100,
    binary_operators: List[str] = None,
    unary_operators: List[str] = None,
    model_selection:str="accuracy",
    select_k_features: int = None,
    scale_features: bool = False,
) -> Dict:
    logger.info("Performing Symbolic Regression...")

    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["square", "log", "sqrt"]

    # Apply scaling if requested
    if scale_features:
        X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
        X_train_data = X_train_scaled
        X_val_data = X_val_scaled
    else:
        X_train_data = X_train
        X_val_data = X_val
        scaler = None

    model = PySRRegressor(
        model_selection=model_selection,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=populations,
        population_size=population_size,
        select_k_features=select_k_features,
        verbosity=0,
    )
    
    model.fit(X_train_data, y_train, variable_names=list(X_train.columns))
    
    # Get equations dataframe
    equations = model.equations_.reset_index().rename(columns={"index": "eq_index"})
    equations = equations.sort_values(by="loss", ascending=True).reset_index(drop=True)
    
    # Calculate metrics for each equation on the val set
    if X_val is not None and y_val is not None:
        val_metrics = []
        for i, row in equations.iterrows():
            eq_index = row['eq_index']
            try:
                # Ensure we're passing numpy arrays, not trying to access .values on a numpy array
                X_val_array = X_val_data if isinstance(X_val_data, np.ndarray) else X_val_data.values
                val_pred = model.predict(X_val_array, index=eq_index)
                
                # Safeguard against NaNs
                valid_mask = np.isfinite(val_pred)
                if valid_mask.all():
                    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
                    val_r2 = r2_score(y_val, val_pred)
                    val_mae = mean_absolute_error(y_val, val_pred)
                    
                    val_metrics.append({
                        'eq_index': eq_index,
                        'val_rmse': val_rmse,
                        'val_r2': val_r2,
                        'val_mae': val_mae
                    })
                else:
                    if valid_mask.any():
                        # Use only valid predictions for metrics
                        val_rmse = np.sqrt(mean_squared_error(y_val[valid_mask], val_pred[valid_mask]))
                        val_r2 = r2_score(y_val[valid_mask], val_pred[valid_mask])
                        val_mae = mean_absolute_error(y_val[valid_mask], val_pred[valid_mask])
                        
                        val_metrics.append({
                            'eq_index': eq_index,
                            'val_rmse': val_rmse,
                            'val_r2': val_r2,
                            'val_mae': val_mae,
                            'valid_ratio': valid_mask.sum() / len(valid_mask)
                        })
            except Exception as e:
                logger.warning(f"Error evaluating equation {i} on val set: {str(e)}")
    
        # Find the best equation based on val RMSE
        if val_metrics:
            val_metrics_df = pd.DataFrame(val_metrics)
            best_val_idx = val_metrics_df['val_rmse'].idxmin()
            best_val_eq_index = val_metrics_df.loc[best_val_idx, 'eq_index']
            
            # Get the best expression based on val performance
            best_expr = simplify_expression(model, best_val_eq_index)
            logger.info(f"Best symbolic expression on val set: {best_expr}")
            
            # Get train and val predictions for the best val model
            X_train_array = X_train_data if isinstance(X_train_data, np.ndarray) else X_train_data.values
            train_pred = model.predict(X_train_array, index=best_val_eq_index)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            X_val_array = X_val_data if isinstance(X_val_data, np.ndarray) else X_val_data.values
            val_pred = model.predict(X_val_array, index=best_val_eq_index)
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
        else:
            # If no equations worked well on val set, use the original best by train loss
            best_expr = model.sympy(equations.iloc[0]['eq_index'])
            logger.warning("No equations performed well on val set, using best from training")
            
            # Predictions with best training model
            X_train_array = X_train_data if isinstance(X_train_data, np.ndarray) else X_train_data.values
            train_pred = model.predict(X_train_array, index=equations.iloc[0]['eq_index'])
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            train_r2 = r2_score(y_train, train_pred)
            
            X_val_array = X_val_data if isinstance(X_val_data, np.ndarray) else X_val_data.values
            val_pred = model.predict(X_val_array, index=equations.iloc[0]['eq_index'])
            val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
            val_r2 = r2_score(y_val, val_pred)
            val_mae = mean_absolute_error(y_val, val_pred)
    else:
        # Without val data, just use the best equation from training
        best_expr = model.sympy(equations.iloc[0]['eq_index'])
        
        X_train_array = X_train_data if isinstance(X_train_data, np.ndarray) else X_train_data.values
        train_pred = model.predict(X_train_array, index=equations.iloc[0]['eq_index'])
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        train_r2 = r2_score(y_train, train_pred)
    
    all_eqs = []
    for i, row in equations.iterrows():
        eq_index = row['eq_index']
        eq_str = simplify_expression(model, eq_index)
        complexity = row['complexity']
        loss = row['loss']
        score = row['score']
        
        try:
            X_train_array = X_train_data if isinstance(X_train_data, np.ndarray) else X_train_data.values
            eq_pred_train = model.predict(X_train_array, index=eq_index)
            
            valid_mask_train = np.isfinite(eq_pred_train)
            if not valid_mask_train.all():
                logger.warning(f"Equation {i}: {valid_mask_train.sum()}/{len(valid_mask_train)} valid predictions on train")

                train_rmse_eq = np.sqrt(mean_squared_error(y_train[valid_mask_train], eq_pred_train[valid_mask_train])) if valid_mask_train.any() else np.nan
                train_r2_eq = r2_score(y_train[valid_mask_train], eq_pred_train[valid_mask_train]) if valid_mask_train.any() else np.nan
            else:
                train_rmse_eq = np.sqrt(mean_squared_error(y_train, eq_pred_train))
                train_r2_eq = r2_score(y_train, eq_pred_train)
            
            val_rmse_eq, val_r2_eq, val_mae_eq = np.nan, np.nan, np.nan
            if X_val is not None and y_val is not None:
                X_val_array = X_val_data if isinstance(X_val_data, np.ndarray) else X_val_data.values
                eq_pred_val = model.predict(X_val_array, index=eq_index)
                
                valid_mask_val = np.isfinite(eq_pred_val)
                if not valid_mask_val.all():
                    logger.warning(f"Equation {i}: {valid_mask_val.sum()}/{len(valid_mask_val)} valid predictions on val")
                    if valid_mask_val.any():
                        val_rmse_eq = np.sqrt(mean_squared_error(y_val[valid_mask_val], eq_pred_val[valid_mask_val]))
                        val_r2_eq = r2_score(y_val[valid_mask_val], eq_pred_val[valid_mask_val])
                        val_mae_eq = mean_absolute_error(y_val[valid_mask_val], eq_pred_val[valid_mask_val])
                else:
                    val_rmse_eq = np.sqrt(mean_squared_error(y_val, eq_pred_val))
                    val_r2_eq = r2_score(y_val, eq_pred_val)
                    val_mae_eq = mean_absolute_error(y_val, eq_pred_val)
        
        except Exception as e:
            logger.warning(f"Error calculating metrics for equation {i}: {str(e)}")
            train_rmse_eq, train_r2_eq = np.nan, np.nan
            val_rmse_eq, val_r2_eq, val_mae_eq = np.nan, np.nan, np.nan
            
        all_eqs.append({
            'equation': eq_str,
            'complexity': complexity,
            'loss': loss,
            'score': score,
            'train_rmse': train_rmse_eq,
            'train_r2': train_r2_eq,
            'val_rmse': val_rmse_eq,
            'val_r2': val_r2_eq,
            'val_mae': val_mae_eq,
            'equation_index': eq_index
        })
    

    valid_mask_train = np.isfinite(train_pred)
    if not valid_mask_train.all():
        logger.warning(f"Best model: {valid_mask_train.sum()}/{len(valid_mask_train)} valid predictions on train")
        train_rmse = np.sqrt(mean_squared_error(y_train[valid_mask_train], train_pred[valid_mask_train])) if valid_mask_train.any() else np.nan
        train_r2 = r2_score(y_train[valid_mask_train], train_pred[valid_mask_train]) if valid_mask_train.any() else np.nan
    
    results = {
        'model': model,
        'best_expr': str(best_expr),
        'train_pred': train_pred,
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'all_equations': pd.DataFrame(all_eqs),
        'scaler': scaler,
    }
    
    if X_val is not None and y_val is not None:
        results.update({
            'val_pred': val_pred,
            'val_rmse': val_rmse,
            'val_r2': val_r2,
            'val_mae': val_mae
        })
        logger.info(
            f"Symbolic Reg Perf: Train RMSE={train_rmse:.4f}, R²={train_r2:.4f} | "
            f"Val RMSE={val_rmse:.4f}, R²={val_r2:.4f}, MAE={val_mae:.4f}"
        )

    return results


def simplify_expression(model, eq_index):
    expr = model.sympy(eq_index)
    simplified = sympy.simplify(expr)
    return str(simplified)