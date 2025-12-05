import os
import numpy as np
import pandas as pd
import logging
from typing import Dict, List

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from pysr import PySRRegressor
import sympy

from utils import scale_data, get_classification_results

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train a Logistic Regression model using GridSearchCV.
    Returns train and val metrics and predictions.
    """
    logger.info("Training Logistic Regression model...")

    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    
    if param_grid is None:
        param_grid = {
            'C': [0.01, 0.1, 1, 10],
        }

    logreg = LogisticRegression(random_state=42, max_iter=1000, penalty='l2')
    grid_search = GridSearchCV(
        estimator=logreg,
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_logreg = grid_search.best_estimator_
    logger.info(f"Logistic Regression best params: {grid_search.best_params_}")

    # Training predictions
    train_pred = best_logreg.predict(X_train_scaled)
    train_proba = best_logreg.predict_proba(X_train_scaled)[:, 1]
    train_metrics = get_classification_results(y_train, train_pred, train_proba, prefix="train_")
    
    # Validation predictions
    val_pred = best_logreg.predict(X_val_scaled)
    val_proba = best_logreg.predict_proba(X_val_scaled)[:, 1]
    val_metrics = get_classification_results(y_val, val_pred, val_proba, prefix="val_")
    
    logger.info(f"Logistic Regression Perf")
    logger.info(f"Train results: {train_metrics}")
    logger.info(f"Validation results: {val_metrics}")

    # Create feature importance dataframe  
    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Coefficient': best_logreg.coef_[0]
    }).sort_values(by='Coefficient', key=abs, ascending=False)

    results = {
        'model': best_logreg,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'val_pred': val_pred,
        'val_proba': val_proba,
        'coefficients': feature_importances
    }
    
    # Add all metrics to results
    results.update(train_metrics)
    results.update(val_metrics)

    return results


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train a Random Forest Classifier with GridSearchCV.
    Returns train and val metrics and predictions.
    """
    logger.info("Training Random Forest classifier...")

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

    # No scaling strictly required for RF, but can be done for consistency
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, 
        param_grid=param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)

    best_rf = grid_search.best_estimator_
    logger.info(f"RF best parameters: {grid_search.best_params_}")

    # Training predictions
    train_pred = best_rf.predict(X_train_scaled)
    train_proba = best_rf.predict_proba(X_train_scaled)[:, 1]
    train_metrics = get_classification_results(y_train, train_pred, train_proba, prefix="train_")
    
    # Validation predictions
    val_pred = best_rf.predict(X_val_scaled)
    val_proba = best_rf.predict_proba(X_val_scaled)[:, 1]
    val_metrics = get_classification_results(y_val, val_pred, val_proba, prefix="val_")

    logger.info(f"RF Perf")
    logger.info(f"Train results: {train_metrics}")
    logger.info(f"Validation results: {val_metrics}")

    feature_importances = pd.DataFrame({
        'Feature': X_train.columns,
        'Importance': best_rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    results = {
        'model': best_rf,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'feature_importance': feature_importances,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'val_pred': val_pred,
        'val_proba': val_proba
    }
    
    # Add all metrics to results
    results.update(train_metrics)
    results.update(val_metrics)

    return results


def train_svm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train an SVM Classifier with GridSearchCV.
    Returns train and val metrics and predictions.
    """
    logger.info("Training SVM classifier...")
    
    X_train_scaled, X_val_scaled, scaler = scale_data(X_train, X_val)
    
    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10, 100],
            'gamma': ['scale', 'auto', 0.1, 0.01],
            'kernel': ['rbf']
        }
    
    svc = SVC(probability=True, random_state=42)  # probability=True for predict_proba
    grid_search = GridSearchCV(
        svc, 
        param_grid, 
        cv=cv, 
        scoring='accuracy',
        n_jobs=-1
    )
    grid_search.fit(X_train_scaled, y_train)
    
    best_svc = grid_search.best_estimator_
    logger.info(f"SVM best params: {grid_search.best_params_}")

    # Training predictions
    train_pred = best_svc.predict(X_train_scaled)
    train_proba = best_svc.predict_proba(X_train_scaled)[:, 1]
    train_metrics = get_classification_results(y_train, train_pred, train_proba, prefix="train_")
    
    # Validation predictions
    val_pred = best_svc.predict(X_val_scaled)
    val_proba = best_svc.predict_proba(X_val_scaled)[:, 1]
    val_metrics = get_classification_results(y_val, val_pred, val_proba, prefix="val_")

    logger.info(f"SVM Perf")
    logger.info(f"Train results: {train_metrics}")
    logger.info(f"Validation results: {val_metrics}")

    results = {
        'model': best_svc,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'val_pred': val_pred,
        'val_proba': val_proba
    }
    
    # Add all metrics to results
    results.update(train_metrics)
    results.update(val_metrics)
    
    return results


def perform_symbolic_classification(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    niterations: int = 200,
    populations: int = 50,
    population_size: int = 100,
    binary_operators: List[str] = None,
    unary_operators: List[str] = None,
    model_selection: str = "accuracy" ,
    select_k_features: int = 10,
    scale_features: bool = False,
) -> Dict:
    """
    Perform a *symbolic 'classification'* approach by:
      1) Letting PySR model y as a continuous 0/1 variable (like regression).
      2) Predicting a continuous output, then thresholding at 0.5 to get a class label.
      3) Computing classification metrics (accuracy, F1, AUC, etc.) on the 0/1 predictions.
    """

    logger.info("Performing Symbolic Classification via threshold=0.5 on continuous output...")

    # Default operators if none are provided
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = []

    # Apply scaling if requested
    if scale_features:
        X_train_data, X_val_data, scaler = scale_data(X_train, X_val)
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
        select_k_features = select_k_features,
        verbosity=1,
    )
    
    model.fit(X_train_data, y_train, variable_names=list(X_train.columns))
    
    # Get equations dataframe
    equations = model.equations_.reset_index().rename(columns={"index": "eq_index"})
    equations = equations.sort_values(by="loss", ascending=True).reset_index(drop=True)
    
    # Calculate metrics for each equation on the val set
    val_metrics = []
    for i, row in equations.iterrows():
        eq_index = row['eq_index']
        try:
            # Calculate predictions for this equation
            val_pred_cont = model.predict(X_val_data, index=eq_index)
            
            # Filter out NaN/infinite values
            valid_mask = np.isfinite(val_pred_cont)
            if not valid_mask.any():
                logger.warning(f"Equation {i}: All predictions are NaN/infinite on val set")
                continue
                
            if not valid_mask.all():
                logger.warning(f"Equation {i}: {valid_mask.sum()}/{len(valid_mask)} valid predictions on val")
                
            # Use only valid predictions for metrics
            val_pred_cont_valid = val_pred_cont[valid_mask] if not valid_mask.all() else val_pred_cont
            y_val_valid = y_val[valid_mask] if not valid_mask.all() else y_val
            
            val_pred_valid = (val_pred_cont_valid >= 0.5).astype(int)
            val_metrics_dict = get_classification_results(y_val_valid, val_pred_valid, val_pred_cont_valid, prefix="")
            
            val_metrics.append({
                'eq_index': eq_index,
                'val_acc': val_metrics_dict['acc'],
                'val_f1': val_metrics_dict['f1'],
                'val_auc': val_metrics_dict['auc'],
                'valid_ratio': valid_mask.sum() / len(valid_mask) if not valid_mask.all() else 1.0
            })
        except Exception as e:
            logger.warning(f"Error evaluating equation {i} on val set: {str(e)}")
            
    # Find the best equation based on val AUC (could also use acc or f1)
    if val_metrics:
        val_metrics_df = pd.DataFrame(val_metrics)
        # Use AUC as the primary metric for classification
        best_val_idx = val_metrics_df['val_auc'].idxmax()
        best_val_eq_index = val_metrics_df.loc[best_val_idx, 'eq_index']
        
        # Get the best expression based on val performance
        best_expr = simplify_expression(model, best_val_eq_index)
        logger.info(f"Best symbolic expression on val set: {best_expr}")
        
        # Get train and val predictions for the best val model
        train_pred_cont = model.predict(X_train_data, index=best_val_eq_index)
        val_pred_cont = model.predict(X_val_data, index=best_val_eq_index)
        
        # Handle NaN values in final predictions
        train_valid_mask = np.isfinite(train_pred_cont)
        val_valid_mask = np.isfinite(val_pred_cont)
        
        if train_valid_mask.all() and val_valid_mask.all():
            # All predictions are valid
            train_pred = (train_pred_cont >= 0.5).astype(int)
            val_pred = (val_pred_cont >= 0.5).astype(int)
            
            train_metrics = get_classification_results(y_train, train_pred, train_pred_cont, prefix="train_")
            val_metrics = get_classification_results(y_val, val_pred, val_pred_cont, prefix="val_")
        else:
            # Some predictions are invalid, filter them out
            if train_valid_mask.any():
                train_pred_cont_valid = train_pred_cont[train_valid_mask]
                train_pred_valid = (train_pred_cont_valid >= 0.5).astype(int)
                y_train_valid = y_train[train_valid_mask]
                train_metrics = get_classification_results(y_train_valid, train_pred_valid, train_pred_cont_valid, prefix="train_")
                
                # Create full arrays with NaN for invalid predictions
                train_pred = np.full_like(train_pred_cont, np.nan)
                train_pred[train_valid_mask] = train_pred_valid
            else:
                train_metrics = {'train_acc': np.nan, 'train_f1': np.nan, 'train_auc': np.nan}
                train_pred = np.full_like(train_pred_cont, np.nan)
            
            if val_valid_mask.any():
                val_pred_cont_valid = val_pred_cont[val_valid_mask]
                val_pred_valid = (val_pred_cont_valid >= 0.5).astype(int)
                y_val_valid = y_val[val_valid_mask]
                val_metrics = get_classification_results(y_val_valid, val_pred_valid, val_pred_cont_valid, prefix="val_")
                
                # Create full arrays with NaN for invalid predictions
                val_pred = np.full_like(val_pred_cont, np.nan)
                val_pred[val_valid_mask] = val_pred_valid
            else:
                val_metrics = {'val_acc': np.nan, 'val_f1': np.nan, 'val_auc': np.nan}
                val_pred = np.full_like(val_pred_cont, np.nan)
    else:
        # If no equations worked well on val set, use the original best by train loss
        best_expr = model.sympy(equations.iloc[0]['eq_index'])
        logger.warning("No equations performed well on val set, using best from training")
        
        # Predictions with best training model
        train_pred_cont = model.predict(X_train_data, index=equations.iloc[0]['eq_index'])
        val_pred_cont = model.predict(X_val_data, index=equations.iloc[0]['eq_index'])
        
        # Handle NaN values similar to above
        train_valid_mask = np.isfinite(train_pred_cont)
        val_valid_mask = np.isfinite(val_pred_cont)
        
        if train_valid_mask.all() and val_valid_mask.all():
            train_pred = (train_pred_cont >= 0.5).astype(int)
            val_pred = (val_pred_cont >= 0.5).astype(int)
            
            train_metrics = get_classification_results(y_train, train_pred, train_pred_cont, prefix="train_")
            val_metrics = get_classification_results(y_val, val_pred, val_pred_cont, prefix="val_")
        else:
            # Handle invalid predictions as above
            if train_valid_mask.any():
                train_pred_cont_valid = train_pred_cont[train_valid_mask]
                train_pred_valid = (train_pred_cont_valid >= 0.5).astype(int)
                y_train_valid = y_train[train_valid_mask]
                train_metrics = get_classification_results(y_train_valid, train_pred_valid, train_pred_cont_valid, prefix="train_")
                
                train_pred = np.full_like(train_pred_cont, np.nan)
                train_pred[train_valid_mask] = train_pred_valid
            else:
                train_metrics = {'train_acc': np.nan, 'train_f1': np.nan, 'train_auc': np.nan}
                train_pred = np.full_like(train_pred_cont, np.nan)
            
            if val_valid_mask.any():
                val_pred_cont_valid = val_pred_cont[val_valid_mask]
                val_pred_valid = (val_pred_cont_valid >= 0.5).astype(int)
                y_val_valid = y_val[val_valid_mask]
                val_metrics = get_classification_results(y_val_valid, val_pred_valid, val_pred_cont_valid, prefix="val_")
                
                val_pred = np.full_like(val_pred_cont, np.nan)
                val_pred[val_valid_mask] = val_pred_valid
            else:
                val_metrics = {'val_acc': np.nan, 'val_f1': np.nan, 'val_auc': np.nan}
                val_pred = np.full_like(val_pred_cont, np.nan)

    # Get all equations and their metrics
    all_eqs = []
    for i, row in equations.iterrows():
        eq_index = row['eq_index']
        eq_str = simplify_expression(model, eq_index)
        complexity = row['complexity']
        loss = row['loss']
        score = row['score']
        
        # Calculate predictions for this specific equation
        try:
            eq_pred_cont_train = model.predict(X_train_data, index=eq_index)
            eq_pred_cont_val = model.predict(X_val_data, index=eq_index)
            
            # Handle train metrics with NaN filtering using utility function
            train_valid_mask_eq = np.isfinite(eq_pred_cont_train)
            if train_valid_mask_eq.any():
                y_train_filtered = y_train[train_valid_mask_eq] if not train_valid_mask_eq.all() else y_train
                eq_pred_cont_train_filtered = eq_pred_cont_train[train_valid_mask_eq] if not train_valid_mask_eq.all() else eq_pred_cont_train
                eq_pred_train_valid = (eq_pred_cont_train_filtered >= 0.5).astype(int)
                
                train_metrics_eq = get_classification_results(y_train_filtered, eq_pred_train_valid, 
                                                            eq_pred_cont_train_filtered, prefix="")
                train_acc_eq, train_f1_eq, train_auc_eq = train_metrics_eq['acc'], train_metrics_eq['f1'], train_metrics_eq['auc']
            else:
                train_acc_eq, train_f1_eq, train_auc_eq = np.nan, np.nan, np.nan
            
            # Handle val metrics with NaN filtering using utility function
            val_valid_mask_eq = np.isfinite(eq_pred_cont_val)
            if val_valid_mask_eq.any():
                y_val_filtered = y_val[val_valid_mask_eq] if not val_valid_mask_eq.all() else y_val
                eq_pred_cont_val_filtered = eq_pred_cont_val[val_valid_mask_eq] if not val_valid_mask_eq.all() else eq_pred_cont_val
                eq_pred_val_valid = (eq_pred_cont_val_filtered >= 0.5).astype(int)
                
                val_metrics_eq = get_classification_results(y_val_filtered, eq_pred_val_valid, 
                                                          eq_pred_cont_val_filtered, prefix="")
                val_acc_eq, val_f1_eq, val_auc_eq = val_metrics_eq['acc'], val_metrics_eq['f1'], val_metrics_eq['auc']
            else:
                val_acc_eq, val_f1_eq, val_auc_eq = np.nan, np.nan, np.nan
        
        except Exception as e:
            logger.warning(f"Error calculating metrics for equation {i}: {str(e)}")
            train_acc_eq, train_f1_eq, train_auc_eq = np.nan, np.nan, np.nan
            val_acc_eq, val_f1_eq, val_auc_eq = np.nan, np.nan, np.nan
            
        all_eqs.append({
            'equation': eq_str,
            'complexity': complexity,
            'loss': loss,
            'score': score,
            'train_acc': train_acc_eq,
            'train_f1': train_f1_eq,
            'train_auc': train_auc_eq,
            'val_acc': val_acc_eq,
            'val_f1': val_f1_eq,
            'val_auc': val_auc_eq,
            'equation_index': eq_index
        })
    
    logger.info(f"Symbolic Classification Perf")
    logger.info(f"Train results: {train_metrics}")
    logger.info(f"Validation results: {val_metrics}")

    # Prepare results dict
    results = {
        'model': model,
        'best_expr': str(best_expr),
        'train_pred_cont': train_pred_cont,
        'train_pred': train_pred,
        'train_proba': train_pred_cont,
        'val_pred_cont': val_pred_cont,
        'val_pred': val_pred,
        'val_proba': val_pred_cont,
        'all_equations': pd.DataFrame(all_eqs),
        'scaler': scaler,  # Include the scaler if scaling was used
    }
    
    # Add all metrics to results
    results.update(train_metrics)
    results.update(val_metrics)

    return results

def simplify_expression(model, eq_index):
    expr = model.sympy(eq_index)
    simplified = sympy.simplify(expr)
    return str(simplified)
