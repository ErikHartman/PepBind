import numpy as np
import pandas as pd
import logging
from typing import Dict, List

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from pysr import PySRRegressor

from utils import scale_data

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def train_logistic_regression(
    X_train: pd.DataFrame, 
    y_train: np.ndarray, 
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train a Logistic Regression model using GridSearchCV.
    Returns train and test metrics and predictions.
    """
    logger.info("Training Logistic Regression model...")

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
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
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='binary')
    train_proba = best_logreg.predict_proba(X_train_scaled)[:, 1]
    logger.info(f"Logistic Regression - Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

    # Create a nice dataframe with feature importance like the RF function does
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
        'train_acc': train_acc,
        'train_f1': train_f1,
        'coefficients': feature_importances
    }

    # Evaluate on test data
    if X_test is not None and y_test is not None:
        test_pred = best_logreg.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='binary')
        test_proba = best_logreg.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)    

        
        results.update({
            'test_pred': test_pred,
            'test_proba': test_proba,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc
        })
        logger.info(
            f"Logistic Regression - Test Accuracy: {test_acc:.4f}, "
            f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}"
        )
        
    return results


def train_random_forest_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train a Random Forest Classifier with GridSearchCV.
    Returns train and test metrics and predictions.
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
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
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
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='binary')
    train_proba = best_rf.predict_proba(X_train_scaled)[:, 1]
    logger.info(f"RF - Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

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
        'train_acc': train_acc,
        'train_f1': train_f1
    }

    # Evaluate on test data
    if X_test is not None and y_test is not None:
        test_pred = best_rf.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='binary')
        test_proba = best_rf.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)


        results.update({
            'test_pred': test_pred,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_proba': test_proba
        })
        logger.info(
            f"RF - Test Accuracy: {test_acc:.4f}, "
            f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}"
        )

    return results


def train_svm_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    cv: int = 5,
    param_grid: Dict = None,
) -> Dict:
    """
    Train an SVM Classifier with GridSearchCV.
    Returns train and test metrics and predictions.
    """
    logger.info("Training SVM classifier...")
    
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
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

    train_pred = best_svc.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='binary')
    train_proba = best_svc.predict_proba(X_train_scaled)[:, 1]
    logger.info(f"SVM - Train Accuracy: {train_acc:.4f}, F1: {train_f1:.4f}")

    results = {
        'model': best_svc,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'train_pred': train_pred,
        'train_proba': train_proba,
        'train_acc': train_acc,
        'train_f1': train_f1
    }

    if X_test is not None and y_test is not None:
        test_pred = best_svc.predict(X_test_scaled)
        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='binary')

        # For AUC
        test_proba = best_svc.predict_proba(X_test_scaled)[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)

        results.update({
            'test_pred': test_pred,
            'test_accuracy': test_acc,
            'test_f1': test_f1,
            'test_auc': test_auc,
            'test_proba': test_proba
        })
        logger.info(
            f"SVM - Test Accuracy: {test_acc:.4f}, "
            f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}"
        )
    
    return results


def perform_symbolic_classification(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame = None,
    y_test: np.ndarray = None,
    niterations: int = 200,
    populations: int = 50,
    population_size: int = 100,
    binary_operators: List[str] = None,
    unary_operators: List[str] = None,
    model_selection: str = "accuracy" ,
    select_k_features: int = 10,
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
        unary_operators = ["square", "log", "sqrt"]

    # Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    model = PySRRegressor(
        model_selection=model_selection,
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=populations,
        population_size=population_size,
        select_k_features = select_k_features,
        verbosity=0
    )
    
    # Fit on the scaled training data, with y in {0,1}
    model.fit(X_train_scaled, y_train, variable_names=list(X_train.columns))
    best_expr = model.sympy()
    logger.info(f"Best symbolic expression found: {best_expr}")

    hall_of_fame = model.equations_
    top_eqs = []
    for _, eq in hall_of_fame.iterrows():
        top_eqs.append({
            'equation': eq['equation'],      # String representation of the equation
            'loss': eq['loss'],
            'complexity': eq['complexity'],
            'score': eq['score'],
        })

    # Predict continuous values on training data
    train_pred_cont = model.predict(X_train_scaled)
    # Threshold at 0.5 to get class predictions (0 or 1)
    train_pred = (train_pred_cont >= 0.5).astype(int)

    # Compute training metrics
    train_acc = accuracy_score(y_train, train_pred)
    train_f1 = f1_score(y_train, train_pred, average='binary')
    # If you want AUC on train, we can do it with the raw continuous output:
    # (But keep in mind it might be <0 or >1)
    train_auc = roc_auc_score(y_train, train_pred_cont)
    # if it can't compute for some reason

    logger.info(
        f"Symbolic Classification (Train) - Accuracy: {train_acc:.4f}, "
        f"F1: {train_f1:.4f}, AUC: {train_auc:.4f}"
    )

    # Prepare results dict
    results = {
        'model': model,
        'scaler': scaler,
        'best_expr': str(best_expr),
        'train_pred_cont': train_pred_cont,  # continuous output
        'train_pred': train_pred,           # thresholded
        'train_proba': train_pred_cont,     # continuous output for AUC
        'train_acc': train_acc,
        'train_f1': train_f1,
        'train_auc': train_auc,
        'top_equations': pd.DataFrame(top_eqs),
    }

    # Evaluate on test data (if provided)
    if X_test is not None and y_test is not None:
        test_pred_cont = model.predict(X_test_scaled)
        test_pred = (test_pred_cont >= 0.5).astype(int)

        test_acc = accuracy_score(y_test, test_pred)
        test_f1 = f1_score(y_test, test_pred, average='binary')
        test_auc = roc_auc_score(y_test, test_pred_cont)
  

        results.update({
            'test_pred_cont': test_pred_cont,
            'test_pred': test_pred,
            'test_accuracy': test_acc,
            'test_proba': test_pred_cont,
            'test_f1': test_f1,
            'test_auc': test_auc
        })

        logger.info(
            f"Symbolic Classification (Test) - Accuracy: {test_acc:.4f}, "
            f"F1: {test_f1:.4f}, AUC: {test_auc:.4f}"
        )

    return results