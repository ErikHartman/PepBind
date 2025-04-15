# classification.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import seaborn as sns
import logging

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
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


def plot_confusion_matrix(cm, class_labels, title: str = 'Confusion Matrix', output_path: str = None):
    """
    Plot a confusion matrix using matplotlib.
    """
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, cmap='Blues', fmt='d',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.title(title)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300)
    plt.close()


def train_logistic_regression_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    output_dir: str = None
) -> Dict:
    """
    Multiclass logistic regression using one-vs-rest or multinomial.
    """
    logger.info("Training Logistic Regression classifier (multiclass)...")

    X_scaled, scaler = scale_data(X)
    
    model_cv = LogisticRegressionCV(
        Cs=10,
        cv=cv_folds,
        penalty='elasticnet',
        multi_class='multinomial',
        max_iter=10000,
        random_state=42,
    )
    model_cv.fit(X_scaled, y)

    train_pred = model_cv.predict(X_scaled)
    train_acc = accuracy_score(y, train_pred)

    logger.info(f"Logistic Regression (train) accuracy: {train_acc:.4f}")
    cm = confusion_matrix(y, train_pred)

    results = {
        'model': model_cv,
        'scaler': scaler,
        'train_acc': train_acc,
        'confusion_matrix': cm,
        'class_report': classification_report(y, train_pred, output_dict=True)
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(
            cm, 
            class_labels=['Shuffled','Random','Real'], 
            title='Logistic Regression Confusion Matrix',
            output_path=os.path.join(output_dir, 'logreg_confusion_matrix.png')
        )
        cr_df = pd.DataFrame(results['class_report']).transpose()
        cr_df.to_csv(os.path.join(output_dir, 'logreg_classification_report.csv'))

    return results


def train_random_forest_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    param_grid: Dict = None,
    output_dir: str = None
) -> Dict:
    """
    Multiclass Random Forest Classifier with grid search cross-validation.
    """
    logger.info("Training Random Forest classifier (multiclass)...")

    X_scaled, scaler = scale_data(X)

    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }

    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf,
        param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_scaled, y)
    best_model = grid_search.best_estimator_
    logger.info(f"RF best params: {grid_search.best_params_}")

    train_pred = best_model.predict(X_scaled)
    train_acc = accuracy_score(y, train_pred)
    logger.info(f"Random Forest (train) accuracy: {train_acc:.4f}")

    cm = confusion_matrix(y, train_pred)
    
    feat_names = X.columns if isinstance(X, pd.DataFrame) else [f'x{i}' for i in range(X.shape[1])]
    feature_importance = pd.DataFrame({
        'Feature': feat_names,
        'Importance': best_model.feature_importances_
    }).sort_values(by='Importance', ascending=False)

    results = {
        'model': best_model,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'train_acc': train_acc,
        'confusion_matrix': cm,
        'class_report': classification_report(y, train_pred, output_dict=True),
        'feature_importance': feature_importance
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(
            cm, 
            class_labels=['Shuffled','Random','Real'], 
            title='Random Forest Confusion Matrix',
            output_path=os.path.join(output_dir, 'rf_confusion_matrix.png')
        )
        feature_importance.to_csv(os.path.join(output_dir, 'rf_feature_importance.csv'), index=False)

        plt.figure(figsize=(6,6))
        sns.barplot(x='Importance', y='Feature', data=feature_importance.head(20))
        plt.title('Top 20 Feature Importances (RF Classifier)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rf_feature_importance.png'), dpi=300)
        plt.close()

        cr_df = pd.DataFrame(results['class_report']).transpose()
        cr_df.to_csv(os.path.join(output_dir, 'rf_classification_report.csv'))

    return results


def train_svc_classifier(
    X: pd.DataFrame,
    y: np.ndarray,
    cv_folds: int = 5,
    param_grid: Dict = None,
    output_dir: str = None
) -> Dict:
    """
    Multiclass SVC with grid search cross-validation.
    """
    logger.info("Training SVC classifier (multiclass)...")

    X_scaled, scaler = scale_data(X)

    if param_grid is None:
        param_grid = {
            'C': [0.1, 1, 10],
            'gamma': ['scale', 'auto', 0.01],
            'kernel': ['rbf']
        }

    svc = SVC(probability=True, random_state=42, decision_function_shape='ovr')
    grid_search = GridSearchCV(
        svc, 
        param_grid, 
        cv=cv_folds, 
        scoring='accuracy', 
        n_jobs=-1, 
        refit=True
    )
    grid_search.fit(X_scaled, y)
    best_svc = grid_search.best_estimator_
    logger.info(f"SVC best params: {grid_search.best_params_}")

    train_pred = best_svc.predict(X_scaled)
    train_acc = accuracy_score(y, train_pred)
    logger.info(f"SVC (train) accuracy: {train_acc:.4f}")

    cm = confusion_matrix(y, train_pred)

    results = {
        'model': best_svc,
        'scaler': scaler,
        'best_params': grid_search.best_params_,
        'train_acc': train_acc,
        'confusion_matrix': cm,
        'class_report': classification_report(y, train_pred, output_dict=True),
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(
            cm,
            class_labels=['Shuffled','Random','Real'],
            title='SVC Confusion Matrix',
            output_path=os.path.join(output_dir, 'svc_confusion_matrix.png')
        )
        cr_df = pd.DataFrame(results['class_report']).transpose()
        cr_df.to_csv(os.path.join(output_dir, 'svc_classification_report.csv'))

    return results


def train_symbolic_classifier(
    X_real: pd.DataFrame,
    X_fake: pd.DataFrame,
    niterations: int = 50,
    output_dir: str = None
) -> Dict:
    """
    Binary symbolic classification (Real=1 vs. Fake=0).
    """
    logger.info("Training Symbolic Classifier for real vs. not real (binary).")

    X_real_cp = X_real.copy()
    X_fake_cp = X_fake.copy()
    X_real_cp['label'] = 1
    X_fake_cp['label'] = 0

    X_bin = pd.concat([X_real_cp, X_fake_cp], axis=0)
    y_bin = X_bin.pop('label').values  

    X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(
        X_bin, y_bin, test_size=0.2, random_state=42, stratify=y_bin
    )

    X_train_scaled, X_test_scaled, scaler = scale_data(X_train_bin, X_test_bin)

    model = PySRRegressor(
        model_selection="best",
        niterations=niterations,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "log", "sqrt"],
        populations=20,
        population_size=50,
        maxsize=30,
        verbosity=1,
        select_k_features=10,
        parallelism="multithreading"
    )
    model.fit(X_train_scaled, y_train_bin, variable_names=list(X_train_bin.columns))

    train_pred_cont = model.predict(X_train_scaled)
    train_pred_label = (train_pred_cont >= 0.5).astype(int)
    train_acc = accuracy_score(y_train_bin, train_pred_label)

    test_pred_cont = model.predict(X_test_scaled)
    test_pred_label = (test_pred_cont >= 0.5).astype(int)
    test_acc = accuracy_score(y_test_bin, test_pred_label)

    cm_train = confusion_matrix(y_train_bin, train_pred_label)
    cm_test = confusion_matrix(y_test_bin, test_pred_label)

    logger.info(f"Symbolic Classifier: train_acc={train_acc:.4f}, test_acc={test_acc:.4f}")
    best_expr = model.sympy()

    results = {
        'model': model,
        'scaler': scaler,
        'best_expr': str(best_expr),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'confusion_matrix_train': cm_train,
        'confusion_matrix_test': cm_test,
        'classification_report_train': classification_report(y_train_bin, train_pred_label, output_dict=True),
        'classification_report_test': classification_report(y_test_bin, test_pred_label, output_dict=True)
    }

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_confusion_matrix(
            cm_train, 
            ['Fake','Real'], 
            title='Symbolic Classifier Train Confusion', 
            output_path=os.path.join(output_dir, 'symclf_confusion_train.png')
        )
        plot_confusion_matrix(
            cm_test, 
            ['Fake','Real'], 
            title='Symbolic Classifier Test Confusion', 
            output_path=os.path.join(output_dir, 'symclf_confusion_test.png')
        )
        with open(os.path.join(output_dir, 'symbolic_classifier_expr.txt'), 'w') as f:
            f.write(f"Best expression: {best_expr}\n")
        cr_train_df = pd.DataFrame(results['classification_report_train']).transpose()
        cr_train_df.to_csv(os.path.join(output_dir, 'symclf_report_train.csv'))
        cr_test_df = pd.DataFrame(results['classification_report_test']).transpose()
        cr_test_df.to_csv(os.path.join(output_dir, 'symclf_report_test.csv'))

    return results
