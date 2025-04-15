# main_classification.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import logging

from classification import (
    train_logistic_regression_classifier,
    train_random_forest_classifier,
    train_svc_classifier,
    train_symbolic_classifier,
    plot_confusion_matrix
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def evaluate(model, X_test, y_test):
    """
    Evaluate the model on the test set.
    """
    X_test_scaled = model['scaler'].transform(X_test)
    test_pred = model['model'].predict(X_test_scaled)
    test_acc = accuracy_score(y_test, test_pred)
    logger.info(f"Test accuracy: {test_acc:.4f}")
    cm_test = confusion_matrix(y_test, test_pred)
    logger.info("\n" + classification_report(y_test, test_pred))
    return cm_test, test_pred

if __name__ == "__main__":
    # Example data
    base_path = "/srv/data1/general/immunopeptides_data/"
    scores_path = os.path.join(base_path, "outputs/binding_score_function/4_processed_scores/")
    output_dir = "./plots/classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_real = pd.read_csv(os.path.join(scores_path, "real_X_train.csv")).set_index("complex_filename")
    X_shuffled = pd.read_csv(os.path.join(scores_path, "shuffled_X_train.csv")).set_index("complex_filename")
    X_random = pd.read_csv(os.path.join(scores_path, "random_X_train.csv")).set_index("complex_filename")

    X_fake = pd.concat([X_shuffled, X_random])

    # 3-class labeling: shuffled=0, random=1, real=2
    X_fake = X_fake.copy()
    X_real_clf = X_real.copy()
    X_fake['label'] = 0
    X_real_clf['label'] = 1

    X_all_clf = pd.concat([X_fake, X_real_clf], axis=0)
    y_all_clf = X_all_clf.pop('label').values

    X_clf_train, X_clf_test, y_clf_train, y_clf_test = train_test_split(
        X_all_clf, y_all_clf, test_size=0.2, random_state=42, stratify=y_all_clf
    )

    # Train logistic regression
    logreg_results = train_logistic_regression_classifier(
        X_clf_train, y_clf_train, cv_folds=5,
        output_dir=os.path.join(output_dir, "logreg")
    )
    # Evaluate
    cm_test, test_pred = evaluate(logreg_results, X_clf_test, y_clf_test)
    logger.info("\n" + classification_report(y_clf_test, test_pred))
    plot_confusion_matrix(
        cm_test, 
        class_labels=['Shuffled','Random','Real'], 
        title='LogReg (Test) Confusion Matrix',
        output_path=os.path.join(output_dir, "logreg", "logreg_test_confusion.png")
    )

    # Train random forest
    rf_results = train_random_forest_classifier(
        X_clf_train, y_clf_train, cv_folds=5,
        output_dir=os.path.join(output_dir, "rf")
    )
    X_clf_test_scaled = rf_results['scaler'].transform(X_clf_test)
    test_pred = rf_results['model'].predict(X_clf_test_scaled)
    test_acc = accuracy_score(y_clf_test, test_pred)
    logger.info(f"RF (test) accuracy: {test_acc:.4f}")

    # Train SVC
    svc_results = train_svc_classifier(
        X_clf_train, y_clf_train, cv_folds=5,
        output_dir=os.path.join(output_dir, "svc")
    )
    X_clf_test_scaled = svc_results['scaler'].transform(X_clf_test)
    test_pred = svc_results['model'].predict(X_clf_test_scaled)
    logger.info(f"SVC (test) accuracy: {accuracy_score(y_clf_test, test_pred):.4f}")

    # Symbolic classification (binary: real vs. not real)
    symclf_results = train_symbolic_classifier(
        X_real,
        X_fake,
        niterations=50,
        output_dir=os.path.join(output_dir, "symbolic_binary")
    )

    logger.info("Classification pipeline complete!")
