# main_regression.py
import os
import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from regression import (
    train_lasso,
    train_random_forest,
    train_svr,
    perform_symbolic_regression,
    compare_models
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    base_path = "/srv/data1/general/immunopeptides_data/"
    scores_path = os.path.join(base_path, "outputs/binding_score_function/4_processed_scores/")
    output_dir = "./plots/regression"
    os.makedirs(output_dir, exist_ok=True)

    # Load real docking data
    X_real = pd.read_csv(os.path.join(scores_path, "real_X_train.csv")).set_index("complex_filename")
    y_real = pd.read_csv(os.path.join(scores_path, "real_y_train.csv"))["pKd"].values

    # Train/test split
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )

    # Lasso
    lasso_results = train_lasso(
        X_train_reg, y_train_reg,
        X_test_reg, y_test_reg,
        output_dir=os.path.join(output_dir, "lasso")
    )

    # RF
    rf_results = train_random_forest(
        X_train_reg, y_train_reg,
        X_test_reg, y_test_reg,
        output_dir=os.path.join(output_dir, "rf")
    )

    # SVR
    svr_results = train_svr(
        X_train_reg, y_train_reg,
        X_test_reg, y_test_reg,
        output_dir=os.path.join(output_dir, "svr")
    )

    # Symbolic
    symbolic_results = perform_symbolic_regression(
        X_train_reg, y_train_reg,
        X_test_reg, y_test_reg,
        niterations=20,
        populations=20,
        population_size=20,
        output_dir=os.path.join(output_dir, "symbolic")
    )

    # Compare
    model_dict = {
        'Lasso': lasso_results,
        'Random Forest': rf_results,
        'SVR': svr_results,
        'Symbolic': symbolic_results
    }
    comparison_df = compare_models(model_dict, output_dir)

    if 'Test R²' in comparison_df.columns:
        best_idx = comparison_df['Test R²'].idxmax()
        best_model_name = comparison_df.loc[best_idx, 'Model']
        best_r2 = comparison_df.loc[best_idx, 'Test R²']
        logger.info(f"\nBest regression model: {best_model_name} (Test R²={best_r2:.4f})")
        if best_model_name == 'Symbolic':
            logger.info(f"Best symbolic expression: {symbolic_results['best_expr']}")

    logger.info("Regression pipeline complete!")
