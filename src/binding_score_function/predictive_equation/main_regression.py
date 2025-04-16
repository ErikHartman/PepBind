import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from regression import (
    train_lasso,
    train_random_forest as train_rf,
    train_svr,
    perform_symbolic_regression as train_symbolic,
)

from plotting import (
    plot_regression_scatter,
    plot_feature_importances,
    plot_model_comparison,
    plot_symbolic_complexity_tradeoff,
)

if __name__ == "__main__":
    base_path = "/srv/data1/general/immunopeptides_data/"
    scores_path = os.path.join(
        base_path, "outputs/binding_score_function/4_processed_scores/"
    )
    output_dir = "./plots/regression"
    os.makedirs(output_dir, exist_ok=True)

    X_real = pd.read_csv(os.path.join(scores_path, "real_X_train.csv")).set_index(
        "complex_filename"
    )
    y_real = pd.read_csv(os.path.join(scores_path, "real_y_train.csv"))["pKd"].values

    print(f"X_real shape: {X_real.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.2, random_state=42
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    lasso_results = train_lasso(X_train, y_train, X_test, y_test)
    rf_results = train_rf(X_train, y_train, X_test, y_test)
    svr_results = train_svr(X_train, y_train, X_test, y_test)
    symb_results = train_symbolic(
        X_train,
        y_train,
        X_test,
        y_test,
        niterations=500,
        populations=50,
        population_size=100,
        model_selection="best",
        select_k_features=15,
        scale_features=True,
    )

    train_preds_df = pd.DataFrame(
        {
            "y_train": y_train,
            "lasso_pred": lasso_results["train_pred"],
            "rf_pred": rf_results["train_pred"],
            "svr_pred": svr_results["train_pred"],
            "symbolic_pred": symb_results["train_pred"],
        }
    )
    train_preds_df.to_csv(
        os.path.join(output_dir, "train_predictions.csv"), index=False
    )

    test_preds_df = pd.DataFrame(
        {
            "y_test": y_test,
            "lasso_pred": lasso_results["test_pred"],
            "rf_pred": rf_results["test_pred"],
            "svr_pred": svr_results["test_pred"],
            "symbolic_pred": symb_results["test_pred"],
        }
    )
    test_preds_df.to_csv(
        os.path.join(output_dir, "test_predictions.csv"), index=False
    )

    plot_regression_scatter(
        y_train,
        lasso_results["train_pred"],
        y_test,
        lasso_results.get("test_pred", None),
        model_name="Lasso",
        output_path=os.path.join(output_dir, "lasso_scatter.png"),
    )

    plot_regression_scatter(
        y_train,
        rf_results["train_pred"],
        y_test,
        rf_results.get("test_pred", None),
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_scatter.png"),
    )

    plot_regression_scatter(
        y_train,
        svr_results["train_pred"],
        y_test,
        svr_results.get("test_pred", None),
        model_name="SVR",
        output_path=os.path.join(output_dir, "svr_scatter.png"),
    )

    plot_regression_scatter(
        y_train,
        symb_results["train_pred"],
        y_test,
        symb_results.get("test_pred", None),
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_scatter.png"),
    )

    plot_feature_importances(
        lasso_results["coefficients"],
        model_name="Lasso",
        output_path=os.path.join(output_dir, "lasso_coef.png"),
    )

    plot_feature_importances(
        rf_results["feature_importance"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_importance.png"),
    )

    # Save top symbolic equations to CSV
    if 'top_equations' in symb_results:
        symb_results['top_equations'].to_csv(
            os.path.join(output_dir, "symbolic_regression_equations.csv"), index=False
        )
        print(f"Saved top symbolic regression equations to {output_dir}/symbolic_regression_equations.csv")
        
    # Save all symbolic equations with metrics
    if 'all_equations' in symb_results:
        symb_results['all_equations'].to_csv(
            os.path.join(output_dir, "symbolic_regression_all_equations.csv"), index=False
        )
        print(f"Saved all symbolic regression equations to {output_dir}/symbolic_regression_all_equations.csv")
        
        # Plot complexity vs accuracy tradeoff
        plot_symbolic_complexity_tradeoff(
            symb_results['all_equations'],
            metric='test_r2',
            model_type='regression',
            output_path=os.path.join(output_dir, "symbolic_regression_complexity_tradeoff.png"),
            lower_is_better=False
        )

    results_dict = {
        "Lasso": lasso_results,
        "RF": rf_results,
        "SVR": svr_results,
        "Symbolic": symb_results,
    }
    comparison = {
        "Model": [],
        "Train RMSE": [],
        "Train R²": [],
        "Test RMSE": [],
        "Test R²": [],
        "Test MAE": [],
    }

    for model_name, result in results_dict.items():
        comparison["Model"].append(model_name)
        comparison["Train RMSE"].append(result.get("train_rmse", np.nan))
        comparison["Train R²"].append(result.get("train_r2", np.nan))
        comparison["Test RMSE"].append(result.get("test_rmse", np.nan))
        comparison["Test R²"].append(result.get("test_r2", np.nan))
        comparison["Test MAE"].append(result.get("test_mae", np.nan))

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    plot_model_comparison(
        comparison_df, output_path=os.path.join(output_dir, "model_comparison.png")
    )

    print("Regression pipeline complete!")
