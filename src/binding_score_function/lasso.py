import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

import dotenv
from adalasso import AdaptiveLasso


def ensure_plot_dir(base_dir="plots", subdir=None):
    """
    Ensure plot directory exists, creating it if needed.
    Returns the full path to the directory.
    """
    if subdir:
        plot_dir = os.path.join(base_dir, subdir)
    else:
        plot_dir = base_dir

    os.makedirs(plot_dir, exist_ok=True)
    return plot_dir


def preprocess_data(scores_csv, fraction_threshold=0.5):
    """
    Loads, merges, and preprocesses the data to extract features (X) and target (y = pKd).
    Removes any entries with 'IC50' and extracts numeric Kd from data such as '=25nM'.
    Only keeps rows where 'fraction_in_binding_site' > fraction_threshold.
    """
    scores_df = pd.read_csv(scores_csv)

    scores_df["PDB code"] = scores_df["pdb_file"].str.replace(".pdb", "", regex=False)
    scores_df = scores_df[~scores_df["Binding data"].str.contains("IC50", na=False)]
    binding_data = scores_df["Binding data"].str.extract(
        r"([=<>]?)(\d+\.?\d*)([a-zA-Z]*)"
    )
    binding_data[1] = pd.to_numeric(binding_data[1], errors="coerce")

    unit_conversion = {
        "fM": 1e-15,
        "pM": 1e-12,
        "nM": 1e-9,
        "uM": 1e-6,
        "mM": 1e-3,
        "M": 1,
    }
    scores_df["Kd_M"] = binding_data[1] * binding_data[2].map(unit_conversion)
    scores_df["pKd"] = -np.log10(scores_df["Kd_M"])

    scores_df = scores_df[scores_df["fraction_in_binding_site"] > fraction_threshold]
    non_feature_cols = [
        "pdb_file",
        "peptide_sequence",
        "Release year",
        "Binding data",
        "Kd_M",
        "pKd",
    ]
    feature_cols = scores_df.select_dtypes(include=[np.number]).columns.difference(
        non_feature_cols
    )

    X = scores_df[feature_cols].copy()
    y = scores_df["pKd"].values
    valid_mask = ~X.isnull().any(axis=1) & ~pd.isnull(y)
    X = X.loc[valid_mask]
    y = y[valid_mask]

    return X, y, feature_cols


def remove_outliers_iqr(X, y, threshold=3.0):
    df = X.copy()
    df["target"] = y
    df = df.reset_index(drop=True)

    keep_mask = np.ones(len(df), dtype=bool)
    for col in df.columns:
        Q1 = df[col].quantile(0.1)
        Q3 = df[col].quantile(0.9)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        col_outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
        keep_mask = keep_mask & ~col_outliers

    df_clean = df[keep_mask]
    X_clean = df_clean.drop("target", axis=1)
    y_clean = df_clean["target"].values

    print(
        f"Removed {len(df) - len(df_clean)} outliers using IQR with threshold={threshold}."
    )
    return X_clean, y_clean, keep_mask


def cross_validate_adaptive_lasso(X, y, alpha_list, gamma_list, n_folds=5):
    """
    Does a grid search over alpha_list and gamma_list for Adaptive Lasso,
    returning a DataFrame with cross-validation results (R^2 and RMSE).
    """
    results = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    for alpha in alpha_list:
        for gamma in gamma_list:
            fold_r2 = []
            fold_rmse = []
            for tr_idx, val_idx in kf.split(X):
                X_train, X_val = X[tr_idx], X[val_idx]
                y_train, y_val = y[tr_idx], y[val_idx]

                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)

                model = AdaptiveLasso(alpha=alpha, gamma=gamma, max_iter=5000)
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_val_scaled)

                fold_r2.append(r2_score(y_val, preds))
                fold_rmse.append(np.sqrt(mean_squared_error(y_val, preds)))

            results.append(
                {
                    "alpha": alpha,
                    "gamma": gamma,
                    "mean_r2": np.mean(fold_r2),
                    "mean_rmse": np.mean(fold_rmse),
                }
            )
    return pd.DataFrame(results)


def cross_validate_random_forest(X, y):
    """
    Performs a simple grid search for a Random Forest regressor.
    Returns the fitted GridSearchCV object.
    """
    rf = RandomForestRegressor(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 5, 10],
        "min_samples_split": [2, 5],
    }
    gs = GridSearchCV(rf, param_grid, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X, y)
    return gs


def cross_validate_svr(X, y):
    """
    Simple grid search for an SVM regressor (SVR) with RBF kernel.
    Returns the fitted GridSearchCV object.
    """
    svr = SVR(kernel="rbf")
    param_grid = {
        "C": [1.0, 10.0, 100.0],
        "epsilon": [0.1, 0.01],
        "gamma": ["scale", 0.01, 0.1, 1.0],
    }
    gs = GridSearchCV(svr, param_grid, cv=5, scoring="r2", n_jobs=-1)
    gs.fit(X, y)
    return gs


def final_train_test_split_and_evaluate(
    X, y, model_func, model_params={}, test_size=0.2
):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = model_func(**model_params)
    model.fit(X_train_scaled, y_train)

    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)

    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

    return {
        "model": model,
        "scaler": scaler,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
    }


def plot_predictions_and_residuals(
    y_true, y_pred, model_name="Model", subdir="results"
):
    """
    Scatter plot of true vs. predicted and histogram of residuals.
    """
    plot_dir = ensure_plot_dir(subdir=subdir)

    # True vs Predicted
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, alpha=0.7)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    plt.title(f"{model_name}: True vs. Predicted pKd")
    plt.xlabel("True pKd")
    plt.ylabel("Predicted pKd")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_name}_true_vs_pred.png"))
    plt.close()


def plot_adalasso_coefficients(model, feature_cols, subdir="results"):
    """
    Bar plot of non-zero coefficients from the fitted Adaptive Lasso.
    """
    plot_dir = ensure_plot_dir(subdir=subdir)
    coefs = pd.Series(model.coef_, index=feature_cols)
    nonzero = coefs[coefs != 0].sort_values(ascending=False)

    plt.figure(figsize=(7, 6))
    plt.barh(nonzero.index[::-1], nonzero.values[::-1])
    plt.title("Adaptive Lasso Non-Zero Coefficients")
    plt.xlabel("Coefficient")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "adalasso_coefficients.png"))
    plt.close()


def plot_feature_importances(model, feature_cols, model_name="Model", subdir="results"):
    """
    Generic bar plot of feature importances for models that have 'feature_importances_'.
    """
    if not hasattr(model, "feature_importances_"):
        print(f"{model_name} does not expose feature_importances_. Skipping this plot.")
        return

    plot_dir = ensure_plot_dir(subdir=subdir)
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    importances = importances.sort_values(ascending=False)

    plt.figure(figsize=(7, 6))
    plt.barh(importances.index[::-1], importances.values[::-1])
    plt.title(f"{model_name} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f"{model_name}_feature_importances.png"))
    plt.close()


def run_analysis():
    dotenv.load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")
    scores_path = os.path.join(
        DATA_DIR, "databases/benchmark_data/new_run/2_scored/scores_and_affinity.csv"
    )

    X, y, feature_cols = preprocess_data(scores_path)
    print(f"Initial data shape: {X.shape}")

    X_clean, y_clean, keep_mask = remove_outliers_iqr(X, y, threshold=3.0)
    print(f"After outlier removal: {X_clean.shape}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_clean)

    alpha_list = [1e-3, 1e-2, 1e-1, 1.0, 5.0, 10.0]
    gamma_list = [1, 2, 3, 4]
    results_lasso = cross_validate_adaptive_lasso(
        X_scaled, y_clean, alpha_list, gamma_list, n_folds=5
    )
    best_lasso = results_lasso.loc[results_lasso["mean_r2"].idxmax()]

    print("\nAdaptive Lasso CV Results:")
    print(results_lasso)
    print(
        f"Best Lasso: R^2={best_lasso['mean_r2']:.3f}, alpha={best_lasso['alpha']}, gamma={best_lasso['gamma']}, RMSE={best_lasso['mean_rmse']:.3f}"
    )

    rf_gs = cross_validate_random_forest(X_scaled, y_clean)
    print("\nRandom Forest CV best params:")
    print(rf_gs.best_params_)
    print(f"Best RF mean CV R^2 = {rf_gs.best_score_:.3f}")

    svr_gs = cross_validate_svr(X_scaled, y_clean)
    print("\nSVR CV best params:")
    print(svr_gs.best_params_)
    print(f"Best SVR mean CV R^2 = {svr_gs.best_score_:.3f}")

    lasso_results = final_train_test_split_and_evaluate(
        X_clean.values,
        y_clean,
        model_func=AdaptiveLasso,
        model_params={
            "alpha": best_lasso["alpha"],
            "gamma": best_lasso["gamma"],
            "max_iter": 5000,
        },
        test_size=0.2,
    )
    print("\nFinal Adaptive Lasso Performance:")
    print(
        f"Train R^2={lasso_results['train_r2']:.3f}, Test R^2={lasso_results['test_r2']:.3f}"
    )
    print(
        f"Train RMSE={lasso_results['train_rmse']:.3f}, Test RMSE={lasso_results['test_rmse']:.3f}"
    )
    plot_predictions_and_residuals(
        lasso_results["y_test"],
        lasso_results["y_pred_test"],
        model_name="Adaptive_Lasso",
    )
    plot_adalasso_coefficients(lasso_results["model"], feature_cols)

    rf_results = final_train_test_split_and_evaluate(
        X_clean.values,
        y_clean,
        model_func=RandomForestRegressor,
        model_params=rf_gs.best_params_,
        test_size=0.2,
    )
    print("\nFinal Random Forest Performance:")
    print(
        f"Train R^2={rf_results['train_r2']:.3f}, Test R^2={rf_results['test_r2']:.3f}"
    )
    print(
        f"Train RMSE={rf_results['train_rmse']:.3f}, Test RMSE={rf_results['test_rmse']:.3f}"
    )
    plot_predictions_and_residuals(
        rf_results["y_test"], rf_results["y_pred_test"], model_name="Random_Forest"
    )
    plot_feature_importances(
        rf_results["model"], feature_cols, model_name="Random_Forest"
    )

    svr_results = final_train_test_split_and_evaluate(
        X_clean.values,
        y_clean,
        model_func=SVR,
        model_params=svr_gs.best_params_,
        test_size=0.2,
    )
    print("\nFinal SVR Performance:")
    print(
        f"Train R^2={svr_results['train_r2']:.3f}, Test R^2={svr_results['test_r2']:.3f}"
    )
    print(
        f"Train RMSE={svr_results['train_rmse']:.3f}, Test RMSE={svr_results['test_rmse']:.3f}"
    )
    plot_predictions_and_residuals(
        svr_results["y_test"], svr_results["y_pred_test"], model_name="SVR"
    )


if __name__ == "__main__":
    run_analysis()
