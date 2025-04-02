import os
import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt

from pysr import PySRRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def preprocess_data(
    data_dir: str = None, fraction_threshold: float = 0.5
) -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """
    Preprocess the input data for symbolic regression.

    """
    if data_dir is None:
        data_dir = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")

    scores_path = os.path.join(
        data_dir, "databases/benchmark_data/new_run/2_scored/benchmark_scores.csv"
    )
    affinity_path = os.path.join(
        data_dir, "databases/benchmark_data/new_run/0_unprocessed/pdbs.csv"
    )

    # Read CSVs
    scores_df = pd.read_csv(scores_path)
    affinity_df = pd.read_csv(affinity_path)

    # Filter by binding site fraction
    scores_df = scores_df[scores_df["fraction_in_binding_site"] > fraction_threshold]

    # Merge with the affinity data
    scores_df["PDB code"] = scores_df["pdb_file"].str.replace(".pdb", "", regex=False)
    scores_df = scores_df.merge(affinity_df, on="PDB code", how="inner")
    scores_df = scores_df[~scores_df["Binding data"].str.contains("IC50", na=False)]

    # Convert binding data to numeric (M)
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
    scores_df["Binding data"] = binding_data[1] * binding_data[2].map(unit_conversion)

    # Prepare features and target
    target_column = "Binding data"
    non_feature_cols = ["pdb_file", "peptide_sequence", "Release year", target_column]
    feature_cols = (
        scores_df.select_dtypes(include=[np.number])
        .columns.difference(non_feature_cols)
        .tolist()
    )

    # pKd = -log10(Kd)
    y = -np.log10(scores_df[target_column].values)
    X = scores_df[feature_cols]

    # Drop rows with missing values in features or target
    mask = ~X.isnull().any(axis=1) & ~pd.Series(y, index=X.index).isna()
    X = X.loc[mask]
    y = y[mask]

    return X, y


def perform_symbolic_regression(
    X: pd.DataFrame,
    y: np.ndarray,
    niterations: int = 200,
    binary_operators: List[str] = None,
    unary_operators: List[str] = None,
) -> PySRRegressor:
    """
    Perform symbolic regression on the provided dataset using PySR.
    """
    if binary_operators is None:
        binary_operators = ["+", "-", "*", "/"]
    if unary_operators is None:
        unary_operators = ["square", "cube", "exp", "log", "abs", "sqrt"]

    model = PySRRegressor(
        model_selection="accuracy",
        niterations=niterations,
        binary_operators=binary_operators,
        unary_operators=unary_operators,
        populations=50,
        population_size=50,
        maxsize=50,
        verbosity=1,
    )
    model.fit(X, y)
    return model


def analyze_symbolic_models(
    model: PySRRegressor,
    X: pd.DataFrame,
    y: np.ndarray,
    num_equations: int = 20,
    save_figure: bool = True,
) -> pd.DataFrame:
    """
    Analyze the top symbolic equations and evaluate them on the test set.
    """
    equations = model.equations_.reset_index().rename(columns={"index": "eq_index"})
    equations = equations.sort_values(by="loss", ascending=True).reset_index(drop=True)

    print("Equations found (sorted by loss):")
    print(equations)

    results = []

    plt.figure(figsize=(20, 20))
    for rank, eq in equations.iterrows():
        if rank >= num_equations:
            break

        # 'eq_index' is the original row in model.equations_.
        original_idx = eq["eq_index"]
        y_pred = model.predict(X, original_idx)

        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        results.append(
            {
                "Equation": eq["equation"],
                "Complexity": eq["complexity"],
                "Loss": eq["loss"],
                "MSE": mse,
                "R²": r2,
            }
        )

        plt.subplot(4, 5, rank + 1)
        plt.scatter(y, y_pred, alpha=0.5)
        plt.plot([y.min(), y.max()], [y.min(), y.max()], "r--")
        plt.xlim(y.min(), y.max())
        plt.ylim(y.min(), y.max())
        plt.gca().set_aspect("equal", adjustable="box")
        plt.xlabel("Actual pKd")
        plt.ylabel("Predicted pKd")
        plt.title(
            f"Eq {rank + 1}: {eq['equation']}\nR² = {r2:.3f}, MSE = {mse:.3f}"
        )

    plt.tight_layout()
    if save_figure:
        plt.savefig("symbolic_regression_results.png")

    best_equations = pd.DataFrame(results)
    print("\nBest equations found:")
    print(best_equations)

    return best_equations



def run_symbolic_regression_analysis(
    data_dir: str = None,
    fraction_threshold: float = 0.5,
    test_size: float = 0.2,
    niterations: int = 200,
    num_equations: int = 5,
    random_state: int = 42,
) -> Tuple[PySRRegressor, pd.DataFrame]:
    """
    End-to-end pipeline for symbolic regression analysis:
    1. Preprocesses data from disk
    2. Splits data into train/test
    3. Trains a PySR model
    4. Analyzes and plots the top discovered equations
    """
    X, y = preprocess_data(data_dir, fraction_threshold)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    print(f"Training data shape: {X_train.shape}")

    print("Starting symbolic regression training...")

    model = perform_symbolic_regression(X_train, y_train, niterations=niterations)
    print("Analyzing symbolic regression models on test set...")
    best_equations = analyze_symbolic_models(
        model, X_test, y_test, num_equations=num_equations
    )

    return model, best_equations


if __name__ == "__main__":
    # Modify these values as needed
    fraction_threshold = 0.5
    iterations = 75
    num_equations = 20

    model, equations = run_symbolic_regression_analysis(
        data_dir=None,
        fraction_threshold=fraction_threshold,
        niterations=iterations,
        num_equations=num_equations,
    )

    # Show plots after saving
    plt.show()
