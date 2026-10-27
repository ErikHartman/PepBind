import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[2]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


DATA_DIR = REPO_ROOT / "data" / "x_y_v2"
OUTPUT_DIR = REPO_ROOT / "plots_cv"
TASKS = ("regression", "classification")
SPLITS = ("train", "val", "test")
FOLDS = 3
RANDOM_STATE = 42
DROP_PREFIX = "boltz"
NITERATIONS = 100
POPULATIONS = 20
POPULATION_SIZE = 20
MODEL_SELECTION = "best"
SELECT_K_FEATURES = 25
SCALE_FEATURES = True

DEFAULT_DROPPED_FEATURES = [
    "intra_all_mean_rmsd",

]


def read_feature_file(data_dir: Path, kind: str, split: str) -> pd.DataFrame:
    path = data_dir / f"{kind}_X_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path}")

    df = pd.read_csv(path)
    columns_to_drop = [col for col in df.columns if col.startswith("Unnamed:")]
    columns_to_drop.append("receptor_contacts")
    df = df.drop(columns=columns_to_drop, errors="ignore")

    if "complex_filename" not in df.columns:
        raise ValueError(f"{path} does not contain a complex_filename column.")

    return df.set_index("complex_filename")


def read_target_file(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"real_y_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing target file: {path}")

    df = pd.read_csv(path)
    df = df.drop(columns=[col for col in df.columns if col.startswith("Unnamed:")], errors="ignore")
    duplicate_complex_columns = [
        col for col in df.columns if col.startswith("complex_filename.") or col == "complex_filename.1"
    ]
    df = df.drop(columns=duplicate_complex_columns, errors="ignore")

    required_columns = {"complex_filename", "pKd"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"{path} is missing required columns: {sorted(missing_columns)}")

    return df[["complex_filename", "pKd"]]


def drop_model_family_features(
    X: pd.DataFrame,
    prefix: str,
    extra_dropped_features: Iterable[str] = DEFAULT_DROPPED_FEATURES,
) -> pd.DataFrame:
    columns_to_drop = [col for col in X.columns if prefix and prefix in col]
    columns_to_drop.extend(extra_dropped_features)
    return X.drop(columns=columns_to_drop, errors="ignore")


def validate_features(X: pd.DataFrame, label: str) -> None:
    non_numeric = X.select_dtypes(exclude=[np.number]).columns.tolist()
    if non_numeric:
        raise ValueError(f"{label} contains non-numeric feature columns: {non_numeric}")

    nan_counts = X.isna().sum()
    nan_columns = nan_counts[nan_counts > 0]
    if not nan_columns.empty:
        raise ValueError(f"{label} contains NaNs after preprocessing:\n{nan_columns}")


def load_regression_data(
    data_dir: Path,
    splits: Iterable[str],
    drop_prefix: str,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    fold_frames = []
    metadata_frames = []

    for split in splits:
        X_split = read_feature_file(data_dir, "real", split)
        y_split = read_target_file(data_dir, split).set_index("complex_filename")
        joined = X_split.join(y_split["pKd"], how="inner")

        if len(joined) != len(X_split):
            missing = sorted(set(X_split.index) - set(joined.index))
            raise ValueError(
                f"Could not match all regression targets for split {split}; "
                f"{len(missing)} complexes are missing pKd values."
            )

        metadata_frames.append(
            pd.DataFrame(
                {
                    "complex_filename": joined.index,
                    "source_split": split,
                    "data_type": "real",
                }
            )
        )
        fold_frames.append(joined)

    combined = pd.concat(fold_frames, axis=0)
    metadata = pd.concat(metadata_frames, axis=0, ignore_index=True)
    y = combined.pop("pKd").values
    X = drop_model_family_features(combined, prefix=drop_prefix)
    validate_features(X, "Regression features")
    return X, y, metadata


def load_classification_data(
    data_dir: Path,
    splits: Iterable[str],
    drop_prefix: str,
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    frames = []
    metadata_frames = []

    for split in splits:
        for kind, label in [("real", 1), ("shuffle", 0), ("random", 0)]:
            X_split = read_feature_file(data_dir, kind, split)
            X_split = X_split.copy()
            X_split["label"] = label
            frames.append(X_split)

            metadata_frames.append(
                pd.DataFrame(
                    {
                        "complex_filename": X_split.index,
                        "source_split": split,
                        "data_type": kind,
                    }
                )
            )

    combined = pd.concat(frames, axis=0, sort=False)
    metadata = pd.concat(metadata_frames, axis=0, ignore_index=True)
    y = combined.pop("label").values
    X = drop_model_family_features(combined, prefix=drop_prefix)
    validate_features(X, "Classification features")
    return X, y, metadata


def symbolic_kwargs() -> Dict:
    return {
        "niterations": NITERATIONS,
        "populations": POPULATIONS,
        "population_size": POPULATION_SIZE,
        "model_selection": MODEL_SELECTION,
        "select_k_features": SELECT_K_FEATURES,
        "scale_features": SCALE_FEATURES,
    }


def save_fold_assignments(
    output_path: Path,
    metadata: pd.DataFrame,
    fold_ids: np.ndarray,
    y: np.ndarray,
    target_column: str,
) -> None:
    assignments = metadata.copy()
    assignments["fold"] = fold_ids
    assignments[target_column] = y
    assignments.to_csv(output_path, index=False)


def selected_row(equations: pd.DataFrame, metric: str) -> Dict:
    if equations.empty or metric not in equations.columns or equations[metric].isna().all():
        return {}
    return equations.loc[equations[metric].idxmax()].to_dict()


def save_metric_summary(metrics: pd.DataFrame, output_path: Path) -> None:
    numeric_metrics = metrics.select_dtypes(include=[np.number])
    numeric_metrics = numeric_metrics.drop(
        columns=["fold", "n_train", "n_val", "selected_equation_index"],
        errors="ignore",
    )
    summary = numeric_metrics.agg(["mean", "std", "min", "max"]).T.reset_index()
    summary = summary.rename(columns={"index": "metric"})
    summary.to_csv(output_path, index=False)


def run_regression_cv() -> None:
    from regression import perform_symbolic_regression

    output_dir = OUTPUT_DIR / "regression"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_regression_data(DATA_DIR, SPLITS, DROP_PREFIX)
    logger.info("Regression CV data: X=%s, y=%s, splits=%s", X.shape, y.shape, SPLITS)

    splitter = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(y), -1, dtype=int)
    summaries = []
    final_equations = []
    all_equations = []
    predictions = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        logger.info("Starting regression fold %s/%s", fold, FOLDS)
        fold_ids[val_idx] = fold

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        results = perform_symbolic_regression(
            X_train,
            y_train,
            X_val,
            y_val,
            **symbolic_kwargs(),
        )

        equations = results["all_equations"].copy()
        equations.insert(0, "fold", fold)
        equations.to_csv(output_dir / f"fold_{fold}_all_equations.csv", index=False)
        all_equations.append(equations)

        selected = selected_row(equations, "val_r2")
        summary = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "best_expr": results.get("best_expr"),
            "selected_equation_index": selected.get("equation_index"),
            "selected_complexity": selected.get("complexity"),
            "selected_loss": selected.get("loss"),
            "selected_score": selected.get("score"),
            "train_rmse": results.get("train_rmse"),
            "train_r2": results.get("train_r2"),
            "train_mae": results.get("train_mae"),
            "train_pearson_r": results.get("train_pearson_r"),
            "train_spearman_r": results.get("train_spearman_r"),
            "train_kendall_tau": results.get("train_kendall_tau"),
            "train_top_k_accuracy_true": results.get("train_top_k_accuracy_true"),
            "val_rmse": results.get("val_rmse"),
            "val_r2": results.get("val_r2"),
            "val_mae": results.get("val_mae"),
            "val_pearson_r": results.get("val_pearson_r"),
            "val_spearman_r": results.get("val_spearman_r"),
            "val_kendall_tau": results.get("val_kendall_tau"),
            "val_top_k_accuracy_true": results.get("val_top_k_accuracy_true"),
        }
        summaries.append(summary)
        final_equations.append(summary)

        predictions.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "complex_filename": X_val.index,
                    "y_true": y_val,
                    "symbolic_pred": results.get("val_pred"),
                }
            )
        )

    save_fold_assignments(output_dir / "fold_assignments.csv", metadata, fold_ids, y, "pKd")
    metrics_df = pd.DataFrame(summaries)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    save_metric_summary(metrics_df, output_dir / "metric_summary.csv")
    pd.DataFrame(final_equations).to_csv(output_dir / "final_equations_by_fold.csv", index=False)
    pd.concat(all_equations, axis=0, ignore_index=True).to_csv(
        output_dir / "all_equations_by_fold.csv", index=False
    )
    pd.concat(predictions, axis=0, ignore_index=True).to_csv(
        output_dir / "cv_predictions.csv", index=False
    )


def run_classification_cv() -> None:
    from classification import perform_symbolic_classification

    output_dir = OUTPUT_DIR / "classification"
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, metadata = load_classification_data(DATA_DIR, SPLITS, DROP_PREFIX)
    logger.info("Classification CV data: X=%s, y=%s, splits=%s", X.shape, y.shape, SPLITS)

    splitter = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)
    fold_ids = np.full(len(y), -1, dtype=int)
    summaries = []
    final_equations = []
    all_equations = []
    predictions = []

    for fold, (train_idx, val_idx) in enumerate(splitter.split(X, y), start=1):
        logger.info("Starting classification fold %s/%s", fold, FOLDS)
        fold_ids[val_idx] = fold

        X_train = X.iloc[train_idx]
        X_val = X.iloc[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        results = perform_symbolic_classification(
            X_train,
            y_train,
            X_val,
            y_val,
            **symbolic_kwargs(),
        )

        equations = results["all_equations"].copy()
        equations.insert(0, "fold", fold)
        equations.to_csv(output_dir / f"fold_{fold}_all_equations.csv", index=False)
        all_equations.append(equations)

        selected = selected_row(equations, "val_auc")
        summary = {
            "fold": fold,
            "n_train": len(train_idx),
            "n_val": len(val_idx),
            "best_expr": results.get("best_expr"),
            "selected_equation_index": selected.get("equation_index"),
            "selected_complexity": selected.get("complexity"),
            "selected_loss": selected.get("loss"),
            "selected_score": selected.get("score"),
            "train_acc": results.get("train_acc"),
            "train_f1": results.get("train_f1"),
            "train_auc": results.get("train_auc"),
            "val_acc": results.get("val_acc"),
            "val_f1": results.get("val_f1"),
            "val_auc": results.get("val_auc"),
        }
        summaries.append(summary)
        final_equations.append(summary)

        predictions.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "complex_filename": X_val.index,
                    "y_true": y_val,
                    "symbolic_pred": results.get("val_pred"),
                    "symbolic_proba": results.get("val_proba"),
                }
            )
        )

    save_fold_assignments(output_dir / "fold_assignments.csv", metadata, fold_ids, y, "label")
    metrics_df = pd.DataFrame(summaries)
    metrics_df.to_csv(output_dir / "fold_metrics.csv", index=False)
    save_metric_summary(metrics_df, output_dir / "metric_summary.csv")
    pd.DataFrame(final_equations).to_csv(output_dir / "final_equations_by_fold.csv", index=False)
    pd.concat(all_equations, axis=0, ignore_index=True).to_csv(
        output_dir / "all_equations_by_fold.csv", index=False
    )
    pd.concat(predictions, axis=0, ignore_index=True).to_csv(
        output_dir / "cv_predictions.csv", index=False
    )


def main() -> None:
    if FOLDS < 2:
        raise ValueError("--folds must be at least 2.")

    logger.info("Writing outputs to %s", OUTPUT_DIR)
    logger.info("Using data from %s", DATA_DIR)
    logger.info("Dropping features containing %r", DROP_PREFIX)
    if "regression" in TASKS:
        run_regression_cv()
    if "classification" in TASKS:
        run_classification_cv()


if __name__ == "__main__":
    main()
