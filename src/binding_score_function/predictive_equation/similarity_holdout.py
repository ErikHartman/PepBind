"""Train, freeze and evaluate symbolic scores on a sequence-disjoint holdout.

The notebook creates and audits the split. This module never consults test rows
during fitting or equation selection. Frozen expressions can be scored without
importing PySR or starting Julia.
"""

from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, r2_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler
import sympy


FAMILIES = ("alphafold", "boltz", "both")
TASKS = ("regression", "classification")
METRICS = ("Pearson r", "Spearman rho", "AUROC", "AP")
BASELINES = tuple(
    f"{family}_{metric}"
    for family in ("alphafold", "boltz")
    for metric in ("iptm", "negative_pae", "negative_interface_dG")
)


def _versions():
    result = {"python": platform.python_version()}
    for package in ("numpy", "pandas", "scipy", "scikit-learn", "sympy", "pysr"):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = "not installed"
    return result


def _score_expression(features, specification):
    """Use the saved sklearn affine transform, including constant columns."""
    names = specification["features"]
    values = features.loc[:, names].to_numpy(dtype=float)
    scaled = values * np.asarray(specification["scale"]) + np.asarray(specification["offset"])
    symbols = [sympy.Symbol(name) for name in names]
    expression = sympy.sympify(
        specification["expression"], locals=dict(zip(names, symbols)), evaluate=False
    )
    unknown = expression.free_symbols - set(symbols)
    if unknown:
        raise ValueError(f"Frozen expression contains unknown variables: {unknown}")
    evaluate = sympy.lambdify(symbols, expression, modules="numpy", cse=False)
    with np.errstate(all="ignore"):
        prediction = np.asarray(evaluate(*scaled.T), dtype=float)
    # Constant equations are legitimate classifier/ranking baselines.
    return np.broadcast_to(prediction, (len(features),)).copy()


def fit_holdout_models(
    features, membership, affinities, feature_schemas, output_dir,
    niterations=100, seed=42,
):
    """Fit six searches and select equations using development observations only.

    Inputs are indexed by complex_filename. ``membership`` has split, kind and
    similarity_group columns. Regression uses real training rows and validation
    R²; classification uses real/decoy training labels and validation AUROC.
    All candidates must produce finite scores on every train/validation row,
    including decoys. Neither the composite nor its family pairing is tuned.

    The returned table lists selected equations. ``output_dir/frozen.json`` is
    the portable scoring artifact. Existing training/frozen artifacts are never
    overwritten, including artifacts from an interrupted fitting attempt.
    """
    output_dir = Path(output_dir)
    frozen_path = output_dir / "frozen.json"
    training_dir = output_dir / "training"
    if frozen_path.exists() or training_dir.exists():
        raise FileExistsError(
            f"Training artifacts already exist in {output_dir}; reuse frozen.json "
            "or choose a new, explicitly documented run directory."
        )
    if niterations < 1:
        raise ValueError("niterations must be positive")
    if not features.index.is_unique or not membership.index.is_unique or not affinities.index.is_unique:
        raise ValueError("Feature, membership and affinity indices must be unique")
    required = {"split", "kind", "similarity_group"}
    if not required.issubset(membership.columns):
        raise ValueError(f"membership requires columns {sorted(required)}")
    # Slice by membership before any feature or affinity examination. The caller
    # may also pass development-only inputs for a stronger physical separation.
    development = membership.loc[membership["split"].isin(["train", "val"])].copy()
    if development.empty or not set(development["split"]) == {"train", "val"}:
        raise ValueError("Both training and validation memberships are required")
    if development["similarity_group"].isna().any():
        raise ValueError("Development rows need similarity groups")
    if development.groupby("similarity_group")["split"].nunique().gt(1).any():
        raise ValueError("A similarity group crosses training and validation")
    if not development["kind"].isin(["real", "shuffle", "random"]).all():
        raise ValueError("Expected real, shuffle or random kinds")
    is_train = development["split"].eq("train").to_numpy()
    is_val = development["split"].eq("val").to_numpy()
    is_real = development["kind"].eq("real").to_numpy()
    y_affinity = np.full(len(development), np.nan)
    y_affinity[is_real] = affinities.loc[development.index[is_real]].to_numpy(dtype=float)
    if not np.isfinite(y_affinity[is_real]).all():
        raise ValueError("Every development real row needs a finite affinity")
    for label, mask in (("train", is_train), ("val", is_val)):
        if np.unique(is_real[mask]).size != 2:
            raise ValueError(f"{label} requires both real and decoy observations")
        if np.sum(mask & is_real) < 3 or np.unique(y_affinity[mask & is_real]).size < 2:
            raise ValueError(f"{label} requires at least three real observations with varied affinities")
    for family in FAMILIES:
        for task in TASKS:
            names = list(feature_schemas[family][task])
            if not names or len(set(names)) != len(names):
                raise ValueError(f"Invalid feature schema: {family}/{task}")
            values = features.loc[development.index, names].to_numpy(dtype=float)
            if not np.isfinite(values).all():
                raise ValueError(f"Nonfinite development features in {family}/{task}")

    # Lazy import makes predict/evaluate independent of Julia installation.
    from pysr import PySRRegressor

    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(exist_ok=False)
    settings = dict(
        niterations=int(niterations), populations=20, population_size=20,
        binary_operators=["+", "-", "*", "/"], unary_operators=[], maxsize=30,
        model_selection="best", random_state=int(seed), deterministic=True,
        parallelism="serial", input_stream="devnull", verbosity=1, progress=False,
    )
    frozen = {
        "format_version": 1, "versions": _versions(), "settings": settings,
        "selection": {
            "regression": "validation R2", "classification": "validation AUROC",
            "tie_break": ["lower complexity", "lower equation index"],
            "candidate_validity": "finite on every training and validation observation",
            "composite": "raw R times raw C; no clipping or extra fitting",
            "manuscript_product": "alphafold_regression * both_classification",
        },
        "models": {},
    }
    selected_rows = []
    for family in FAMILIES:
        for task in TASKS:
            key = f"{family}_{task}"
            folder = training_dir / family / task
            folder.mkdir(parents=True, exist_ok=False)
            names = list(feature_schemas[family][task])
            panel = features.loc[development.index, names]
            train_mask = is_train & is_real if task == "regression" else is_train
            val_mask = is_val & is_real if task == "regression" else is_val
            y = y_affinity if task == "regression" else is_real.astype(float)
            scaler = MinMaxScaler(clip=False).fit(panel.loc[train_mask])
            scaled = scaler.transform(panel)
            model = PySRRegressor(
                **settings, select_k_features=min(25, len(names)),
                output_directory=str(folder / "pysr"), run_id="search",
            )
            # PySR's feature selector may use NumPy's global random stream.
            rng_state = np.random.get_state()
            try:
                np.random.seed(seed)
                model.fit(scaled[train_mask], y[train_mask], variable_names=names)
            finally:
                np.random.set_state(rng_state)
            mask = np.asarray(model.selection_mask_, dtype=bool)
            if mask.shape != (len(names),):
                raise ValueError(f"Unexpected PySR feature-selection mask for {key}: {mask.shape}")
            pd.DataFrame({
                "feature": names, "selected": mask, "scale": scaler.scale_,
                "offset": scaler.min_, "data_min": scaler.data_min_,
                "data_max": scaler.data_max_,
            }).to_csv(folder / "scaling_params.csv", index=False)
            specification = {
                "family": family, "task": task, "features": names,
                "selected_feature_mask": mask.tolist(),
                "selected_features": [name for name, keep in zip(names, mask) if keep],
                "scale": scaler.scale_.tolist(), "offset": scaler.min_.tolist(),
                "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
                "n_train": int(train_mask.sum()), "n_validation": int(val_mask.sum()),
            }
            candidates = []
            predictions = {}
            for equation_index, row in model.equations_.iterrows():
                index = int(equation_index)
                candidate = {
                    "equation_index": index, "complexity": int(row["complexity"]),
                    "training_loss": float(row["loss"]), "equation": str(row["equation"]),
                    "expression": "", "valid_development": False,
                    "n_nonfinite": np.nan, "validation_metric": np.nan, "error": "",
                }
                try:
                    pred = np.asarray(model.predict(scaled, index=index), dtype=float)
                    pred = np.broadcast_to(pred, (len(panel),)).copy()
                    candidate["n_nonfinite"] = int((~np.isfinite(pred)).sum())
                    candidate["expression"] = str(model.sympy(index=index))
                    if not np.isfinite(pred).all():
                        candidate["error"] = "Nonfinite development predictions"
                    else:
                        # Verify the portable scorer before it becomes frozen.
                        portable = _score_expression(panel, dict(specification, expression=candidate["expression"]))
                        if not np.allclose(portable, pred, rtol=1e-5, atol=1e-6):
                            raise ValueError("Portable expression predictions differ from PySR predictions")
                        metric = (
                            r2_score(y[val_mask], pred[val_mask]) if task == "regression"
                            else roc_auc_score(y[val_mask], pred[val_mask])
                        )
                        candidate["validation_metric"] = float(metric)
                        candidate["valid_development"] = bool(np.isfinite(metric))
                        predictions[index] = pred
                except Exception as error:
                    candidate["error"] = f"{type(error).__name__}: {error}"
                candidates.append(candidate)
            candidates = pd.DataFrame(candidates)
            candidates.to_csv(folder / "candidates.csv", index=False)
            eligible = candidates.loc[candidates["valid_development"]].sort_values(
                ["validation_metric", "complexity", "equation_index"], ascending=[False, True, True]
            )
            if eligible.empty:
                raise RuntimeError(f"No valid {key} candidates; inspect {folder / 'candidates.csv'}")
            best = eligible.iloc[0]
            specification.update({
                "equation_index": int(best["equation_index"]),
                "complexity": int(best["complexity"]), "expression": best["expression"],
                "validation_metric": float(best["validation_metric"]),
            })
            frozen["models"][key] = specification
            pd.DataFrame({
                "complex_filename": development.index, "split": development["split"].to_numpy(),
                "kind": development["kind"].to_numpy(),
                "score": predictions[specification["equation_index"]],
            }).to_csv(folder / "selected_development_predictions.csv", index=False)
            selected_rows.append({
                "score_id": key, "family": family, "task": task,
                **{name: specification[name] for name in (
                    "equation_index", "complexity", "expression", "validation_metric", "n_train", "n_validation"
                )},
                "selection_metric": "R2" if task == "regression" else "AUROC",
            })
    selected = pd.DataFrame(selected_rows)
    selected.to_csv(output_dir / "selected_equations.csv", index=False)
    with frozen_path.open("x") as handle:
        json.dump(frozen, handle, indent=2, allow_nan=False)
        handle.write("\n")
    return selected


def predict_frozen(features, frozen_path):
    """Score a frame using frozen expressions and training-only scaling."""
    with Path(frozen_path).open() as handle:
        frozen = json.load(handle)
    if frozen.get("format_version") != 1:
        raise ValueError("Unsupported frozen score format")
    if not features.index.is_unique:
        raise ValueError("Feature indices must be unique")
    scores = pd.DataFrame(index=features.index)
    for key, specification in frozen["models"].items():
        scores[key] = _score_expression(features, specification)
    with np.errstate(all="ignore"):
        for family in FAMILIES:
            scores[f"{family}_product"] = scores[f"{family}_regression"] * scores[f"{family}_classification"]
        scores["manuscript_product"] = scores["alphafold_regression"] * scores["both_classification"]
    for family in ("alphafold", "boltz"):
        scores[f"{family}_iptm"] = features[f"{family}_iptm"]
        scores[f"{family}_negative_pae"] = -features[f"{family}_peptide_pae"]
        scores[f"{family}_negative_interface_dG"] = -features[f"{family}_interface_dG"]
    return scores


def _metric_values(scores, labels, affinity):
    real = labels == 1
    values = np.full((scores.shape[1], len(METRICS)), np.nan)
    varied_affinity = real.sum() >= 3 and np.unique(affinity[real]).size >= 2
    two_classes = np.unique(labels).size == 2
    for index, score in enumerate(scores.T):
        if varied_affinity and np.isfinite(score[real]).all() and np.unique(score[real]).size >= 2:
            values[index, 0] = pearsonr(affinity[real], score[real]).statistic
            values[index, 1] = spearmanr(affinity[real], score[real]).statistic
        if two_classes and np.isfinite(score).all():
            values[index, 2] = roc_auc_score(labels, score)
            values[index, 3] = average_precision_score(labels, score)
    return values


def evaluate_holdout(predictions, membership, affinities, n_bootstrap=1000, seed=2026):
    """Return long-form metrics and paired composite-minus-baseline differences.

    Whole similarity groups are resampled with replacement, using the same draw
    for every score. Percentile 95% intervals describe the fixed fitted models
    conditional on this test split; fitting variability is not included. Metrics
    requiring any nonfinite prediction are unavailable, without row removal.
    Undefined bootstrap replicates are counted and excluded from interval
    quantiles; no interval is supplied with fewer than two held-out groups.
    """
    if n_bootstrap < 0:
        raise ValueError("n_bootstrap must be nonnegative")
    if not membership.index.is_unique or not predictions.index.is_unique or not affinities.index.is_unique:
        raise ValueError("Membership, predictions and affinity indices must be unique")
    panel = membership.loc[membership["split"].eq("test")].copy() if "split" in membership else membership.copy()
    if panel.empty or panel["similarity_group"].isna().any():
        raise ValueError("Nonempty test membership with similarity groups is required")
    if not panel["kind"].isin(["real", "shuffle", "random"]).all():
        raise ValueError("Expected real, shuffle or random kinds")
    scores = predictions.loc[panel.index].to_numpy(dtype=float)
    names = predictions.columns.tolist()
    labels = panel["kind"].eq("real").to_numpy(dtype=int)
    real = labels == 1
    affinity = np.full(len(panel), np.nan)
    affinity[real] = affinities.loc[panel.index[real]].to_numpy(dtype=float)
    if not np.isfinite(affinity[real]).all():
        raise ValueError("Every real test observation requires a finite affinity")
    point = _metric_values(scores, labels, affinity)
    group_names = panel["similarity_group"].unique()
    group_rows = [np.flatnonzero(panel["similarity_group"].eq(group).to_numpy()) for group in group_names]
    draws = np.full((n_bootstrap, len(names), len(METRICS)), np.nan)
    rng = np.random.default_rng(seed)
    if len(group_rows) >= 2:
        for replicate in range(n_bootstrap):
            sampled = rng.integers(0, len(group_rows), size=len(group_rows))
            indices = np.concatenate([group_rows[index] for index in sampled])
            draw = _metric_values(scores[indices], labels[indices], affinity[indices])
            # A resample omitting failed rows must not rescue an unavailable score.
            draw[~np.isfinite(point)] = np.nan
            draws[replicate] = draw
    metric_rows = []
    for index, name in enumerate(names):
        for metric_index, metric in enumerate(METRICS):
            distribution = draws[:, index, metric_index]
            valid = distribution[np.isfinite(distribution)]
            interval = np.quantile(valid, [.025, .975]) if len(valid) >= 2 else [np.nan, np.nan]
            metric_rows.append({
                "score_id": name, "metric": metric, "estimate": point[index, metric_index],
                "ci_low": interval[0], "ci_high": interval[1], "n_valid_bootstrap": len(valid),
                "n_bootstrap": n_bootstrap, "n_affinity": int(real.sum()),
                "n_decoys": int((~real).sum()), "n_groups": len(group_names),
                "n_affinity_groups": panel.loc[real, "similarity_group"].nunique(),
                "n_nonfinite_affinity": int((~np.isfinite(scores[real, index])).sum()),
                "n_nonfinite_all": int((~np.isfinite(scores[:, index])).sum()),
            })
    comparisons = {}
    for family in FAMILIES:
        comparisons[f"{family}_product"] = [f"{family}_regression", f"{family}_classification", *BASELINES]
    comparisons["manuscript_product"] = ["alphafold_regression", "both_classification", *BASELINES]
    difference_rows = []
    for composite, baselines in comparisons.items():
        if composite not in names:
            continue
        composite_index = names.index(composite)
        for baseline in baselines:
            if baseline not in names:
                continue
            baseline_index = names.index(baseline)
            for metric_index, metric in enumerate(METRICS):
                differences = draws[:, composite_index, metric_index] - draws[:, baseline_index, metric_index]
                valid = differences[np.isfinite(differences)]
                interval = np.quantile(valid, [.025, .975]) if len(valid) >= 2 else [np.nan, np.nan]
                difference_rows.append({
                    "composite": composite, "baseline": baseline, "metric": metric,
                    "delta": point[composite_index, metric_index] - point[baseline_index, metric_index],
                    "ci_low": interval[0], "ci_high": interval[1],
                    "n_valid_bootstrap": len(valid), "n_bootstrap": n_bootstrap,
                    "n_groups": len(group_names),
                })
    return pd.DataFrame(metric_rows), pd.DataFrame(difference_rows)
