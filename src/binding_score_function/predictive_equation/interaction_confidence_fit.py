"""Fit complete symbolic candidate lists using training interactions/confidence.

The caller merges the former training and validation memberships before calling
this module. Only training rows are accepted, and no equation is selected here.
Saved candidates use a training-only affine transform and can be evaluated with
``similarity_holdout._score_expression`` without importing PySR or Julia.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

if __package__:
    from .similarity_holdout import _score_expression, _versions
else:
    from similarity_holdout import _score_expression, _versions


INTERACTION_SUFFIXES = (
    "interface_sasa", "interface_dG", "interface_delta_hbond_unsat",
    "packstat", "distance_score", "n_contacts",
)
CONFIDENCE_SUFFIXES = {
    "alphafold": (
        "iptm", "peptide_plddt", "interface_peptide_plddt", "peptide_pae",
        "ipsae_max", "ipsae_min",
    ),
    "boltz": (
        "iptm", "confidence_score", "complex_plddt", "complex_iplddt",
        "complex_pde", "complex_ipde", "ligand_iptm", "protein_iptm",
        "chain_0_ptm", "chain_1_ptm", "peptide_plddt",
        "interface_peptide_plddt", "peptide_pae", "peptide_pde",
        "ipsae_max", "ipsae_min",
    ),
}
FEATURE_SCHEMAS = {
    family: [f"{family}_{suffix}" for suffix in INTERACTION_SUFFIXES + confidence]
    for family, confidence in CONFIDENCE_SUFFIXES.items()
}
FEATURE_SCHEMAS["both"] = FEATURE_SCHEMAS["alphafold"] + FEATURE_SCHEMAS["boltz"]
TASKS = ("regression", "classification")


def fit_candidate_lists(
    features, membership, affinities, output_dir, niterations=100, seed=42, families=None,
):
    """Fit requested family searches and persist every final hall-of-fame equation.

    All inputs are indexed by ``complex_filename``. ``membership`` must contain
    only ``split == 'train'``, with ``kind`` and ``similarity_group`` columns.
    Features must have exactly those rows; affinities must have exactly the real
    training rows. Regression is fitted on real rows, classification on all rows
    with real/decoy targets 1/0. Each search has its own training-only scaler.
    By default ``families=None`` fits AlphaFold, Boltz and their combined feature
    set, each for both tasks (six searches). An explicit nonempty family list
    fits only that subset, for example ``families=['both']`` for two additional
    searches saved in a separate output directory. All 34 combined features are
    retained without feature selection.

    Return the dictionary atomically written to ``candidate_models.json``.
    ``models`` maps ``{family}_{task}`` to scaler/schema metadata and a complete
    ``candidates`` list. There is no selected equation. Training-panel finite
    diagnostics are reported without filtering or ranking candidates. Export
    mismatches fail the run. Existing or interrupted training artifacts cannot
    be overwritten; callers may load a completed JSON artifact explicitly.
    """
    output_dir = Path(output_dir)
    frozen_path = output_dir / "candidate_models.json"
    temporary_path = output_dir / ".candidate_models.json.tmp"
    training_dir = output_dir / "training"
    if any(path.exists() for path in (frozen_path, temporary_path, training_dir)):
        raise FileExistsError(
            f"Training artifacts already exist in {output_dir}; load "
            "candidate_models.json or choose a new documented run directory."
        )
    if not isinstance(niterations, (int, np.integer)) or niterations < 1:
        raise ValueError("niterations must be a positive integer")
    if families is None:
        families = tuple(FEATURE_SCHEMAS)
    elif isinstance(families, str):
        raise ValueError("families must be a nonempty sequence of distinct allowed family names")
    else:
        try:
            families = tuple(families)
        except TypeError as error:
            raise ValueError("families must be a sequence of allowed family names") from error
    if (
        not families or not all(isinstance(family, str) for family in families)
        or len(set(families)) != len(families)
        or any(family not in FEATURE_SCHEMAS for family in families)
    ):
        raise ValueError(f"families must contain distinct names from {list(FEATURE_SCHEMAS)}")
    if not features.index.is_unique or not membership.index.is_unique or not affinities.index.is_unique:
        raise ValueError("Feature, membership and affinity indices must be unique")
    required = {"split", "kind", "similarity_group"}
    if not required.issubset(membership.columns):
        raise ValueError(f"membership requires columns {sorted(required)}")
    if membership.empty or not membership["split"].eq("train").all():
        raise ValueError("Only nonempty training membership is accepted; merge validation first")
    if membership["similarity_group"].isna().any():
        raise ValueError("Training rows need similarity groups")
    if not membership["kind"].isin(["real", "shuffle", "random"]).all():
        raise ValueError("Expected real, shuffle or random kinds")
    if not features.columns.is_unique:
        raise ValueError("Feature columns must be unique")
    if len(features) != len(membership) or not features.index.isin(membership.index).all():
        raise ValueError("Feature rows must match training membership exactly")
    is_real = membership["kind"].eq("real").to_numpy()
    if np.unique(is_real).size != 2:
        raise ValueError("Training requires both real and decoy observations")
    real_ids = membership.index[is_real]
    if len(affinities) != len(real_ids) or not affinities.index.isin(real_ids).all():
        raise ValueError("Affinity rows must match real training membership exactly")
    y_affinity = affinities.loc[real_ids].to_numpy(dtype=float)
    if not np.isfinite(y_affinity).all():
        raise ValueError("Every real training row needs a finite affinity")
    if len(y_affinity) < 3 or np.unique(y_affinity).size < 2:
        raise ValueError("Training requires at least three real rows with varied affinities")
    for family in families:
        names = FEATURE_SCHEMAS[family]
        missing = set(names) - set(features.columns)
        if missing:
            raise ValueError(f"Missing allowed {family} features: {sorted(missing)}")
        values = features.loc[membership.index, names].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Nonfinite training features in {family}")

    settings = dict(
        niterations=int(niterations), populations=20, population_size=20,
        binary_operators=["+", "-", "*", "/"], unary_operators=[], maxsize=30,
        random_state=int(seed), deterministic=True, parallelism="serial",
        input_stream="devnull", verbosity=1, progress=False, select_k_features=None,
    )
    frozen = {
        "format_version": 1, "versions": _versions(), "settings": settings,
        "families": list(families),
        "candidate_policy": "Every final hall-of-fame row; no equation selection or filtering",
        "feature_policy": "Explicit interaction and confidence allowlist; no feature selection",
        "diagnostic_panel": "All training rows, including decoys; never used for candidate selection",
        "models": {},
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    training_dir.mkdir(exist_ok=False)
    membership.to_csv(training_dir / "membership.csv", index_label="complex_filename")
    print(f"Initializing PySR for {len(families) * len(TASKS)} training-only searches.", flush=True)
    # Lazy import keeps all portable scoring and mocked tests independent of Julia.
    from pysr import PySRRegressor

    for family in families:
        allowed_names = FEATURE_SCHEMAS[family]
        for task in TASKS:
            key = f"{family}_{task}"
            folder = training_dir / family / task
            folder.mkdir(parents=True, exist_ok=False)
            names = list(allowed_names)
            panel = features.loc[membership.index, names]
            fit_mask = is_real if task == "regression" else np.ones(len(panel), dtype=bool)
            y = y_affinity if task == "regression" else is_real.astype(float)
            scaler = MinMaxScaler(clip=False).fit(panel.loc[fit_mask])
            scaled = scaler.transform(panel)
            print(f"Fitting {key}: {fit_mask.sum()} rows, {len(names)} features.", flush=True)
            model = PySRRegressor(
                **settings, output_directory=str(folder / "pysr"), run_id="search",
            )
            rng_state = np.random.get_state()
            try:
                np.random.seed(seed)
                model.fit(scaled[fit_mask], y, variable_names=names)
            finally:
                np.random.set_state(rng_state)
            pd.DataFrame({
                "feature": names, "selected": True, "scale": scaler.scale_,
                "offset": scaler.min_, "data_min": scaler.data_min_,
                "data_max": scaler.data_max_,
            }).to_csv(folder / "scaling_params.csv", index=False)
            specification = {
                "family": family, "task": task, "features": names,
                "selected_feature_mask": [True] * len(names), "selected_features": names,
                "scale": scaler.scale_.tolist(), "offset": scaler.min_.tolist(),
                "data_min": scaler.data_min_.tolist(), "data_max": scaler.data_max_.tolist(),
                "n_train": int(fit_mask.sum()), "n_training_panel": len(panel),
                "candidates": [],
            }
            for equation_index, row in model.equations_.iterrows():
                index = int(equation_index)
                expression = str(model.sympy(index=index))
                pred = np.asarray(model.predict(scaled, index=index), dtype=float)
                pred = np.broadcast_to(pred, (len(panel),)).copy()
                portable = _score_expression(panel, dict(specification, expression=expression))
                finite = np.isfinite(pred)
                masks_match = all(
                    np.array_equal(operation(portable), operation(pred))
                    for operation in (np.isfinite, np.isnan, np.isposinf, np.isneginf)
                )
                values_match = np.allclose(portable[finite], pred[finite], rtol=1e-5, atol=1e-6)
                if not masks_match or not values_match:
                    raise ValueError(
                        f"Portable expression predictions differ from PySR for {key} "
                        f"equation {index}; refusing to freeze incomplete candidate lists"
                    )
                loss = float(row["loss"])
                # Strict JSON represents a nonfinite loss as null with its exact
                # diagnostic retained separately; the candidate itself is kept.
                candidate = {
                    "equation_index": index, "complexity": int(row["complexity"]),
                    "training_loss": loss if np.isfinite(loss) else None,
                    "training_loss_status": "finite" if np.isfinite(loss) else str(loss),
                    "equation": str(row["equation"]), "expression": expression,
                    "finite_training_panel": bool(finite.all()),
                    "n_nonfinite": int((~finite).sum()),
                    "n_nonfinite_real": int((~finite & is_real).sum()),
                    "n_nonfinite_decoy": int((~finite & ~is_real).sum()),
                    "training_status": "finite" if finite.all() else "nonfinite",
                    "exporter_verified": True,
                }
                specification["candidates"].append(candidate)
            if not specification["candidates"]:
                raise RuntimeError(f"PySR returned an empty candidate list for {key}")
            pd.DataFrame(specification["candidates"]).to_csv(folder / "candidates.csv", index=False)
            frozen["models"][key] = specification
            print(f"Saved {len(specification['candidates'])} candidates for {key}.", flush=True)

    with temporary_path.open("x") as handle:
        json.dump(frozen, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temporary_path.replace(frozen_path)
    return frozen
