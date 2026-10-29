"""Evaluate every frozen expression and within-family R × C pair."""

from itertools import product

import numpy as np
import pandas as pd

if __package__:
    from .interaction_confidence_fit import FEATURE_SCHEMAS
    from .similarity_holdout import METRICS, _metric_values, _score_expression
else:
    from interaction_confidence_fit import FEATURE_SCHEMAS
    from similarity_holdout import METRICS, _metric_values, _score_expression


def evaluate_candidates(models, features, membership, affinities):
    """Return candidate and baseline metrics on the saved test partition.

    Pearson uses real complexes; AUROC uses real complexes and decoys. Undefined
    metrics remain NaN. All six model lists and every R × C pairing are retained.
    """
    families = {"alphafold": "AF", "boltz": "Boltz", "both": "Both"}
    tasks = {"regression": "R", "classification": "C"}
    assert all(frame.index.is_unique for frame in (features, membership, affinities))
    assert set(features.index) == set(membership.index)
    assert set(affinities.index) == set(membership.index[membership.kind.eq("real")])
    assert set(membership.split) == {"train", "test"}
    assert membership.kind.isin(["real", "shuffle", "random"]).all()
    assert membership.similarity_group.notna().all()
    assert membership.groupby("similarity_group").split.nunique().eq(1).all()
    assert set(models["models"]) == {f"{family}_{task}" for family in families for task in tasks}
    test = membership.loc[membership.split.eq("test")]
    features = features.loc[test.index]
    labels = test.kind.eq("real").to_numpy(dtype=int)
    affinity = affinities.reindex(test.index).to_numpy()
    assert set(labels) == {0, 1} and np.isfinite(affinity[labels == 1]).all()

    predictions, metadata = {}, []
    for family, prefix in families.items():
        groups = []
        for task, kind in tasks.items():
            specification = models["models"][f"{family}_{task}"]
            assert specification["features"] == FEATURE_SCHEMAS[family]
            group = []
            for candidate in specification["candidates"]:
                name = f"{prefix}-{kind}{candidate['equation_index']}"
                assert name not in predictions
                predictions[name] = _score_expression(
                    features, dict(specification, expression=candidate["expression"]))
                metadata.append((name, family, kind))
                group.append(name)
            assert group
            groups.append(group)
        for regression, classification in product(*groups):
            name = f"{regression} × {classification}"
            with np.errstate(all="ignore"):
                predictions[name] = predictions[regression] * predictions[classification]
            metadata.append((name, family, "R × C"))

    scores = pd.DataFrame(predictions, index=test.index)
    estimates = pd.DataFrame(_metric_values(scores.to_numpy(), labels, affinity), columns=METRICS)
    metrics = pd.concat([pd.DataFrame(metadata, columns=["candidate_id", "family", "kind"]),
                         estimates], axis=1).melt(
        id_vars=["candidate_id", "family", "kind"], var_name="metric", value_name="estimate")
    metrics = metrics.loc[
        (metrics.metric.eq("Pearson r") & metrics.kind.isin(["R", "R × C"]))
        | (metrics.metric.eq("AUROC") & metrics.kind.isin(["C", "R × C"]))].reset_index(drop=True)

    baseline_rows = []
    for family in ["alphafold", "boltz"]:
        for column, sign, label in [("iptm", 1, "ipTM"), ("peptide_pae", -1, "−peptide pAE"),
                                    ("interface_dG", -1, "−interface dG")]:
            score = sign * features[f"{family}_{column}"].to_numpy()[:, None]
            values = _metric_values(score, labels, affinity)[0]
            for index, metric in [(0, "Pearson r"), (2, "AUROC")]:
                baseline_rows.append((family, f"{families[family]} {label}", metric, values[index]))
    baselines = pd.DataFrame(baseline_rows, columns=["family", "baseline", "metric", "estimate"])
    baselines = pd.concat([baselines, baselines.assign(family="both")], ignore_index=True)
    return metrics, baselines
