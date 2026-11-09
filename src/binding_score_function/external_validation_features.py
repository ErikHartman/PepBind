#!/usr/bin/env python3
"""Cache missing frozen-equation inputs from existing external AlphaFold models.

Run in the BoPep/PyRosetta environment. This only scores saved model_1 structures;
it never docks, fits a model, or changes the original validation results.
"""
import argparse
import csv
import hashlib
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FEATURES = Path("/srv/data1/er8813ha/bopep/external_val/features.csv")
OUTPUT = ROOT / "data/external_val/analysis/supplemental_features.csv"
SCORES = ("alphafold_rosetta_score", "instability_index", "hydrophobic_aa_percent", "uHrel")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect_features(features_path, output_path):
    """Score every complete, unique input pair and atomically write its cache."""
    from bopep import ComplexScorer

    features_path = Path(features_path).resolve()
    output_path = Path(output_path).resolve()
    with features_path.open(newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records or any(row.get("status") != "complete" for row in records):
        raise ValueError("Expected a nonempty features.csv with every row complete")
    identifiers = [row["complex_id"] for row in records]
    pairs = [(row["target_id"], row["peptide_sequence"]) for row in records]
    if len(set(identifiers)) != len(records) or len(set(pairs)) != len(records):
        raise ValueError("Complex IDs and target/peptide pairs must be unique")

    scorer = ComplexScorer()
    source_hash = sha256(features_path)
    rows = []
    for index, record in enumerate(records, start=1):
        target, sequence = record["target_id"], record["peptide_sequence"]
        processed = features_path.parent / "docking" / target / "processed" / f"{target}_{sequence}"
        model = processed / "alphafold_model_1.pdb"
        if not model.is_file():
            raise FileNotFoundError(model)
        values = scorer.score(
            scores_to_include=list(SCORES), processed_dir=str(processed),
            sequence=sequence, receptor_chain="A", sequence_chain="B",
        )[sequence]
        missing = [name for name in SCORES
                   if values.get(name) is None or not math.isfinite(float(values[name]))]
        if missing:
            raise ValueError(f"{record['complex_id']}: missing/nonfinite scores {missing}")
        rows.append({
            "complex_id": record["complex_id"], "target_id": target,
            "peptide_sequence": sequence, "model_path": str(model),
            "model_sha256": sha256(model), "features_sha256": source_hash,
            **{name: float(values[name]) for name in SCORES},
        })
        print(f"Supplemental features: {index}/{len(records)} {record['complex_id']}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(output_path)
    print(f"Saved {len(rows)} complete feature rows to {output_path}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    collect_features(args.features, args.output)


if __name__ == "__main__":
    main()
