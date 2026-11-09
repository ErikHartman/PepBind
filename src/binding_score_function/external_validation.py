#!/usr/bin/env python3
"""Dock the prepared external cohort with BoPep's AlphaFold backend and save scores.

One call per receptor; BoPep distributes its peptides across the allocated GPUs.
Terminal caps are recorded in the CSV but modeled as canonical uncapped chains.
No binding-site residues, experimental complex poses, or model fitting are used.
"""
import argparse
from collections import defaultdict
import csv
import logging
import math
import os
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path("/srv/data1/er8813ha/bopep/external_val")
DOCKING_CONFIG = {
    "num_models": 5,
    "num_recycles": 20,
    "recycle_early_stop_tolerance": 0.1,
    "amber": True,
    "num_relax": 1,
    "save_raw": True,
    "overwrite_results": False,
}


def load_cohort(csv_path):
    """Post-cutoff pairs excluding the same assayed receptor domains."""
    csv_path = Path(csv_path).resolve()
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = []
    for row in rows:
        if (row["temporal_scope"] == "AF2_and_Boltz2_temporal_candidate"
                and row["score_original_development_same_native_domain"] == "False"
                and row["score_revision2_fit_same_native_domain"] == "False"):
            row["target_pdb"] = str((csv_path.parent / row["target_structure"]).resolve())
            selected.append(row)
    if not selected:
        raise ValueError("No external pairs passed the cohort filter")
    return selected


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def dock_and_score(records, output_dir):
    from bopep import ComplexScorer
    from bopep.docking.alphafold_docker import AlphaFoldDocker
    from predictive_equation.interaction_confidence_fit import FEATURE_SCHEMAS

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selected_complexes.csv", records)
    batches = defaultdict(list)
    for row in records:
        batches[row["target_id"]].append(row)
    gpu_ids = [gpu.strip() for gpu in (os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3").split(",")]
    features = FEATURE_SCHEMAS["alphafold"]
    score_names = [name.replace("peptide_", "sequence_") for name in features]
    scorer, results = ComplexScorer(), []
    logging.info("Docking %d pairs against %d targets, using GPUs %s", len(records), len(batches), gpu_ids)

    for target_id, batch in batches.items():
        target_dir = output_dir / "docking" / target_id
        # ColabFold converts the four-character PDB template name to mmCIF.
        template = output_dir / "templates" / target_id / "targ.pdb"
        template.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(batch[0]["target_pdb"], template)
        docker = AlphaFoldDocker(output_dir=str(target_dir), gpu_ids=gpu_ids, **DOCKING_CONFIG)
        try:
            docker.dock(
                [row["peptide_sequence"] for row in batch],
                target_structure=str(template),
                target_sequence=batch[0]["target_sequence"],
                target_name=target_id,
            )
        except Exception:
            logging.exception("Docking failed for target %s; recording individual outcomes", target_id)

        for record in batch:
            row = dict(record)
            row["chemistry_model"] = "Canonical uncapped chain; experimental terminal caps remain in metadata"
            processed = target_dir / "processed" / f"{target_id}_{row['peptide_sequence']}"
            try:
                raw = scorer.score(scores_to_include=score_names, processed_dir=str(processed),
                                   sequence=row["peptide_sequence"], receptor_chain="A", sequence_chain="B")
                scores = raw[row["peptide_sequence"]]
                for feature, name in zip(features, score_names):
                    row[feature] = scores.get(name)
                missing = [name for name in features if row[name] is None or not math.isfinite(float(row[name]))]
                for metric in ("peptide_pae", "interface_dG"):
                    value = row.get(f"alphafold_{metric}")
                    row[f"alphafold_negative_{metric}"] = -float(value) if value is not None else None
                if missing:
                    raise ValueError(f"Missing or nonfinite scores: {missing}")
                row.update(status="complete", error="")
            except Exception as error:
                row.update(status="failed", error=str(error))
                logging.exception("Scoring failed for %s", row["complex_id"])
            results.append(row)
        write_csv(output_dir / "features.csv", results)

    failed = sum(row["status"] == "failed" for row in results)
    logging.info("Finished: %d complete, %d failed. Results: %s", len(results) - failed, failed, output_dir / "features.csv")
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "data/external_val/complexes.csv")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    return bool(dock_and_score(load_cohort(args.csv), args.output_dir))


if __name__ == "__main__":
    raise SystemExit(main())
