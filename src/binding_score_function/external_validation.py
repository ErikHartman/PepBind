#!/usr/bin/env python3
"""Dock the prepared external cohort with BoPep's shared AlphaFold pipeline.

One call per receptor; BoPep distributes its peptides across the allocated GPUs.
The modeled receptor is the observed template chain, as in the Boltz pipeline.
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
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_models(processed, receptor_sequence, peptide_sequence):
    """Reject missing or stale models before extracting any scores."""
    from bopep.structure.parser import get_chain_sequences

    processed = Path(processed)
    if not (processed / "alphafold_metrics.json").is_file():
        raise FileNotFoundError(processed / "alphafold_metrics.json")
    for index in range(1, DOCKING_CONFIG["num_models"] + 1):
        model = processed / f"alphafold_model_{index}.pdb"
        if not model.is_file():
            raise FileNotFoundError(model)
        sequences = get_chain_sequences(str(model))
        if sequences.get("A") != receptor_sequence or sequences.get("B") != peptide_sequence:
            raise ValueError(f"AlphaFold model chains differ from the template receptor/peptide: {model}")


def dock_and_score(records, output_dir, gpu_ids=None, reference_sequences=None):
    from bopep import ComplexScorer, Docker
    from predictive_equation.interaction_confidence_fit import FEATURE_SCHEMAS

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "selected_complexes.csv", records)
    batches = defaultdict(list)
    for row in records:
        batches[row["target_id"]].append(row)
    if gpu_ids is None:
        gpu_ids = (os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3").split(",")
    gpu_ids = [str(gpu).strip() for gpu in gpu_ids]
    if not gpu_ids or not all(gpu_ids) or len(set(gpu_ids)) != len(gpu_ids):
        raise ValueError("GPU IDs must be nonempty and unique")
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
        docker = Docker(kwargs={"models": ["alphafold"], "output_dir": str(target_dir),
                                "gpu_ids": gpu_ids, **DOCKING_CONFIG})
        docker.set_target_structure(target_structure_path=str(template), keep_chains="A")
        # Preserve the four-character template filename required by ColabFold,
        # while naming output folders by the biological target.
        docker.target_name = target_id
        receptor_sequence = docker.target_sequence
        if reference_sequences is not None and receptor_sequence != reference_sequences[target_id]:
            raise ValueError(f"Template receptor differs from the saved Boltz receptor: {target_id}")
        for record in batch:
            processed = target_dir / "processed" / f"{target_id}_{record['peptide_sequence']}"
            if (processed / "alphafold_metrics.json").exists():
                # BoPep's cache check alone does not verify receptor identity.
                validate_models(processed, receptor_sequence, record["peptide_sequence"])
        try:
            docker.dock_sequences([row["peptide_sequence"] for row in batch])
        except Exception:
            logging.exception("Docking failed for target %s; recording individual outcomes", target_id)

        for record in batch:
            row = dict(record)
            row.update(modeled_target_sequence=receptor_sequence,
                       modeled_target_length=len(receptor_sequence),
                       receptor_query_policy="Shared BoPep Docker: observed template-chain sequence")
            row["chemistry_model"] = "Canonical uncapped chain; experimental terminal caps remain in metadata"
            processed = target_dir / "processed" / f"{target_id}_{row['peptide_sequence']}"
            try:
                validate_models(processed, receptor_sequence, row["peptide_sequence"])
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
    parser.add_argument("--gpu-ids", help="Comma-separated IDs; defaults to CUDA_VISIBLE_DEVICES")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    gpu_ids = args.gpu_ids.split(",") if args.gpu_ids else None
    return bool(dock_and_score(load_cohort(args.csv), args.output_dir, gpu_ids=gpu_ids))


if __name__ == "__main__":
    raise SystemExit(main())
