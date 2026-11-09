#!/usr/bin/env python3
"""Add Boltz predictions and frozen-v2 scoring inputs for the 53 FP pairs.

BoPep distributes each receptor's peptide batch across the allocated GPUs, as in
the AlphaFold run. Existing predictions and features are retained. Terminal caps are
metadata only: both predictors model canonical uncapped peptide chains.
"""
import argparse
from collections import defaultdict
import csv
import hashlib
import json
import logging
import math
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
FEATURES = Path("/srv/data1/er8813ha/bopep/external_val/features.csv")
OUTPUT = ROOT / "data/external_val/analysis/boltz_features.csv"
CACHE = Path("/srv/data1/general/tmp/cache")
STUDY = "ASHI_2026_FP"
CLASSIFICATION_FEATURES = (
    "inter_model_rmsd", "delta_net_charge_frac", "isoelectric_point",
    "positively_charged_aa_percent", "negatively_charged_aa_percent", "gravy",
    "boltz_peptide_plddt", "boltz_chain_0_ptm", "boltz_confidence_score",
)
ADDITIONAL_CLASSIFICATION_FEATURES = (
    "boltz_complex_iplddt", "boltz_ipsae_max", "boltz_n_contacts",
)
REGRESSION_FEATURES = (
    "boltz_chain_1_ptm", "boltz_interface_sasa", "boltz_distance_score",
    "intra_boltz_mean_rmsd", "turn_fraction",
)
BOLTZ_BASELINE_FEATURES = (
    "boltz_iptm", "boltz_peptide_pae", "boltz_interface_dG",
)
ALL_FEATURES = tuple(dict.fromkeys(
    CLASSIFICATION_FEATURES + REGRESSION_FEATURES + BOLTZ_BASELINE_FEATURES
    + ADDITIONAL_CLASSIFICATION_FEATURES
))
DOCKING_CONFIG = {
    "recycling_steps": 20, "diffusion_samples": 5, "output_format": "pdb",
    "sampling_steps": 200, "step_scale": 1.638,
    "save_raw": True, "overwrite_results": False, "force": True, "threshold": 2,
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def api_score_name(name):
    """Map frozen training names to the current BoPep scorer API."""
    # Both names mean ProteinAnalysis.secondary_structure_fraction()[1].
    if name == "turn_fraction":
        return "loop_fraction"
    return name.replace("peptide_", "sequence_")


def processed_dir(features_path, record):
    target, peptide = record["target_id"], record["peptide_sequence"]
    return Path(features_path).parent / "docking" / target / "processed" / f"{target}_{peptide}"


def load_cohort(features_path):
    """Use the completed AF cohort's single study, retaining the plotting outlier."""
    with Path(features_path).open(newline="") as handle:
        records = [row for row in csv.DictReader(handle) if row["study_id"] == STUDY]
    if len(records) != 53 or len({row["target_id"] for row in records}) != 16:
        raise ValueError("Expected exactly 53 ASHI_2026_FP pairs across 16 targets")
    if len({row["complex_id"] for row in records}) != len(records):
        raise ValueError("External complex IDs must be unique")
    if len({(row["target_id"], row["peptide_sequence"]) for row in records}) != len(records):
        raise ValueError("External receptor/peptide pairs must be unique")
    for row in records:
        if row["status"] != "complete":
            raise ValueError(f"AlphaFold features incomplete: {row['complex_id']}")
        for path in (Path(row["target_pdb"]),
                     processed_dir(features_path, row) / "alphafold_model_1.pdb",
                     processed_dir(features_path, row) / "alphafold_metrics.json"):
            if not path.is_file():
                raise FileNotFoundError(path)
    return records


def gpu_ids_from_environment(value=None):
    ids = [item.strip() for item in (value or os.environ.get("CUDA_VISIBLE_DEVICES") or "0,1,2,3").split(",")]
    if not all(ids) or len(set(ids)) != len(ids):
        raise ValueError("GPU IDs must be a nonempty, unique comma-separated list")
    return ids


def prepare_templates(records, output_dir):
    """Give stripped native coordinates the sequence metadata Boltz requires."""
    import gemmi

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    templates = {}
    for record in records:
        target = record["target_id"]
        if target in templates:
            continue
        structure = gemmi.read_structure(record["target_pdb"])
        structure.setup_entities()
        for chain in structure[0]:
            polymer = chain.get_polymer()
            entity = structure.get_entity_of(polymer)
            # The extracted PDB has no SEQRES. Assign the observed template
            # sequence required by Boltz's mmCIF parser.
            entity.full_sequence = [residue.name for residue in polymer]
        structure.assign_label_seq_id()
        path = output_dir / f"{target}.cif"
        structure.make_mmcif_document().write_file(str(path))
        templates[target] = str(path.resolve())
    return templates


def get_docker_class():
    """Use the public BoPep Docker with its model configured to Boltz."""
    from bopep import Docker
    return Docker


def native_target_sequence(record):
    """The unmodified BoPep Boltz backend uses the longest template chain."""
    from bopep.structure.parser import get_chain_sequences

    sequences = get_chain_sequences(record["target_pdb"])
    if not sequences:
        raise ValueError(f"No native template sequence: {record['target_id']}")
    return max(sequences.values(), key=len)


def write_progress(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def validate_models(features_path, record):
    from bopep.structure.parser import get_chain_sequences

    folder = processed_dir(features_path, record)
    model = folder / "boltz_model_1.pdb"
    for index in range(1, DOCKING_CONFIG["diffusion_samples"] + 1):
        if not (folder / f"boltz_model_{index}.pdb").is_file():
            raise FileNotFoundError(folder / f"boltz_model_{index}.pdb")
    if not (folder / "boltz_metrics.json").is_file():
        raise FileNotFoundError(folder / "boltz_metrics.json")
    sequences = get_chain_sequences(str(model))
    if sequences.get("A") != native_target_sequence(record) or sequences.get("B") != record["peptide_sequence"]:
        raise ValueError(f"Boltz chains differ from native template/peptide sequences: {record['complex_id']}")
    return model


def dock_and_score(records, features_path, output, gpu_ids, score_only=False):
    """Resume method-specific docking, then score and atomically cache all rows."""
    from bopep import ComplexScorer

    features_path, output = Path(features_path).resolve(), Path(output).resolve()
    if features_path == output:
        raise ValueError("Boltz output must not overwrite the original AlphaFold feature table")
    source_hash = sha256(features_path)
    rows = []
    for record in records:
        af_model = processed_dir(features_path, record) / "alphafold_model_1.pdb"
        rows.append({
            **{key: record[key] for key in ("complex_id", "target_id", "peptide_sequence")},
            "target_sequence": record["target_sequence"],
            "modeled_target_sequence": "", "modeled_target_length": "",
            "receptor_query_policy": "Native BoPep Boltz: longest observed template-chain sequence",
            "features_sha256": source_hash,
            "alphafold_model_path": str(af_model), "alphafold_model_sha256": sha256(af_model),
            "boltz_model_path": "", "boltz_model_sha256": "", "boltz_metrics_sha256": "",
            **{feature: "" for feature in ALL_FEATURES},
            "boltz_status": "pending", "error": "",
        })
    write_progress(output, rows)
    row_by_id = {row["complex_id"]: row for row in rows}
    docker_class = None if score_only else get_docker_class()
    templates = {} if score_only else prepare_templates(records, features_path.parent / "templates_boltz")
    batches = defaultdict(list)
    for record in records:
        batches[record["target_id"]].append(record)
    for target, batch in batches.items():
        if not score_only:
            for record in batch:
                row_by_id[record["complex_id"]]["boltz_status"] = "docking"
            write_progress(output, rows)
            try:
                docker = docker_class(kwargs={
                    "models": ["boltz"],
                    "output_dir": str(features_path.parent / "docking" / target),
                    "gpu_ids": gpu_ids, "cache": os.environ.get("BOLTZ_CACHE", str(CACHE)),
                    **DOCKING_CONFIG,
                })
                docker.set_target_structure(target_structure_path=templates[target], keep_chains="A")
                docker.dock_sequences([record["peptide_sequence"] for record in batch])
            except Exception:
                logging.exception("Boltz docking failed for %s; checking individual pair outputs", target)
        for record in batch:
            identifier = record["complex_id"]
            try:
                validate_models(features_path, record)
                modeled = native_target_sequence(record)
                row_by_id[identifier].update(boltz_status="docked", modeled_target_sequence=modeled,
                                              modeled_target_length=len(modeled))
                logging.info("Boltz ready: %s", identifier)
            except Exception as error:
                row_by_id[identifier].update(boltz_status="failed", error=str(error))
                logging.exception("Boltz docking/validation failed for %s", identifier)
        write_progress(output, rows)

    # Collect all frozen-equation inputs and baseline metrics after docking.
    scorer = ComplexScorer()
    api_names = [api_score_name(name) for name in ALL_FEATURES]
    for record in records:
        row = row_by_id[record["complex_id"]]
        if row["boltz_status"] == "failed":
            continue
        try:
            folder = processed_dir(features_path, record)
            if sha256(row["alphafold_model_path"]) != row["alphafold_model_sha256"]:
                raise ValueError("AlphaFold structure changed during Boltz docking")
            values = scorer.score(
                scores_to_include=api_names, processed_dir=str(folder),
                sequence=record["peptide_sequence"], receptor_chain="A", sequence_chain="B",
            )[record["peptide_sequence"]]
            missing = [name for name in api_names
                       if values.get(name) is None or not math.isfinite(float(values[name]))]
            if missing:
                raise ValueError(f"Missing/nonfinite score inputs: {missing}")
            row.update({name: float(values[api]) for name, api in zip(ALL_FEATURES, api_names)})
            model = folder / "boltz_model_1.pdb"
            row.update(boltz_model_path=str(model), boltz_model_sha256=sha256(model),
                       boltz_metrics_sha256=sha256(folder / "boltz_metrics.json"),
                       boltz_status="complete", error="")
            logging.info("Boltz scoring features complete: %s", record["complex_id"])
        except Exception as error:
            row.update(boltz_status="failed", error=str(error))
            logging.exception("Boltz feature scoring failed for %s", record["complex_id"])
        write_progress(output, rows)
    if sha256(features_path) != source_hash:
        raise ValueError("Original AlphaFold feature table changed during this run")
    failed = sum(row["boltz_status"] != "complete" for row in rows)
    logging.info("Boltz finished: %d complete, %d failed; %s", len(rows) - failed, failed, output)
    return failed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--gpu-ids", help="Comma-separated IDs; defaults to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--score-only", action="store_true", help="Score existing models without docking")
    parser.add_argument("--dry-run", action="store_true", help="Read-only cohort/cache preflight")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = load_cohort(args.features)
    gpu_ids = gpu_ids_from_environment(args.gpu_ids)
    cache = Path(os.environ.get("BOLTZ_CACHE", str(CACHE)))
    if args.dry_run:
        print(json.dumps({
            "study_id": STUDY, "pairs": len(records), "targets": len({r["target_id"] for r in records}),
            "gpu_ids": gpu_ids, "batching": "Native BoPep, one batch per receptor",
            "complex_ids": [r["complex_id"] for r in records],
            "source_features_sha256": sha256(args.features), "output": str(args.output),
            "boltz_cache": str(cache), "checkpoint_available": (cache / "boltz2_conf.ckpt").is_file(),
            "existing_boltz_pairs": sum((processed_dir(args.features, r) / "boltz_model_1.pdb").is_file() for r in records),
            "docking_config": DOCKING_CONFIG,
        }, indent=2))
        return 0
    os.environ.setdefault("BOLTZ_CACHE", str(cache))
    return bool(dock_and_score(records, args.features, args.output, gpu_ids, args.score_only))


if __name__ == "__main__":
    raise SystemExit(main())
