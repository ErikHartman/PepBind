#!/usr/bin/env python3
"""Rerun external AlphaFold inputs that differ from the saved Boltz receptors.

Create a new result tree, retain unaffected predictions, and refresh all dependent
feature caches. The original structures and checked-in analysis inputs are never
overwritten. Boltz structures are reused; its feature cache is rescored because
cross-predictor RMSD depends on the regenerated AlphaFold structures.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
FEATURES = Path("/srv/data1/er8813ha/bopep/external_val/features.csv")
BOLTZ_FEATURES = ROOT / "data/external_val/analysis/boltz_features.csv"
STUDY = "ASHI_2026_FP"
KEYS = ("complex_id", "target_id", "peptide_sequence")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def processed_dir(root, record):
    target, peptide = record["target_id"], record["peptide_sequence"]
    return Path(root) / "docking" / target / "processed" / f"{target}_{peptide}"


def unique_rows(rows, description):
    identifiers = [row["complex_id"] for row in rows]
    pairs = [(row["target_id"], row["peptide_sequence"]) for row in rows]
    if not rows or len(set(identifiers)) != len(rows) or len(set(pairs)) != len(rows):
        raise ValueError(f"{description} requires nonempty, unique complex IDs and pairs")


def inspect_inputs(features_path, boltz_features_path):
    """Read-only preflight; select mismatches from actual saved model chains."""
    from bopep.structure.parser import get_chain_sequences

    features_path, boltz_features_path = Path(features_path).resolve(), Path(boltz_features_path).resolve()
    records, boltz = read_rows(features_path), read_rows(boltz_features_path)
    unique_rows(records, "AlphaFold inputs")
    unique_rows(boltz, "Boltz cache")
    if any(row.get("status") != "complete" for row in records):
        raise ValueError("All source AlphaFold rows must be complete")
    study = {row["complex_id"]: row for row in records if row["study_id"] == STUDY}
    if {tuple(row[key] for key in KEYS) for row in study.values()} != {
        tuple(row[key] for key in KEYS) for row in boltz
    }:
        raise ValueError("Boltz cache must contain exactly the same ASHI study pairs")
    source_hash = sha256(features_path)
    selected_ids, references, changes = set(), {}, []
    for cached in boltz:
        record = study[cached["complex_id"]]
        if cached.get("boltz_status") != "complete" or cached.get("features_sha256") != source_hash:
            raise ValueError(f"Incomplete or stale Boltz cache: {record['complex_id']}")
        folder = processed_dir(features_path.parent, record)
        af_model, boltz_model = folder / "alphafold_model_1.pdb", folder / "boltz_model_1.pdb"
        for path_key, expected in [("alphafold_model_path", af_model), ("boltz_model_path", boltz_model)]:
            if Path(cached[path_key]).resolve() != expected.resolve():
                raise ValueError(f"Cache model path differs from source tree: {record['complex_id']}")
            if sha256(expected) != cached[path_key.replace("_path", "_sha256")]:
                raise ValueError(f"Cache model hash differs: {expected}")
        if sha256(folder / "boltz_metrics.json") != cached["boltz_metrics_sha256"]:
            raise ValueError(f"Boltz metrics changed: {record['complex_id']}")
        for index in range(1, 6):
            if not (folder / f"boltz_model_{index}.pdb").is_file():
                raise FileNotFoundError(folder / f"boltz_model_{index}.pdb")
        af_chains = get_chain_sequences(str(af_model))
        boltz_chains = get_chain_sequences(str(boltz_model))
        expected = cached["modeled_target_sequence"]
        native = get_chain_sequences(record["target_pdb"])
        if not native or max(native.values(), key=len) != expected or boltz_chains.get("A") != expected:
            raise ValueError(f"Boltz receptor differs from native template: {record['complex_id']}")
        if any(chains.get("B") != record["peptide_sequence"] for chains in (af_chains, boltz_chains)):
            raise ValueError(f"Saved peptide chain differs from intended input: {record['complex_id']}")
        if not af_chains.get("A"):
            raise ValueError(f"Missing AlphaFold receptor: {record['complex_id']}")
        target = record["target_id"]
        if target in references and references[target] != expected:
            raise ValueError(f"Multiple modeled receptor sequences for {target}")
        references[target] = expected
        if af_chains["A"] != expected:
            selected_ids.add(record["complex_id"])
            changes.append({"complex_id": record["complex_id"], "target_id": target,
                            "old_receptor_length": len(af_chains["A"]), "new_receptor_length": len(expected)})
    selected = [dict(row, modeled_target_sequence=references[row["target_id"]])
                for row in records if row["complex_id"] in selected_ids]
    for record in records:
        if not processed_dir(features_path.parent, record).is_dir():
            raise FileNotFoundError(processed_dir(features_path.parent, record))
    return records, selected, references, changes


def validate_output_root(output_dir, features_path, boltz_features_path):
    output_dir = Path(output_dir).resolve()
    for source in (Path(features_path).resolve().parent, Path(boltz_features_path).resolve().parent):
        if output_dir == source or output_dir in source.parents or source in output_dir.parents:
            raise ValueError("Output directory must be separate from both source data trees")
    if output_dir.exists():
        raise FileExistsError(f"Use a fresh output directory; refusing to overwrite {output_dir}")
    return output_dir


def write_manifest(path, manifest, status, **updates):
    manifest.update(status=status, updated_utc=datetime.now(timezone.utc).isoformat(), **updates)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def software_hashes():
    import bopep

    paths = [Path(__file__), *[Path(__file__).with_name(name) for name in (
        "external_validation.py", "external_validation_features.py", "external_validation_boltz.py")]]
    package = Path(bopep.__file__).resolve().parent
    paths.extend(package / name for name in (
        "docking/docker.py", "docking/alphafold_docker.py", "structure/parser.py",
        "scoring/complex_scorer.py"))
    return {str(path.resolve()): sha256(path) for path in paths}


def run(features_path, boltz_features_path, output_dir, gpu_ids, dry_run=False):
    from external_validation import DOCKING_CONFIG, dock_and_score, write_csv

    features_path, boltz_features_path = Path(features_path).resolve(), Path(boltz_features_path).resolve()
    output_dir = validate_output_root(output_dir, features_path, boltz_features_path)
    records, selected, references, changes = inspect_inputs(features_path, boltz_features_path)
    logging.info("Selected %d pairs across %d targets for AlphaFold rerun on GPUs %s",
                 len(selected), len({row["target_id"] for row in selected}), gpu_ids)
    for change in changes:
        logging.info("Matched-template input: %s (%d -> %d receptor residues)",
                     change["complex_id"], change["old_receptor_length"], change["new_receptor_length"])
    if dry_run:
        return changes
    if not selected:
        raise ValueError("No receptor-sequence mismatches require rerunning")
    manifest = {"source_features": str(features_path), "source_features_sha256": sha256(features_path),
                "source_boltz_features": str(boltz_features_path),
                "source_boltz_features_sha256": sha256(boltz_features_path),
                "selected_pairs": changes, "n_source_pairs": len(records), "gpu_ids": list(gpu_ids),
                "software_sha256": software_hashes(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_nodelist": os.environ.get("SLURM_JOB_NODELIST"),
                "alphafold_docking_config": DOCKING_CONFIG,
                "receptor_policy": "Shared BoPep Docker: observed native template-chain sequence",
                "boltz_policy": "Reuse structures; rescore features and cross-predictor RMSD"}
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path = output_dir / "run_manifest.json"
    write_manifest(manifest_path, manifest, "staging")
    try:
        selected_ids = {row["complex_id"] for row in selected}
        for record in records:
            def ignore_old_af(directory, names):
                return [name for name in names if name.startswith("alphafold_")]
            shutil.copytree(processed_dir(features_path.parent, record), processed_dir(output_dir, record),
                            ignore=ignore_old_af if record["complex_id"] in selected_ids else None)
        write_manifest(manifest_path, manifest, "docking_and_scoring_alphafold")
        failed = dock_and_score(selected, output_dir, gpu_ids=gpu_ids, reference_sequences=references)
        if failed:
            raise RuntimeError(f"{failed} AlphaFold pairs failed; partial rows retained in features.csv")
        updated = read_rows(output_dir / "features.csv")
        unique_rows(updated, "Regenerated AlphaFold rows")
        updates = {row["complex_id"]: row for row in updated}
        if set(updates) != selected_ids or any(row.get("status") != "complete" for row in updated):
            raise ValueError("Regenerated AlphaFold rows must exactly match the selected pairs")
        for original in records:
            if original["complex_id"] in updates:
                replacement = updates[original["complex_id"]]
                if any(replacement[key] != original[key] for key in KEYS + ("target_sequence", "pKd", "Kd_M")):
                    raise ValueError(f"Regeneration changed source labels or identities: {original['complex_id']}")
        write_csv(output_dir / "rerun_features.csv", updated)
        merged = [updates.get(row["complex_id"], row) for row in records]
        temporary = output_dir / "features.csv.tmp"
        write_csv(temporary, merged)
        temporary.replace(output_dir / "features.csv")
        from bopep.structure.parser import get_chain_sequences
        for record in merged:
            if record["study_id"] == STUDY:
                chains = get_chain_sequences(str(processed_dir(output_dir, record) / "alphafold_model_1.pdb"))
                if chains.get("A") != references[record["target_id"]] or chains.get("B") != record["peptide_sequence"]:
                    raise ValueError(f"Final AlphaFold chains mismatch: {record['complex_id']}")
        write_manifest(manifest_path, manifest, "refreshing_dependent_features")
        from external_validation_features import collect_features
        from external_validation_boltz import dock_and_score as rescore_boltz, load_cohort
        analysis = output_dir / "analysis"
        new_features = output_dir / "features.csv"
        collect_features(new_features, analysis / "supplemental_features.csv")
        if rescore_boltz(load_cohort(new_features), new_features, analysis / "boltz_features.csv", gpu_ids, score_only=True):
            raise RuntimeError("Boltz feature refresh failed; original structures were not redocked")
        if sha256(features_path) != manifest["source_features_sha256"] or sha256(boltz_features_path) != manifest["source_boltz_features_sha256"]:
            raise ValueError("Source input tables changed during the rerun")
        outputs = [new_features, analysis / "supplemental_features.csv", analysis / "boltz_features.csv"]
        write_manifest(manifest_path, manifest, "complete", output_sha256={str(path): sha256(path) for path in outputs})
    except Exception as error:
        write_manifest(manifest_path, manifest, "failed", error=str(error))
        raise
    return changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--boltz-features", type=Path, default=BOLTZ_FEATURES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-ids", help="Comma-separated GPU IDs; defaults to all visible GPUs")
    parser.add_argument("--dry-run", action="store_true", help="Validate and report inputs without writing or docking")
    args = parser.parse_args()
    from external_validation_boltz import gpu_ids_from_environment
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.features, args.boltz_features, args.output_dir, gpu_ids_from_environment(args.gpu_ids), args.dry_run)


if __name__ == "__main__":
    main()
