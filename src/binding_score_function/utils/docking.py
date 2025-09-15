import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from bopep import Docker
from Bio.Data import PDBData
from Bio.PDB import MMCIFParser, PDBParser


logger = logging.getLogger(__name__)

def _get_template_structure_path(template_dir: str, pdb_code: str) -> str:
    """
    Prefer a CIF template if present; otherwise fall back to PDB.
    """
    cif_path = os.path.join(template_dir, f"{pdb_code}.cif")
    if os.path.exists(cif_path):
        return cif_path
    pdb_path = os.path.join(template_dir, f"{pdb_code}.pdb")
    if os.path.exists(pdb_path):
        return pdb_path
    raise FileNotFoundError(f"No template found for {pdb_code} (.cif or .pdb) in {template_dir}")


def _select_longest_chain(structure_path: str):
    """Return ID of longest non-empty standard AA chain."""

    parser = MMCIFParser(QUIET=True) if structure_path.lower().endswith('.cif') else PDBParser(QUIET=True)
    structure = parser.get_structure("tmp", structure_path)
    model = next(structure.get_models())
    max_length = 0
    longest_chain = None
    three_to_one_dict = PDBData.protein_letters_3to1
    for chain in model:
        length = 0
        for r in chain:
            if r.id[0] == ' ':
                resname = r.resname.strip().upper()
                if resname in three_to_one_dict:
                    _ = three_to_one_dict[resname]
                    length += 1
        if length > max_length:
            longest_chain = chain.id
            max_length = length
    if longest_chain is None:
        raise ValueError(f"No valid standard amino acid chain found in {structure_path}")
    return longest_chain


def dock_complexes(
    template_pdb_dir: str,
    processed_df: pd.DataFrame,
    docking_config: Dict[str, Any],
    min_length: int = 7,
    max_length: int = 45,
) -> None:
    """
    Dock peptides to protein templates and score the interactions.
    """
    docking_tasks = []
    for _, row in processed_df.iterrows():
        pdb_code = row["pdb_code"]
        peptide_sequence = row.get("peptide_sequence")

        if peptide_sequence is None or pd.isna(peptide_sequence) or peptide_sequence == "":
            logger.warning(f"Skipping {pdb_code} - missing peptide sequence")
            continue

        if len(peptide_sequence) < min_length or len(peptide_sequence) > max_length:
            logger.warning(f"Skipping {pdb_code} - peptide length ({len(peptide_sequence)}) outside 7-40 range")
            continue

        docking_tasks.append((pdb_code, peptide_sequence))

    logger.info(f"Found {len(docking_tasks)} complexes to dock")

    parallel_config = docking_config.copy()
    gpu_ids = docking_config.get("gpu_ids") # No default, should be provided
    if not gpu_ids:
        raise ValueError("No GPU IDs provided in docking configuration")
    num_gpus = len(gpu_ids)

    # Process a single docking task
    def process_task(args):
        idx, (pdb_code, peptide_sequence) = args
        if peptide_sequence == '' or peptide_sequence is None or peptide_sequence == "nan":
            raise ValueError(f"Peptide sequence is empty for {idx} {pdb_code}")
        
        
        gpu_id = gpu_ids[idx % num_gpus]

        # Configure for this specific GPU
        this_config = parallel_config.copy()
        this_config["gpu_ids"] = [gpu_id]

        if not docking_config["overwrite_results"]:
            output_path = os.path.join(
                docking_config["output_dir"], f"{pdb_code}_{peptide_sequence}.pdb"
            )
            if os.path.exists(output_path):
                logger.info(f"Skipping {pdb_code}_{peptide_sequence}.pdb: already docked")
                return False

        logger.info(
            f"Processing {idx+1}/{len(docking_tasks)}: {pdb_code} {peptide_sequence} on GPU {gpu_id}"
        )
        try:
            target_structure_path = _get_template_structure_path(template_pdb_dir, pdb_code)
            keep_chains_arg = _select_longest_chain(target_structure_path)
            print(f"Selected chain {keep_chains_arg} for {pdb_code}")
            docker = Docker(kwargs=this_config)
            docker.set_target_structure(
                target_structure_path=target_structure_path,
                keep_chains=keep_chains_arg,
            )
            docker.dock_peptides([peptide_sequence])
            return True

        except Exception as e:
            logger.warning(f"Could not dock {pdb_code}: {e}")
            return False

    # Run tasks in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        tasks = [(i, task) for i, task in enumerate(docking_tasks)]
        for result in executor.map(process_task, tasks):
            if result:
                successful += 1

    logger.info(f"Completed docking: {successful}/{len(docking_tasks)} successful")
