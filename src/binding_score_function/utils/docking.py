import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from bopep import Docker

logger = logging.getLogger(__name__)

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
    gpu_ids = docking_config.get("gpu_ids", ["0"])
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
            target_structure_path = os.path.join(template_pdb_dir, f"{pdb_code}.pdb")
            docker = Docker(docker_kwargs=this_config)
            docker.set_target_structure(
                target_structure_path=target_structure_path,
                strip_template=True,
                get_first_model=True,
                keep_chains="A",
            )
            docker.dock_peptides([peptide_sequence])
            return True

        except Exception as e:
            logger.warning(f"Could not score existing docking for {pdb_code}: {e}")
            return False

    # Run tasks in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        tasks = [(i, task) for i, task in enumerate(docking_tasks)]
        for result in executor.map(process_task, tasks):
            if result:
                successful += 1

    logger.info(f"Completed docking: {successful}/{len(docking_tasks)} successful")
