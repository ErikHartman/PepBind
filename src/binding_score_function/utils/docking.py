import os
import logging
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any
from bopep import Docker

logger = logging.getLogger(__name__)

def dock_complexes(
    processed_df: pd.DataFrame,
    docking_dir: str,
    docking_config: Dict[str, Any],
) -> None:
    """
    Dock peptides to protein templates and score the interactions.
    """
    docked_pdbs_dir = os.path.join(docking_dir, "docked_pdbs")
    os.makedirs(docked_pdbs_dir, exist_ok=True)

    docking_tasks = []
    for _, row in processed_df.iterrows():
        pdb_code = row["PDB code"]
        peptide_sequence = row.get("peptide_sequence")
        docking_tasks.append((pdb_code, peptide_sequence))

    logger.info(f"Found {len(docking_tasks)} complexes to dock and score")

    parallel_config = docking_config.copy()
    parallel_config["pdb_dir"] = docked_pdbs_dir
    gpu_ids = docking_config.get("gpu_ids", ["0"])
    num_gpus = len(gpu_ids)

    # Process a single docking task
    def process_task(args):
        idx, (pdb_code, peptide_sequence) = args
        gpu_id = gpu_ids[idx % num_gpus]

        # Configure for this specific GPU
        this_config = parallel_config.copy()
        this_config["gpu_ids"] = [gpu_id]

        logger.info(
            f"Processing {idx+1}/{len(docking_tasks)}: {pdb_code} on GPU {gpu_id}"
        )
        try:

            docker = Docker(docker_kwargs=this_config)
            docker.set_target_structure(
                target_structure_path=pdb_code,
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
