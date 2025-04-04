"""
Main script to generate the benchmark dataset for the binding score function.
This script downloads PDB files, preprocesses them, and performs docking and scoring.
"""

import os
import logging
from typing import Dict
from dotenv import load_dotenv
import pandas as pd
import pyrosetta

from utils.preprocessing import process_pdbs
from utils.download_pdbs import download_pdbs
from utils.scoring import dock_and_score_all_pdbs

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

pyrosetta.init(options="-mute all")


def setup_paths() -> Dict[str, str]:
    """
    Set up and return all paths.
    Uses environment variables if available.

    Returns:
        Dictionary containing all path definitions
    """
    load_dotenv()
    base_dir = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")

    paths = {
        "base_dir": base_dir,
        "index_dir": os.path.abspath(
            os.path.join(base_dir, "inputs/pdbbind_index_files")
        ),
        "output_dir": os.path.abspath(
            os.path.join(base_dir, "outputs/binding_score_function")
        ),
    }

    paths["unprocessed_dir"] = os.path.join(paths["output_dir"], "0_unprocessed")
    paths["processed_dir"] = os.path.join(paths["output_dir"], "1_processed")
    paths["scored_dir"] = os.path.join(paths["output_dir"], "2_scored")

    for path_name, path_value in paths.items():
        if path_name.endswith("_dir") and not os.path.exists(path_value):
            os.makedirs(path_value, exist_ok=True)
            logger.info(f"Created directory: {path_value}")

    return paths


def chmod_dir(directory: str) -> None:
    """
    Recursively sets 777 permissions on all files and subdirectories.
    """
    import os

    for root, dirs, files in os.walk(directory):
        for d in dirs:
            os.chmod(os.path.join(root, d), 0o777)
        for f in files:
            os.chmod(os.path.join(root, f), 0o777)


def main() -> None:
    """
    Main function to run the complete pipeline.
    Always runs all steps without command line arguments.
    """
    dock = True
    # Set up paths
    paths = setup_paths()

    # Configuration for docking
    docking_config = {
        "num_models": 5,
        "num_recycles": 50,
        "recycle_early_stop_tolerance": 0.1,
        "amber": True,
        "num_relax": 1,
        "gpu_ids": ["2", "3"],
        "overwrite_results": False,
    }

    min_peptide_length = 7
    max_peptide_length = 40

    # Step 1: Download PDBs
    logger.info("Downloading PDB files...")
    download_pdbs(
        paths["index_dir"],
        paths["unprocessed_dir"],
        min_peptide_length,
        max_peptide_length,
    )
    logger.info("PDB download completed.")

    # Step 2: Preprocess PDBs
    logger.info("Preprocessing PDB files...")
    process_pdbs(paths["unprocessed_dir"], paths["processed_dir"])
    logger.info(
        f"Preprocessing completed. Number of PDB files: "
        f"{len(os.listdir(os.path.join(paths['processed_dir'], 'pdbs')))}"
    )

    # Ensure the processed pdbs.csv includes the new columns
    unprocessed_pdbs_csv = os.path.join(paths["unprocessed_dir"], "pdbs.csv")
    processed_pdbs_csv = os.path.join(paths["processed_dir"], "pdbs.csv")
    
    if os.path.exists(unprocessed_pdbs_csv) and os.path.exists(processed_pdbs_csv):
        unprocessed_df = pd.read_csv(unprocessed_pdbs_csv)
        processed_df = pd.read_csv(processed_pdbs_csv)
        
        # Make sure processed CSV has peptide_sequence and uniprot_id columns
        if 'peptide_sequence' in unprocessed_df.columns and 'peptide_sequence' not in processed_df.columns:
            processed_df = processed_df.merge(
                unprocessed_df[['PDB code', 'peptide_sequence', 'uniprot_id']], 
                on='PDB code', 
                how='left'
            )
            processed_df.to_csv(processed_pdbs_csv, index=False)
            logger.info("Updated processed pdbs.csv with peptide sequences and UniProt IDs")
    if dock:
        # Step 3: Dock and score
        logger.info("Docking and scoring peptides...")
        dock_and_score_all_pdbs(
            os.path.join(paths["processed_dir"], "pdbs"),
            paths["scored_dir"],
            docking_config,
        )
        logger.info("All peptides have been docked and scored.")

    # Final step: create and output merged CSV
    scores_file = os.path.join(paths["scored_dir"], "scores.csv")
    affinity_file = os.path.join(paths["unprocessed_dir"], "pdbs.csv")
    
    if os.path.exists(scores_file) and os.path.exists(affinity_file):
        scores_df = pd.read_csv(scores_file)
        affinity_df = pd.read_csv(affinity_file)

        scores_df["PDB code"] = scores_df["pdb_file"].str.replace(".pdb", "", regex=False)
        scores_and_affinity = scores_df.merge(affinity_df, on="PDB code", how="inner")

        output_merge_file = os.path.join(paths["scored_dir"], "scores_and_affinity.csv")
        scores_and_affinity.to_csv(output_merge_file, index=False)
        logger.info("Final merged CSV saved as %s", output_merge_file)

    logger.info("Pipeline completed successfully.")
    chmod_dir(paths["output_dir"])
    logger.info("All output files are now globally accessible.")


if __name__ == "__main__":
    main()
