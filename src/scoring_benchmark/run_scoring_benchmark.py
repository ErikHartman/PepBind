import os
import logging
from bopep import Scorer
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb
import pandas as pd
from utils import remove_peptide_from_complex, compare_binding_site
import re

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def ensure_dir_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def run_benchmark(pdb_path):
    """
    Run a docking benchmark on the provided PDB structure and return the scoring results.

    Args:
        pdb_path (str): Path to the PDB file to benchmark

    Returns:
        dict: Dictionary containing scoring results or None if an error occurred
    """
    try:
        # Define output directories
        protein_template_dir = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs_erik/stripped_protein_templates"
        docked_peptides_dir = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs_erik/docked_peptides"

        # Ensure directories exist
        ensure_dir_exists(protein_template_dir)
        ensure_dir_exists(docked_peptides_dir)

        output_pdb_path = os.path.join(protein_template_dir, os.path.basename(pdb_path))

        logger.info(f"Processing PDB: {os.path.basename(pdb_path)}")

        # Extract peptide sequence
        peptide_sequence = extract_sequence_from_pdb(pdb_path, chain_id="B")
        if not peptide_sequence:
            logger.error(f"Failed to extract peptide sequence from {pdb_path}")
            return None

        # Remove peptide from complex
        protein_template_path = remove_peptide_from_complex(
            pdb_path, output_pdb_path=output_pdb_path, protein_chain="A"
        )
        if not os.path.exists(protein_template_path):
            logger.error(
                f"Failed to create protein template at {protein_template_path}"
            )
            return None

        logger.info(f"Peptide sequence: {peptide_sequence}")
        logger.info(f"Protein template path: {protein_template_path}")

        # Set up docking parameters
        docker_kwargs = {
            "num_models": 5,
            "num_recycles": 10,
            "recycle_early_stop_tolerance": 0.1,
            "amber": True,
            "num_relax": 1,
            "pdb_dir": docked_peptides_dir,
            "gpu_ids": ["3"],
            "overwrite_results": False,
        }

        # Run docking
        docker = Docker(docker_kwargs)
        docker.set_target_structure(protein_template_path)
        dock_dir = docker.dock_peptides([peptide_sequence])[0]

        # Score results
        scorer = Scorer()
        scores = scorer.score(
            scores_to_include=["interface_sasa", "rosetta_score"], colab_dir=dock_dir
        )

        pdb_pattern = re.compile(
                r".*_relaxed_rank_001_.*\.pdb"
            )  
        docked_top_pdb_file = os.path.join(
            dock_dir,
            [f for f in os.listdir(dock_dir) if pdb_pattern.search(f)][0],
        )

        # Compare binding sites
        in_same_binding_site, overlap = compare_binding_site(pdb_path, docked_top_pdb_file)
        scores["in_same_binding_site"] = in_same_binding_site
        scores["overlap"] = overlap
        scores["pdb_file"] = os.path.basename(pdb_path)
        scores["peptide_sequence"] = peptide_sequence

        logger.info(f"Completed benchmark for {os.path.basename(pdb_path)}")
        return scores

    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}", exc_info=True)
        return None


if __name__ == "__main__":
    data_dir = os.path.abspath(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs"
    )

    # Ensure data directory exists
    if not os.path.exists(data_dir):
        logger.error(f"Data directory does not exist: {data_dir}")
        exit(1)

    # Ensure output directory for CSV exists
    results_dir = os.path.dirname(os.path.abspath("benchmark_scores.csv"))
    ensure_dir_exists(results_dir)

    pdb_files = [
        os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".pdb")
    ]
    logger.info(f"Number of PDB files to process: {len(pdb_files)}")

    for i, pdb in enumerate(pdb_files):
        logger.info(f"Processing file {i+1}/{len(pdb_files)}: {os.path.basename(pdb)}")
        scores = run_benchmark(pdb)
        if scores:
            scores_df = pd.DataFrame([scores])
            scores_df.to_csv(
                "benchmark_scores.csv",
                mode="a",
                header=not os.path.exists("benchmark_scores.csv"),
                index=False,
            )
        else:
            logger.warning(f"No scores obtained for {pdb}")

    logger.info("Benchmark completed")
