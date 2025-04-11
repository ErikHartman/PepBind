import os
import logging
import argparse
from typing import Dict
from dotenv import load_dotenv
import pyrosetta
from binding_score_function.utils.decoy_peptides import generate_decoy_dataset
from binding_score_function.utils.docking import dock_complexes
from binding_score_function.utils.download_pdbs import download_pdbs
from binding_score_function.utils.preprocessing import process_pdbs
from binding_score_function.utils.scoring import score_pdbs_in_dir

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)
pyrosetta.init(options="-mute all")


def main():
    """Main function to run the pipeline based on command line arguments"""
    args = parse_arguments()

    logger.info(f"Arguments: {args}")

    load_dotenv()
    paths = setup_directory_structure()

    # Default configuration for docking
    docking_config = {
        "num_models": 5,
        "num_recycles": 50,
        "recycle_early_stop_tolerance": 0.1,
        "amber": True,
        "num_relax": 1,
        "gpu_ids": ["2", "3"],
        "overwrite_results": False,
        "output_dir": os.path.join(paths["2_docked"], "pdbs"),
    }

    decoy_docking_config = {
        "num_models": 5,
        "num_recycles": 10,
        "recycle_early_stop_tolerance": 0.1,
        "amber": True,
        "num_relax": 1,
        "gpu_ids": ["2", "3"],
        "overwrite_results": False,
    }

    # Process based on arguments
    try:
        if args.generate_data or args.all:
            download_pdbs(
                pdbbind_index_files_path=paths["index_dir"],
                output_pdb_dir=paths["0_complexes"],
                min_peptide_length=args.min_length,
                max_peptide_length=args.max_length,
                overwrite=args.force_redownload,
            )

            process_pdbs(
                raw_pdbs_dir=os.path.join(paths["0_complexes"], "pdbs"),
                processed_dir=paths["1_processed_complexes"],
            )
            logger.info("Data generation completed")
            # Now we have raw_pdbs_dir/pdbs and raw_pdbs_dir/pdb.csv
            # as well as processed_pdbs_dir/pdbs and processed_pdbs_dir/pdb.csv
            # Time to dock the complexes in processed_pdbs_dir/pdbs

        if args.dock_score or args.all:
            dock_complexes(
                processed_dir=paths["1_processed_complexes"],
                docking_dir=paths["2_docked"],
                docking_config=docking_config,
            )
            score_pdbs_in_dir(
                docking_dir=docking_config["output_dir"],
                output_csv_path=os.path.join(paths["3_scores"], "scores.csv"),
                binding_residue_distance_cutoff=5.0,
                max_workers=4,
            )

        if args.decoys or args.all:
            decoys_df = generate_decoy_dataset(
                docking_dir=paths["2_docked"], n_decoys=200
            )
            dock_complexes()  # dock decoys

        # Set permissions on output files
        set_permissions_to_777(paths["output_dir"])

        logger.info("Pipeline completed successfully")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Peptide binding score function pipeline"
    )

    parser.add_argument(
        "--generate-data", action="store_true", help="Download and process PDB files"
    )
    parser.add_argument(
        "--dock-score", action="store_true", help="Dock peptides to templates and score"
    )
    parser.add_argument(
        "--decoys", action="store_true", help="Generate and score decoy peptides"
    )
    parser.add_argument("--all", action="store_true", help="Run the complete pipeline")

    parser.add_argument(
        "--min-length", type=int, default=7, help="Minimum peptide length (default: 7)"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=40,
        help="Maximum peptide length (default: 40)",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force redownloading PDB files even if they exist",
    )

    return parser.parse_args()


def setup_directory_structure() -> Dict[str, str]:
    load_dotenv()
    base_dir = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")

    # Define primary directories
    paths = {
        "base_dir": base_dir,
        "index_dir": os.path.abspath(
            os.path.join(base_dir, "inputs/pdbbind_index_files")
        ),
        "output_dir": os.path.abspath(
            os.path.join(base_dir, "outputs/binding_score_function")
        ),
        "0_complexes": os.path.abspath(
            os.path.join(base_dir, "outputs/binding_score_function/0_complexes")
        ),
        "1_processed_complexes": os.path.abspath(
            os.path.join(
                base_dir, "outputs/binding_score_function/1_processed_complexes"
            )
        ),
        "2_docked": os.path.abspath(
            os.path.join(base_dir, "outputs/binding_score_function/2_docked")
        ),
        "3_scores": os.path.abspath(
            os.path.join(base_dir, "outputs/binding_score_function/3_scores")
        ),
        "4_processed_scores": os.path.abspath(
            os.path.join(base_dir, "outputs/binding_score_function/4_processed_scores")
        ),
    }

    for path_name, path_value in paths.items():
        if not os.path.exists(path_value):
            os.makedirs(path_value, exist_ok=True)
            logger.info(f"created directory: {path_name} at {path_value}")
        else:
            logger.info(f"{path_name} already exists: {path_value}")

    logger.info("All output dirs now exist")
    return paths


def set_permissions_to_777(directory: str) -> None:
    """
    Recursively sets read/write/execute permissions (777) on all files and subdirectories.
    """
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions for {d}: {str(e)}")

        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o777)
            except Exception as e:
                logger.warning(f"Could not set permissions for {f}: {str(e)}")

    logger.info(f"Set permissions to 777 for all files in {directory}")


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


if __name__ == "__main__":
    main()
