import os
import logging
import argparse
from typing import Dict
from dotenv import load_dotenv
import pandas as pd
import pyrosetta
from utils.decoy_peptides import generate_decoy_dataset
from utils.docking import dock_complexes
from utils.download_pdbs import download_pdbs
from utils.process import process_pdbs
from utils.scoring import score_pdbs_in_dir

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
    "models": ["alphafold", "boltz"],
    "num_models": 5,
    "num_recycles": 20,
    "recycle_early_stop_tolerance": 0.1,
    "amber": True,
    "num_relax": 1,
    "gpu_ids": ["1", "2"],
    "overwrite_results": False,
    "output_dir": os.path.join(paths["2_docked"], "pdbs"),

    # Boltz-specific options (add as needed)
    "recycling_steps": 20,
    "diffusion_samples": 5,
    "output_format": "pdb",     
    "cache": "/srv/data1/general/tmp/cache",
    }

    n_decoy_shuffle = 150
    n_decoy_random = 150

    decoy_docking_config = {
    "models": ["alphafold", "boltz"],
    "num_models": 5,
    "num_recycles": 20,
    "recycle_early_stop_tolerance": 0.1,
    "amber": True,
    "num_relax": 1,
    "gpu_ids": ["1", "2"],
    "overwrite_results": False,
    "output_dir": os.path.join(paths["2_docked_decoy"]),

    # Boltz-specific options (add as needed)
    "recycling_steps": 20,
    "diffusion_samples": 5,
    "output_format": "pdb",
    "cache": "/srv/data1/general/tmp/cache",
    }

    processed_downloaded_df = pd.DataFrame()
    downloaded_df = pd.DataFrame()
    decoys_df = pd.DataFrame()
    scores_df = pd.DataFrame()
    decoy_scores_df = pd.DataFrame()

    # Process based on arguments
    try:
        if args.download_pdbs or args.all:
            logger.info("Downloading pdbs")
            downloaded_df = download_pdbs(
                pdbbind_index_files_path=paths["index_dir"],
                output_pdb_dir=paths["0_complexes"],
                min_peptide_length=args.min_length,
                max_peptide_length=args.max_length,
                overwrite=args.force_redownload,
            )
            downloaded_df.to_csv(os.path.join(paths["0_complexes"], "pdbs.csv"))

        if args.process or args.all:
            logger.info("Processing pdbs")
            manual_csv_path = paths["manually_curated_pdbs"]
            if not os.path.isfile(manual_csv_path):
                logger.info(f"Manually curated PDBs file not found at {manual_csv_path}. Not including manually curated PDBs.")
                manual_csv_path = None
            processed_downloaded_df = process_pdbs(
                raw_pdbs_dir=os.path.join(paths["0_complexes"], "pdbs"),
                pdb_csv_dir=os.path.join(paths["0_complexes"], "pdbs.csv"),
                manual_csv_path=manual_csv_path,
            )
            processed_downloaded_df["peptide_length"] = processed_downloaded_df[
                "peptide_sequence"
            ].apply(len)
            processed_downloaded_df = processed_downloaded_df[
                (processed_downloaded_df["peptide_length"] >= args.min_length)
                & (processed_downloaded_df["peptide_length"] <= args.max_length)
            ]

            logger.info(f"Processed PDBs (n={len(processed_downloaded_df.index)}):")
            logger.info(f"{processed_downloaded_df.head(5)}")
            processed_downloaded_df.to_csv(
                os.path.join(paths["1_processed_complexes"], "processed_pdbs.csv"),
            )

        if args.dock or args.all:
            logger.info("Starting docking")
            if processed_downloaded_df.empty:
                logger.info(
                    "Reading processed PDBs from CSV since not processed in this run"
                )
                processed_downloaded_df = pd.read_csv(
                    os.path.join(paths["1_processed_complexes"], "processed_pdbs.csv")
                )
            dock_complexes(
                template_pdb_dir=os.path.join(paths["0_complexes"], "pdbs"),
                processed_df=processed_downloaded_df,
                docking_config=docking_config,
                min_length=args.min_length,
                max_length=args.max_length,
            )
            logger.info("Docking completed")

        if args.score or args.all:
            logger.info("Starting scoring")
            scores_df = score_pdbs_in_dir(
                docking_dir=os.path.join(paths["2_docked"], "pdbs", "processed"),
                complexes_dir=os.path.join(paths["0_complexes"], "pdbs"),
                binding_residue_distance_cutoff=5.0,
                max_workers=20,
            )
            scores_df["is_decoy"] = False
            scores_df.to_csv(os.path.join(paths["3_scores"], "scores.csv"), index=False)
            logger.info("Scoring completed")

        if args.decoys or args.all:
            logger.info("Generating decoy dataset")
            if len(os.listdir(paths["2_docked_decoy"])) != 0: # This is a bit safer than checking exact lengths (I accidentally wiped our decoy data once)
                logger.info("Decoy dataset already exists, skipping generation")
            else:
                decoys_df_shuffle = generate_decoy_dataset(
                    docking_dir=paths["2_docked"],
                    n_decoys=n_decoy_shuffle,
                    decoy_method="shuffle",
                    min_length=args.min_length,
                    max_length=args.max_length,
                )
                decoys_df_random = generate_decoy_dataset(
                    docking_dir=paths["2_docked"],
                    n_decoys=n_decoy_random,
                    decoy_method="random",
                    min_length=args.min_length,
                    max_length=args.max_length,
                )

                decoys_df = pd.concat(
                    [decoys_df_shuffle, decoys_df_random], ignore_index=True
                )
                decoys_df.to_csv(
                    os.path.join(paths["1_processed_complexes"], "decoys.csv"),
                    index=False,
                )
                logger.info("Docking decoy dataset")
                dock_complexes(
                    template_pdb_dir=os.path.join(paths["0_complexes"], "pdbs"),
                    processed_df=decoys_df,
                    docking_config=decoy_docking_config,
                    max_length=args.max_length,
                    min_length=args.min_length,
                )

            logger.info("Scoring decoy dataset")
            decoy_scores_df = score_pdbs_in_dir(
                docking_dir=os.path.join(paths["2_docked"], "decoy_pdbs", "processed"),
                complexes_dir=os.path.join(paths["0_complexes"], "pdbs"),
                binding_residue_distance_cutoff=5.0,
                max_workers=20,
            )
            decoy_scores_df["is_decoy"] = True
            decoy_scores_df.to_csv(
                os.path.join(paths["3_scores"], "decoy_scores.csv"), index=False
            )

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
        "--download-pdbs", action="store_true", help="Download and process PDB files"
    )
    parser.add_argument(
        "--process", action="store_true", help="Process downloaded PDB files"
    )
    parser.add_argument(
        "--dock", action="store_true", help="Dock peptides to templates"
    )
    parser.add_argument("--score", action="store_true", help="Score docked peptides")
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
        default=45,
        help="Maximum peptide length (default: 45)",
    )

    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force redownloading PDB files even if they exist",
    )

    return parser.parse_args()


def setup_directory_structure() -> Dict[str, str]:
    load_dotenv()
    base_dir = os.getenv("DATA_DIR", "/srv/data1/ma7631si/immunopeptides_data/")
    output_dir = os.getenv("OUTPUT_DIR", "/srv/data1/ma7631si/immunopeptides_data/outputs/binding_score_function/")
    paths = {
        "base_dir": base_dir,
        "manually_curated_pdbs": os.path.abspath(
            os.path.join(base_dir, "inputs/manually_curated_pdbs.csv")
        ),
        "index_dir": os.path.abspath(
        "/home/ma7631si/home/immunopeptides/data/"
        ),
        "output_dir": os.path.abspath(output_dir),
        "0_complexes": os.path.abspath(
            os.path.join(output_dir, "0_complexes")
        ),
        "1_processed_complexes": os.path.abspath(
            os.path.join(
                output_dir, "1_processed_complexes"
            )
        ),
        "2_docked": os.path.abspath(
            os.path.join(output_dir, "2_docked")
        ),
        "3_scores": os.path.abspath(
            os.path.join(output_dir, "3_scores")
        ),
        "4_processed_scores": os.path.abspath(
            os.path.join(output_dir, "4_processed_scores")
        ),
    }
    paths.update(
        {
            "2_docked_pdbs": os.path.join(paths["2_docked"], "pdbs"),
            "2_docked_decoy": os.path.join(paths["2_docked"], "decoy_pdbs"),
        }
    )

    for path_name, path_value in paths.items():
        if not os.path.exists(path_value):
            os.makedirs(path_value, exist_ok=True)
            logger.info(f"created directory: {path_name} at {path_value}")
        else:
            logger.info(f"{path_name} already exists: {path_value}")

    logger.info("All output dirs now exist")
    return paths


def set_permissions_to_777(directory: str) -> None:
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


if __name__ == "__main__":
    main()
