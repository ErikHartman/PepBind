"""
Main script for the peptide binding score function pipeline.

This script provides three main functions:
1. generate_data: Download and process PDB files, create stripped templates
2. dock_and_score: Dock peptides to templates and score the interactions
3. generate_and_score_decoys: Create decoy peptides and score them
"""

import os
import logging
import argparse
from typing import Dict, Any
from dotenv import load_dotenv
import pandas as pd
import pyrosetta

from data_generation import download_and_process_pdb_files
from docking import dock_and_score_complexes
from decoys import generate_and_score_decoys
from common_utils import setup_directory_structure, set_permissions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize PyRosetta silently
pyrosetta.init(options="-mute all")


def generate_data(
    index_dir: str,
    output_dir: str,
    min_peptide_length: int = 7,
    max_peptide_length: int = 40,
    force_redownload: bool = False
) -> None:
    """
    Generate data by downloading PDB files, processing them, and creating stripped templates.
    
    Args:
        index_dir: Directory containing PDBBind index files
        output_dir: Directory to save outputs
        min_peptide_length: Minimum peptide length to include
        max_peptide_length: Maximum peptide length to include
        force_redownload: If True, redownload PDBs even if they exist
    """
    logger.info("Starting data generation...")
    
    # Create necessary directories
    complexes_dir = os.path.join(output_dir, "0_complexes")
    os.makedirs(complexes_dir, exist_ok=True)
    
    # Download and process PDB files
    download_and_process_pdb_files(
        index_dir=index_dir,
        output_dir=complexes_dir,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        force_redownload=force_redownload
    )
    
    logger.info("Data generation completed")


def dock_and_score(
    output_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_existing: bool = True
) -> None:
    """
    Dock peptides to protein templates and score the interactions.
    
    Args:
        output_dir: Directory where outputs are saved
        docking_config: Configuration parameters for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residue detection
        skip_existing: If True, skip complexes that have already been processed
    """
    logger.info("Starting docking and scoring...")
    
    # Define input and output directories
    complexes_dir = os.path.join(output_dir, "0_complexes")
    docked_dir = os.path.join(output_dir, "1_docked")
    
    # Create necessary directories
    os.makedirs(docked_dir, exist_ok=True)
    
    # Check if docking results already exist and should be skipped
    results_file = os.path.join(docked_dir, "real_scores.csv")
    if os.path.exists(results_file) and skip_existing:
        logger.info(f"Docking results file already exists at {results_file}")
        logger.info("You can run with --no-skip to force re-docking")
        
        # Check if the metadata merge has been done
        merged_file = os.path.join(docked_dir, "real_scores_with_metadata.csv")
        if not os.path.exists(merged_file):
            logger.info("Merging existing scores with metadata...")
            from docking import merge_scores_with_metadata
            merge_scores_with_metadata(docked_dir, complexes_dir)
        return
    
    # Dock and score complexes
    dock_and_score_complexes(
        complexes_dir=complexes_dir,
        docked_dir=docked_dir,
        docking_config=docking_config,
        binding_residue_distance_cutoff=binding_residue_distance_cutoff,
        skip_existing=skip_existing
    )
    
    logger.info("Docking and scoring completed")


def generate_and_dock_decoys(
    output_dir: str,
    decoy_config: Dict[str, Any],
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_existing: bool = True
) -> None:
    """
    Generate decoy peptides and dock them to protein templates.
    
    Args:
        output_dir: Directory where outputs are saved
        decoy_config: Configuration for decoy generation
        docking_config: Configuration parameters for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residue detection
        skip_existing: If True, skip decoys that have already been processed
    """
    logger.info("Starting decoy generation and scoring...")
    
    # Define input and output directories
    complexes_dir = os.path.join(output_dir, "0_complexes")
    docked_dir = os.path.join(output_dir, "1_docked")
    
    # Check if decoy results already exist and should be skipped
    decoy_results_file = os.path.join(docked_dir, "decoy_scores.csv")
    if os.path.exists(decoy_results_file) and skip_existing:
        logger.info(f"Decoy results file already exists at {decoy_results_file}")
        logger.info("Skipping re-docking for existing entries, but will still process new or incomplete tasks")
        
        # Check if the combined file has been created
        combined_file = os.path.join(docked_dir, "all_scores.csv")
        if not os.path.exists(combined_file):
            logger.info("Combining existing real and decoy scores...")
            combine_all_scores(docked_dir)
        # Removed the return to allow processing of incomplete entries
    
    # Generate and score decoys
    generate_and_score_decoys(
        complexes_dir=complexes_dir,
        docked_dir=docked_dir,
        decoy_config=decoy_config,
        docking_config=docking_config,
        binding_residue_distance_cutoff=binding_residue_distance_cutoff,
        skip_existing=skip_existing
    )
    
    # Combine real and decoy scores
    combine_all_scores(docked_dir)
    
    logger.info("Decoy generation and scoring completed")


def combine_all_scores(docked_dir: str) -> None:
    """
    Combine scores from real peptides and decoys into a single file.
    
    Args:
        docked_dir: Directory containing docking results
    """
    real_scores_file = os.path.join(docked_dir, "real_scores.csv")
    decoy_scores_file = os.path.join(docked_dir, "decoy_scores.csv")
    combined_file = os.path.join(docked_dir, "all_scores.csv")
    
    if os.path.exists(real_scores_file) and os.path.exists(decoy_scores_file):
        # Read both files
        real_df = pd.read_csv(real_scores_file)
        decoy_df = pd.read_csv(decoy_scores_file)
        
        logger.info(f"Read {len(real_df)} entries from {real_scores_file}")
        logger.info(f"Read {len(decoy_df)} entries from {decoy_scores_file}")
        
        # Make sure real peptides have the is_decoy flag set to False
        if 'is_decoy' not in real_df.columns:
            real_df['is_decoy'] = False
            logger.info("Added 'is_decoy=False' flag to real peptides")
            
        # Combine and save
        combined_df = pd.concat([real_df, decoy_df], ignore_index=True)
        combined_df.to_csv(combined_file, index=False)
        
        logger.info(f"Combined {len(real_df)} real scores and {len(decoy_df)} decoy scores into {combined_file}")
        logger.info(f"Total entries in combined file: {len(combined_df)}")
    else:
        missing_files = []
        if not os.path.exists(real_scores_file):
            missing_files.append(real_scores_file)
        if not os.path.exists(decoy_scores_file):
            missing_files.append(decoy_scores_file)
        logger.warning(f"Could not combine scores: missing input files: {', '.join(missing_files)}")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Peptide binding score function pipeline')
    
    parser.add_argument('--generate-data', action='store_true', 
                        help='Download and process PDB files')
    parser.add_argument('--dock-score', action='store_true',
                        help='Dock peptides to templates and score')
    parser.add_argument('--decoys', action='store_true',
                        help='Generate and score decoy peptides')
    parser.add_argument('--all', action='store_true',
                        help='Run the complete pipeline')
                        
    parser.add_argument('--min-length', type=int, default=7,
                        help='Minimum peptide length (default: 7)')
    parser.add_argument('--max-length', type=int, default=40,
                        help='Maximum peptide length (default: 40)')
    
    parser.add_argument('--force-redownload', action='store_true',
                        help='Force redownloading PDB files even if they exist')
    parser.add_argument('--no-skip', dest='skip_existing', action='store_false',
                        help='Do not skip existing docking results')
    parser.set_defaults(skip_existing=True)
                        
    return parser.parse_args()


def main():
    """Main function to run the pipeline based on command line arguments"""
    args = parse_arguments()
    
    # Load environment variables
    load_dotenv()
    
    # Set up directory structure
    paths = setup_directory_structure()
    
    # Default configuration for docking
    docking_config = {
        "num_models": 5,
        "num_recycles": 50,
        "recycle_early_stop_tolerance": 0.1,
        "amber": True,
        "num_relax": 1,
        "gpu_ids": ["2", "3"],
        "overwrite_results": not args.skip_existing,
    }
    
    # Default configuration for decoy generation
    decoy_config = {
        "n_templates": 200,
        "n_decoys_per_template": 1,
        "min_peptide_length": args.min_length,
        "max_peptide_length": args.max_length,
        "random_seed": 42
    }

    decoy_docking_config = {
        "num_models": 5,
        "num_recycles": 10,
        "recycle_early_stop_tolerance": 0.1,
        "amber": True,
        "num_relax": 1,
        "gpu_ids": ["2", "3"],
        "overwrite_results": not args.skip_existing,
    }
    
    # Process based on arguments
    try:
        if args.generate_data or args.all:
            generate_data(
                index_dir=paths["index_dir"],
                output_dir=paths["output_dir"],
                min_peptide_length=args.min_length,
                max_peptide_length=args.max_length,
                force_redownload=args.force_redownload
            )
            
        if args.dock_score or args.all:
            dock_and_score(
                output_dir=paths["output_dir"],
                docking_config=docking_config,
                binding_residue_distance_cutoff=5.0,
                skip_existing=args.skip_existing
            )
            
        if args.decoys or args.all:
            generate_and_dock_decoys(
                output_dir=paths["output_dir"],
                decoy_config=decoy_config,
                docking_config=decoy_docking_config,
                binding_residue_distance_cutoff=5.0,
                skip_existing=args.skip_existing
            )
            
        # Set permissions on output files
        set_permissions(paths["output_dir"])
        
        logger.info("Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)


if __name__ == "__main__":
    main()
