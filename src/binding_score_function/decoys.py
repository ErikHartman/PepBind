"""
Module for generating and scoring decoy peptides.

This includes creating decoy peptides by shuffling real peptides,
docking them to protein templates, and scoring the interactions.
"""

import os
import logging
import random
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


def generate_and_score_decoys(
    complexes_dir: str,
    docked_dir: str,
    decoy_config: Dict[str, Any],
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_existing: bool = True
) -> None:
    """
    Generate decoy peptides and score them against protein templates.
    
    Args:
        complexes_dir: Directory containing complexes data
        docked_dir: Directory to save docking results
        decoy_config: Configuration for decoy generation
        docking_config: Configuration for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residue detection
        skip_existing: If True, skip decoys that have already been processed
    """
    # Create necessary directories - use decoy_pdbs directory as requested
    decoy_pdbs_dir = os.path.join(docked_dir, "decoy_pdbs")
    os.makedirs(decoy_pdbs_dir, exist_ok=True)
    logger.info(f"Created directory for decoy docking results: {decoy_pdbs_dir}")
    
    # Load metadata
    metadata_file = os.path.join(complexes_dir, "metadata.csv")
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return
    
    metadata_df = pd.read_csv(metadata_file)
    logger.info(f"Loaded metadata with {len(metadata_df)} entries")
    
    # Get protein templates
    stripped_templates_dir = os.path.join(complexes_dir, "stripped_templates")
    if not os.path.exists(stripped_templates_dir):
        logger.error(f"Stripped templates directory not found: {stripped_templates_dir}")
        return
    
    # Check if decoy configuration already exists
    decoy_config_file = os.path.join(docked_dir, "decoy_complexes.csv")
    if os.path.exists(decoy_config_file) and skip_existing:
        logger.info(f"Decoy configuration file already exists at {decoy_config_file}")
        logger.info("Using existing decoy configuration")
        try:
            decoy_df = pd.read_csv(decoy_config_file)
            logger.info(f"Loaded {len(decoy_df)} existing decoy configurations")
        except Exception as e:
            logger.error(f"Error loading existing decoy configuration: {e}")
            return
    else:
        # Generate new decoy dataset
        # Get protein templates and real peptide sequences
        protein_templates = []
        for _, row in metadata_df.iterrows():
            pdb_code = row["PDB code"]
            template_path = os.path.join(stripped_templates_dir, f"{pdb_code}.pdb")
            
            if os.path.exists(template_path):
                protein_templates.append(template_path)
        
        logger.info(f"Found {len(protein_templates)} protein templates for potential decoy generation")
        
        # Sample templates if needed
        n_templates = decoy_config.get("n_templates", 10)
        if len(protein_templates) > n_templates:
            random.seed(decoy_config.get("random_seed", 42))
            protein_templates = random.sample(protein_templates, n_templates)
        
        logger.info(f"Using {len(protein_templates)} protein templates for decoy generation")
        
        # Get real peptide sequences
        real_peptides = metadata_df["peptide_sequence"].dropna().tolist()
        min_length = decoy_config.get("min_peptide_length", 7)
        max_length = decoy_config.get("max_peptide_length", 40)
        
        real_peptides = [p for p in real_peptides if min_length <= len(p) <= max_length]
        logger.info(f"Found {len(real_peptides)} real peptide sequences for shuffling (length {min_length}-{max_length})")
        
        # Generate decoy dataset
        decoy_df = generate_decoy_dataset(
            protein_templates=protein_templates,
            real_peptides=real_peptides,
            n_decoys_per_template=decoy_config.get("n_decoys_per_template", 3),
            min_length=min_length,
            max_length=max_length,
            seed=decoy_config.get("random_seed", 42)
        )
        
        # Save decoy configuration
        decoy_df.to_csv(decoy_config_file, index=False)
        logger.info(f"Saved decoy configuration with {len(decoy_df)} entries to {decoy_config_file}")
    
    # Load existing results to skip already processed entries
    results_file = os.path.join(docked_dir, "decoy_scores.csv")
    processed_decoys = set()
    if os.path.exists(results_file) and skip_existing:
        try:
            existing_results = pd.read_csv(results_file)
            # Create identifier from pdb_code and decoy_peptide
            processed_decoys = set(existing_results["pdb_code"] + "_" + existing_results["decoy_peptide"])
            logger.info(f"Found {len(processed_decoys)} already processed decoys in {results_file}")
        except Exception as e:
            logger.warning(f"Could not read existing decoy results file: {e}")
    
    # Filter out already processed decoys
    decoy_tasks = []
    for _, row in decoy_df.iterrows():
        if pd.isna(row.get("decoy_peptide")) or pd.isna(row.get("pdb_code")):
            logger.warning(f"Skipping decoy task with missing data: {row}")
            continue
            
        decoy_peptide = row["decoy_peptide"]
        pdb_code = row["pdb_code"]
        
        # Skip if already processed
        if skip_existing and f"{pdb_code}_{decoy_peptide}" in processed_decoys:
            logger.debug(f"Skipping decoy {pdb_code}_{decoy_peptide} as it's already processed")
            continue
            
        decoy_tasks.append(row)
    
    if not decoy_tasks:
        logger.info("No new decoy tasks to perform")
        return
        
    # Configure docking - use the decoy-specific output directory
    parallel_config = docking_config.copy()
    parallel_config["pdb_dir"] = decoy_pdbs_dir
    
    # Create a lock for thread-safe CSV writing
    results_lock = threading.Lock()
    
    # Get GPU IDs for parallel processing
    gpu_ids = docking_config.get("gpu_ids", ["0"])
    num_gpus = len(gpu_ids)
    
    logger.info(f"Docking and scoring {len(decoy_tasks)} decoys using {num_gpus} GPUs")
    
    # Process a single decoy
    def process_decoy(args):
        idx, row = args
        gpu_id = gpu_ids[idx % num_gpus]
        
        # Configure for this specific GPU
        this_config = parallel_config.copy()
        this_config["gpu_ids"] = [gpu_id]
        
        protein_template = row["protein_template"]
        decoy_peptide = row["decoy_peptide"]
        pdb_code = row["pdb_code"]
        
        logger.info(f"Processing decoy {idx+1}/{len(decoy_tasks)}: {pdb_code} - {decoy_peptide} on GPU {gpu_id}")
        
        # Check if decoy is already docked - check in the decoy-specific directory
        docked_dir_path = os.path.join(decoy_pdbs_dir, f"{pdb_code}_{decoy_peptide}")
        if skip_existing and os.path.exists(docked_dir_path) and os.listdir(docked_dir_path):
            logger.info(f"Docking result for {decoy_peptide} already exists in {docked_dir_path}. Skipping docking...")
            try:
                # Still try to score if possible
                scores = dock_and_score_peptide(
                    protein_template_path=protein_template,
                    peptide_sequence=decoy_peptide,
                    output_dir=docked_dir,
                    docking_config=this_config,  # Contains the decoy-specific output dir
                    binding_residue_distance_cutoff=binding_residue_distance_cutoff,
                    skip_docking=True,
                    is_decoy=True
                )
                
                if scores:
                    # Add metadata
                    scores["pdb_code"] = pdb_code
                    scores["decoy_peptide"] = decoy_peptide
                    scores["original_peptide"] = row.get("original_peptide", "")
                    scores["is_decoy"] = True
                    
                    # Write to CSV
                    with results_lock:
                        scores_df = pd.DataFrame([scores])
                        scores_df.to_csv(
                            results_file,
                            index=False
                        )
                    return True
            except Exception as e:
                logger.warning(f"Could not score existing docking for {pdb_code}_{decoy_peptide}: {e}")
            
            return False
        
        try:
            # Dock and score
            scores = dock_and_score_peptide(
                protein_template_path=protein_template,
                peptide_sequence=decoy_peptide,
                output_dir=docked_dir,
                docking_config=this_config,  # Contains the decoy-specific output dir
                binding_residue_distance_cutoff=binding_residue_distance_cutoff,
                is_decoy=True
            )
            
            # Save results
            if scores:
                # Add metadata
                scores["pdb_code"] = pdb_code
                scores["decoy_peptide"] = decoy_peptide
                scores["original_peptide"] = row.get("original_peptide", "")
                scores["is_decoy"] = True
                
                # Write to CSV
                with results_lock:
                    scores_df = pd.DataFrame([scores])
                    scores_df.to_csv(
                        results_file,
                        index=False
                    )
                logger.info(f"Saved scores for decoy {pdb_code}_{decoy_peptide} to {results_file}")
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error processing decoy {pdb_code} - {decoy_peptide}: {str(e)}")
            return False
    
    # Run tasks in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        tasks = [(i, row) for i, row in enumerate(decoy_tasks)]
        
        for result in executor.map(process_decoy, tasks):
            if result:
                successful += 1
    
    logger.info(f"Completed decoy docking and scoring: {successful}/{len(decoy_tasks)} successful")
    logger.info(f"Results saved to {results_file}")


def generate_decoy_dataset(
    protein_templates: List[str], 
    real_peptides: List[str],
    n_decoys_per_template: int = 3,
    min_length: int = 7,
    max_length: int = 40,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a dataset of decoy peptides by shuffling real peptides.
    
    Args:
        protein_templates: List of protein template PDB paths
        real_peptides: List of real peptide sequences
        n_decoys_per_template: Number of decoys per template
        min_length: Minimum peptide length
        max_length: Maximum peptide length
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame of decoy configurations
    """
    # Import here to avoid circular imports
    from utils.decoy_peptides import generate_decoy_dataset as gdd
    
    return gdd(
        protein_templates=protein_templates,
        real_peptides=real_peptides,
        n_decoys_per_template=n_decoys_per_template,
        min_length=min_length,
        max_length=max_length,
        seed=seed
    )


def dock_and_score_peptide(
    protein_template_path: str,
    peptide_sequence: str,
    output_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_docking: bool = False,
    is_decoy: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Dock a peptide to a protein template and score the interaction.
    
    Args:
        protein_template_path: Path to the protein template PDB file
        peptide_sequence: Peptide sequence to dock
        output_dir: Directory to save output files
        docking_config: Configuration for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residues
        skip_docking: If True, skip docking and only score if results exist
        is_decoy: If True, this is a decoy peptide
        
    Returns:
        Dictionary of scores if successful, None otherwise
    """
    # Import here to avoid circular imports
    from utils.scoring import dock_and_score_peptide
    
    return dock_and_score_peptide(
        protein_template_path=protein_template_path,
        peptide_sequence=peptide_sequence,
        output_dir=output_dir,
        docking_config=docking_config,
        binding_residue_distance_cutoff=binding_residue_distance_cutoff,
        skip_docking=skip_docking,
        is_decoy=is_decoy
    )
