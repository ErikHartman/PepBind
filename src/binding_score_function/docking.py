"""
Module for docking and scoring peptides to protein templates.

This includes loading peptide sequences from metadata,
docking them to protein templates, and scoring the interactions.
"""

import os
import logging
import pandas as pd
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Optional, Any

logger = logging.getLogger(__name__)


def dock_and_score_complexes(
    complexes_dir: str,
    docked_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_existing: bool = True
) -> None:
    """
    Dock peptides to protein templates and score the interactions.
    
    Args:
        complexes_dir: Directory containing complexes data
        docked_dir: Directory to save docking results
        docking_config: Configuration parameters for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residue detection
        skip_existing: If True, skip complexes that have already been processed
    """
    # Create necessary directories
    docked_pdbs_dir = os.path.join(docked_dir, "docked_pdbs")
    os.makedirs(docked_pdbs_dir, exist_ok=True)
    
    # Load metadata
    metadata_file = os.path.join(complexes_dir, "metadata.csv")
    if not os.path.exists(metadata_file):
        logger.error(f"Metadata file not found: {metadata_file}")
        return
    
    metadata_df = pd.read_csv(metadata_file)
    
    # Get protein templates and peptide sequences
    stripped_templates_dir = os.path.join(complexes_dir, "stripped_templates")
    if not os.path.exists(stripped_templates_dir):
        logger.error(f"Stripped templates directory not found: {stripped_templates_dir}")
        return
    
    # Load existing results to skip already processed entries
    results_file = os.path.join(docked_dir, "real_scores.csv")
    processed_pdbs = set()
    if os.path.exists(results_file) and skip_existing:
        try:
            existing_results = pd.read_csv(results_file)
            processed_pdbs = set(existing_results["pdb_code"].tolist())
            logger.info(f"Found {len(processed_pdbs)} already processed PDB entries")
        except Exception as e:
            logger.warning(f"Could not read existing results file: {e}")
    
    # Prepare docking tasks
    docking_tasks = []
    for _, row in metadata_df.iterrows():
        pdb_code = row["PDB code"]
        
        # Skip if already processed
        if skip_existing and pdb_code in processed_pdbs:
            logger.debug(f"Skipping {pdb_code} as it's already processed")
            continue
            
        peptide_sequence = row.get("peptide_sequence")
        
        if not peptide_sequence or pd.isna(peptide_sequence):
            logger.warning(f"No peptide sequence found for {pdb_code}")
            continue
        
        template_path = os.path.join(stripped_templates_dir, f"{pdb_code}.pdb")
        if not os.path.exists(template_path):
            logger.warning(f"Template not found for {pdb_code}")
            continue
        
        docking_tasks.append((pdb_code, template_path, peptide_sequence))
    
    if not docking_tasks:
        logger.info("No new docking tasks to perform")
        return
        
    logger.info(f"Found {len(docking_tasks)} complexes to dock and score")
    
    # Run docking and scoring in parallel
    # Configure docking
    parallel_config = docking_config.copy()
    parallel_config["pdb_dir"] = docked_pdbs_dir
    
    # Create a lock for thread-safe CSV writing
    results_lock = threading.Lock()
    
    # Get GPU IDs for parallel processing
    gpu_ids = docking_config.get("gpu_ids", ["0"])
    num_gpus = len(gpu_ids)
    
    logger.info(f"Docking and scoring {len(docking_tasks)} complexes using {num_gpus} GPUs")
    
    # Process a single docking task
    def process_task(args):
        idx, (pdb_code, template_path, peptide_sequence) = args
        gpu_id = gpu_ids[idx % num_gpus]
        
        # Configure for this specific GPU
        this_config = parallel_config.copy()
        this_config["gpu_ids"] = [gpu_id]
        
        logger.info(f"Processing {idx+1}/{len(docking_tasks)}: {pdb_code} on GPU {gpu_id}")
        
        # Check if peptide is already docked for this complex
        docked_dir_path = os.path.join(docked_pdbs_dir, f"{pdb_code}_{peptide_sequence}")
        if skip_existing and os.path.exists(docked_dir_path) and os.listdir(docked_dir_path):
            logger.info(f"Docking result for {peptide_sequence} already exists in {docked_dir_path}. Skipping...")
            try:
                # Still try to score if possible
                scores = dock_and_score_peptide(
                    protein_template_path=template_path,
                    peptide_sequence=peptide_sequence,
                    output_dir=docked_dir,
                    docking_config=this_config,
                    binding_residue_distance_cutoff=binding_residue_distance_cutoff,
                    skip_docking=True
                )
                
                if scores:
                    scores["pdb_code"] = pdb_code
                    scores["is_decoy"] = False
                    
                    with results_lock:
                        scores_df = pd.DataFrame([scores])
                        scores_df.to_csv(
                            results_file,
                            index=False
                        )
                    return True
            except Exception as e:
                logger.warning(f"Could not score existing docking for {pdb_code}: {e}")
                
            return False
        
        try:
            # Dock and score
            scores = dock_and_score_peptide(
                protein_template_path=template_path,
                peptide_sequence=peptide_sequence,
                output_dir=docked_dir,
                docking_config=this_config,
                binding_residue_distance_cutoff=binding_residue_distance_cutoff
            )
            
            # Save results
            if scores:
                # Add metadata
                scores["pdb_code"] = pdb_code
                scores["is_decoy"] = False
                
                # Write to CSV
                with results_lock:
                    scores_df = pd.DataFrame([scores])
                    scores_df.to_csv(
                        results_file,
                        index=False
                    )
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"Error processing {pdb_code}: {str(e)}")
            return False
    
    # Run tasks in parallel
    successful = 0
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        tasks = [(i, task) for i, task in enumerate(docking_tasks)]
        
        for result in executor.map(process_task, tasks):
            if result:
                successful += 1
    
    logger.info(f"Completed docking and scoring: {successful}/{len(docking_tasks)} successful")
    
    # Merge scores with metadata
    merge_scores_with_metadata(docked_dir, complexes_dir)


def dock_and_score_peptide(
    protein_template_path: str,
    peptide_sequence: str,
    output_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_docking: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Dock a peptide to a protein template and score the interaction.
    
    Args:
        protein_template_path: Path to the protein template PDB file
        peptide_sequence: Peptide sequence to dock
        output_dir: Directory to save output files
        docking_config: Configuration for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residues
        skip_docking: If True, skip docking and only score if results already exist
        
    Returns:
        Dictionary of scores if successful, None otherwise
    """
    # Import here to avoid circular imports
    from utils.scoring import docker_and_score_peptide
    
    return docker_and_score_peptide(
        protein_template_path=protein_template_path,
        peptide_sequence=peptide_sequence,
        output_dir=output_dir,
        docking_config=docking_config,
        binding_residue_distance_cutoff=binding_residue_distance_cutoff,
        skip_docking=skip_docking
    )


def merge_scores_with_metadata(docked_dir: str, complexes_dir: str) -> None:
    """
    Merge docking scores with metadata from complexes.
    
    Args:
        docked_dir: Directory containing docking results
        complexes_dir: Directory containing complexes data
    """
    scores_file = os.path.join(docked_dir, "real_scores.csv")
    metadata_file = os.path.join(complexes_dir, "metadata.csv")
    merged_file = os.path.join(docked_dir, "real_scores_with_metadata.csv")
    
    if os.path.exists(scores_file) and os.path.exists(metadata_file):
        # Read files
        scores_df = pd.read_csv(scores_file)
        metadata_df = pd.read_csv(metadata_file)

        logger.info(f"Merging scores ({len(scores_df)} entries) with metadata ({len(metadata_df)} entries)")
        
        # Check which columns exist in the metadata
        valid_columns = ["PDB code"]
        for col in ["Binding data", "Kd_M", "pKd"]:
            if col in metadata_df.columns:
                valid_columns.append(col)
        
        logger.info(f"Using metadata columns for merge: {valid_columns}")
        
        # Only merge with columns that actually exist in the metadata file
        if len(valid_columns) > 1:
            # Merge on PDB code
            merged_df = scores_df.merge(
                metadata_df[valid_columns],
                left_on="pdb_code",
                right_on="PDB code",
                how="left"
            )
            
            # Save merged file
            merged_df.to_csv(merged_file, index=False)
            logger.info(f"Merged scores with metadata: {merged_file} ({len(merged_df)} entries)")
            
            # Log how many entries had binding data
            non_null_binding = merged_df["Binding data"].notna().sum() if "Binding data" in merged_df.columns else 0
            logger.info(f"Entries with binding data: {non_null_binding}")
        else:
            # Just copy the scores file as-is if no metadata columns to merge
            scores_df.to_csv(merged_file, index=False)
            logger.warning("No binding data columns found in metadata, copied scores without merging")
    else:
        missing_files = []
        if not os.path.exists(scores_file):
            missing_files.append(scores_file)
        if not os.path.exists(metadata_file):
            missing_files.append(metadata_file)
        logger.warning(f"Could not merge scores with metadata: missing input files: {', '.join(missing_files)}")
