"""
Module for generating data for the binding score function.

This includes downloading PDB files based on PDBBind index files,
processing them, and creating stripped templates for docking.
"""

import os
import logging
import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select
import numpy as np

logger = logging.getLogger(__name__)

def download_and_process_pdb_files(
    index_dir: str,
    output_dir: str,
    min_peptide_length: int = 7,
    max_peptide_length: int = 40,
    force_redownload: bool = False
) -> None:
    """
    Download PDB files, process them, and create stripped templates.
    
    Args:
        index_dir: Directory containing PDBBind index files
        output_dir: Directory to save outputs
        min_peptide_length: Minimum peptide length to include
        max_peptide_length: Maximum peptide length to include
        force_redownload: If True, redownload PDBs even if they exist
    """
    # Create subdirectories
    raw_pdbs_dir = os.path.join(output_dir, "raw_pdbs")
    processed_pdbs_dir = os.path.join(output_dir, "processed_pdbs")
    stripped_templates_dir = os.path.join(output_dir, "stripped_templates")
    
    os.makedirs(raw_pdbs_dir, exist_ok=True)
    os.makedirs(processed_pdbs_dir, exist_ok=True)
    os.makedirs(stripped_templates_dir, exist_ok=True)
    
    # Check if we need to download PDB files
    raw_pdbs_exists = os.path.exists(os.path.join(raw_pdbs_dir, "pdbs")) and \
                      len(os.listdir(os.path.join(raw_pdbs_dir, "pdbs"))) > 0
    
    if raw_pdbs_exists and not force_redownload:
        logger.info("Raw PDB files already exist, skipping download")
    else:
        # Download PDB files
        logger.info("Downloading PDB files...")
        download_pdbs(
            pdbbind_index_files_path=index_dir,
            output_pdb_dir=raw_pdbs_dir,
            min_peptide_length=min_peptide_length,
            max_peptide_length=max_peptide_length,
            force_redownload=force_redownload
        )
    
    # Check if processing is needed
    processed_exists = len(os.listdir(processed_pdbs_dir)) > 0
    if processed_exists:
        logger.info("Processed PDB files already exist, skipping processing")
    else:
        # Process PDB files to ensure correct structure
        logger.info("Processing PDB files...")
        process_pdbs(raw_pdbs_dir, output_dir)
    
    # Check if stripped templates need to be created
    templates_exist = len(os.listdir(stripped_templates_dir)) > 0
    if templates_exist:
        logger.info("Stripped templates already exist, skipping creation")
    else:
        # Create stripped templates (protein only, no peptide)
        logger.info("Creating stripped templates...")
        create_stripped_templates(output_dir, stripped_templates_dir)
    
    # Always update metadata to ensure it's complete
    update_metadata(raw_pdbs_dir, os.path.join(output_dir, "processed_pdbs"), output_dir)
    
    logger.info("Data generation completed")


def download_pdbs(
    pdbbind_index_files_path: str, 
    output_pdb_dir: str, 
    min_peptide_length: int = 7,
    max_peptide_length: int = 40,
    force_redownload: bool = False
) -> None:
    """
    Download PDB files based on PDBBind index files.
    
    This function is a wrapper around the actual download function,
    which is imported from the download_pdbs module.
    
    Args:
        pdbbind_index_files_path: Directory containing PDBBind index files
        output_pdb_dir: Directory to save PDB files
        min_peptide_length: Minimum peptide length to include
        max_peptide_length: Maximum peptide length to include
        force_redownload: If True, redownload PDBs even if they exist
    """
    # Import here to avoid circular imports
    from utils.download_pdbs import download_pdbs as dp
    
    dp(
        pdbbind_index_files_path=pdbbind_index_files_path,
        output_pdb_dir=output_pdb_dir,
        min_peptide_length=min_peptide_length,
        max_peptide_length=max_peptide_length,
        overwrite=force_redownload
    )


def process_pdbs(raw_pdbs_dir: str, output_dir: str) -> None:
    """
    Process PDB files to ensure they have the correct structure.
    
    This includes selecting the first model from NMR structures and
    ensuring that the peptide is in chain B and the protein in chain A.
    
    Args:
        raw_pdbs_dir: Directory containing raw PDB files
        output_dir: Directory to save processed files
    """
    # Import here to avoid circular imports
    from utils.preprocessing import process_pdbs as pp
    
    processed_dir = os.path.join(output_dir, "processed_pdbs")
    os.makedirs(processed_dir, exist_ok=True)
    
    pp(raw_pdbs_dir, processed_dir)
    
    # Update metadata with processed PDBs
    update_metadata(raw_pdbs_dir, processed_dir, output_dir)


def update_metadata(raw_pdbs_dir: str, processed_dir: str, output_dir: str) -> None:
    """
    Update metadata files with information from processed PDBs.
    
    Args:
        raw_pdbs_dir: Directory containing raw PDB files
        processed_dir: Directory containing processed PDB files
        output_dir: Directory to save metadata files
    """
    raw_metadata_file = os.path.join(raw_pdbs_dir, "pdbs.csv")
    processed_metadata_file = os.path.join(processed_dir, "pdbs.csv")
    output_metadata_file = os.path.join(output_dir, "metadata.csv")
    
    if os.path.exists(raw_metadata_file) and os.path.exists(processed_metadata_file):
        # Read metadata files
        raw_df = pd.read_csv(raw_metadata_file)
        processed_df = pd.read_csv(processed_metadata_file)
        
        # Merge metadata
        merged_df = processed_df.merge(
            raw_df[["PDB code", "Binding data"]],
            on="PDB code",
            how="left"
        )
        
        # Add standardized binding data columns
        if "Binding data" in merged_df.columns:
            binding_data = convert_binding_data_to_standard_units(merged_df["Binding data"])
            merged_df["Kd_M"] = binding_data["Kd_M"]
            merged_df["pKd"] = binding_data["pKd"]
        
        # Save merged metadata
        merged_df.to_csv(output_metadata_file, index=False)
        logger.info(f"Updated metadata saved to {output_metadata_file}")
    else:
        logger.warning("Could not update metadata: missing input files")


def convert_binding_data_to_standard_units(binding_data_series: pd.Series) -> pd.DataFrame:
    """
    Convert binding affinity data from various units to standard M units and calculate pKd.
    
    Args:
        binding_data_series: Series containing binding data strings
        
    Returns:
        DataFrame with columns for comparison operator, value, unit, Kd_M, and pKd
    """

    
    # Define unit conversion factors to convert to Molar
    unit_conversion = {
        "fM": 1e-15,
        "pM": 1e-12,
        "nM": 1e-9,
        "uM": 1e-6,
        "mM": 1e-3,
        "M": 1,
    }
    
    # Extract components from binding data strings
    binding_data = binding_data_series.str.extract(r"([=<>]?)(\d+\.?\d*)([a-zA-Z]*)")
    binding_data.columns = ["operator", "value", "unit"]
    
    # Convert values to numeric
    binding_data["value"] = pd.to_numeric(binding_data["value"], errors="coerce")
    
    # Convert to molar units
    binding_data["Kd_M"] = binding_data["value"] * binding_data["unit"].map(unit_conversion)
    
    # Calculate pKd (-log10 of Kd in M)
    binding_data["pKd"] = -np.log10(binding_data["Kd_M"])
    
    return binding_data


def create_stripped_templates(processed_dir: str, stripped_templates_dir: str) -> None:
    """
    Create stripped templates (protein only, no peptide) from processed PDB files.
    
    Args:
        processed_dir: Directory containing processed PDB files
        stripped_templates_dir: Directory to save stripped templates
    """
    processed_pdbs_dir = os.path.join(processed_dir, "processed_pdbs")
    
    if not os.path.exists(processed_pdbs_dir):
        logger.error(f"Processed PDBs directory not found: {processed_pdbs_dir}")
        return
    
    # Define a class to select only chain A (protein)
    class ChainASelect(Select):
        def accept_chain(self, chain):
            return chain.id == "A"
    
    # Process each PDB file
    pdb_files = [f for f in os.listdir(processed_pdbs_dir) if f.endswith(".pdb")]
    for pdb_file in pdb_files:
        input_path = os.path.join(processed_pdbs_dir, pdb_file)
        output_path = os.path.join(stripped_templates_dir, pdb_file)
        
        try:
            # Parse the PDB file
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("structure", input_path)
            
            # Write only chain A to the output file
            io = PDBIO()
            io.set_structure(structure)
            io.save(output_path, ChainASelect())
            
            logger.debug(f"Created stripped template: {output_path}")
        except Exception as e:
            logger.error(f"Error creating stripped template for {pdb_file}: {str(e)}")
    
    logger.info(f"Created {len(pdb_files)} stripped templates")
