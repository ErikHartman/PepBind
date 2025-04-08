"""
Utility functions for generating decoy peptides to serve as negative examples
in the binding affinity benchmark.
"""

import random
import numpy as np
from typing import List
import pandas as pd
import logging

logger = logging.getLogger(__name__)

# Standard amino acid one-letter codes
STANDARD_AA = 'ACDEFGHIKLMNPQRSTVWY'

def shuffle_sequence(sequence: str) -> str:
    """
    Create a decoy peptide by shuffling the amino acids in a sequence.
    
    Args:
        sequence (str): Original peptide sequence
        
    Returns:
        str: Shuffled peptide sequence
    """
    amino_acids = list(sequence)
    random.shuffle(amino_acids)
    shuffled_sequence = ''.join(amino_acids)
    return shuffled_sequence

def generate_decoy_dataset(
    protein_templates: List[str], 
    real_peptides: List[str],
    n_decoys_per_template: int = 3,
    min_length: int = 7,
    max_length: int = 15,
    seed: int = 42
) -> pd.DataFrame:
    """
    Generate a dataset of decoy peptides by shuffling real peptides.
    
    Args:
        protein_templates (List[str]): List of protein template PDB filenames
        real_peptides (List[str]): List of real peptide sequences to shuffle
        n_decoys_per_template (int): Number of decoys to generate per template
        min_length (int): Minimum peptide length to include (filters real_peptides)
        max_length (int): Maximum peptide length to include (filters real_peptides)
        seed (int): Random seed for reproducibility
        
    Returns:
        pd.DataFrame: DataFrame containing template and decoy pairs
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # Filter peptides by length constraints
    filtered_peptides = [p for p in real_peptides if min_length <= len(p) <= max_length]
    
    if not filtered_peptides:
        logger.warning(f"No peptides found in length range {min_length}-{max_length}")
        return pd.DataFrame()
    
    logger.info(f"Using {len(filtered_peptides)} peptides for shuffling (after filtering by length)")
    
    decoy_data = []
    
    for template in protein_templates:
        template_base = template.split('/')[-1].replace('.pdb', '')
        
        # Randomly select peptides to shuffle
        peptides_to_shuffle = random.sample(
            filtered_peptides, 
            min(n_decoys_per_template, len(filtered_peptides))
        )
        
        for peptide in peptides_to_shuffle:
            shuffled = shuffle_sequence(peptide)
            decoy_data.append({
                "protein_template": template,
                "pdb_code": template_base,
                "decoy_peptide": shuffled,
                "original_peptide": peptide,
                "is_decoy": True,
            })
    
    df = pd.DataFrame(decoy_data)
    logger.info(f"Generated {len(df)} shuffled decoy peptides")
    return df

def save_decoy_dataset(decoy_df: pd.DataFrame, output_path: str) -> None:
    """
    Save the decoy dataset to a CSV file.
    
    Args:
        decoy_df (pd.DataFrame): DataFrame containing decoy data
        output_path (str): Path to save the CSV file
    """
    decoy_df.to_csv(output_path, index=False)
    logger.info(f"Saved decoy dataset with {len(decoy_df)} entries to {output_path}")
