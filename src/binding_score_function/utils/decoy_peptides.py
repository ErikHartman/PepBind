import random
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)

def shuffle_sequence(sequence: str) -> str:
    amino_acids = list(sequence)
    random.shuffle(amino_acids)
    shuffled_sequence = ''.join(amino_acids)
    return shuffled_sequence

def generate_decoy_dataset(
    docking_dir: str,
    n_decoys: int = 5,
) -> pd.DataFrame:
    """
    Generate a dataset of decoy peptides by shuffling real peptides.
    """
    random.seed(42)
    np.random.seed(42)

    docked_complexes = os.listdir(os.path.join(docking_dir, "docked_pdbs"))
    decoy_data = []
    
    decoy_complexes = random.sample(
        docked_complexes, 
        min(n_decoys, len(docked_complexes))
    )
    for complex in decoy_complexes:
        peptide = complex.split("_")[1]
        shuffled_peptide = shuffle_sequence(peptide)
        
        decoy_data.append({
            "complex": complex,
            "pdb_code": complex.split("_")[0],
            "decoy_peptide": shuffled_peptide,
            "original_peptide": peptide,
            "is_decoy": True,
        })
    
    df = pd.DataFrame(decoy_data)
    logger.info(f"Generated {len(df)} shuffled decoy peptides")
    return df
