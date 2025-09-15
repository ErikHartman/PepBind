import random
import numpy as np
import pandas as pd
import logging
import os

logger = logging.getLogger(__name__)


def shuffle_sequence(sequence: str) -> str:
    amino_acids = list(sequence)
    random.shuffle(amino_acids)
    shuffled_sequence = "".join(amino_acids)
    return shuffled_sequence


def generate_random_sequence(min_length: int = 10, max_length: int = 30) -> str:
    """
    Generate a random amino acid sequence with length drawn from uniform distribution.
    Uses amino acid frequencies observed in natural proteins.
    """
    # Amino acid frequencies based on UniProt KB/Swiss-Prot data
    aa_frequencies = {
        'A': 8.25,  # Ala
        'R': 5.53,  # Arg
        'N': 4.06,  # Asn
        'D': 5.45,  # Asp
        'C': 1.37,  # Cys
        'Q': 3.93,  # Gln
        'E': 6.75,  # Glu
        'G': 7.07,  # Gly
        'H': 2.27,  # His
        'I': 5.96,  # Ile
        'L': 9.66,  # Leu
        'K': 5.84,  # Lys
        'M': 2.42,  # Met
        'F': 3.86,  # Phe
        'P': 4.70,  # Pro
        'S': 6.56,  # Ser
        'T': 5.34,  # Thr
        'W': 1.08,  # Trp
        'Y': 2.92,  # Tyr
        'V': 6.87   # Val
    }


    amino_acids = list(aa_frequencies.keys())
    probabilities = list(aa_frequencies.values())
    probabilities = [freq / sum(probabilities) for freq in probabilities]
    length = random.randint(min_length, max_length)
    random_sequence = "".join(
        np.random.choice(amino_acids, size=length, p=probabilities)
    )

    return random_sequence


def generate_decoy_dataset(
    docking_dir: str,
    n_decoys: int = 5,
    decoy_method: str = "shuffle",
    min_length: int = 10,
    max_length: int = 30,
) -> pd.DataFrame:
    """
    Generate a dataset of decoy peptides.
    """
    random.seed(42)
    np.random.seed(42)

    docked_complexes = os.listdir(os.path.join(docking_dir, "pdbs", "processed"))
    decoy_data = []

    decoy_complexes = random.sample(
        docked_complexes, min(n_decoys, len(docked_complexes))
    )

    for complex in decoy_complexes:
        pdb_code = complex.split("_")[0]

        if decoy_method == "shuffle":
            peptide = complex.split("_")[1]
            decoy_peptide = shuffle_sequence(peptide)
            original_peptide = peptide
        elif decoy_method == "random":
            decoy_peptide = generate_random_sequence(min_length, max_length)
            original_peptide = (
                complex.split("_")[1] if len(complex.split("_")) > 1 else ""
            )
        else:
            raise ValueError(f"Unknown decoy method: {decoy_method}")

        decoy_data.append(
            {
                "complex": complex,
                "pdb_code": pdb_code,
                "peptide_sequence": decoy_peptide,
                "decoy_peptide": decoy_peptide,
                "original_peptide": original_peptide,
                "is_decoy": True,
                "decoy_type": decoy_method,
            }
        )

    df = pd.DataFrame(decoy_data)
    logger.info(f"Generated {len(df)} decoy peptides using method: {decoy_method}")
    return df
