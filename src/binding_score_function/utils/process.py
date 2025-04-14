import os
from typing import Union
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
import pandas as pd
import logging
import numpy as np
from Bio.PDB.Polypeptide import is_aa
import io

logger = logging.getLogger(__name__)

residue_map = {
        "ALA": "A",
        "ARG": "R",
        "ASN": "N",
        "ASP": "D",
        "CYS": "C",
        "GLN": "Q",
        "GLU": "E",
        "GLY": "G",
        "HIS": "H",
        "ILE": "I",
        "LEU": "L",
        "LYS": "K",
        "MET": "M",
        "PHE": "F",
        "PRO": "P",
        "SER": "S",
        "THR": "T",
        "TRP": "W",
        "TYR": "Y",
        "VAL": "V",
    }


def throws_rosetta_error(pdb_file: str) -> bool:
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception:
        return True

def is_peptide_cyclic(pdb_file, cutoff: float = 1.7) -> bool:
    """
    Detect if the peptide (chain B) in the given PDB file is cyclic.
    Checks if there's a covalent bond (distance < cutoff Å) between the
    C-terminal carbonyl carbon and N-terminal nitrogen atoms.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_file)
    model = structure[0]
    
    try:
        chain = model["B"]
    except KeyError:
        return False

    residues = [res for res in chain if is_aa(res, standard=True)]

    # N-terminal residue N atom
    n_term_res = residues[0]
    n_atom = n_term_res["N"] if "N" in n_term_res else None

    # C-terminal residue C atom
    c_term_res = residues[-1]
    c_atom = c_term_res["C"] if "C" in c_term_res else None

    if n_atom is None or c_atom is None:
        return False

    distance = np.linalg.norm(n_atom.coord - c_atom.coord)

    is_cyclic = distance <= cutoff
    return is_cyclic

def ensure_peptide_is_chain_b(pdb_path: str, output_pdb_path: str) -> str:
    """
    Ensures that the shortest chain is labeled as chain B (peptide) and
    the longest chain is labeled as chain A (protein).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]

    # Determine chain lengths (only counting standard residues)
    chain_lengths = {}
    for chain in model:
        residue_count = sum(
            1 for residue in chain.get_residues() if residue.id[0] == " "
        )
        chain_lengths[chain.id] = residue_count

    # Identify shortest chain as new 'B' and another chain as 'A'
    sorted_chains = sorted(chain_lengths.items(), key=lambda x: x[1])
    peptide_chain_id = sorted_chains[0][0]
    protein_chain_id = sorted_chains[-1][0]

    # Create a new structure with renamed chains
    new_structure = Structure("complex")
    new_model = Model(0)
    new_structure.add(new_model)

    for chain in model:
        new_chain = Chain("")
        if chain.id == peptide_chain_id:
            new_chain.id = "B"
        elif chain.id == protein_chain_id:
            new_chain.id = "A"
        else:
            # Skip or assign a different ID if needed
            continue

        # Copy all residues to the new chain
        for residue in chain:
            new_chain.add(residue.copy())

        new_model.add(new_chain)

    io = PDBIO()
    io.set_structure(new_structure)
    io.save(output_pdb_path)
    return output_pdb_path


def process_pdbs(raw_pdbs_dir: str, pdb_csv_dir: str, manual_csv_path: str = None) -> pd.DataFrame:
    # Find all PDB files in raw_pdbs_dir
    pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".pdb")
    ]

    # Check for Rosetta compatibility in parallel
    with ThreadPoolExecutor(max_workers=72) as executor:
        error_results = list(executor.map(throws_rosetta_error, pdb_files))
        logger.info("Files that Rosetta throws an error for:")
        for pdb_file, has_error in zip(pdb_files, error_results):
            if has_error:
                logger.info(pdb_file)

    # Remove files with Rosetta errors
    for pdb_file, has_error in zip(pdb_files, error_results):
        if has_error:
            os.remove(pdb_file)
            logger.info(f"Removed {pdb_file}")

    # Ensure peptide is in chain B for all remaining files
    remaining_pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".pdb")
    ]

    for pdb_file in remaining_pdb_files:
        output_pdb_path = os.path.join(raw_pdbs_dir, os.path.basename(pdb_file))
        ensure_peptide_is_chain_b(pdb_file, output_pdb_path)

    for pdb_file in remaining_pdb_files:
        if is_peptide_cyclic(pdb_file):
            logger.info(f"File {pdb_file} is cyclic. Removing...")
            os.remove(pdb_file)
            

    # Load the CSV data
    pdbs_df = pd.read_csv(os.path.join(pdb_csv_dir))
    
    # Load manually curated data if provided
    manual_data = None
    if manual_csv_path and os.path.exists(manual_csv_path):
        manual_data = pd.read_csv(manual_csv_path)
        # Standardize column names
        manual_data['pdb_code'] = manual_data['pdb_code'].str.replace('.pdb', '')
        logger.info(f"Loaded {len(manual_data)} manually curated entries")
    
    # Filter to only valid PDB files
    valid_pdb_filenames = set(os.listdir(raw_pdbs_dir))
    pdbs_df = pdbs_df[
        pdbs_df["pdb_code"].apply(lambda x: f"{x}.pdb" in valid_pdb_filenames)
    ]

    # Remove any manually curated PDB codes from the main dataframe 
    # We'll handle them specially
    manual_pdb_codes = []
    if manual_data is not None:
        manual_pdb_codes = manual_data["pdb_code"].unique().tolist()
        pdbs_df = pdbs_df[~pdbs_df["pdb_code"].isin(manual_pdb_codes)]
    
    # Process regular PDB entries
    logger.info(f"Processing {len(pdbs_df)} regular PDB entries")
    protein_sequences = []
    peptide_sequences = []
    
    for pdb_code in pdbs_df["pdb_code"]:
        pdb_path = os.path.join(raw_pdbs_dir, f"{pdb_code}.pdb")
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_path)
        model = structure[0]
        
        protein_sequence = ""
        peptide_sequence = ""
        
        for chain in model:
            seq = "".join(
                residue_map.get(res.resname, "X")
                for res in chain
                if res.id[0] == " "
            )
            if chain.id == "A":
                protein_sequence = seq
            elif chain.id == "B":
                peptide_sequence = seq
        
        protein_sequences.append(protein_sequence)
        peptide_sequences.append(peptide_sequence)
    
    pdbs_df["protein_sequence"] = protein_sequences
    pdbs_df["peptide_sequence"] = peptide_sequences
    
    # Now, handle manually curated entries
    if manual_data is not None and len(manual_pdb_codes) > 0:
        logger.info(f"Processing {len(manual_data)} manually curated entries")
        manual_entries = []
        
        # For each manually curated PDB, extract the protein sequence once
        protein_seq_map = {}
        for pdb_code in manual_pdb_codes:
            if f"{pdb_code}.pdb" in valid_pdb_filenames:
                pdb_path = os.path.join(raw_pdbs_dir, f"{pdb_code}.pdb")
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("complex", pdb_path)
                model = structure[0]
                
                for chain in model:
                    if chain.id == "A":
                        protein_seq = "".join(
                            residue_map.get(res.resname, "X")
                            for res in chain
                            if res.id[0] == " "
                        )
                        protein_seq_map[pdb_code] = protein_seq
                        break
        
        # Now create separate entries for each peptide sequence in manual data
        for _, row in manual_data.iterrows():
            pdb_code = row["pdb_code"]
            if pdb_code in protein_seq_map:
                new_row = row.copy()
                new_row["protein_sequence"] = protein_seq_map[pdb_code]
                # Use the peptide sequence from the CSV - this is the key fix!
                new_row["peptide_sequence"] = row["peptide_sequence"]
                manual_entries.append(new_row)
            else:
                logger.warning(f"Could not find protein sequence for manual entry: {pdb_code}")
        
        # Add the manual entries to the dataframe
        if manual_entries:
            manual_df = pd.DataFrame(manual_entries)
            pdbs_df = pd.concat([pdbs_df, manual_df], ignore_index=True)
            logger.info(f"Added {len(manual_entries)} manually curated peptide sequences")
    
    # Final check - make sure all peptides meet length requirements
    pdbs_df = pdbs_df[pdbs_df["peptide_sequence"].apply(lambda x: len(x) >= 7 and len(x) <= 40)]
    
    # Clean up
    pdbs_df.dropna(subset=["protein_sequence", "peptide_sequence"], inplace=True)
    pdbs_df.reset_index(drop=True, inplace=True)
    
    return pdbs_df

