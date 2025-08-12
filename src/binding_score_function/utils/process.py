import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import MMCIFParser

import pandas as pd
import logging
import numpy as np
from Bio.PDB.Polypeptide import is_aa


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
    # Try loading directly (Rosetta may support CIF)
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception:
        pass

    return True

def is_peptide_cyclic(pdb_file, cutoff: float = 1.7) -> bool:
    """
    Detect if the peptide (chain B) in the given CIF file is cyclic.
    Checks if there's a covalent bond (distance < cutoff Å) between the
    C-terminal carbonyl carbon and N-terminal nitrogen atoms.
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_file)
    model = structure[0]
    
    try:
        chain = model["B"]
    except KeyError:
        return False

    residues = [res for res in chain if is_aa(res, standard=True)]
    
    # Check if there are at least 2 residues (needed for cyclic check)
    if len(residues) < 2:
        return False

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

def process_pdbs(raw_pdbs_dir: str, pdb_csv_dir: str, manual_csv_path: str = None) -> pd.DataFrame:
    pdbs_df = pd.read_csv(os.path.join(pdb_csv_dir))

    # If the CSV is empty or lacks 'pdb_code', infer from filenames
    if pdbs_df.empty or "pdb_code" not in pdbs_df.columns:
        logger.warning("pdbs.csv is empty or missing 'pdb_code'. Inferring codes from CIF files in raw directory.")
        inferred_codes = [
            os.path.splitext(f)[0]
            for f in os.listdir(raw_pdbs_dir)
            if f.endswith(".cif")
        ]
        pdbs_df = pd.DataFrame({"pdb_code": inferred_codes})

    pdb_files_in_raw_pdbs_dir = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".cif")
    ]

    # Check for Rosetta compatibility in parallel
    with ThreadPoolExecutor(max_workers=72) as executor:
        error_results = list(executor.map(throws_rosetta_error, pdb_files_in_raw_pdbs_dir))
        logger.info("Files that Rosetta throws an error for:")
        for pdb_file, has_error in zip(pdb_files_in_raw_pdbs_dir, error_results):
            if has_error:
                logger.info(pdb_file)

    # Remove files with Rosetta errors
    for pdb_file, has_error in zip(pdb_files_in_raw_pdbs_dir, error_results):
        if has_error:
            os.remove(pdb_file)
            logger.info(f"Removed {pdb_file}")

    remaining_pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".cif")
    ]

    for pdb_file in remaining_pdb_files:
        if is_peptide_cyclic(pdb_file):
            logger.info(f"File {pdb_file} is cyclic. Removing...")
            os.remove(pdb_file)

    remaining_pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".cif")]
            
    
    # Load manually curated data if provided
    manual_data = None
    if manual_csv_path:
        manual_data = pd.read_csv(manual_csv_path)
        manual_data['pdb_code'] = manual_data['pdb_code'].str.replace('.pdb', '')
        logger.info(f"Loaded {len(manual_data)} manually curated entries")
    
    # Filter to only valid CIF files
    valid_pdb_filenames = set(os.listdir(raw_pdbs_dir))

    pdbs_df = pdbs_df[
        pdbs_df["pdb_code"].apply(lambda x: f"{x}.cif" in valid_pdb_filenames)
    ]

    manual_pdb_codes = []
    if manual_data is not None:
        manual_pdb_codes = manual_data["pdb_code"].unique().tolist()
        pdbs_df = pdbs_df[~pdbs_df["pdb_code"].isin(manual_pdb_codes)]
    
    # Process regular CIF entries
    logger.info(f"Processing {len(pdbs_df)} regular CIF entries")
    protein_sequences = []
    peptide_sequences = []
    
    for pdb_code in pdbs_df["pdb_code"]:
        pdb_path = os.path.join(raw_pdbs_dir, f"{pdb_code}.cif")
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("complex", pdb_path)
        model = structure[0]

            # Only allow chains A and B
        allowed_chains = {"A", "B"}
        present_chains = {chain.id for chain in model}
        if not present_chains.issubset(allowed_chains):
            logger.info(f"Skipping {pdb_code}: contains chains other than A and B ({present_chains})")
            protein_sequences.append("")
            peptide_sequences.append("")
            continue
        
        # Build a dict of chain_id -> sequence
        chain_seqs = {}
        for chain in model:
            seq = "".join(
                residue_map.get(res.resname, "X")
                for res in chain
                if res.id[0] == " "
            )
            chain_seqs[chain.id] = seq
        # Sort chains by length
        sorted_chains = sorted(chain_seqs.items(), key=lambda x: len(x[1]))
        if len(sorted_chains) >= 2:
            peptide_sequence = sorted_chains[0][1]
            protein_sequence = sorted_chains[-1][1]
        else:
            peptide_sequence = ""
            protein_sequence = ""
        protein_sequences.append(protein_sequence)
        peptide_sequences.append(peptide_sequence)
    
    pdbs_df["protein_sequence"] = protein_sequences
    pdbs_df["peptide_sequence"] = peptide_sequences
    
    # Now, handle manually curated entries
    if manual_data is not None and len(manual_pdb_codes) > 0:
        logger.info(f"Processing {len(manual_data)} manually curated entries")
        manual_entries = []
        
        # For each manually curated CIF, extract the protein sequence once
        protein_seq_map = {}
        for pdb_code in manual_pdb_codes:
            if f"{pdb_code}.cif" in valid_pdb_filenames:
                pdb_path = os.path.join(raw_pdbs_dir, f"{pdb_code}.cif")
                parser = MMCIFParser(QUIET=True)
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
    pdbs_df.dropna(subset=["protein_sequence", "peptide_sequence"], inplace=True)
    pdbs_df.reset_index(drop=True, inplace=True)
    
    return pdbs_df

