import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
import pandas as pd

def throws_rosetta_error(pdb_file: str) -> bool:
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception:
        return True


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
        residue_count = sum(1 for residue in chain.get_residues() if residue.id[0] == " ")
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


def process_pdbs(raw_pdbs_dir: str) -> None:
    # Find all PDB files in raw_pdbs_dir
    pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".pdb")
    ]
    
    # Check for Rosetta compatibility in parallel
    with ThreadPoolExecutor(max_workers=72) as executor:
        error_results = list(executor.map(throws_rosetta_error, pdb_files))
        print("Files that Rosetta throws an error for:")
        for pdb_file, has_error in zip(pdb_files, error_results):
            if has_error:
                print(pdb_file)
    
    # Remove files with Rosetta errors
    for pdb_file, has_error in zip(pdb_files, error_results):
        if has_error:
            os.remove(pdb_file)
            print(f"Removed {pdb_file}")

    # Ensure peptide is in chain B for all remaining files
    remaining_pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".pdb")
    ]
    
    for pdb_file in remaining_pdb_files:
        output_pdb_path = os.path.join(raw_pdbs_dir, os.path.basename(pdb_file))
        ensure_peptide_is_chain_b(pdb_file, output_pdb_path)
        
    # Update pdbs.csv file to reflect only valid PDB files
    pdbs_csv_path = os.path.join(raw_pdbs_dir, "pdbs.csv")
    if os.path.exists(pdbs_csv_path):
        pdbs_df = pd.read_csv(pdbs_csv_path)
        valid_pdb_filenames = set(os.listdir(raw_pdbs_dir))
        pdbs_df = pdbs_df[pdbs_df['pdb_id'].apply(lambda x: f"{x}.pdb" in valid_pdb_filenames)]

        residue_map = {
            "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
            "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
            "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"
        }

        protein_sequences, peptide_sequences = [], []
        for pdb_id in pdbs_df['pdb_id']:
            pdb_path = os.path.join(raw_pdbs_dir, f"{pdb_id}.pdb")
            parser = PDBParser(QUIET=True)
            structure = parser.get_structure("complex", pdb_path)
            model = structure[0]  # this gets the first model

            protein_sequence, peptide_sequence = "", ""
            for chain in model:
                seq = "".join(residue_map.get(res.resname, "X")
                              for res in chain if res.id[0] == " ")
                if chain.id == "A":
                    protein_sequence = seq
                elif chain.id == "B":
                    peptide_sequence = seq

            protein_sequences.append(protein_sequence)
            peptide_sequences.append(peptide_sequence)

        pdbs_df['protein_sequence'] = protein_sequences
        pdbs_df['peptide_sequence'] = peptide_sequences

        return pdbs_df
    else:
        print(f"No pdbs.csv file found in {raw_pdbs_dir}")
        return pd.DataFrame()


