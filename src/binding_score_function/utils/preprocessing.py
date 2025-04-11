import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from Bio.PDB import PDBParser, PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain

"""
Utils for preprocessing PDB files for PyRosetta compatibility.
This includes selecting the first model from NMR structures,
checking for PyRosetta errors, and ensuring peptide chains are correctly labeled.
"""


def is_rosetta_error(pdb_file: str) -> bool:
    # Initialize PyRosetta with mute option to suppress all output. Turn on if needed for debugging.
    
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


def process_pdbs(raw_pdbs_dir: str, processed_dir: str) -> None:
    # Find all PDB files in raw_pdbs_dir
    pdb_files = [
        os.path.join(raw_pdbs_dir, filename)
        for filename in os.listdir(raw_pdbs_dir)
        if filename.endswith(".pdb")
    ]
    
    # Check for Rosetta compatibility in parallel
    with ThreadPoolExecutor(max_workers=72) as executor:
        error_results = list(executor.map(is_rosetta_error, pdb_files))
        print("Files that Rosetta throws an error for:")
        for pdb_file, has_error in zip(pdb_files, error_results):
            if has_error:
                print(pdb_file)
    
    # Remove files with Rosetta errors
    for pdb_file, has_error in zip(pdb_files, error_results):
        if has_error:
            os.remove(pdb_file)
            print(f"Removed {pdb_file}")
    
    valid_count = len([result for result in error_results if not result])
    error_count = len([result for result in error_results if result])

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
        with open(pdbs_csv_path, "r") as f:
            pdbs_csv_lines = f.readlines()
        header = pdbs_csv_lines[0]
        valid_pdb_filenames = set(os.listdir(raw_pdbs_dir))
        filtered_csv_lines = [
            line for line in pdbs_csv_lines 
            if line.strip().split(",")[0] + ".pdb" in valid_pdb_filenames
        ]
        header = header.strip() + ",protein_sequence,peptide_sequence\n"

        residue_map = {
            "ALA":"A","ARG":"R","ASN":"N","ASP":"D","CYS":"C","GLN":"Q","GLU":"E","GLY":"G",
            "HIS":"H","ILE":"I","LEU":"L","LYS":"K","MET":"M","PHE":"F","PRO":"P","SER":"S",
            "THR":"T","TRP":"W","TYR":"Y","VAL":"V"
        }

        with open(os.path.join(processed_dir, "pdbs.csv"), "w") as f:
            f.write(header)
            for line in filtered_csv_lines:
                pdb_id = line.strip().split(",")[0]
                pdb_path = os.path.join(raw_pdbs_dir, pdb_id + ".pdb")
                parser = PDBParser(QUIET=True)
                structure = parser.get_structure("complex", pdb_path)
                model = structure[0] # this gets the first model

                protein_sequence, peptide_sequence = "", ""
                for chain in model:
                    seq = "".join(residue_map.get(res.resname, "X")
                                  for res in chain if res.id[0] == " ")
                    if chain.id == "A":
                        protein_sequence = seq
                    elif chain.id == "B":
                        peptide_sequence = seq

                f.write(line.strip() + f",{protein_sequence},{peptide_sequence}\n")
    else:
        print(f"No pdbs.csv file found in {raw_pdbs_dir}")

    # Report results
    print(f"Wrote {len(pdb_files)} PDB files to {processed_dir}")
    print(f"Kept {valid_count} valid files in {raw_pdbs_dir}")
    print(f"Removed {error_count} files with Rosetta errors")


