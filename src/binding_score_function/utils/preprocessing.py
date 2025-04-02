import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple
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
    """ 
    Checks if the PDB file throws an error when loaded with PyRosetta 
    
    Args:
        pdb_file: Path to the PDB file
    
    Returns:
        True if an error is thrown, False otherwise
    """
    # Initialize PyRosetta with mute option to suppress all output. Turn on if needed for debugging.
    
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception:
        return True


def select_first_model(input_folder: str, output_folder: Optional[str] = None) -> None:
    """
    Selects the first model from NMR structures in the PDB files if multiple,
    and writes the result to the output folder instead of modifying original files.

    Args:
        input_folder: Path to the folder containing the PDB files
        output_folder: Path to the folder where processed files will be saved
                      (defaults to a new folder to avoid overwriting)
    """
    # Create a default output folder to avoid overwriting input files
    if output_folder is None or output_folder == input_folder:
        output_folder = os.path.join(os.path.dirname(input_folder), "first_model_pdbs")
        print(f"Warning: Using {output_folder} to avoid overwriting original files")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]
    for filename in pdb_files:
        with open(os.path.join(input_folder, filename), "r") as file:
            lines = file.readlines()

        with open(os.path.join(output_folder, filename), "w") as file:
            for line in lines:
                if line.startswith("ENDMDL"):
                    break
                file.write(line)


def ensure_peptide_is_chain_b(pdb_path: str, output_pdb_path: str) -> str:
    """
    Ensures that the shortest chain is labeled as chain B (peptide) and
    the longest chain is labeled as chain A (protein).
    
    Args:
        pdb_path: Path to the input PDB file
        output_pdb_path: Path to save the output PDB file
    
    Returns:
        Path to the output PDB file
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


def process_pdbs(input_dir: str, output_dir: str) -> None:
    """
    Preprocess PDB files by selecting the first NMR model from each file and checking for Rosetta compatibility.
    This function takes PDB files from the input directory, extracts the first model from each file,
    saves these models to the output directory, and then removes any files that cause errors when processed
    with Rosetta.
    
    Args:
        input_dir: Path to the input directory containing a "pdbs" subdirectory with PDB files to process
        output_dir: Path to the directory where processed PDB files will be saved in a "pdbs" subdirectory
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    input_pdb_folder = os.path.join(input_dir, "pdbs")
    output_pdb_folder = os.path.join(output_dir, "pdbs")
    os.makedirs(output_pdb_folder, exist_ok=True)

    select_first_model(input_pdb_folder, output_pdb_folder)
    print("First models selected from all cleaned PDB files")

    # Find all PDB files in output directory
    pdb_files = [
        os.path.join(output_pdb_folder, filename)
        for filename in os.listdir(output_pdb_folder)
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
        os.path.join(output_pdb_folder, filename)
        for filename in os.listdir(output_pdb_folder)
        if filename.endswith(".pdb")
    ]
    
    for pdb_file in remaining_pdb_files:
        output_pdb_path = os.path.join(output_pdb_folder, os.path.basename(pdb_file))
        ensure_peptide_is_chain_b(pdb_file, output_pdb_path)

    # Update pdbs.csv file to reflect only valid PDB files
    pdbs_csv_path = os.path.join(input_dir, "pdbs.csv")
    if os.path.exists(pdbs_csv_path):
        with open(pdbs_csv_path, "r") as f:
            pdbs_csv_lines = f.readlines()
        
        valid_pdb_filenames = set(os.listdir(output_pdb_folder))
        filtered_csv_lines = [
            line for line in pdbs_csv_lines 
            if line.strip().split(",")[0] + ".pdb" in valid_pdb_filenames
        ]
        
        with open(os.path.join(output_dir, "pdbs.csv"), "w") as f:
            f.writelines(filtered_csv_lines)
    else:
        print(f"No pdbs.csv file found in {input_dir}")

    # Report results
    print(f"Kept {valid_count} valid files in {output_pdb_folder}")
    print(f"Removed {error_count} files with Rosetta errors")


