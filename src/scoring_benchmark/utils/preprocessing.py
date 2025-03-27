import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import PDBParser, PDBIO



def is_rosetta_error(pdb_file):
    """ 
    
    Checks if the PDB file throws an error when loaded with PyRosetta 
    
    :param pdb_file: Path to the PDB file
    :return: True if an error is thrown, False otherwise

    """
    # Initialize PyRosetta with mute option to suppress all output. Turn on if needed for debugging.
    pyrosetta.init(options="-mute all")
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception as e:
        return True


def select_first_model(input_folder, output_folder):
    """
    Selects the first model from NMR structures in the PDB files if multiple,
    and writes the result to the output folder instead of modifying original files.

    :param input_folder: Path to the folder containing the PDB files
    :param output_folder: Path to the folder where processed files will be saved
                         
    :return: None
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
    return None




def ensure_peptide_is_chain_b(pdb_path, output_pdb_path):
    from Bio.PDB.Structure import Structure
    from Bio.PDB.Model import Model
    from Bio.PDB.Chain import Chain
    
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]
    
    # Determine chain lengths (only counting standard residues)
    chain_lengths = {}
    for chain in model:
        length = sum(1 for r in chain.get_residues() if r.id[0] == " ")
        chain_lengths[chain.id] = length
    
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



def preprocess_pdbs(input_dir, output_dir):
    """
    Preprocess PDB files by selecting the first NMR model from each file and checking for Rosetta compatibility.
    This function takes PDB files from the input directory, extracts the first model from each file,
    saves these models to the output directory, and then removes any files that cause errors when processed
    with Rosetta.
    
    Parameters:
    ----------
    input_dir : str
        Path to the input directory containing a "pdbs" subdirectory with PDB files to process.
    output_dir : str
        Path to the directory where processed PDB files will be saved in a "pdbs" subdirectory.
    
    Returns:
    -------
    None
        The function doesn't return any value but prints information about the processing results.
    
    Notes:
    -----
    - Uses multithreading (up to 72 workers) to check Rosetta compatibility in parallel
    - Removes PDB files that cause Rosetta errors
    - Prints summary statistics of valid and error files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    input_pdb_folder = os.path.join(input_dir, "pdbs")
    output_pdb_folder = os.path.join(output_dir, "pdbs")
    os.makedirs(output_pdb_folder, exist_ok=True)

    select_first_model(input_pdb_folder, output_pdb_folder)
    print("First models selected from all cleaned PDB files")

    # Dumps single model PDBs to output directory where we then check for Rosetta compatibility

    pdb_files = [
        os.path.join(output_pdb_folder, f)
        for f in os.listdir(output_pdb_folder)
        if f.endswith(".pdb")
    ]
    
    # Check for Rosetta compatibility


    with ThreadPoolExecutor(max_workers=72) as executor:
        
        error_files = list(executor.map(is_rosetta_error, pdb_files))
        print("Files that Rosetta throws an error for:")
        for pdb_file, is_error in zip(pdb_files, error_files):
            if is_error:
                print(pdb_file)

    
    
    for pdb_file, is_error in zip(pdb_files, error_files):
        if is_error:
            os.remove(pdb_file)
            print(f"Removed {pdb_file}")
    
    valid_count = len([f for f in error_files if not f])
    error_count = len([f for f in error_files if f])


    # Ensure peptide is in chain B
    for pdb_file in pdb_files:
        output_pdb_path = os.path.join(output_pdb_folder, os.path.basename(pdb_file))
        ensure_peptide_is_chain_b(pdb_file, output_pdb_path)


    # Using the pdbs.csv from the input directory, filter out the files that are not in the output directory and output the new file to the output directory
    # This is to ensure that the pdbs.csv file is in sync with the PDB files in the output directory
    pdbs_csv_path = os.path.join(input_dir, "pdbs.csv")
    if os.path.exists(pdbs_csv_path):
        with open(pdbs_csv_path, "r") as f:
            pdbs_csv = f.readlines()
        pdbs_csv = [line for line in pdbs_csv if line.strip().split(",")[0] + ".pdb" in os.listdir(output_pdb_folder)]
        with open(os.path.join(output_dir, "pdbs.csv"), "w") as f:
            f.writelines(pdbs_csv)
    else:
        print(f"No pdbs.csv file found in {input_dir}")

    # Report results
    print(f"Kept {valid_count} valid files in {output_pdb_folder}")
    print(f"Removed {error_count} files with Rosetta errors")
    return None


