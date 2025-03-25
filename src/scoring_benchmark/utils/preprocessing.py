import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import PDBParser, PDBIO
import os


from ..run import output_dir, INDEX_dir



def is_rosetta_error(pdb_file):
    """ 
    
    Checks if the PDB file throws an error when loaded with PyRosetta 
    
    :param pdb_file: Path to the PDB file
    :return: True if an error is thrown, False otherwise

    """
    pyrosetta.init()
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception as e:
        return True


def select_first_model(input_folder):
    """
    
    Selects the first model from NMR structures in the PDB files if multiple.

    :param input_folder: Path to the folder containing the PDB files
    :return: None
    """
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]
    for filename in pdb_files:
        with open(os.path.join(input_folder, filename), "r") as file:
            lines = file.readlines()

        with open(os.path.join(input_folder, filename), "w") as file:
            for line in lines:
                if line.startswith("ENDMDL"):
                    break
                file.write(line)




def ensure_peptide_is_chain_b(pdb_path, output_pdb_path):
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
    
    # Rename chain IDs if needed
    for chain in model:
        if chain.id == peptide_chain_id:
            chain.id = "B"
        elif chain.id == protein_chain_id:
            chain.id = "A"
            
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path)
    return output_pdb_path



def preprocess_pdbs(input_folder):
    """

    1. Removes any PDBs causing errors with Rosetta. Requires confirmation before removing files.
    2. Selects the first model from NMR structures. 
    3. Ensures that the peptide is in chain B.
    4. Split the PDB files into protein and peptide chains.
    
    
    

    :param input_folder: Path to the folder containing the PDB files
    :return: output_folder: Path to the folder containing the preprocessed PDB files

    """
    select_first_model(input_folder)
    print("First models selected from all cleaned PDB files")

    pdb_files = [
        os.path.join(input_folder, f)
        for f in os.listdir(input_folder)
        if f.endswith(".pdb")
    ]
    with ThreadPoolExecutor(max_workers=72) as executor:
        error_files = list(executor.map(is_rosetta_error, pdb_files))
        print("Files that Rosetta throws an error for:")
        for pdb_file, is_error in zip(pdb_files, error_files):
            if is_error:
                print(pdb_file)

    confirm = input("Confirm removal of the above (y/n): ")
    if confirm.lower() == 'y':
        for pdb_file, is_error in zip(pdb_files, error_files):
            if is_error:
                os.remove(pdb_file)
                print(f"Removed {pdb_file}")
    else:
        print("Aborted file removal.")


