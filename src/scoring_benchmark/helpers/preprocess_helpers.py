import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor


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

# Currently not implemented in clean_pdbs
def harmonize_chains(pdb_file):
    """
    
    Ensures that the peptide chain in the PDB file is labeled as chain 'B'.
    If the longest chain is on the B strand, it flips A and B.

    :param pdb_file: Path to the PDB file

    """
    with open(pdb_file, "r") as file:
        lines = file.readlines()

    chain_lengths = {"A": 0, "B": 0}
    for line in lines:
        if line.startswith("ATOM") or line.startswith("HETATM"):
            chain_id = line[21]
            if chain_id in chain_lengths:
                chain_lengths[chain_id] += 1

    if chain_lengths["B"] > chain_lengths["A"]:
        with open(pdb_file, "w") as file:
            for line in lines:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    if line[21] == "A":
                        line = line[:21] + "B" + line[22:]
                    elif line[21] == "B":
                        line = line[:21] + "A" + line[22:]
                file.write(line)


def find_pdbs_with_longer_B(directory):
    """Finds PDB files where Chain B is longer than Chain A based on SEQRES records."""
    longer_B_files = []

    for pdb_file in os.listdir(directory):
        if pdb_file.endswith(".pdb"):
            chain_lengths = {"A": 0, "B": 0}

            with open(os.path.join(directory, pdb_file), "r") as f:
                for line in f:
                    if line.startswith("SEQRES"):
                        parts = line.split()
                        chain = parts[2]
                        num_residues = int(parts[3])

                        if chain in chain_lengths:
                            chain_lengths[chain] = num_residues

            if chain_lengths["B"] > chain_lengths["A"]:
                longer_B_files.append(pdb_file)

    return longer_B_files






def clean_pdbs(input_folder):
    """

    Cleans the PDB files in the input folder. User confirmation is required to remove the files.

    Removes any PDBs causing errors with Rosetta using is_rosetta_error.
    Removes any models after the first one in NMR models using select_first_model.

    :param input_folder: Path to the folder containing the PDB files
    :return: None



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

