import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor


# This script cleans the downloaded PDBs retrieved with curate_benchmark_data.py. It removes any PDBs that Rosetta throws an error for when loading, and removes any models after the first one in NMR models.

input_folder = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs"
output_folder = os.path.join(input_folder, "../cleaned_pdbs")

def is_rosetta_error(pdb_file):
    pyrosetta.init()
    try:
        pyrosetta.pose_from_file(pdb_file)
        return False
    except Exception as e:
        return True
    


# Some of the PDBs are NMR models, which have multiple models in the same file. We only want to keep the first model. Filter through cleaned PDBs and keep only the first model.
# This function will remove all models after the first one in each cleaned PDB file.
def select_first_model(input_folder):
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]
    for filename in pdb_files:
        with open(os.path.join(input_folder, filename), 'r') as file:
            lines = file.readlines()
        
        with open(os.path.join(input_folder, filename), 'w') as file:
            for line in lines:
                if line.startswith("ENDMDL"):
                    break
                file.write(line)



if __name__ == "__main__":

    select_first_model(input_folder)
    print("First models selected from all cleaned PDB file")


    pdb_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(".pdb")]
    with ThreadPoolExecutor(max_workers=72) as executor:
        error_files = list(executor.map(is_rosetta_error, pdb_files))
        print("Files that Rosetta throws an error for:")
        for pdb_file, is_error in zip(pdb_files, error_files):
            if is_error:
                print(pdb_file)
        # Remove all files that Rosetta throws an error for (e.g non-standard residues)
        for pdb_file, is_error in zip(pdb_files, error_files):
            if is_error:
                os.remove(pdb_file)
                print(f"Removed {pdb_file}")

        
