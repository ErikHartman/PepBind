import os
import pyrosetta
from concurrent.futures import ThreadPoolExecutor

def clean_pdb(input_pdb, output_pdb):
    pyrosetta.init("-ignore_unrecognized_res true")
    try:
        pose = pyrosetta.pose_from_pdb(input_pdb)
        pose.dump_pdb(output_pdb)
        print(f"Cleaned PDB saved to {output_pdb}")
    except Exception as e:
        print(f"Error processing {input_pdb}: {e}")

def clean_pdbs_in_folder(input_folder):
    output_folder = os.path.join(input_folder, "../cleaned_pdbs")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    pdb_files = [f for f in os.listdir(input_folder) if f.endswith(".pdb")]
    with ThreadPoolExecutor(max_workers=72) as executor:
        futures = []
        for filename in pdb_files:
            input_pdb = os.path.join(input_folder, filename)
            output_pdb = os.path.join(output_folder, f"cleaned_{filename}")
            futures.append(executor.submit(clean_pdb, input_pdb, output_pdb))
        
        for future in futures:
            future.result()  # Wait for all threads to complete

# Cleans PDB files in the input folder
input_folder = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs"
# clean_pdbs_in_folder(input_folder)


def count_cleaned_pdbs(output_folder):
    pdb_files = [f for f in os.listdir(output_folder) if f.endswith(".pdb")]
    return len(pdb_files)

output_folder = os.path.join(input_folder, "../cleaned_pdbs")
num_cleaned_pdbs = count_cleaned_pdbs(output_folder)
print(f"Number of cleaned PDB files: {num_cleaned_pdbs}")