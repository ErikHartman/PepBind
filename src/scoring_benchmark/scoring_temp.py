# This file is for scoring files without running the entire flow. 
# #It is currently used for building out the LASSO regression model.

import os
import dotenv

dotenv.load_dotenv()

DATA_DIR = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")

from utils.scoring import run_benchmark
from bopep.docking.utils import extract_sequence_from_pdb
from utils.preprocessing import ensure_peptide_is_chain_b

old_run_pdbs = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/1_preprocessed/test_pdbs")
clean_pdbs = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/1_preprocessed/test_pdbs/test_pdbs_clean")
test_scores = os.path.join(DATA_DIR, "databases/benchmark_data/new_run/2_scored")

def main():

    # Create clean_pdbs directory if it doesn't exist
    os.makedirs(clean_pdbs, exist_ok=True)
    
    # Process all PDB files in old_run_pdbs directory
    for filename in os.listdir(old_run_pdbs):
        if filename.endswith('.pdb'):
            input_path = os.path.join(old_run_pdbs, filename)
            output_path = os.path.join(clean_pdbs, filename)
            ensure_peptide_is_chain_b(input_path, output_path)

    # Run the benchmark on the cleaned PDB files only for PDBs with peptide chains in 2_scored/docked_peptides

    for filename in os.listdir(clean_pdbs):
        if filename.endswith('.pdb'):
            input_path = os.path.join(clean_pdbs, filename)
            output_path = os.path.join(test_scores, filename)
            ensure_peptide_is_chain_b(input_path, output_path)
            # Extract peptide sequence
            peptide_sequence = extract_sequence_from_pdb(output_path, chain_id="B")
            docked_peptides_dir = os.path.join(test_scores, "pdbs/docked_peptides")
            if os.path.exists(docked_peptides_dir):
                # Check if peptide sequence exists in any directory name in docked_peptides
                peptide_found = False
                for dirname in os.listdir(docked_peptides_dir):
                    if peptide_sequence in dirname:
                        peptide_found = True
                        break
                
                # If peptide sequence not found in any directory name, remove the PDB file
                if not peptide_found:
                    os.remove(input_path)
                    print(f"Removed {filename} as peptide sequence {peptide_sequence} was not found in docked_peptides")


    
            


    run_benchmark(clean_pdbs, test_scores)

if __name__ == "__main__":
    main()