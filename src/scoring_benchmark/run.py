import os
import pandas as pd
from dotenv import load_dotenv 

from utils.plots import plot_peptide_lengths
from utils.preprocessing import preprocess_pdbs
from utils.download import download_pdbs    
from utils.scoring import run_benchmark
from utils.lasso import run_lasso

# Loads path to the data directory specified by the user in the .env file. Otherwise defaults to the specified path.
try: 
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")
except:
    raise Exception("Please set the DATA_DIR environment variable. Create a .env file in the root directory and set DATA_DIR to the path of the data directory.")

INDEX_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "index"))
output_dir = os.path.abspath(os.path.join(DATA_DIR, "databases/benchmark_data/new_run"))

# Staged directories to allow for easy tracking of progress and avoid overwriting files
unprocessed_dir = os.path.join(output_dir, "0_unprocessed")
preprocessed_dir = os.path.join(output_dir, "1_preprocessed")
scored_dir = os.path.join(output_dir, "2_scored")

max_peptide_length = 40

def main():
    download_pdbs(INDEX_dir, unprocessed_dir, max_peptide_length)
    

    print("Commencing preprocessing of PDB files...")
    
    preprocess_pdbs(unprocessed_dir, preprocessed_dir)
    print("Preprocessing completed.")
    print(f"Number of PDB files: {len(os.listdir(os.path.join(preprocessed_dir, 'pdbs')))}")
    plot_peptide_lengths(preprocessed_dir)
    print("Peptide length distribution plot saved.")

    print("Commencing docking and scoring benchmark...")
    # run_benchmark(preprocessed_dir, scored_dir) 
    print("Benchmark completed.")

    # run_lasso(scored_dir, output_dir) # Only stub for now
    print("LASSO model completed.")



if __name__ == "__main__":
    main()
    

