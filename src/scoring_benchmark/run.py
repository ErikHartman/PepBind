import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from utils.plots import plot_peptide_lengths
from utils.preprocessing import preprocess_pdbs
from utils.download import (
    load_files, 
    download_pdbs    
)
from utils.scoring import run_benchmark
from utils.lasso import run_lasso

# Loads path to the data directory specified by the user in the .env file. Otherwise defaults to the specified path.
try: 
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR", "/srv/data1/general/immunopeptides_data/")
except:
    raise Exception("Please set the DATA_DIR environment variable.")

INDEX_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "index"))
output_dir = os.path.abspath(os.path.join(DATA_DIR, "databases/benchmark_data/new_run"))
unprocessed_dir = os.path.join(output_dir, "0_unprocessed")
preprocessed_dir = os.path.join(output_dir, "1_preprocessed")
scored_dir = os.path.join(output_dir, "2_scored")

max_peptide_length = 40

def main():
    df_filtered = load_files()
    pdbs_dir = download_pdbs(df_filtered, max_peptide_length)
    print(f"Downloaded PDB files and removed peptides longer than {max_peptide_length} amino acids.")

    print("Commencing preprocessing of PDB files...")
    
    preprocess_pdbs(unprocessed_dir, preprocessed_dir)
    print("Preprocessing completed.")
    print(f"Number of PDB files: {len(os.listdir(pdbs_dir))}")
    plot_peptide_lengths(preprocessed_dir)
    print("Peptide length distribution plot saved.")

    print("Commencing docking and scoring benchmark...")
    run_benchmark(preprocessed_dir, output_dir) 
    print("Benchmark completed.")

    run_lasso(scored_dir, output_dir) # Only stub for now
    print("LASSO model completed.")



if __name__ == "__main__":
    main()
    

