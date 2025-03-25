import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from utils.plots import plot_peptide_lengths
from utils.preprocessing import clean_pdbs
from utils.download import {
    download_pdbs
    load_files
}

# Loads path to the data directory specified by the user in the .env file
try: 
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")
except:
    raise Exception("Please set the DATA_DIR environment variable.")

INDEX_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "index"))
output_dir = os.path.abspath(os.path.join(DATA_DIR, "databases/benchmark_data/new_run"))



def preprocess_pdbs_dir(pdbs_dir):
    clean_pdbs(pdbs_dir)
    print("PDB files downloaded and preprocessed.")
    print(f"Number of PDB files: {len(os.listdir(pdbs_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scoring benchmark workflow.")
    parser.add_argument('-d', '--download', action='store_true', help='Parse INDEX files and downloads the PDB files')
    parser.add_argument('-plt', '--plot', action='store_true', help='Plot peptide lengths')
    parser.add_argument('--skip_preprocess', action='store_true', help='Skip preprocessing of PDB files')

    args = parser.parse_args()

    if args.download:
        df_filtered = load_files()
        pdbs_dir = download_pdbs(df_filtered)
    if not args.skip_preprocess:
        if 'pdbs_dir' not in locals():
            if 'df_filtered' not in locals():
                df_filtered = load_files()
            pdbs_dir = download_pdbs(df_filtered)
        preprocess_pdbs_dir(pdbs_dir)
    if args.plot:
        pdbs_dir = os.path.join(output_dir, "pdbs")
        if pdbs_dir is None:
            raise Exception("PDB directory is empty.")
        plot_peptide_lengths(pdbs_dir)

    if not any(vars(args).values()):
        df_filtered = load_files()
        pdbs_dir = download_pdbs(df_filtered)
        preprocess_pdbs_dir(pdbs_dir)
        plot_peptide_lengths(pdbs_dir)
