import os
import argparse
import pandas as pd
from dotenv import load_dotenv

from utils.plots import plot_peptide_lengths
from utils.preprocessing import clean_pdbs
from utils.download import ( 
    convert_to_dataframe, 
    filter_large_binders, 
    download_pdbs_in_batch
)

# Loads path to the data directory specified by the user in the .env file
try: 
    load_dotenv()
    DATA_DIR = os.getenv("DATA_DIR")
except:
    raise Exception("Please set the DATA_DIR environment variable.")

INDEX_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "index"))
output_dir = os.path.abspath(os.path.join(DATA_DIR, "databases/benchmark_data/new_run"))

def load_files():
    protein_ligands = os.path.join(INDEX_dir, "INDEX_PL.2020")
    protein_protein = os.path.join(INDEX_dir, "INDEX_PP.2020")
    df_pl = convert_to_dataframe(protein_ligands)
    df_pp = convert_to_dataframe(protein_protein)
    df = pd.concat([df_pl, df_pp])
    df_filtered = filter_large_binders(df)
    print("Loaded INDEX files and removed large binders.")
    print(df_filtered.head())
    return df_filtered

def download_pdbs(df_filtered):
    pdbs_dir = os.path.join(output_dir, "pdbs")
    if not os.listdir(pdbs_dir):
        download_pdbs_in_batch(df_filtered["PDB code"], pdbs_dir)
    else:
        print(f"{pdbs_dir} is not empty. Skipping download.")
    return pdbs_dir

def preprocess_pdbs_dir(pdbs_dir):
    clean_pdbs(pdbs_dir)
    print("PDB files downloaded and preprocessed.")
    print(f"Number of PDB files: {len(os.listdir(pdbs_dir))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the scoring benchmark workflow.")
    parser.add_argument('-l', '--load', action='store_true', help='Load the index files')
    parser.add_argument('-d', '--download', action='store_true', help='Download PDB files')
    parser.add_argument('-p', '--preprocess', action='store_true', help='Preprocess PDB files for docking')
    parser.add_argument('-plt', '--plot', action='store_true', help='Plot peptide lengths')

    args = parser.parse_args()

    if args.load:
        df_filtered = load_files()
    if args.download:
        if 'df_filtered' not in locals():
            df_filtered = load_files()
        pdbs_dir = download_pdbs(df_filtered)
    if args.preprocess:
        if 'pdbs_dir' not in locals():
            if 'df_filtered' not in locals():
                df_filtered = load_files()
            pdbs_dir = download_pdbs(df_filtered)
        preprocess_pdbs_dir(pdbs_dir)

    if args.plot:
        if 'pdbs_dir' not in locals():
            if 'df_filtered' not in locals():
                df_filtered = load_files()
            pdbs_dir = download_pdbs(df_filtered)
            plot_peptide_lengths(pdbs_dir)
    
    else:
        preprocess_pdbs_dir(pdbs_dir)
        df_filtered = load_files()
        pdbs_dir = download_pdbs(df_filtered)
        plot_peptide_lengths(pdbs_dir)
        

