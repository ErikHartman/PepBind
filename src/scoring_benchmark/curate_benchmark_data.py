import os
import io
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd

from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa
import numpy as np

def is_peptide_cyclic(pdb_file, cutoff=1.7):
    """
    Detect if the peptide (specified by chain ID) in the given PDB file is cyclic.
    Checks if there's a covalent bond (distance < cutoff Å) between the
    C-terminal carbonyl carbon and N-terminal nitrogen atoms.

    :param pdb_file: Path to PDB file
    :param peptide_chain_id: Chain ID of the peptide
    :param cutoff: Maximum distance (in Å) to consider atoms covalently bonded
    :return: True if cyclic, False otherwise
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_file)
    model = structure[0]


    chain = model["B"]

    residues = [res for res in chain if is_aa(res, standard=True)]

    # N-terminal residue N atom
    n_term_res = residues[0]
    n_atom = n_term_res['N'] if 'N' in n_term_res else None

    # C-terminal residue C atom
    c_term_res = residues[-1]
    c_atom = c_term_res['C'] if 'C' in c_term_res else None

    if n_atom is None or c_atom is None:
        # Missing backbone atoms, cannot confirm cyclicity
        return False

    distance = np.linalg.norm(n_atom.coord - c_atom.coord)

    # Typical covalent bond distances C-N ~1.3-1.5Å, using 1.7Å as generous cutoff
    is_cyclic = distance <= cutoff
    return is_cyclic


def has_peptide_and_protein(pdb_contents, peptide_max_length=40):
    """
    Checks whether the PDB (provided as a string) contains exactly two polypeptide chains:
      1) One chain with fewer than `peptide_max_length` amino acids,
      2) One chain with >= `peptide_max_length` amino acids.

    Returns:
      (bool, str): 
        bool -> indicates if condition is met
        str  -> reason for exclusion if not met (or "meets_condition" if passed)
    """
    parser = PDB.PDBParser(QUIET=True)

    pdb_file_handle = io.StringIO(pdb_contents)
    structure = parser.get_structure("temp_struct", pdb_file_handle)

    chain_residue_counts = []
    model = structure[0]  

    for chain in model:
        count_aa = 0
        for residue in chain.get_residues():
            if is_aa(residue, standard=True):
                count_aa += 1
        if count_aa > 0:
            chain_residue_counts.append(count_aa)

    if len(chain_residue_counts) != 2:
        return False, "not_two_chains"

    chain_residue_counts.sort()
    if chain_residue_counts[0] >= peptide_max_length:
        return False, "no_peptide"
    if chain_residue_counts[1] < peptide_max_length:
        return False, "no_protein"

    return True, "meets_condition"


def download_and_check_pdb(pdb_id, output_dir, peptide_max_length=40):
    """
    Downloads a PDB from RCSB, checks if it has exactly two chains
    (peptide+protein) using `has_peptide_and_protein`.
    Also checks if the peptide is cyclic. If so, it is not saved.
    If it meets the condition, saves it to `output_dir`.

    :param pdb_id: PDB code (e.g. '1ABC')
    :param output_dir: Directory where PDB files should be saved
    :param peptide_max_length: length threshold that separates peptide from protein
    :return: (bool, str) => (did_save, reason)
             did_save is True if PDB was saved, False otherwise.
             reason is "meets_condition" or the reason for exclusion.
    """
    url = f'https://files.rcsb.org/download/{pdb_id}.pdb'
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdb_text = response.text
            passes, reason = has_peptide_and_protein(pdb_text, peptide_max_length=peptide_max_length)
            is_cyclic = is_peptide_cyclic(io.StringIO(pdb_text))
            if passes and (not is_cyclic): # Only save if passes criterion
                file_path = os.path.join(output_dir, f'{pdb_id}.pdb')
                with open(file_path, 'w') as file:
                    file.write(pdb_text)
                print(f'{pdb_id}.pdb was downloaded', end="\r")
                return True, reason
            else:
                print(f'{pdb_id}.pdb wasnt downloaded', end="\r")
                return False, reason
        else:
            return False, f"download_failed_{response.status_code}"
    except Exception as e:
        return False, f"download_exception_{str(e)}"


def download_pdbs_in_batch(pdb_ids, output_dir, peptide_max_length=40, max_workers=5):
    """
    Downloads multiple PDB IDs in parallel (up to `max_workers` threads),
    checks if each meets the "peptide+protein" condition, and saves only
    those that pass.
    
    :param pdb_ids: iterable of PDB codes
    :param output_dir: directory where passing PDB files get stored
    :param peptide_max_length: length threshold to distinguish peptide vs protein
    :param max_workers: number of parallel downloads
    """
    os.makedirs(output_dir, exist_ok=True)
    results = []

    total_attempts = len(pdb_ids)
    saved_count = 0
    reason_counts = {}

    def worker(pdb_id):
        did_save, reason = download_and_check_pdb(
            pdb_id=pdb_id,
            output_dir=output_dir,
            peptide_max_length=peptide_max_length
        )
        return pdb_id, did_save, reason

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for pdb_id, did_save, reason in executor.map(worker, pdb_ids):
            results.append((pdb_id, did_save, reason))
    
    for pdb_id, did_save, reason in results:
        if did_save:
            saved_count += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    print(f"Total PDBs attempted: {total_attempts}")
    print(f"PDBs saved (pass criterion): {saved_count}")
    print("Reasons for skip/failure:")
    for k, v in reason_counts.items():
        if k != "meets_condition":
            print(f"  {k}: {v}")

    print(f"  meets_condition: {reason_counts.get('meets_condition', 0)}")


if __name__ == "__main__":

    data_dir = os.path.abspath("/srv/data1/general/immunopeptides_data/")

    protein_protein_df = pd.read_csv(os.path.join(data_dir, 'databases/benchmark_data/peptide_like_proteins.csv'))
    protein_protein_df['source'] = 'protein_protein'

    protein_ligand_df = pd.read_csv(os.path.join(data_dir, 'databases/benchmark_data/merged_peptide_like_ligands.csv'))
    protein_ligand_df['source'] = 'protein_ligand'
 
    peptides_df = pd.concat([protein_protein_df, protein_ligand_df])

    peptides_to_parse = peptides_df["PDB code"].unique()
    
    output_dir = os.path.join(data_dir, "databases/benchmark_data/pdbs")

    if not len(os.listdir(output_dir)) == 0:
        print(f"Output directory contains {len(os.listdir(output_dir))} files.")
        print(f"Please clear it before running this script.")
        exit()

    os.makedirs(output_dir, exist_ok=True)

    download_pdbs_in_batch(
        pdb_ids=peptides_to_parse,
        output_dir=output_dir,
        peptide_max_length=40,
        max_workers=12
    )
