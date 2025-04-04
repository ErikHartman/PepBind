import os
import io
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import Dict, List, Tuple, Union
import numpy as np

from Bio import PDB
from Bio.PDB import PDBParser
from Bio.PDB.Polypeptide import is_aa

"""
Utils for downloading PDB files from RCSB
and filtering them based on peptide/protein criteria.
"""


def convert_to_index_file_to_dataframe(input_file: str) -> pd.DataFrame:
    """
    Converts the PDB-bind INDEX file to a pandas DataFrame
    
    Args:
        input_file: Path to the PDB-bind INDEX file
    
    Returns:
        DataFrame with PDB-bind data
    """
    data = []
    with open(input_file, "r") as infile:
        for line in infile:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pdb_code = parts[0]
            resolution = parts[1]
            release_year = parts[2]
            binding_data = parts[3]
            reference = parts[5]
            ligand_name = " ".join(parts[6:])
            data.append(
                [
                    pdb_code,
                    resolution,
                    release_year,
                    binding_data,
                    reference,
                    ligand_name,
                ]
            )

    df = pd.DataFrame(
        data,
        columns=[
            "PDB code",
            "Resolution",
            "Release year",
            "Binding data",
            "Reference",
            "Ligand name",
        ],
    )
    return df


def remove_long_and_short_binders(df: pd.DataFrame, min_length:int = 7, max_length: int = 40) -> pd.DataFrame:
    """
    Filters out large binders using the specification in the PDB-bind INDEX file.
    
    Args:
        df: DataFrame with PDB-bind data
        max_length: Maximum length of the binder
    
    Returns:
        DataFrame with large binders removed
    """

    def is_valid_ligand(ligand_name: str) -> bool:
        if "(" in ligand_name and "-mer)" in ligand_name:
            try:
                peptide_length = int(ligand_name.split("-mer")[0].split("(")[-1]) # ARE WE SURE THIS IS ROBUST? NOTE
                if peptide_length < min_length:
                    return False
                if peptide_length > max_length:
                    return False
                return True
            except ValueError:
                return True
        return True

    return df[df["Ligand name"].apply(is_valid_ligand)]


def is_peptide_cyclic(pdb_file: Union[str, io.StringIO], cutoff: float = 1.7) -> bool:
    """
    Detect if the peptide (chain B) in the given PDB file is cyclic.
    Checks if there's a covalent bond (distance < cutoff Å) between the
    C-terminal carbonyl carbon and N-terminal nitrogen atoms.
    
    Args:
        pdb_file: Path to PDB file or StringIO object containing PDB data
        cutoff: Maximum distance (in Å) to consider atoms covalently bonded
    
    Returns:
        True if cyclic, False otherwise
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("peptide", pdb_file)
    model = structure[0]

    # Try to get chain B, return False if it doesn't exist
    try:
        chain = model["B"]
    except KeyError:
        return False

    residues = [res for res in chain if is_aa(res, standard=True)]
    
    # Need at least 2 residues to check cyclicity
    if len(residues) < 2:
        return False

    # N-terminal residue N atom
    n_term_res = residues[0]
    n_atom = n_term_res["N"] if "N" in n_term_res else None

    # C-terminal residue C atom
    c_term_res = residues[-1]
    c_atom = c_term_res["C"] if "C" in c_term_res else None

    if n_atom is None or c_atom is None:
        # Missing backbone atoms, cannot confirm cyclicity
        return False

    distance = np.linalg.norm(n_atom.coord - c_atom.coord)

    # Typical covalent bond distances C-N ~1.3-1.5Å, using 1.7Å as generous cutoff
    is_cyclic = distance <= cutoff
    return is_cyclic


def has_peptide_and_protein(pdb_contents: str, peptide_max_length: int = 40) -> Tuple[bool, str]:
    """
    Checks whether the PDB (provided as a string) contains exactly two polypeptide chains:
      1) One chain with fewer than `peptide_max_length` amino acids,
      2) One chain with >= `peptide_max_length` amino acids.
    
    Args:
        pdb_contents: String containing PDB file contents
        peptide_max_length: Maximum amino acid length to be considered a peptide
    
    Returns:
        Tuple of (bool, str) where:
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


def download_pdb_from_rcsb(
    pdb_id: str, 
    output_dir: str, 
    peptide_max_length: int = 40
) -> Tuple[bool, str]:
    """
    Downloads a PDB from RCSB, checks if it has exactly two chains
    (peptide+protein) using `has_peptide_and_protein`.
    Also checks if the peptide is cyclic. If so, it is not saved.
    If it meets the condition, saves it to `output_dir`.
    
    Args:
        pdb_id: PDB code (e.g. '1ABC')
        output_dir: Directory where PDB files should be saved
        peptide_max_length: length threshold that separates peptide from protein
    
    Returns:
        Tuple of (bool, str) where:
          bool -> True if PDB was saved, False otherwise
          str -> "meets_condition" or the reason for exclusion
    """

    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            pdb_text = response.text
            meets_criteria, reason = has_peptide_and_protein(
                pdb_text, peptide_max_length=peptide_max_length
            )
            is_cyclic = is_peptide_cyclic(io.StringIO(pdb_text))
            
            if meets_criteria and (not is_cyclic):  # Only save if passes criterion
                file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
                with open(file_path, "w") as file:
                    file.write(pdb_text)
                print(f"{pdb_id}.pdb was downloaded", end="\r")
                return True, reason
            else:
                rejection_reason = reason if not meets_criteria else "cyclic_peptide"
                print(f"{pdb_id}.pdb wasn't downloaded: {rejection_reason}", end="\r")
                return False, rejection_reason
        else:
            return False, f"download_failed_{response.status_code}"
    except Exception as e:
        return False, f"download_exception_{str(e)}"


def parallell_download(
    pdb_ids: List[str],
    output_dir: str,
    peptide_max_length: int = 40,
    max_workers: int = 5,
    overwrite: bool = False
) -> None:
    """
    Downloads multiple PDB IDs in parallel (up to `max_workers` threads),
    checks if each meets the "peptide+protein" condition, and saves only
    those that pass.
    
    Args:
        pdb_ids: List of PDB codes to download
        output_dir: Directory where passing PDB files get stored
        peptide_max_length: Length threshold to distinguish peptide vs protein
        max_workers: Number of parallel downloads
        overwrite: If False, skip downloading PDBs that already exist in output_dir
    """

    os.makedirs(output_dir, exist_ok=True)
    results = []

    total_attempts = len(pdb_ids)
    print(f"Total PDBs to download: {total_attempts}")
    saved_count = 0
    reason_counts: Dict[str, int] = {}

    def worker(pdb_id: str) -> Tuple[str, bool, str]:
        file_path = os.path.join(output_dir, f"{pdb_id}.pdb")
        if not overwrite and os.path.exists(file_path):
            return pdb_id, False, "already_exists"
        did_save, reason = download_pdb_from_rcsb(
            pdb_id=pdb_id, output_dir=output_dir, peptide_max_length=peptide_max_length
        )
        return pdb_id, did_save, reason

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for pdb_id, did_save, reason in executor.map(worker, pdb_ids):
            results.append((pdb_id, did_save, reason))

    for pdb_id, did_save, reason in results:
        if did_save:
            saved_count += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
    print(f"PDBs saved (pass criterion): {saved_count}")
    print("Reasons for skip/failure:")
    for reason, count in reason_counts.items():
        if reason != "meets_condition":
            print(f"  {reason}: {count}")

    print(f"  meets_condition: {reason_counts.get('meets_condition', 0)}")


def download_pdbs(
    pdbbind_index_files_path: str, 
    output_pdb_dir: str, 
    min_peptide_length: int = 7,
    max_peptide_length: int = 40
) -> None:
    """
    Loads the PDB-bind INDEX files, filters out large binders,
    and downloads the filtered PDB files.
    
    Args:
        pdbbind_index_files_path: Directory containing the PDB-bind INDEX files
        output_pdb_dir: Directory where PDB files should be saved
        max_peptide_length: Maximum amino acid length to be considered a peptide
    """
    # Load and filter INDEX files
    protein_ligands_path = os.path.join(pdbbind_index_files_path, "INDEX_PL.2020")
    protein_protein_path = os.path.join(pdbbind_index_files_path, "INDEX_PP.2020")
    
    df_pl = convert_to_index_file_to_dataframe(protein_ligands_path)
    df_pp = convert_to_index_file_to_dataframe(protein_protein_path)
    
    df_combined = pd.concat([df_pl, df_pp])
    df_filtered = remove_long_and_short_binders(df_combined, min_peptide_length, max_peptide_length)
    
    print(f"Filtered PDB files to those with peptides shorter than {max_peptide_length} amino acids.")

    # Create the output directory for PDBs
    pdbs_dir = os.path.join(output_pdb_dir, "pdbs")
    os.makedirs(pdbs_dir, exist_ok=True)

    # Download filtered PDBs if directory is empty
    if not os.listdir(pdbs_dir):
        parallell_download(df_filtered["PDB code"].tolist(), pdbs_dir, max_peptide_length)
    else:
        print(f"{pdbs_dir} already has files. Assuming download complete and skipping...")

    # Update the filtered dataframe to only include successfully downloaded PDBs
    pdb_filenames = set(os.listdir(pdbs_dir))
    downloaded_pdb_codes = [filename.split(".")[0] for filename in pdb_filenames if filename.endswith(".pdb")]
    
    df_downloaded = df_filtered[df_filtered["PDB code"].isin(downloaded_pdb_codes)]
    df_downloaded.to_csv(os.path.join(output_pdb_dir, "pdbs.csv"), index=False)

    print(f"Filtered PDB files saved to {os.path.join(output_pdb_dir, 'pdbs.csv')}")
    
    print(f"Successfully downloaded {len(downloaded_pdb_codes)} PDB files.")
