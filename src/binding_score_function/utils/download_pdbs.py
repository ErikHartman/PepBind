import os
import io
import requests
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from typing import Dict, List, Tuple
import logging

from Bio import PDB
from Bio.PDB.Polypeptide import is_aa
from Bio.PDB import MMCIFParser

logger = logging.getLogger(__name__)


def convert_to_index_file_to_dataframe(input_file: str) -> pd.DataFrame:
    """
    Converts the PDB-bind INDEX file to a pandas DataFrame
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
            "pdb_code",
            "resolution",
            "release_year",
            "binding_data",
            "reference",
            "ligand_name",
        ],
    )
    return df


def remove_long_and_short_binders_from_dataframe(
    df: pd.DataFrame, min_length: int = 7, max_length: int = 40
) -> pd.DataFrame:
    """
    Filters out large binders using the specification in the PDB-bind INDEX file.
    """

    def is_valid_ligand(ligand_name: str) -> bool:
        if "(" in ligand_name and "-mer)" in ligand_name:
            try:
                peptide_length = int(
                    ligand_name.split("-mer")[0].split("(")[-1]
                )
                if peptide_length < min_length:
                    return False
                if peptide_length > max_length:
                    return False
                return True
            except ValueError:
                return True
        return True

    return df[df["ligand_name"].apply(is_valid_ligand)]


def has_peptide_and_protein(
    pdb_contents: str, max_peptide_length: int = 40, min_peptide_length: int = 7
) -> Tuple[bool, str]:
    """
    Checks whether the PDB (provided as a string) contains exactly two polypeptide chains:
      1) One chain with fewer than `max_peptide_length` amino acids,
      2) One chain with >= `max_peptide_length` amino acids.
    """
    parser = PDB.PDBParser(QUIET=True)

    try:
        pdb_file_handle = io.StringIO(pdb_contents)
        structure = parser.get_structure("temp_struct", pdb_file_handle)
    except Exception:
        return False, "pdb_parse_failed"

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

    # Since chain A might be peptide or protein and chain B might be peptide or protein,
    chain_residue_counts.sort() # here [0] is the peptide and [1] is the protein
    if chain_residue_counts[0] >= max_peptide_length: # peptide is too long
        return False, "peptide_too_long"
    if chain_residue_counts[0] < min_peptide_length: # peptide is too short
        return False, "peptide_too_short"


    return True, "meets_condition"


def has_peptide_and_protein_cif(
    cif_contents: str, max_peptide_length: int = 40, min_peptide_length: int = 7
) -> Tuple[bool, str]:
    """
    Checks whether the CIF (provided as a string) contains exactly two polypeptide chains:
      1) One chain with fewer than `max_peptide_length` amino acids (and >= min_peptide_length),
      2) One chain with >= `max_peptide_length` amino acids.
    Returns (True, "meets_condition") if passed, else (False, reason).
    """
    try:
        parser = MMCIFParser(QUIET=True)
        cif_handle = io.StringIO(cif_contents)
        structure = parser.get_structure("temp_struct", cif_handle)
    except Exception:
        return False, "cif_parse_failed"

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
        return False, "invalid_chain_count"

    chain_residue_counts.sort()  # [0] peptide, [1] protein
    if chain_residue_counts[0] >= max_peptide_length:
        return False, "peptide_too_long"
    if chain_residue_counts[0] < min_peptide_length:
        return False, "peptide_too_short"

    return True, "meets_condition"


def load_manually_curated_pdbs(manual_csv_path: str) -> pd.DataFrame:
    """
    Load manually curated PDB-peptide pairs with binding data
    """
    if not os.path.exists(manual_csv_path):
        logger.info(f"No manually curated PDBs found at {manual_csv_path}")
        return pd.DataFrame()

    df = pd.read_csv(manual_csv_path)

    # Standardize column names to match the PDBBind data format
    df["pdb_code"] = df["pdb_code"].str.replace(".pdb", "")

    # Ensure required columns exist
    if "binding_data" not in df.columns or "peptide_sequence" not in df.columns:
        logger.warning("Manually curated CSV missing required columns")
        return pd.DataFrame()

    # Add other required columns that might be used in the pipeline
    if "resolution" not in df.columns:
        df["resolution"] = "NA"
    if "release_year" not in df.columns:
        df["release_year"] = "NA"
    if "reference" not in df.columns:
        df["reference"] = "manually_curated"
    if "ligand_name" not in df.columns:
        df["ligand_name"] = df["peptide_sequence"].apply(lambda x: f"({len(x)}-mer)")

    return df


def download_from_rcsb(
    pdb_code: str,
    output_dir: str,
    max_peptide_length: int = 40,
    min_peptide_length: int = 7,
) -> Tuple[bool, str]:
    """
    Downloads a CIF from RCSB, checks for exactly two chains (peptide+protein)
    using has_peptide_and_protein_cif, and saves it to `output_dir` only if it meets the criteria.
    """
    url = f"https://files.rcsb.org/download/{pdb_code}.cif"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            cif_text = response.text
            meets_criteria, reason = has_peptide_and_protein_cif(
                cif_text,
                max_peptide_length=max_peptide_length,
                min_peptide_length=min_peptide_length,
            )
            if meets_criteria:
                file_path = os.path.join(output_dir, f"{pdb_code}.cif")
                with open(file_path, "w") as file:
                    file.write(cif_text)
                logger.info(f"{pdb_code}.cif was downloaded")
                return True, "downloaded"
            else:
                logger.info(f"{pdb_code}.cif wasn't downloaded: {reason}")
                return False, reason
        else:
            logger.info(f"{pdb_code}.cif wasn't downloaded: download_failed_{response.status_code}")
            return False, f"download_failed_{response.status_code}"
    except Exception as e:
        logger.info(f"{pdb_code}.cif wasn't downloaded: download_exception_{str(e)}")
        return False, f"download_exception_{str(e)}"


def parallell_download(
    pdb_codes: List[str],
    output_dir: str,
    max_peptide_length: int = 40,
    min_peptide_length: int = 7,
    max_workers: int = 5,
    overwrite: bool = False,
) -> None:
    """
    Downloads multiple PDB IDs in parallel (up to `max_workers` threads),
    validates they contain peptide+protein by chain length, and saves only passing CIFs.
    """
    results = []
    total_attempts = len(pdb_codes)
    logger.info(f"Total CIFs to download: {total_attempts}")
    saved_count = 0
    reason_counts: Dict[str, int] = {}

    def worker(pdb_code: str) -> Tuple[str, bool, str]:
        file_path = os.path.join(output_dir, f"{pdb_code}.cif")
        if not overwrite and os.path.exists(file_path):
            return pdb_code, False, "already_exists"
        did_save, reason = download_from_rcsb(
            pdb_code=pdb_code,
            output_dir=output_dir,
            max_peptide_length=max_peptide_length,
            min_peptide_length=min_peptide_length,
        )
        return pdb_code, did_save, reason

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for pdb_code, did_save, reason in executor.map(worker, pdb_codes):
            results.append((pdb_code, did_save, reason))

    for pdb_code, did_save, reason in results:
        if did_save:
            saved_count += 1
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    logger.info(f"CIFs saved: {saved_count}")
    logger.info("Reasons for skip/failure:")
    for reason, count in reason_counts.items():
        if reason != "downloaded":
            logger.info(f"  {reason}: {count}")
    logger.info(f"  downloaded: {reason_counts.get('downloaded', 0)}")


def download_pdbs(
    pdbbind_index_files_path: str,
    output_pdb_dir: str,
    min_peptide_length: int = 7,
    max_peptide_length: int = 40,
    overwrite: bool = False,
    manual_csv_path: str = None,
) -> pd.DataFrame:
    """
    Loads the PDB-bind INDEX files and manual curated data,
    filters out large binders, and downloads the filtered PDB files.
    """
    # Load PDBBind data
    protein_ligands_path = os.path.join(
        pdbbind_index_files_path, "INDEX_general_PP.2020R1.lst"
    )  # protein ligand
    protein_protein_path = os.path.join(
        pdbbind_index_files_path, "INDEX_general_PL.2020R1.lst"
    )  # protein peptide

    df_pl = convert_to_index_file_to_dataframe(protein_ligands_path)
    df_pp = convert_to_index_file_to_dataframe(protein_protein_path)
    df_combined = pd.concat([df_pl, df_pp])

    if manual_csv_path and os.path.exists(manual_csv_path):
        df_manual = load_manually_curated_pdbs(manual_csv_path)
        logger.info(f"Loaded {len(df_manual)} manually curated PDB entries")
        unique_manual_pdbs = df_manual["pdb_code"].unique()
        df_combined = pd.concat([df_combined, df_manual])
    else:
        unique_manual_pdbs = []

    # Filter by peptide length
    df_filtered = remove_long_and_short_binders_from_dataframe(
        df_combined, min_peptide_length, max_peptide_length
    )

    # Create output directory if it doesn't exist
    pdbs_dir = os.path.join(output_pdb_dir, "pdbs")
    os.makedirs(pdbs_dir, exist_ok=True)

    if not os.listdir(pdbs_dir) or overwrite:
        # Ensure we download all manually curated PDBs first
        all_pdb_codes = df_filtered["pdb_code"].tolist()

        # Move manually curated PDBs to the front of the list
        for pdb in unique_manual_pdbs:
            if pdb in all_pdb_codes:
                all_pdb_codes.remove(pdb)
            all_pdb_codes.insert(0, pdb)

        parallell_download(
            pdb_codes=all_pdb_codes,
            output_dir=pdbs_dir,
            max_peptide_length=max_peptide_length,
            max_workers=20,
            overwrite=overwrite,
        )
    else:
        logger.info(
            f"{pdbs_dir} already has files. Assuming download complete and skipping..."
        )

    # Return the filtered dataframe of successfully downloaded PDBs
    cif_filenames = set(os.listdir(pdbs_dir))
    downloaded_pdb_codes = [
        filename.split(".")[0]
        for filename in cif_filenames
        if filename.endswith(".cif")
    ]

    df_downloaded = df_filtered[df_filtered["pdb_code"].isin(downloaded_pdb_codes)]
    return df_downloaded
