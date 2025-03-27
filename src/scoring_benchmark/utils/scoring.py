from Bio.PDB import PDBParser, PDBIO, Select
import logging
import os
import logging
from bopep import Scorer
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb
import pandas as pd



# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("benchmark.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

def remove_peptide_from_complex(pdb_path, output_pdb_path, protein_chain="A"):
    """
    Remove a specified peptide chain from a PDB file, leaving only the protein
    chain (and any other chains not specified as the peptide).

    """

    # Parse the structure
    parser = PDBParser(QUIET=True)  # QUIET avoids a lot of console output
    structure = parser.get_structure("complex", pdb_path)

    # We will create a custom selection class that excludes the peptide chain
    class PeptideRemoverSelect(Select):
        def accept_chain(self, chain):
            # Keep chain if it is the protein chain
            if chain.id == protein_chain:
                return True  # include
            return False  # exclude

    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb_path, select=PeptideRemoverSelect())

    return output_pdb_path


def get_contact_residues(
    structure, protein_chain_id="A", peptide_chain_id="B", cutoff=4.0
):
    """
    Identify residues on the protein chain that are within 'cutoff' angstroms
    of any atom in the peptide chain.
    """
    # Extract chain objects
    model = structure[0]  # assume only one model
    protein_chain = model[protein_chain_id]
    peptide_chain = model[peptide_chain_id]

    contact_residues = set()

    # Get all atoms from the peptide
    peptide_atoms = list(peptide_chain.get_atoms())

    # For each residue in the protein chain, check if it's within 'cutoff' of the peptide
    for residue in protein_chain.get_residues():
        # Skip heteroatoms or water
        if not residue.id[0] == " ":  # Standard residues have id[0] == " "
            continue
        for atom in residue:
            for pep_atom in peptide_atoms:
                dist = atom - pep_atom  # operator- gives distance
                if dist <= cutoff:
                    contact_residues.add((protein_chain_id, residue.id[1]))
                    # Break to avoid double-counting the same residue
                    break
            else:
                # If we never broke out, keep checking next residue atom
                continue
            # Once we've added the residue, no need to check additional atoms
            break

    return contact_residues


def get_interface_residues_in_pdb(
    pdb_path,
    protein_chain="A",
    peptide_chain="B",
    cutoff=5.0
):
    """
    Given a PDB file with a protein (chain A) and a peptide (chain B),
    return the residue indices on the protein chain within 'cutoff'
    angstroms of the peptide.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    contact_residues = get_contact_residues(
        structure, 
        protein_chain_id=protein_chain, 
        peptide_chain_id=peptide_chain, 
        cutoff=cutoff
    )
    
    residue_indices = sorted([res_id for (_, res_id) in contact_residues])
    
    return residue_indices






def ensure_dir_exists(directory):
    """Ensure that a directory exists, creating it if necessary."""
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def benchmark_pdb(pdb_path, output_dir):
    """
    Run a docking benchmark on the provided PDB structure and return the scoring results.

    Args:
        pdb_path (str): Path to the PDB file to benchmark
        output_dir (str): Path to the output directory

    Returns:
        dict: Dictionary containing scoring results or None if an error occurred
    """
    try:
        # Define output directories
        protein_template_dir = os.path.join(output_dir, "pdbs/stripped_protein_templates")
        docked_peptides_dir = os.path.join(output_dir, "pdbs/docked_peptides")

        # Ensure directories exist
        ensure_dir_exists(protein_template_dir)
        ensure_dir_exists(docked_peptides_dir)

        output_pdb_path = os.path.join(protein_template_dir, os.path.basename(pdb_path))

        logger.info(f"Processing PDB: {os.path.basename(pdb_path)}")

        # Extract peptide sequence
        peptide_sequence = extract_sequence_from_pdb(pdb_path, chain_id="B")
        if not peptide_sequence:
            logger.error(f"Failed to extract peptide sequence from {pdb_path}")
            return None

        # Remove peptide from complex
        protein_template_path = remove_peptide_from_complex(
            pdb_path, output_pdb_path=output_pdb_path, protein_chain="A"
        )
        if not os.path.exists(protein_template_path):
            logger.error(
                f"Failed to create protein template at {protein_template_path}"
            )
            return None

        logger.info(f"Peptide sequence: {peptide_sequence}")
        logger.info(f"Protein template path: {protein_template_path}")

        # Set up docking parameters
        docker_kwargs = {
            "num_models": 5,
            "num_recycles": 10,
            "recycle_early_stop_tolerance": 0.1,
            "amber": True,
            "num_relax": 1,
            "pdb_dir": docked_peptides_dir,
            "gpu_ids": ["3"],
            "overwrite_results": False,
        }

        # Run docking
        docker = Docker(docker_kwargs)
        docker.set_target_structure(protein_template_path)
        dock_dir = docker.dock_peptides([peptide_sequence])[0]

        binding_site_residues = get_interface_residues_in_pdb(pdb_path)
        # I realized that we actually don't need the compare_binding_sites function!
        # Since we can use the in_binding_site with the binding_site_residues in the scorer :) 

        # Score results
        scorer = Scorer()
        # Get all available scores except distance_to_peptide
        scores_to_include = [score for score in scorer.available_scores if score != "distance_score"]
        
        # Score results with the filtered list
        scores = scorer.score(
            scores_to_include=scores_to_include,
            colab_dir=dock_dir,
            binding_site_residue_indices=binding_site_residues,
        )
        scores = scores[peptide_sequence]
        scores["pdb_file"] = os.path.basename(pdb_path)
        scores["peptide_sequence"] = peptide_sequence

        logger.info(f"Completed benchmark for {os.path.basename(pdb_path)}")
        return scores

    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}", exc_info=True)
        return None

def run_benchmark(input_dir, output_dir):

    """
    
    Run the benchmark on all PDB files in the specified directory.

    Args:
        input_dir (str): Path to the directory containing PDB files to process
    
    
    """
    # Ensure data directory exists
    if not os.path.exists(input_dir):
        logger.error(f"Data directory does not exist: {input_dir}")
        exit(1)

    # Ensure output directory for CSV exists
    results_dir = os.path.dirname(os.path.abspath("benchmark_scores.csv"))
    ensure_dir_exists(results_dir)

    pdb_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".pdb")
    ]
    logger.info(f"Number of PDB files to process: {len(pdb_files)}")

    for i, pdb in enumerate(pdb_files):
        logger.info(f"Processing file {i+1}/{len(pdb_files)}: {os.path.basename(pdb)}")
        scores = benchmark_pdb(pdb, output_dir)
        if scores:
            scores_df = pd.DataFrame([scores])
            scores_df.to_csv(
                "benchmark_scores.csv", # Change output path
                mode="a",
                header=not os.path.exists("benchmark_scores.csv"),
                index=False,
            )
        else:
            logger.warning(f"No scores obtained for {pdb}")

    logger.info("Benchmark completed")

