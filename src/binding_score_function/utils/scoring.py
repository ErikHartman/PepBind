from Bio.PDB import PDBParser, PDBIO, Select, Structure
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Any
from bopep import Scorer
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import threading

"""
Utils for scoring and peptide-protein docking
and filtering them based on peptide/protein criteria.
"""


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def remove_peptide_from_complex(
    pdb_path: str, output_pdb_path: str, protein_chain: str = "A"
) -> str:
    """
    Remove a specified peptide chain from a PDB file, leaving only the protein
    chain (and any other chains not specified as the peptide).

    Args:
        pdb_path: Path to the input PDB file
        output_pdb_path: Path to save the output PDB file
        protein_chain: Chain ID of the protein (default: "A")

    Returns:
        str: Path to the output PDB file
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
    structure: Structure,
    protein_chain_id: str = "A",
    peptide_chain_id: str = "B",
    cutoff: float = 4.0,
) -> Set[Tuple[str, int]]:
    """
    Identify residues on the protein chain that are within 'cutoff' angstroms
    of any atom in the peptide chain.

    Args:
        structure: BioPython structure object
        protein_chain_id: Chain ID of the protein (default: "A")
        peptide_chain_id: Chain ID of the peptide (default: "B")
        cutoff: Distance cutoff in angstroms (default: 4.0)

    Returns:
        Set of tuples (chain_id, residue_id) for contact residues
    """
    # Extract chain objects
    model = structure[0]  # assume only one model
    protein_chain = model[protein_chain_id]
    peptide_chain = model[peptide_chain_id]

    min_protein_residue_id = min(
        residue.id[1]
        for residue in protein_chain.get_residues()
        if residue.id[0] == " "
    )

    contact_residues = set()
    peptide_atoms = list(peptide_chain.get_atoms())

    # For each residue in the protein chain, check if it's within 'cutoff' of the peptide
    for residue in protein_chain.get_residues():
        # Skip heteroatoms or water
        if not residue.id[0] == " ":  # Standard residues have id[0] == " "
            continue
        for atom in residue:
            for peptide_atom in peptide_atoms:
                distance = atom - peptide_atom  # operator- gives distance
                if distance <= cutoff:
                    contact_residues.add(
                        (protein_chain_id, residue.id[1] - min_protein_residue_id)
                    )
                    break
            else:
                # If we never broke out, keep checking next residue atom
                continue
            # Once we've added the residue, no need to check additional atoms
            break

    return contact_residues


def get_interface_residues_in_pdb(
    pdb_path: str,
    protein_chain: str = "A",
    peptide_chain: str = "B",
    cutoff: float = 5.0,
) -> List[int]:
    """
    Given a PDB file with a protein (chain A) and a peptide (chain B),
    return the residue indices on the protein chain within 'cutoff'
    angstroms of the peptide.

    Args:
        pdb_path: Path to the PDB file
        protein_chain: Chain ID of the protein (default: "A")
        peptide_chain: Chain ID of the peptide (default: "B")
        cutoff: Distance cutoff in angstroms (default: 5.0)

    Returns:
        List of residue indices on the protein chain
    """

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)

    contact_residues = get_contact_residues(
        structure,
        protein_chain_id=protein_chain,
        peptide_chain_id=peptide_chain,
        cutoff=cutoff,
    )

    residue_indices = sorted([residue_id for (_, residue_id) in contact_residues])

    return residue_indices


def ensure_dir_exists(directory: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        directory: Path to the directory to create
    """
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)


def score_pdb(
    pdb_path: str,
    output_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
) -> Optional[Dict[str, Any]]:
    """
    Run a docking and scoring on the provided PDB structure and return the scoring results.

    Args:
        pdb_path: Path to the PDB file to score
        output_dir: Path to the output directory
        docking_config: Configuration parameters for the docking process

    Returns:
        Dictionary containing scoring results or None if an error occurred
    """
    try:
        # Define directories for intermediate outputs
        protein_template_dir = os.path.join(
            output_dir, "pdbs/stripped_protein_templates"
        )
        docked_peptides_dir = os.path.join(output_dir, "pdbs/docked_peptides")

        ensure_dir_exists(protein_template_dir)
        ensure_dir_exists(docked_peptides_dir)

        protein_filename = os.path.basename(pdb_path)
        output_protein_path = os.path.join(protein_template_dir, protein_filename)

        logger.info(f"Processing PDB: {protein_filename}")

        # Extract peptide sequence from chain B
        peptide_sequence = extract_sequence_from_pdb(pdb_path, chain_id="B")
        if not peptide_sequence:
            logger.error(f"Failed to extract peptide sequence from {pdb_path}")
            return None

        # Remove peptide, keep only protein chain A
        protein_template_path = remove_peptide_from_complex(
            pdb_path, output_pdb_path=output_protein_path, protein_chain="A"
        )
        if not os.path.exists(protein_template_path):
            logger.error(
                f"Failed to create protein template at {protein_template_path}"
            )
            return None

        logger.info(f"Peptide sequence: {peptide_sequence}")
        logger.info(f"Protein template path: {protein_template_path}")

        # Update docking_config with the output directory
        docking_config_for_run = docking_config.copy()
        docking_config_for_run["pdb_dir"] = docked_peptides_dir

        # Perform docking
        docker = Docker(docking_config_for_run)
        docker.set_target_structure(protein_template_path)
        dock_dir = docker.dock_peptides([peptide_sequence])[0]

        # Identify binding site residues
        binding_site_residues = get_interface_residues_in_pdb(
            pdb_path,
            protein_chain="A",
            peptide_chain="B",
            cutoff=binding_residue_distance_cutoff,
        )
        true_binding_site_residues = get_interface_residues_in_pdb(pdb_path)
        logger.info(
            f"Binding site residues: {binding_site_residues} within {binding_residue_distance_cutoff} Å of peptide"
        )
        logger.info(
            f"True binding site residues: {true_binding_site_residues} within {binding_residue_distance_cutoff} Å of peptide"
        )
        logger.info(
            f"Overlap with true binding site: {len(set(binding_site_residues) & set(true_binding_site_residues))/len(true_binding_site_residues) * 100:.2f}%"
        )

        # Configure scoring
        scorer = Scorer()
        scores_to_include = scorer.available_scores

        # Calculate scores
        scores = scorer.score(
            scores_to_include=scores_to_include,
            colab_dir=dock_dir,
            binding_site_residue_indices=binding_site_residues,
        )
        scores = scores[peptide_sequence]
        scores["pdb_file"] = protein_filename
        scores["peptide_sequence"] = peptide_sequence

        logger.info(f"Completed scoring for {protein_filename}")
        return scores

    except Exception as e:
        logger.error(f"Error processing {pdb_path}: {str(e)}", exc_info=True)
        return None


def dock_and_score_peptide(
    protein_template_path: str,
    peptide_sequence: str,
    output_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
    skip_docking: bool = False,
    is_decoy: bool = False
) -> Optional[Dict[str, Any]]:
    """
    Dock a single peptide sequence to a protein template and score the interaction.
    
    Args:
        protein_template_path: Path to the protein template PDB file
        peptide_sequence: Peptide sequence to dock
        output_dir: Directory to save output files
        docking_config: Configuration for docking
        binding_residue_distance_cutoff: Cutoff distance for binding residues
        skip_docking: If True, skip docking and only score if results exist
        is_decoy: If True, this is a decoy peptide and uses a different directory
        
    Returns:
        Dict[str, Any] or None: Scores if successful, None otherwise
    """
    try:
        # Define directories for output - use different directory for decoys
        if is_decoy:
            docked_peptides_dir = os.path.join(output_dir, "decoy_pdbs")
        else:
            docked_peptides_dir = os.path.join(output_dir, "docked_pdbs")
        
        # Use the directory specified in docking_config if it exists, otherwise use our default
        if "pdb_dir" in docking_config:
            # If docking_config specifies a directory, make sure it's consistent with is_decoy
            if is_decoy and "decoy_pdbs" not in docking_config["pdb_dir"]:
                logger.warning(f"Overriding docking_config pdb_dir to use decoy_pdbs for decoy peptide")
                docking_config["pdb_dir"] = docked_peptides_dir
            elif not is_decoy and "docked_pdbs" not in docking_config["pdb_dir"]:
                logger.warning(f"Overriding docking_config pdb_dir to use docked_pdbs for real peptide")
                docking_config["pdb_dir"] = docked_peptides_dir
        else:
            # If no directory is specified, use our default
            docking_config["pdb_dir"] = docked_peptides_dir

        # Ensure directory exists
        ensure_dir_exists(docking_config["pdb_dir"])
        logger.info(f"Using output directory: {docking_config['pdb_dir']}")
            
        protein_filename = os.path.basename(protein_template_path)
        protein_basename = os.path.splitext(protein_filename)[0]
        
        logger.info(f"Target is set to:  {protein_template_path}")
        
        # Check if already docked
        expected_dock_dir = os.path.join(docking_config["pdb_dir"], f"{protein_basename}_{peptide_sequence}")
        dock_dir = expected_dock_dir
        
        if skip_docking and os.path.exists(expected_dock_dir) and os.listdir(expected_dock_dir):
            logger.info(f"Docking result for {peptide_sequence} already exists in {expected_dock_dir}. Skipping docking...")
        else:
            # Perform docking
            logger.info(f"Docking peptide {peptide_sequence} to {protein_filename}")
            docker = Docker(docking_config)
            docker.set_target_structure(protein_template_path)
            dock_dir = docker.dock_peptides([peptide_sequence])[0]
        
        # Look for the docked PDB file
        expected_pdb = os.path.join(
            dock_dir,
            f"{protein_basename}_{peptide_sequence}_relaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb",
        )

        if os.path.exists(expected_pdb):
            docked_pdb = expected_pdb
        else:
            # If the expected PDB file doesn't exist, try to find any resulting PDB file
            pdb_files = [f for f in os.listdir(dock_dir) if f.endswith(".pdb")]
            if pdb_files:
                docked_pdb = os.path.join(dock_dir, pdb_files[0])
            else:
                logger.error(f"No docked PDB file found in {dock_dir}")
                return None

        # Identify binding site residues based on the docked structure
        binding_site_residues = get_interface_residues_in_pdb(
            docked_pdb,
            protein_chain="A",
            peptide_chain="B",
            cutoff=binding_residue_distance_cutoff,
        )

        logger.info(f"Identified {len(binding_site_residues)} binding site residues")

        # Configure scoring
        scorer = Scorer()
        scores_to_include = scorer.available_scores

        # Calculate scores
        scores = scorer.score(
            scores_to_include=scores_to_include,
            colab_dir=dock_dir,
            binding_site_residue_indices=binding_site_residues,
        )

        if peptide_sequence in scores:
            scores = scores[peptide_sequence]
            scores["pdb_file"] = protein_filename
            scores["peptide_sequence"] = peptide_sequence
            logger.info(f"Successfully scored {peptide_sequence} with {protein_filename}")
            return scores
        else:
            logger.error(f"No scores found for peptide {peptide_sequence}")
            return None

    except Exception as e:
        logger.error(
            f"Error docking and scoring peptide {peptide_sequence} with {protein_template_path}: {str(e)}"
        )
        return None


def dock_and_score_all_pdbs(
    input_pdb_dir: str,
    processed_file_dir: str,
    docking_config: Dict[str, Any],
    binding_residue_distance_cutoff: float = 5.0,
) -> None:
    """
    Run the docking and scoring on all PDB files in the specified directory,
    utilizing multiple GPUs for parallel processing.

    Args:
        input_pdb_dir: Path to the directory containing PDB files to process
        processed_file_dir: Path to the output directory for processed files
        docking_config: Configuration parameters for the docking process
                       (including gpu_ids for parallel processing)
    """

    if not os.path.exists(input_pdb_dir):
        logger.error(f"Data directory does not exist: {input_pdb_dir}")
        exit(1)

    results_file = os.path.join(processed_file_dir, "scores.csv")
    ensure_dir_exists(os.path.dirname(results_file))

    pdb_files = [
        os.path.join(input_pdb_dir, filename)
        for filename in os.listdir(input_pdb_dir)
        if filename.endswith(".pdb")
    ]
    logger.info(f"Number of PDB files to dock and score: {len(pdb_files)}")

    gpu_ids = docking_config.get("gpu_ids", ["0"])
    num_gpus = len(gpu_ids)
    logger.info(f"Using {num_gpus} GPUs for parallel processing: {gpu_ids}")

    # Create a lock for thread-safe CSV writing
    results_lock = threading.Lock()

    # Process a single PDB file with a specific GPU
    def process_file(args):
        idx, pdb_file = args
        gpu_id = gpu_ids[idx % num_gpus]

        gpu_docking_config = docking_config.copy()
        gpu_docking_config["gpu_ids"] = [gpu_id]

        logger.info(
            f"Processing file {idx+1}/{len(pdb_files)}: {os.path.basename(pdb_file)} on GPU {gpu_id}"
        )
        try:
            scores = score_pdb(
                pdb_file,
                processed_file_dir,
                gpu_docking_config,
                binding_residue_distance_cutoff,
            )
            if scores:
                with results_lock:
                    scores_df = pd.DataFrame([scores])
                    scores_df.to_csv(
                        results_file,
                        mode="a",
                        header=not os.path.exists(results_file),
                        index=False,
                    )
                return True
            else:
                logger.warning(f"No scores obtained for {pdb_file}")
                return False
        except Exception as e:
            logger.error(
                f"Error processing {os.path.basename(pdb_file)} on GPU {gpu_id}: {str(e)}",
                exc_info=True,
            )
            return False

    successful_files = 0
    with ThreadPoolExecutor(max_workers=num_gpus) as executor:
        tasks = [(i, pdb_file) for i, pdb_file in enumerate(pdb_files)]

        for result in executor.map(process_file, tasks):
            if result:
                successful_files += 1

    logger.info(
        f"Scoring completed: {successful_files}/{len(pdb_files)} files processed successfully"
    )


if __name__ == "__main__":
    original_pdb = "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/1_processed/pdbs/2djy.pdb"
    docked_pdb = "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/2_scored/pdbs/docked_peptides/2djy_GPLGSELESPPPPYSRYPMD/2djy_GPLGSELESPPPPYSRYPMD_relaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb"

    print(get_interface_residues_in_pdb(original_pdb))
    print(get_interface_residues_in_pdb(docked_pdb))
