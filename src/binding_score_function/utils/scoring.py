from Bio.PDB import PDBParser, PDBIO, Select, Structure
import logging
import os
from typing import Dict, List, Optional, Set, Tuple, Any
from bopep import Scorer
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb
import pandas as pd


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


def remove_peptide_from_complex(pdb_path: str, output_pdb_path: str, protein_chain: str = "A") -> str:
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
    cutoff: float = 4.0
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

    contact_residues = set()

    # Get all atoms from the peptide
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
    pdb_path: str, 
    protein_chain: str = "A", 
    peptide_chain: str = "B", 
    cutoff: float = 5.0
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
    docking_config: Dict[str, Any]
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
        protein_template_dir = os.path.join(output_dir, "pdbs/stripped_protein_templates")
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
            logger.error(f"Failed to create protein template at {protein_template_path}")
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
        binding_site_residues = get_interface_residues_in_pdb(pdb_path)

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


def dock_and_score_all_pdbs(
    input_pdb_dir: str, 
    processed_file_dir: str, 
    docking_config: Dict[str, Any]
) -> None:
    """
    Run the docking and scoring on all PDB files in the specified directory.

    Args:
        input_pdb_dir: Path to the directory containing PDB files to process
        processed_file_dir: Path to the output directory for processed files
        docking_config: Configuration parameters for the docking process
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

    for i, pdb_file in enumerate(pdb_files):
        logger.info(f"Processing file {i+1}/{len(pdb_files)}: {os.path.basename(pdb_file)}")
        scores = score_pdb(pdb_file, processed_file_dir, docking_config)
        if scores:
            scores_df = pd.DataFrame([scores])
            scores_df.to_csv(
                results_file,
                mode="a",
                header=not os.path.exists(results_file),
                index=False,
            )
        else:
            logger.warning(f"No scores obtained for {pdb_file}")

    logger.info("Scoring completed")
