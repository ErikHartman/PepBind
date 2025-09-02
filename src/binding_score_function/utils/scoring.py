import logging
import os
from typing import Dict, Optional, Any, Tuple
from bopep import Scorer
from bopep import get_binding_site
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
from Bio.PDB import PDBParser, MMCIFParser

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def determine_chain_assignments(pdb_file: str) -> Tuple[str, str]:
    """
    Determine which chain is the receptor and which is the peptide.
    
    Parameters
    ----------
    pdb_file : str
        Path to the PDB file
        
    Returns
    -------
    tuple
        (receptor_chain_id, peptide_chain_id)
    """
    if pdb_file.endswith('.cif'):
        parser = MMCIFParser(QUIET=True, auth_residues=False)
    else:
        parser = PDBParser(QUIET=True)
    
    try:
        structure = parser.get_structure("structure", pdb_file)
        model = structure[0]
        
        # Get chains A and B
        try:
            chain_a = model["A"]
        except KeyError:
            chain_a = None
            
        try:
            chain_b = model["B"]
        except KeyError:
            chain_b = None
        
        if not chain_a or not chain_b:
            logger.warning(f"Missing chain A or B in {pdb_file}, using default assignment")
            return "A", "B"
        
        # Count residues in each chain (excluding hetero residues)
        count_a = len([res for res in chain_a.get_residues() if res.id[0] == " "])
        count_b = len([res for res in chain_b.get_residues() if res.id[0] == " "])
        
        # Receptor is the longer chain, peptide is the shorter chain
        if count_a >= count_b:
            receptor_chain = "A"
            peptide_chain = "B"
        else:
            receptor_chain = "B"
            peptide_chain = "A"
            
        logger.info(f"Chain assignment for {os.path.basename(pdb_file)}: "
                   f"receptor={receptor_chain} ({count_a if receptor_chain=='A' else count_b} residues), "
                   f"peptide={peptide_chain} ({count_b if peptide_chain=='B' else count_a} residues)")
        
        return receptor_chain, peptide_chain
        
    except Exception as e:
        logger.warning(f"Error determining chain assignments for {pdb_file}: {e}, using default")
        return "A", "B"



def score_pdb(
    docking_result_path: str,
    original_pdb_path: str,
    binding_residue_distance_cutoff: float = 5.0,
) -> Optional[Dict[str, Any]]:
    try:
        complex_filename = os.path.basename(docking_result_path)
        logger.info(f"Processing PDB: {complex_filename}")
        
        # Determine which chain is receptor vs peptide based on length
        receptor_chain, peptide_chain = determine_chain_assignments(original_pdb_path)
        
        # Identify binding site residues
        _, binding_site_residues, _, _ = get_binding_site(
            original_pdb_path,
            receptor_chain=receptor_chain,
            peptide_chain=peptide_chain,
            threshold=binding_residue_distance_cutoff,
        )

        logger.info(f"Binding site residues for {complex_filename}: {binding_site_residues}")

        if binding_site_residues is None:
            logger.error(
                f"Binding site residues not found for {complex_filename}. Skipping scoring."
            )
            return None

        # Configure scoring
        scorer = Scorer()
        scores_to_include = scorer.get_available_scores(processed_dir=docking_result_path, binding_site_residue_indices=binding_site_residues, template_pdb=original_pdb_path)
        logging.info(f"Scores to include for {complex_filename}: {scores_to_include}")

        # Calculate scores
        scores = scorer.score(
            scores_to_include=scores_to_include,
            template_pdb=original_pdb_path,
            processed_dir=docking_result_path,
            binding_site_residue_indices=binding_site_residues,
        )
        scores = list(scores.values())
        assert len(scores) == 1
        scores = scores[0] # should only be one complex
        scores["complex_filename"] = complex_filename

        logger.info(f"Completed scoring for {complex_filename}")
        return scores

    except Exception as e:
        logger.error(f"Error processing {docking_result_path}: {str(e)}", exc_info=True)
        return None
    

def score_pdbs_in_dir(
    docking_dir: str,
    complexes_dir: str,
    binding_residue_distance_cutoff: float = 5.0,
    max_workers: int = 60,
) -> pd.DataFrame:
    """
    Score all PDB files in the docking directory and save results to a CSV.
    """
    colab_docking_dirs = [
        os.path.join(docking_dir, f)
        for f in os.listdir(docking_dir)
    ]

    logger.info(f"Found {len(colab_docking_dirs)} PDB files to score in {docking_dir}")

    def process_pdb(colab_docking_dirs):
        pdb_id = os.path.basename(colab_docking_dirs).split("_")[0]
        base_path = os.path.join(
            complexes_dir, f"{pdb_id}.cif"
        )
        return score_pdb(colab_docking_dirs, base_path, binding_residue_distance_cutoff)

    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_pdb, colab_docking_dirs):
            if result:
                scores.append(result)

    # Convert scores to a DataFrame and save to CSV
    scores_df = pd.DataFrame(scores)
    if scores_df.empty:
        logger.warning("No scores were generated.")
        raise ValueError
    return scores_df


if __name__ == "__main__":
    pass
