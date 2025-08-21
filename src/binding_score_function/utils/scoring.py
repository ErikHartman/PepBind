import logging
import os
from typing import Dict, Optional, Any
from bopep import Scorer
from bopep import get_binding_site
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)



def score_pdb(
    docking_result_path: str,
    original_pdb_path: str,
    binding_residue_distance_cutoff: float = 5.0,
) -> Optional[Dict[str, Any]]:
    try:
        complex_filename = os.path.basename(docking_result_path)
        logger.info(f"Processing PDB: {complex_filename}")

        # Identify binding site residues
        _, binding_site_residues, _, _ = get_binding_site(
            original_pdb_path,
            receptor_chain="A",
            peptide_chain="B",
            threshold=binding_residue_distance_cutoff,
        )
        # Configure scoring
        scorer = Scorer()
        scores_to_include = scorer.available_scores

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
    max_workers: int = 20,
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
