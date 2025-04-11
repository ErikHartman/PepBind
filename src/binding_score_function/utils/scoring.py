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
    docking_pdb_path: str,
    binding_residue_distance_cutoff: float = 5.0,
) -> Optional[Dict[str, Any]]:
    try:
        complex_filename = os.path.basename(docking_pdb_path)
        logger.info(f"Processing PDB: {complex_filename}")

        # Identify binding site residues
        _, binding_site_residues, _, _ = get_binding_site(
            docking_pdb_path,
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
            colab_dir=docking_pdb_path,
            binding_site_residue_indices=binding_site_residues,
        )
        scores = scores[complex_filename]
        scores["complex_filename"] = complex_filename

        logger.info(f"Completed scoring for {complex_filename}")
        return scores

    except Exception as e:
        logger.error(f"Error processing {docking_pdb_path}: {str(e)}", exc_info=True)
        return None
    

def score_pdbs_in_dir(
    docking_dir: str,
    output_csv_path: str,
    binding_residue_distance_cutoff: float = 5.0,
    max_workers: int = 4,
) -> None:
    """
    Score all PDB files in the docking directory and save results to a CSV.
    """
    docking_pdbs_dir = os.path.join(docking_dir, "docked_pdbs")
    pdb_files = [
        os.path.join(docking_pdbs_dir, f)
        for f in os.listdir(docking_pdbs_dir)
        if f.endswith(".pdb")
    ]

    logger.info(f"Found {len(pdb_files)} PDB files to score in {docking_pdbs_dir}")

    def process_pdb(pdb_path):
        return score_pdb(pdb_path, binding_residue_distance_cutoff)

    scores = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for result in executor.map(process_pdb, pdb_files):
            if result:
                scores.append(result)

    # Convert scores to a DataFrame and save to CSV
    scores_df = pd.DataFrame(scores)
    scores_df.to_csv(output_csv_path, index=False)

    logger.info(f"Scoring completed. Results saved to {output_csv_path}")


if __name__ == "__main__":
    original_pdb = "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/1_processed/pdbs/2djy.pdb"
    docked_pdb = "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/2_scored/pdbs/docked_peptides/2djy_GPLGSELESPPPPYSRYPMD/2djy_GPLGSELESPPPPYSRYPMD_relaxed_rank_001_alphafold2_multimer_v3_model_5_seed_000.pdb"

    receptor_binding_site_atoms, receptor_binding_site_residue_indices, peptide_binding_site_residue_indices, peptide_atoms = get_binding_site(original_pdb)
    print(receptor_binding_site_residue_indices)
    receptor_binding_site_atoms, receptor_binding_site_residue_indices, peptide_binding_site_residue_indices, peptide_atoms = get_binding_site(docked_pdb)
    print(receptor_binding_site_residue_indices)
