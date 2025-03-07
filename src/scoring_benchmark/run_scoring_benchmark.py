import os
from bopep import Scorer
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb
import pandas as pd
from multiprocessing import Pool, Value, Lock
import traceback

count = Value('i', 1)
lock = Lock()

def process_pdb(pdb_path):
    global count
    scorer = Scorer()
    with lock:
        print(f"Processing: {pdb_path}")
        print(f"Count: {count.value}")
        count.value += 1
    
    try:
        pdb_scores = scorer.score(scores_to_include=["interface_sasa"], pdb_file=pdb_path)
        pdb_scores['pdb'] = os.path.basename(pdb_path)
        return pdb_scores
    except Exception as e:
        with lock:
            print(f"Error processing {pdb_path}: {e}")
            traceback.print_exc()
        return None



# Main file to run scoring benchmark

if __name__ == "__main__":
    """
    TODO:
    Run scoring benchmark on all PDBs in the benchmark dataset.

    Stub for how benchmark should work:

        from bopep import Docker
        from bopep.docking.utils import extract_sequence_from_pdb

        scorer = Scorer()
        for pdb in pdbs:
            protein_sequence = extract_sequence_from_pdb(pdb, chain_id="A")
            peptide_sequence = extract_sequence_from_pdb(pdb, chain_id="B")

            # We have to remove the peptide from the protein chain to use this as a template when folding with ColabFold
            protein_template = remove_peptide_from_pdb(pdb, protein_chain="A", peptide_chain="B")


            docker_kwargs = {} # Look at implementation in BoPep for how to set this up
            docker = Docker(docker_kwargs)

            # Dock the peptide to the protein
            dock_dir = docker.dock_peptides([peptide_sequence])

            # Extract scores from the docking directory
            scores = scorer.score(..., colab_path = dock_dir)

            # Check if the peptide was docked in the same binding site as the original peptide
            in_same_binding_site = compare_binding_site(pdb, docking_dir) # Not implemented yet

            scores['in_same_binding_site'] = in_same_binding_site

    """
    data_dir = os.path.abspath(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data"
    )

    pdb_dir = os.path.join(data_dir, "pdbs")
    
    pdb_paths = [os.path.join(pdb_dir, pdb) for pdb in os.listdir(pdb_dir)]
    
    num_processes = 20
    
    with Pool(processes=num_processes) as pool:
        all_scores = pool.map(process_pdb, pdb_paths)
    
    all_scores = [score for score in all_scores if score is not None]
    scores_df = pd.DataFrame(all_scores)
    scores_df.set_index('pdb', inplace=True)

    output_dir = os.path.join(data_dir, "scores_sasa.csv")
    scores_df.to_csv(output_dir)
