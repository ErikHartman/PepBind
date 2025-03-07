import os
from bopep import Scorer
from bopep.docking.docker import Docker
from bopep.docking.utils import extract_sequence_from_pdb
import pandas as pd
from multiprocessing import Pool, Value, Lock
import traceback
from bopep import Docker
from bopep.docking.dock_peptides import dock_peptide 
from bopep.docking.utils import extract_sequence_from_pdb
from utils import remove_peptide_from_complex, compare_binding_site

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



def benchmark(pdb_path):
        
    output_pdb_path = os.path.join(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs/protein_template",
        os.path.basename(pdb_path)
    )
    peptide_sequence = extract_sequence_from_pdb(pdb_path, chain_id="B")
    protein_template = remove_peptide_from_complex(pdb_path, output_pdb_path=output_pdb_path, protein_chain="A")
    
    docker_kwargs = {
        "num_models": 5,
        "num_recycles": 3,
        "recycle_early_stop_tolerance": 0.5,
        "amber": True,
        "num_relax": 2,
        "pdb_dir": "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs/docked_peptides",
        "gpu_ids": ["0"],
        "overwrite_results": False
    }
    docker = Docker(docker_kwargs)
    docker.set_target_structure(protein_template) 
    dock_dir = docker.dock_peptides([peptide_sequence])[0]
    
    scorer = Scorer()
    scores = scorer.score(scores_to_include=["interface_sasa", "rosetta_score"], colab_dir=dock_dir)
    in_same_binding_site, overlap = compare_binding_site(pdb_path, dock_dir)
    scores['in_same_binding_site'] = in_same_binding_site
    scores['overlap'] = overlap
    

if __name__ == "__main__":
    """
    TODO:
    Run scoring benchmark on all PDBs in the benchmark dataset.

    Stub for how benchmark should work:

        from bopep import Docker
        from bopep.docking.utils import extract_sequence_from_pdb
        from .utils import remove_peptide_from_pdb, compare_binding_site

        scorer = Scorer()
        for pdb in pdbs:
            peptide_sequence = extract_sequence_from_pdb(pdb, chain_id="B")

            # We have to remove the peptide from the protein chain to use this as a template when folding with ColabFold
            protein_template = remove_peptide_from_pdb(pdb, protein_chain="A") # implemented in utils

            docker_kwargs = {} # Look at implementation in BoPep for how to set this up
            docker = Docker(docker_kwargs)

            # Dock the peptide to the protein
            dock_dir = docker.dock_peptides([peptide_sequence])

            # Extract scores from the docking directory
            scores = scorer.score(..., colab_path = dock_dir)

            # Check if the peptide was docked in the same binding site as the original peptide
            in_same_binding_site, overlap = compare_binding_site(pdb, pdb_docked) # Implemented in utils

            scores['in_same_binding_site'] = in_same_binding_site
            scores['overlap'] = overlap

    """
    data_dir = os.path.abspath(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs"
    )
    
    pdb_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.pdb')]
    for pdb in pdb_files:
        benchmark(pdb)
        







