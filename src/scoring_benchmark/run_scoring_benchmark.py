import os
from bopep import Scorer
import pandas as pd
from multiprocessing import Pool

def process_pdb(pdb_path):
    scorer = Scorer()
    print(pdb_path)
    try:
        pdb_scores = scorer.calculate_rosetta_scores(pdb_path)
        pdb_scores['pdb'] = os.path.basename(pdb_path)
        return pdb_scores
    except:
        return None

if __name__ == "__main__":
    """
    TODO: Figure out why this breaks for certain pdbs.
    """
    data_dir = os.path.abspath(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data"
    )
    pdb_dir = os.path.join(data_dir, "pdbs")
    
    pdb_paths = [os.path.join(pdb_dir, pdb) for pdb in os.listdir(pdb_dir)]
    
    num_processes = 72
    
    with Pool(processes=num_processes) as pool:
        all_scores = pool.map(process_pdb, pdb_paths)
    
    all_scores = [score for score in all_scores if score is not None]
    scores_df = pd.DataFrame(all_scores)
    scores_df.set_index('pdb', inplace=True)

    output_dir = os.path.join(data_dir, "scores.csv")
    scores_df.to_csv(output_dir)
