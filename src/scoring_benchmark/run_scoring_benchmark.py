import os
from bopep import Scorer
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


# if __name__ == "__main__":
#     """
#     TODO: Figure out why this breaks for certain pdbs.
#     - Reason 1: Cyclical peptides
#     - Read
#     """
#     data_dir = os.path.abspath(
#         "/srv/data1/general/immunopeptides_data/databases/benchmark_data"
#     )
#     pdb_dir = os.path.join(data_dir, "pdbs")
    
#     pdb_paths = [os.path.join(pdb_dir, pdb) for pdb in os.listdir(pdb_dir)]
    
#     num_processes = 36
    
#     with Pool(processes=num_processes) as pool:
#         all_scores = pool.map(process_pdb, pdb_paths)
    
#     all_scores = [score for score in all_scores if score is not None]
#     scores_df = pd.DataFrame(all_scores)
#     scores_df.set_index('pdb', inplace=True)

#     output_dir = os.path.join(data_dir, "scores.csv")
#     scores_df.to_csv(output_dir)



# Single file processing, to test the process_pdb function which is currently not working

data_dir = os.path.abspath(
        "/srv/data1/general/immunopeptides_data/databases/benchmark_data"
    )
single_pdb_path = "/srv/data1/general/immunopeptides_data/databases/benchmark_data/pdbs/6gd5.pdb"
single_score = process_pdb(single_pdb_path)

if single_score is not None:
    single_score_df = pd.DataFrame([single_score])
    single_score_df.set_index('pdb', inplace=True)
    single_output_path = os.path.join(data_dir, "single_score.csv")
    single_score_df.to_csv(single_output_path)