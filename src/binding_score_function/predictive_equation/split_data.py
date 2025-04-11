import pandas as pd
import os

if __name__ == "__main__":
    base_path = os.path.abspath("/srv/data1/general/immunopeptides_data/outputs/binding_score_function/1_docked")

    decoy_data = pd.read_csv(os.path.join(base_path, "decoy_scores.csv"))
    
    all_scores = pd.read_csv(os.path.join(base_path, "all_scores.csv"))

    number_of_decoys_in_decoy_data = len(decoy_data[decoy_data["is_decoy"] == True])
    print(f"Number of decoys in all scores: {number_of_decoys_in_decoy_data}")

    number_of_non_decoys_in_all_scores = len(all_scores[all_scores["is_decoy"] == False])
    print(f"Number of non-decoys in all scores: {number_of_non_decoys_in_all_scores}")