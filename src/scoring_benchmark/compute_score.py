"""
Run LASSO to get predictive equation for Kd from in scilio metrics

1. Read data
2. Manually introduce new features (?) like e.g. log(rosetta_score)
3. Plot correlations
4. Train (adaptive) LASSO model (glmnet?)
5. Evaluate model
"""

import pandas as pd

if __name__ == "__main__":
    # Read data
    df = pd.read_csv("/srv/data1/general/immunopeptides_data/databases/benchmark_data/benchmark_scores.csv")
    print("Number of peptides in binding site: ", df["in_binding_site"].sum())
    print("Number of peptides not in binding site: ", (~df["in_binding_site"]).sum())

    
