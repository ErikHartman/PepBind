# Scoring benchmark

[Link to nice Nature paper that does the same thing.](https://www.nature.com/articles/s43016-024-01089-5#MOESM1)

## TODO: @Malcolm

- Clean the script sand combine scripts nicely. Rename things to make the workflow neater etc.
  - Reduce intermediary outputs to a minimum. Only keep what is necessary.
  - Name variables consistently. Including column names. Two columns with the same thing should never have different names.
  - If there are several steps, prefix files with a step index. E.g, `0_*.csv` `1_*.csv` etc. Where 0 is input to produce 1.
- Create a "run script" that runs everything.
- Delete all old files.
- Re-run everything.
- Work on the script `compute_score.py`.
  - The script should take the final output and return linear regression weights.
  - It should also compute metrics.
  - It should also produce plots.

Lastly we'll want to populate this README to explain what we have done.


## Introduction

This repository contains the code to reproduce the scoring benchmark. 

## Setup

To run the benchmark, you need to set DATA_DIR in the .env file to the directory where you want to save the data.

### Data

The data used in this benchmark is based on the collection on http://www.pdbbind.org.cn/. The INDEX files contain complex PDB IDs along with their corresponding affinity values. Using this data, we fetch the PDB files from RCSB and extract the protein and ligand sequences. The sequences are then docked and scored. Lastly, a LASSO regression model is trained to predict the affinity values and the performance of the model is evaluated.


### Running the benchmark

By executing the following command, the benchmark will be run and the results will be saved in data directory that is defined in the .env file.

```bash
python3 run.py
``` 

Run.py will execute the following steps:

1. Parse the index files and download the PDB files.
2. Extract the protein and ligand sequences.
3. Dock the ligand to the protein. *NOTE: This step is not yet implemented.*
4. Score the docked ligand. *NOTE: This step is not yet implemented.*
5. Train a LASSO regression model. *NOTE: This step is not yet implemented.*
6. Evaluate the model. *NOTE: This step is not yet implemented.*


Note that run.py has flags that can be used to skip certain steps. For example, if you want to run only the download steps, you can use the following command: 

```bash
python3 run.py -d
```

The available flags are:

- `-l` or `--load`: Load the index files.
- `-d` or `--download`: Download PDB files.
- `-p` or `--preprocess`: Preprocess PDB files for docking.
- `-plt` or `--plot`: Plot peptide lengths.



