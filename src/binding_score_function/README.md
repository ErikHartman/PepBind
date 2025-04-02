# Binding Affinity Score Function

[Link to nice Nature paper that does the same thing.](https://www.nature.com/articles/s43016-024-01089-5#MOESM1)

## Introduction

This repository contains the code to reproduce the binding affinity scoring benchmark. The codebase has been refactored for better organization, type annotations, and centralized path management.

## Setup

To run the benchmark, you need to create a `.env` file with the following variables:

```sh
DATA_DIR=/path/to/your/data/directory
```

If no `.env` file is provided, the code will default to `/srv/data1/general/immunopeptides_data/`.

### Data

The data used in this benchmark is based on the collection on <http://www.pdbbind.org.cn/>. The INDEX files contain complex PDB IDs along with their corresponding affinity values. Using this data, we fetch the PDB files from RCSB and extract the protein and ligand sequences. The sequences are then docked and scored.

### Project Structure

- `get_target_pdbs.py`: Main script that orchestrates the entire workflow
- `utils/`: Directory containing utility modules:
  - `download_pdbs.py`: Functions for downloading PDB files
  - `preprocessing.py`: Functions for preprocessing PDB files
  - `scoring.py`: Functions for docking and scoring peptides

### Generating the data

By executing the following command, the complete benchmark will be run:

```bash
python download_dock_and_score.py 
```

### Pipeline Steps

1. **Download**: Parse the index files and download the PDB files.
2. **Preprocess**: Extract the protein and peptide sequences.
3. **Dock and Score**: Dock the peptide to the protein and score the interaction.
4. **Analysis**: Train regression models to predict binding affinity (see `lasso.py`).

### Data Output Structure

The benchmark generates output in a staged directory structure:

- `0_unprocessed/`: Raw downloaded PDB files
- `1_processed/`: Preprocessed PDB files ready for docking
- `2_scored/`: Docking results and scores
