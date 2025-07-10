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

The project has been refactored to provide a cleaner, more modular design:

- `main.py`: Entry point with three main functions:
  - `generate_data`: Download PDBs, create stripped templates
  - `dock_and_score`: Dock peptides to templates and score
  - `generate_and_score_decoys`: Create and score decoys
- `data_generation.py`: Functions for data generation
- `docking.py`: Functions for docking and scoring
- `decoys.py`: Functions for decoy generation and scoring
- `utils.py`: Common utility functions
- `utils/`: Directory containing detailed implementation modules

### Pipeline Steps

The pipeline consists of three main steps:

1. **Data Generation**: Downloads PDB files based on PDBBind INDEX files, processes them to ensure correct chain labeling, and creates stripped protein templates.

2. **Docking and Scoring**: Takes the processed data and docks the peptides to the corresponding protein templates. The resulting complexes are scored using various metrics.

3. **Decoy Generation and Scoring**: Creates decoy peptides by shuffling the sequences of real peptides, docks them to protein templates, and scores them. These serve as negative examples in the benchmark.

### Decoy Generation

The pipeline can generate decoy peptides by shuffling the amino acid sequences of real peptides. These decoys maintain the same amino acid composition but disrupt the sequence-specific binding properties.

Decoy configuration parameters:

- `n_templates`: Number of protein templates to use
- `n_decoys_per_template`: Number of decoy peptides per template
- `min_peptide_length`/`max_peptide_length`: Length constraints for peptides
- `random_seed`: Random seed for reproducibility

### Data Output Structure

The benchmark generates output in a staged directory structure:

- `0_complexes/`: Original PDB complexes, stripped templates, and metadata
- `1_docked/`: All docking results and scores (both real and decoy peptides)

The final combined dataset is saved as `all_scores.csv` in the docked directory, with an `is_decoy` column to distinguish between real and decoy peptides.

## Usage

You can run the pipeline with various command-line arguments:

```bash
# Run the complete pipeline
python main.py --all

# Only generate data
python main.py --generate-data

# Only dock and score real peptides
python main.py --dock-score

# Only generate and score decoys
python main.py --decoys

# Customize peptide length range
python main.py --all --min-length 7 --max-length 30

# Force redownloading of PDB files
python main.py --generate-data --force-redownload

# Force re-docking of already processed complexes
python main.py --dock-score --no-skip
```

### Resuming Interrupted Runs

The code is designed to be resumable and will skip already processed files by default:

- When generating data, it checks if files already exist before downloading
- When docking, it skips complexes that have already been processed
- When generating decoys, it can reuse existing decoy configuration files

To force reprocessing, use the `--force-redownload` or `--no-skip` flags.
