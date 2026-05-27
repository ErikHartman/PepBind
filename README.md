# Peptide-Protein Binding Prediction Score

This repository contains code and outputs for building a peptide-protein binding
score function. At a high level, the project takes known peptide-protein
complexes, prepares receptor/peptide structures, generates decoy peptides,
docks real and decoy peptides, scores the resulting complexes, and then trains
models that turn those scores into predictive equations.

## Repository Layout

```text
.
|-- data/
|   |-- INDEX_general_*.lst      # PDBBind index files used as input
|   `-- x_y_v2/                  # Processed feature/label matrices
|-- plots_v1/
|-- plots_v2_alphafold/
|-- plots_v2_boltz/
|-- plots_v2_both/               # Model outputs, comparisons, and figures
|-- src/
|   `-- binding_score_function/
|       |-- main.py              # End-to-end data, docking, scoring, decoy pipeline
|       |-- utils/               # Downloading, processing, docking, scoring, decoys
|       |-- predictive_equation/ # Regression/classification model training
|       `-- notebooks/           # Exploratory analyses and equation development
|-- binding_scoring_slurm.sh     # Example SLURM job for scoring/decoy pipeline
|-- predictive_equation_slurm.sh # Example SLURM job for predictive models
`-- requirements.txt
```

## Main Pipeline

The main workflow lives in `src/binding_score_function/main.py`.

It coordinates scripts for:

1. Downloading PDB structures from PDBBind/RCSB inputs.
2. Processing complexes into receptor/peptide structures.
3. Docking peptides against receptor templates with AlphaFold/Boltz-backed
   configuration through `bopep`.
4. Scoring docked structures with interface, binding-site, confidence, and
   Rosetta-derived features.
5. Generating decoy peptide sets, including shuffled and random decoys.
6. Scoring decoys so real and fake binders can be compared downstream.

The supporting implementation is in `src/binding_score_function/utils/`:

- `download_pdbs.py`: fetches structures listed in the PDBBind index files.
- `process.py`: prepares downloaded complexes and extracts peptide metadata.
- `docking.py`: docks peptides against receptor templates.
- `scoring.py`: computes structure-derived features from docked complexes.
- `decoy_peptides.py`: creates shuffled and random peptide decoy datasets.

The pipeline writes staged outputs such as downloaded complexes, processed
complex metadata, docked structures, and score tables. 

## Predictive Equations

The predictive modeling code lives in
`src/binding_score_function/predictive_equation/`.

This part of the repository takes scored complexes and converts them into
feature matrices for two modeling tasks:

- **Regression:** predict binding affinity, represented as pKd.
- **Classification:** distinguish real peptide binders from decoys.

The main scripts are:

- `create_X_y.py`: builds train/validation/test feature and label splits.
- `main_regression.py`: trains affinity prediction models.
- `main_classification.py`: trains real-vs-decoy classification models.
- `test_model_on_X_test.py`: applies selected models/equations to held-out data.

The model families include regularized linear models, random forests, support
vector models, and symbolic regression/classification via PySR. The symbolic
models are useful for extracting compact equations from the broader feature
set.

## Outputs

The repository includes generated result folders for several analysis variants:

- `plots_v1/`
- `plots_v2_alphafold/`
- `plots_v2_boltz/`
- `plots_v2_both/`

These directories contain model comparison tables, validation predictions,
probability outputs, feature importances, symbolic equation exports, and SVG/PNG
figures. They are useful as reference outputs for comparing model families and
feature sets.

The processed training data used by the predictive equation scripts is stored
under `data/x_y_v2/`, split into real, shuffled-decoy, and random-decoy feature
matrices. Note that the plots_v1 contains the tables and accompanying data from 
the first version of this project. However, the disk that kept the original source
structures crashed, and we therefore re-generated all structures in v2.

## Running

The core pipeline can also be run directly:

```bash
python src/binding_score_function/main.py --all
```

or step by step:

```bash
python src/binding_score_function/main.py --download-pdbs
python src/binding_score_function/main.py --process
python src/binding_score_function/main.py --dock
python src/binding_score_function/main.py --score
python src/binding_score_function/main.py --decoys
```

The predictive equation scripts can be run from the repository root:

```bash
python src/binding_score_function/predictive_equation/main_classification.py
python src/binding_score_function/predictive_equation/main_regression.py
```

Some dependencies and runtime tools, notably `bopep`, PyRosetta, PySR/Julia,
and GPU-backed structure prediction tooling, need to be installed in addition to the Python packages listed in `requirements.txt`.
