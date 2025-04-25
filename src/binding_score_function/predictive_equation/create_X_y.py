from pathlib import Path
import numpy as np
import pandas as pd
from Bio.PDB import PDBParser, Superimposer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os

# --- Constants & Configurations ---
BASE_DIR = Path("/srv/data1/general/immunopeptides_data/outputs/binding_score_function_prod")
DIR_SCORES = BASE_DIR / "3_scores"
DIR_COMPLEXES = BASE_DIR / "1_processed_complexes"
DIR_PROCESSED = BASE_DIR / "4_processed_scores"
PLOTS_DIR = Path.home() / "immunopeptides" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

UNIT_CONVERSION = {
    "fm": 1e-15,
    "pm": 1e-12,
    "nm": 1e-9,
    "um": 1e-6,
    "μm": 1e-6,
    "mm": 1e-3,
    "m": 1,
}

# --- Utility Functions ---
def process_binding_data(binding_series: pd.Series) -> pd.Series:
    """
    Convert binding strings (e.g. "< 5 µM") to pKd values.
    """
    print(f"Processing {len(binding_series)} entries; {binding_series.isna().sum()} missing.")
    parsed = binding_series.str.extract(r"[=<>~]?\s*(?P<value>\d+\.?\d*)\s*(?P<unit>[µa-zA-Z]*)")
    parsed["value"] = pd.to_numeric(parsed["value"], errors="coerce")
    parsed["unit"] = parsed["unit"].str.lower()
    kd_molar = parsed["value"] * parsed["unit"].map(UNIT_CONVERSION)
    pKd = -np.log10(kd_molar)

    nan_count = kd_molar.isna().sum()
    print(f"Converted: {nan_count} entries failed (NaN).")
    if nan_count:
        print(binding_series[kd_molar.isna()].head())
    return pKd


def check_peptide_rmsd(orig_pdb: Path, dock_pdb: Path, chain: str = "B") -> float:
    """Calculate RMSD between C-alpha atoms of two peptide chains."""
    parser = PDBParser(QUIET=True)
    orig = parser.get_structure("orig", orig_pdb)
    dock = parser.get_structure("dock", dock_pdb)

    def get_ca_atoms(chain_obj):
        return [res["CA"] for res in chain_obj.get_residues() if res.id[0] == " " and "CA" in res]

    orig_atoms = get_ca_atoms(orig[0][chain])
    dock_atoms = get_ca_atoms(dock[0][chain])

    sup = Superimposer()
    sup.set_atoms(orig_atoms, dock_atoms)
    return sup.rms


def plot_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str,
    fname: Path,
    shuffled: pd.DataFrame = None,
    random: pd.DataFrame = None,
):
    """Plot correlations (kind='correlation') or distributions (kind='distribution')."""
    n = len(X.columns)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    plt.subplots_adjust(hspace=0.5)

    for i, col in enumerate(X.columns):
        ax = axes[i // cols, i % cols]
        if kind == "correlation":
            sns.scatterplot(x=X[col], y=y, ax=ax)
            sns.regplot(x=X[col], y=y, ax=ax, scatter=False, color="red")
            ax.set_ylabel("pKd")
        elif kind == "distribution":
            sns.histplot(X[col], kde=True, bins=50, alpha=0.5, label="Real", ax=ax)
            if shuffled is not None:
                sns.histplot(shuffled[col], kde=True, bins=50, alpha=0.5, label="Shuffled", ax=ax)
            if random is not None:
                sns.histplot(random[col], kde=True, bins=50, alpha=0.5, label="Random", ax=ax)
            ax.legend(frameon=False)
        ax.set_title(col)

    plt.savefig(fname)
    plt.close(fig)


def split_and_save_real(df: pd.DataFrame):
    """Prepare features and labels for real complexes, plot correlations, split, and return train/test."""
    df = df.set_index("complex_filename")
    X = df.drop(columns=["in_binding_site", "is_decoy", "pKd", "fraction_in_binding_site", "in_binding_site_score"])
    y = df["pKd"].astype(float)

    plot_matrix(
        X,
        y,
        kind="correlation",
        fname=PLOTS_DIR / "corr_real.png",
    )
    return train_test_split(X, y, test_size=0.15, random_state=42)


def split_and_save_decoys(decoy_df: pd.DataFrame) -> dict:
    """Split shuffled and random decoys into train/test sets."""
    df = decoy_df.drop(columns=["is_decoy", "in_binding_site", "in_binding_site_score", "fraction_in_binding_site"])
    results = {}
    for t in ["shuffle", "random"]:
        subset = df[df.decoy_type == t].set_index("complex_filename").drop(columns="decoy_type")
        tr = subset.sample(frac=0.85, random_state=42)
        te = subset.drop(tr.index)
        results[f"{t}_X_train"] = tr
        results[f"{t}_X_test"] = te
    return results


def save_if_not_exists(df: pd.DataFrame, path: Path):
    if path.exists():
        print(f"{path.name} exists; skipping.")
    else:
        df.to_csv(path)

def set_permissions_to_777(directory: str) -> None:
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            try:
                os.chmod(os.path.join(root, d), 0o777)
            except Exception as e:
                print(f"Could not set permissions for {d}: {str(e)}")

        for f in files:
            try:
                os.chmod(os.path.join(root, f), 0o777)
            except Exception as e:
                print(f"Could not set permissions for {f}: {str(e)}")

    print(f"Set permissions to 777 for all files in {directory}")



def main():
    pdbs = pd.read_csv(DIR_COMPLEXES / "processed_pdbs.csv")
    pdbs["complex_filename"] = pdbs.pdb_code + '_' + pdbs.peptide_sequence

    real = pd.read_csv(DIR_SCORES / "scores.csv")
    dec_meta = pd.read_csv(DIR_COMPLEXES / "decoys.csv")
    dec_meta["complex_filename"] = dec_meta.pdb_code + '_' + dec_meta.peptide_sequence
    dec_scores = pd.read_csv(DIR_SCORES / "decoy_scores.csv")
    decoys = dec_meta[["complex_filename", "decoy_type"]].merge(
        dec_scores, on="complex_filename", how="inner"
    )

    bind_map = pd.Series(pdbs.binding_data.values, index=pdbs.complex_filename)
    real["pKd"] = process_binding_data(real.complex_filename.map(bind_map))
    real.dropna(inplace=True)

    for idx, row in real[~real.in_binding_site].iterrows():
        pdb_id, _ = row.complex_filename.split("_", 1)
        orig = BASE_DIR / '0_complexes' / 'pdbs' / f"{pdb_id}.pdb"
        dock_dir = BASE_DIR / '2_docked' / 'pdbs' / row.complex_filename
        relaxed = next(dock_dir.glob('*_relaxed_*'), None)
        if relaxed and check_peptide_rmsd(orig, relaxed) < 2:
            real.at[idx, 'in_binding_site'] = True

    real_bs = real[real.in_binding_site]
    decoy_splits = split_and_save_decoys(decoys)

    X_tr, X_te, y_tr, y_te = split_and_save_real(real_bs)
    save_if_not_exists(X_tr, DIR_PROCESSED / 'real_X_train.csv')
    save_if_not_exists(X_te, DIR_PROCESSED / 'real_X_test.csv')
    save_if_not_exists(y_tr, DIR_PROCESSED / 'real_y_train.csv')
    save_if_not_exists(y_te, DIR_PROCESSED / 'real_y_test.csv')

    set_permissions_to_777(DIR_PROCESSED)

    for name, df in decoy_splits.items():
        save_if_not_exists(df, DIR_PROCESSED / f"{name}.csv")

    # Distribution plot against decoys
    plot_matrix(
        X_tr,
        y_tr,
        kind='distribution',
        fname=PLOTS_DIR / 'dist_real_vs_decoys.png',
        shuffled=decoy_splits['shuffle_X_train'],
        random=decoy_splits['random_X_train'],
    )

if __name__ == "__main__":
    main()
