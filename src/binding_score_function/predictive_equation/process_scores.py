import pandas as pd
import numpy as np
import os
from Bio.PDB import PDBParser, Superimposer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split


def process_binding_data(binding_series: pd.Series):
    """
    Process binding data strings into pKd values.
    """
    print(f"Total binding values: {len(binding_series)}")
    print(f"NA values before processing: {binding_series.isna().sum()}")

    binding_data = binding_series.str.extract(
        r"([=<>~]?)\s*(\d+\.?\d*(?:[eE][+-]?\d+)?)\s*([µa-zA-Z]*)"
    )

    print(binding_data.head())

    binding_data[1] = pd.to_numeric(binding_data[1], errors="coerce")
    binding_data[2] = binding_data[2].str.lower()

    unit_conversion = {
        "fm": 1e-15,
        "pm": 1e-12,
        "nm": 1e-9,
        "um": 1e-6,
        "μm": 1e-6,
        "mm": 1e-3,
        "m": 1,
    }

    def safe_convert(row):
        value, unit = row[1], row[2]
        if pd.isna(value):
            return float("nan")
        if pd.isna(unit) or unit == "":
            return float("nan")
        return value * unit_conversion.get(unit.lower(), float("nan"))

    Kd_M_numeric = binding_data.apply(safe_convert, axis=1)
    pKd = -np.log10(Kd_M_numeric)

    print(f"NaN count in Kd values: {Kd_M_numeric.isna().sum()}")
    print(f"NaN count in pKd values: {pKd.isna().sum()}")

    if Kd_M_numeric.isna().sum() > 0:
        print("Examples of binding data that caused NaN:")
        print(binding_series[Kd_M_numeric.isna()].head())

    return pKd


def check_peptide_rmsd(
    original_pdb_path: str, complex_pdb_path: str, peptide_chain_id: str = "B"
) -> float:
    parser = PDBParser(QUIET=True)

    original_structure = parser.get_structure("original", original_pdb_path)
    docked_structure = parser.get_structure("docked", complex_pdb_path)
    original_peptide = original_structure[0][peptide_chain_id]
    docked_peptide = docked_structure[0][peptide_chain_id]

    original_peptide_atoms = [
        res["CA"]
        for res in original_peptide.get_residues()
        if res.id[0] == " " and "CA" in res
    ]
    docked_peptide_atoms = [
        res["CA"]
        for res in docked_peptide.get_residues()
        if res.id[0] == " " and "CA" in res
    ]

    super_imposer = Superimposer()
    super_imposer.set_atoms(original_peptide_atoms, docked_peptide_atoms)
    return super_imposer.rms


def split_and_save_real_scores(real_scores_in_binding_site: pd.DataFrame):

    real_scores_in_binding_site = real_scores_in_binding_site.copy()
    y = real_scores_in_binding_site["pKd"].astype(float)
    y["complex_filename"] = real_scores_in_binding_site[
        "complex_filename"
    ]
    X = real_scores_in_binding_site.drop(
        columns=[
            "in_binding_site",
            "is_decoy",
            "pKd",
            "fraction_in_binding_site",
            "in_binding_site_score",
        ]
    )
    X.set_index("complex_filename", inplace=True)

    plot_correlation_with_X_and_y(
        X,
        y,
        save_path="/home/er8813ha/immunopeptides/plots/correlation_with_X_and_y.png",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    return X_train, X_test, y_train, y_test


def plot_correlation_with_X_and_y(X, y, save_path=None):
    """
    subplots
    """
    nr_columns = len(X.columns)
    nr_cols = 5
    nr_rows = int(np.ceil(nr_columns / nr_cols))

    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)
    for i, column in enumerate(X.columns):
        ax = axs[i // nr_cols, i % nr_cols]
        sns.scatterplot(x=X[column].astype(float).values, y=y.values, ax=ax)
        # draw line of best fit
        sns.regplot(
            x=X[column].astype(float).values,
            y=y.values,
            ax=ax,
            scatter=False,
            color="red",
        )
        ax.set_xlabel(column)
        ax.set_ylabel("pKd")
        ax.set_title(f"Correlation with {column}")
    plt.savefig(save_path)


def split_and_save_decoys(decoys : pd.DataFrame):
    decoys = decoys.copy()
    decoys.drop(
        columns=[
            "is_decoy",
            "in_binding_site",
            "in_binding_site_score",
            "fraction_in_binding_site",
        ], inplace=True
    )
    print("Decoys columns")
    print(decoys.columns)

    shuffled_X = decoys[decoys["decoy_type"] == "shuffle"].set_index("complex_filename")
    random_X= decoys[decoys["decoy_type"] == "random"].set_index("complex_filename")
    shuffled_X.drop(columns=["decoy_type"], inplace=True)
    random_X.drop(columns=["decoy_type"], inplace=True)
    shuffled_X_train = shuffled_X.sample(frac=0.85, random_state=42)
    shuffled_X_test = shuffled_X.drop(shuffled_X_train.index)
    random_X_train = random_X.sample(frac=0.85, random_state=42)
    random_X_test = random_X.drop(random_X_train.index)
    
    print("Shuffled X train")
    print(shuffled_X_train)


    return shuffled_X_train, random_X_train, shuffled_X_test, random_X_test

def plot_distribution(real_X, shuffled_X, random_X, save_path=None):
    nr_columns = len(real_X.columns)
    nr_cols = 5
    nr_rows = int(np.ceil(nr_columns / nr_cols))

    fig, axs = plt.subplots(nr_rows, nr_cols, figsize=(20, 20))
    fig.subplots_adjust(hspace=0.5)
    for i, column in enumerate(real_X.columns):
        ax = axs[i // nr_cols, i % nr_cols]

        sns.histplot(x = real_X[column].astype(float).values, ax=ax, color="blue", label="Real", kde=True, bins=50, alpha=.5)
        sns.histplot(x = shuffled_X[column].astype(float).values, ax=ax, color="red", label="Shuffled", kde=True, bins=50, alpha=.5)
        sns.histplot(x = random_X[column].astype(float).values, ax=ax, color="orange", label="Random", kde=True, bins=50, alpha=.5)
        ax.set_title(f"{column}")
        ax.legend(frameon=False)
    plt.savefig(save_path)


if __name__ == "__main__":
    base_dir = os.path.dirname(
        "/srv/data1/general/immunopeptides_data/outputs/binding_score_function/"
    )
    scores = os.path.join(base_dir, "3_scores")
    processed_complexes = os.path.join(base_dir, "1_processed_complexes")
    processed_scores = os.path.join(base_dir, "4_processed_scores")

    peptide_data = pd.read_csv(os.path.join(processed_complexes, "processed_pdbs.csv"))
    peptide_data["complex_filename"] = (
        peptide_data["pdb_code"] + "_" + peptide_data["peptide_sequence"]
    )
    real_scores = pd.read_csv(os.path.join(scores, "scores.csv"))

    decoys_data = pd.read_csv(os.path.join(processed_complexes, "decoys.csv"))
    decoys_data["complex_filename"] = (
        decoys_data["pdb_code"] + "_" + decoys_data["peptide_sequence"]
    )
    print("Decoys data")
    print(decoys_data.columns)
    print(decoys_data.head())
    
    decoys_scores = pd.read_csv(os.path.join(scores, "decoy_scores.csv"))
    print("Decoys scores")
    print(decoys_scores.columns)
    print(decoys_scores.head())
    decoys_scores = decoys_data[["complex_filename", "decoy_type"]].merge(decoys_scores, left_on="complex_filename", right_on="complex_filename",how="inner" )

    print("Decoys scores")  
    print(decoys_scores.columns)
    print(decoys_scores.head())

    binding_data_map = dict(
        zip(peptide_data["complex_filename"], peptide_data["binding_data"])
    )
    binding_data_series = real_scores["complex_filename"].map(binding_data_map)
    real_scores["pKd"] = process_binding_data(binding_data_series)
    real_scores.dropna(how="any", inplace=True, axis=0)

    print("Real data")
    print(peptide_data.head())
    print(peptide_data.tail())
    print(peptide_data.columns)

    not_in_binding_site = real_scores[real_scores["in_binding_site"] == False]
    for _, row in not_in_binding_site.iterrows():
        
        complex_filename = row["complex_filename"]
        print(complex_filename)
        pdb_id = complex_filename.split("_")[0]
        original_pdb_path = os.path.join(
            base_dir, "0_complexes", "pdbs", f"{pdb_id}.pdb"
        )
        complex_colabdir_path = os.path.join(
            base_dir, "2_docked", "pdbs", complex_filename
        )

        # get the relaxed structure by searching for _relaxed_
        relaxed_pdb_path = [
            os.path.join(complex_colabdir_path, f)
            for f in os.listdir(complex_colabdir_path)
            if "_relaxed_" in f
        ]

        rmsd = check_peptide_rmsd(original_pdb_path, relaxed_pdb_path[0])
        print(f"RMSD for {complex_filename}: {rmsd}")
        if rmsd < 2:
            "they are in the same binding site"
            real_scores.loc[
                real_scores["complex_filename"] == complex_filename, "in_binding_site"
            ] = True

    not_in_binding_site = real_scores[real_scores["in_binding_site"] == False]

    real_scores_in_binding_site = real_scores[real_scores["in_binding_site"] == True]

    print(real_scores_in_binding_site.columns)

    X_train, X_test, y_train, y_test = split_and_save_real_scores(
        real_scores_in_binding_site
    )

    shuffled_X_train, random_X_train, shuffled_X_test, random_X_test = split_and_save_decoys(decoys_scores)

    plot_distribution(X_train, shuffled_X_train, random_X_train, save_path="/home/er8813ha/immunopeptides/plots/score_distribution.png")

    if "real_X_train.csv" in os.listdir(processed_scores):
        print("real_X_train.csv already exists. Delete it first.")
    else:
        X_train.to_csv(os.path.join(processed_scores, "real_X_train.csv"))
    if "real_X_test.csv" in os.listdir(processed_scores):
        print("real_X_test.csv already exists. Delete it first.")
    else:
        X_test.to_csv(os.path.join(processed_scores, "real_X_test.csv"))
    if "real_y_train.csv" in os.listdir(processed_scores):
        print("real_y_train.csv already exists. Delete it first.")
    else:
        y_train.to_csv(os.path.join(processed_scores, "real_y_train.csv"))
    if "real_y_test.csv" in os.listdir(processed_scores):
        print("real_y_test.csv already exists. Delete it first.")
    else:
        y_test.to_csv(os.path.join(processed_scores, "real_y_test.csv"))

    

    if "shuffled_X_train.csv" in os.listdir(processed_scores):
        print("shuffled_X_train.csv already exists. Delete it first.")
    else:
        shuffled_X_train.to_csv(os.path.join(processed_scores, "shuffled_X_train.csv"))
    if "random_X_train.csv" in os.listdir(processed_scores):
        print("random_X_train.csv already exists. Delete it first.")
    else:
        random_X_train.to_csv(os.path.join(processed_scores, "random_X_train.csv"))
    if "shuffled_X_test.csv" in os.listdir(processed_scores):
        print("shuffled_X_test.csv already exists. Delete it first.")
    else:
        shuffled_X_test.to_csv(os.path.join(processed_scores, "shuffled_X_test.csv"))
    if "random_X_test.csv" in os.listdir(processed_scores):
        print("random_X_test.csv already exists. Delete it first.")
    else:
        random_X_test.to_csv(os.path.join(processed_scores, "random_X_test.csv"))


    print("Not in binding site")
    print(not_in_binding_site["complex_filename"])

    print("Deocys")
    print(decoys_data.head())
    print(decoys_data.tail())
    print(decoys_data.columns)
