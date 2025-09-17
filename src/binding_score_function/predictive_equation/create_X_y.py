from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import os
from scipy import stats
import argparse


BASE_DIR = Path("/mnt/biomsarchive/biomsarchive/Data/personal/er8813ha/immunopeptides/results_ver2_boltz_af")
DIR_SCORES = BASE_DIR / "3_scores"
DIR_COMPLEXES = BASE_DIR / "1_processed_complexes"
DIR_PROCESSED = BASE_DIR / "4_processed_scores"
PLOTS_DIR = Path.home() / "immunopeptides" / "plots_v2"
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

def process_binding_data(binding_series: pd.Series) -> pd.Series:
    """
    Convert binding strings (e.g. "< 5 µM") to pKd values.
    """
    print(f"Processing {len(binding_series)} entries; {binding_series.isna().sum()} missing.")
    
    # Analyze what would be excluded with the new regex
    new_regex = r"[=]?\s*(?P<value>\d+\.?\d*)\s*(?P<unit>[µa-zA-Z]*)"
    new_parsed = binding_series.str.extract(new_regex)

    parsed = new_parsed
    parsed["value"] = pd.to_numeric(parsed["value"], errors="coerce")
    parsed["unit"] = parsed["unit"].str.lower()
    kd_molar = parsed["value"] * parsed["unit"].map(UNIT_CONVERSION)
    pKd = -np.log10(kd_molar)

    nan_count = kd_molar.isna().sum()
    print(f"Converted: {nan_count} entries failed (NaN).")
    if nan_count:
        print(binding_series[kd_molar.isna()].head())
    return pKd


def plot_matrix(
    X: pd.DataFrame,
    y: pd.Series,
    kind: str,
    fname: Path,
    shuffled: pd.DataFrame = None,
    random: pd.DataFrame = None,
):
    n = len(X.columns)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(30, 30))
    plt.subplots_adjust(hspace=0.5)

    for i, col in enumerate(X.columns):
        ax = axes[i // cols, i % cols]
        
        if kind == "correlation":
            # Plot scatter with regression line
            sns.scatterplot(x=X[col], y=y, ax=ax)
            sns.regplot(x=X[col], y=y, ax=ax, scatter=False, color="red")
            
            # Calculate and add Pearson correlation annotation
            correlation, p_value = stats.pearsonr(X[col], y)

            # Format annotation text with correlation value and significance stars
            annotation_text = f"r = {correlation:.2f}"
            ax.annotate(annotation_text, xy=(0.05, 0.95), xycoords='axes fraction', 
                        fontsize=10, ha='left', va='top')
            
            ax.set_ylabel("pKd")
        
        elif kind == "distribution":
            # Plot histograms for distribution comparison
            sns.histplot(X[col], kde=True, bins=50, alpha=0.5, label="Real", ax=ax, color="#2C8C99")
            if shuffled is not None:
                sns.histplot(shuffled[col], kde=True, bins=50, alpha=0.5, label="Shuffled", ax=ax, color="#E88873")
            if random is not None:
                sns.histplot(random[col], kde=True, bins=50, alpha=0.5, label="Random", ax=ax, color="#F46036")
            ax.legend(frameon=False)
            
        ax.set_title(col)

    plt.savefig(fname)
    plt.close(fig)

    fig, axs = plt.subplots(1,2, figsize=(6,3))
    if kind == "correlation":
        sns.regplot(x=X["alphafold_iptm"], y=y, ax=axs[0], scatter_kws={'s':5}, line_kws={"color": "#E88873"})
        sns.regplot(x=X["alphafold_interface_dG"], y=y, ax=axs[1], scatter_kws={'s':5}, line_kws={"color": "#E88873"})
        axs[0].annotate(f"r = {stats.pearsonr(X['alphafold_iptm'], y)[0]:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
        axs[1].annotate(f"r = {stats.pearsonr(X['alphafold_interface_dG'], y)[0]:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', ha='left', va='top')
    elif kind == "distribution":
        sns.histplot(X["alphafold_iptm"], kde=True, bins=50,  label="Real", color ="#2C8C99", ax=axs[0])
        sns.histplot(shuffled["alphafold_iptm"], kde=True, bins=50,  label="Shuffled", color="#E88873", ax=axs[0])
        sns.histplot(random["alphafold_iptm"], kde=True, bins=50,  label="Random", color="#F46036", ax=axs[0])
        sns.histplot(X["alphafold_interface_dG"], kde=True, bins=50, label="Real",color ="#2C8C99", ax=axs[1])
        sns.histplot(shuffled["alphafold_interface_dG"], kde=True, bins=50,  label="Shuffled", color="#E88873", ax=axs[1])
        sns.histplot(random["alphafold_interface_dG"], kde=True, bins=50, label="Random", color="#F46036", ax=axs[1])


    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(fname.with_suffix(".iptm.svg"))

def plot_feature_correlation_matrix(
    real_df: pd.DataFrame,
    shuffled_df: pd.DataFrame,
    random_df: pd.DataFrame,
    fname: Path,
    sample_size: int = 1000
):
    """
    Create a correlation matrix plot showing relationships between all pairs of features,
    with data points colored by type (real/shuffled/random).

    """
    # Ensure all dataframes have the same columns
    common_cols = list(set(real_df.columns) & set(shuffled_df.columns) & set(random_df.columns))
    
    # Remove non-numeric columns
    numeric_cols = []
    for col in common_cols:
        if pd.api.types.is_numeric_dtype(real_df[col]):
            numeric_cols.append(col)

    
    print(f"Creating correlation matrix for {len(numeric_cols)} features: {numeric_cols}")
    
    # Sample data if too large (for performance)
    real_sample = real_df[numeric_cols].sample(n=min(len(real_df), sample_size), random_state=42)
    shuffled_sample = shuffled_df[numeric_cols].sample(n=min(len(shuffled_df), sample_size), random_state=42)
    random_sample = random_df[numeric_cols].sample(n=min(len(random_df), sample_size), random_state=42)
    
    # Add data type labels
    real_sample = real_sample.copy()
    real_sample['data_type'] = 'Real'
    shuffled_sample = shuffled_sample.copy()
    shuffled_sample['data_type'] = 'Shuffled'
    random_sample = random_sample.copy()
    random_sample['data_type'] = 'Random'
    
    combined_df = pd.concat([real_sample, shuffled_sample, random_sample], ignore_index=True)
    
    colors = {'Real': '#2C8C99', 'Shuffled': '#E88873', 'Random': '#F46036'}
    
    n_features = len(numeric_cols)
    
    fig, axes = plt.subplots(n_features, n_features, figsize=(3*n_features, 3*n_features))
    
    for i, col1 in enumerate(numeric_cols):
        for j, col2 in enumerate(numeric_cols):
            ax = axes[i, j]
            
            if i == j:

                for data_type, color in colors.items():
                    subset = combined_df[combined_df['data_type'] == data_type]
                    ax.hist(subset[col1], bins=30, alpha=0.6, color=color, density=True)
                ax.set_xlabel(col1)
                ax.set_ylabel("Density")
                
            else:
     
                for data_type, color in colors.items():
                    subset = combined_df[combined_df['data_type'] == data_type]
                    sns.regplot(x=subset[col2], y=subset[col1], 
                                ax=ax, color=color)
                ax.set_xlabel(col2)
                ax.set_ylabel(col1)
                    
    plt.tight_layout()
    plt.savefig(fname, dpi=200, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Correlation matrix plot saved to: {fname}")



def split_and_save_real(df: pd.DataFrame):
    """
    Prepare features and labels for real complexes, plot correlations, 
    and return train/val/test splits.
    """
    X = df.drop(columns=[        
        "alphafold_in_binding_site", "boltz_in_binding_site", "is_decoy_x", "is_decoy_y", "pKd",
        "alphafold_in_binding_site_score", "boltz_in_binding_site_score", "alphafold_receptor_contacts", "boltz_receptor_contacts"
        ], errors='ignore')
    y = df["pKd"].astype(float)

    plot_matrix(
        X,
        y,
        kind="correlation",
        fname=PLOTS_DIR / "corr_real.png",
    )
    

    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42)
    
    # Create DataFrames for y values that preserve the complex_filename index
    y_train_df = pd.DataFrame({'complex_filename': X_train.index, 'pKd': y_train})
    y_val_df = pd.DataFrame({'complex_filename': X_val.index, 'pKd': y_val})
    y_test_df = pd.DataFrame({'complex_filename': X_test.index, 'pKd': y_test})
    
    return X_train, X_val, X_test, y_train_df, y_val_df, y_test_df


def split_and_save_decoys(decoy_df: pd.DataFrame) -> dict:
    """
    Split shuffled and random decoys into train/val/test sets.
    """
    df = decoy_df.copy()
    results = {}
    
    for t in ["shuffle", "random"]:
        subset = df[df.decoy_type == t].set_index("complex_filename").drop(columns="decoy_type")
        temp_subset, test_subset = train_test_split(subset, test_size=0.1, random_state=42)
        train_subset, val_subset = train_test_split(temp_subset, test_size=0.2, random_state=42)
        
        results[f"{t}_X_train"] = train_subset
        results[f"{t}_X_train_with_index"] = train_subset.reset_index()
        
        results[f"{t}_X_val"] = val_subset
        results[f"{t}_X_val_with_index"] = val_subset.reset_index()
        
        results[f"{t}_X_test"] = test_subset
        results[f"{t}_X_test_with_index"] = test_subset.reset_index()
    
    return results


def save_if_not_exists(df: pd.DataFrame, path: Path):
    if path.exists():
        print(f"{path.name} exists; skipping.")
    else:
        df.to_csv(path)

def set_permissions_to_777(directory) -> None:
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


def validate_processed_files(
    original_real,
    original_decoys,
    X_train,
    X_val,
    X_test,
    y_train_df,
    y_val_df,
    y_test_df,
    decoy_splits,
):
    """
    Validate that the processed files maintain data integrity with the original data.
    
    Args:
        original_real: Original real data before processing
        original_decoys: Original decoy data before processing
        X_train, X_val, X_test: Feature DataFrames for real complexes
        y_train_df, y_val_df, y_test_df: Label DataFrames for real complexes
        decoy_splits: Dictionary of feature DataFrames for decoys
    """
    validation_results = {}
    
    # Check 1: Verify all complexes are accounted for in train/val/test splits
    all_real_complexes = set(X_train.index) | set(X_val.index) | set(X_test.index)
    original_real_binding_site = set(original_real[original_real.alphafold_in_binding_site]['complex_filename'])
    
    validation_results["all_real_complexes_accounted_for"] = (
        len(all_real_complexes) == len(original_real_binding_site)
    )
    validation_results["missing_complexes"] = original_real_binding_site - all_real_complexes
    validation_results["extra_complexes"] = all_real_complexes - original_real_binding_site
    
    # Check 2: Verify feature values integrity for real data
    feature_checks = {}
    for complex_id in X_train.index:
        original_row = original_real[original_real.complex_filename == complex_id]
        if len(original_row) == 0:
            continue
            
        for col in X_train.columns:
            if col in original_row.columns:
                values_match = np.isclose(X_train.loc[complex_id, col], original_row[col].values[0], rtol=1e-5, atol=1e-8)
                if not values_match:
                    feature_checks[f"{complex_id}:{col}"] = {
                        "processed": float(X_train.loc[complex_id, col]),
                        "original": float(original_row[col].values[0]),
                        "diff": float(X_train.loc[complex_id, col] - original_row[col].values[0])
                    }
    
    validation_results["feature_value_mismatches"] = feature_checks
    validation_results["feature_integrity"] = len(feature_checks) == 0
    
    # Check 3: Verify pKd values integrity
    pkd_checks = {}
    for complex_id, pkd in y_train_df.set_index("complex_filename")["pKd"].items():
        original_row = original_real[original_real.complex_filename == complex_id]
        if len(original_row) == 0:
            continue
            
        if "pKd" in original_row.columns:
            values_match = np.isclose(pkd, original_row["pKd"].values[0], rtol=1e-5, atol=1e-8)
            if not values_match:
                pkd_checks[complex_id] = {
                    "processed": float(pkd),
                    "original": float(original_row["pKd"].values[0]),
                    "diff": float(pkd - original_row["pKd"].values[0])
                }
    
    validation_results["pkd_value_mismatches"] = pkd_checks
    validation_results["pkd_integrity"] = len(pkd_checks) == 0
    
    # Check 4: Verify decoy data integrity
    decoy_checks = {}
    for decoy_type in ["shuffle", "random"]:
        decoy_df = decoy_splits[f"{decoy_type}_X_train"]
        for complex_id in decoy_df.index:
            original_row = original_decoys[original_decoys.complex_filename == complex_id]
            if len(original_row) == 0:
                continue
                
            for col in decoy_df.columns:
                if col in original_row.columns:
                    values_match = np.isclose(decoy_df.loc[complex_id, col], original_row[col].values[0], rtol=1e-5, atol=1e-8)
                    if not values_match:
                        decoy_checks[f"{decoy_type}:{complex_id}:{col}"] = {
                            "processed": float(decoy_df.loc[complex_id, col]),
                            "original": float(original_row[col].values[0]),
                            "diff": float(decoy_df.loc[complex_id, col] - original_row[col].values[0])
                        }
    
    validation_results["decoy_value_mismatches"] = decoy_checks
    validation_results["decoy_integrity"] = len(decoy_checks) == 0
    
    # Check 5: Verify no overlap between train, val, and test sets
    train_val_overlap = set(X_train.index) & set(X_val.index)
    train_test_overlap = set(X_train.index) & set(X_test.index)
    val_test_overlap = set(X_val.index) & set(X_test.index)
    
    validation_results["no_train_val_overlap"] = len(train_val_overlap) == 0
    validation_results["no_train_test_overlap"] = len(train_test_overlap) == 0
    validation_results["no_val_test_overlap"] = len(val_test_overlap) == 0
    validation_results["train_val_overlap"] = train_val_overlap
    validation_results["train_test_overlap"] = train_test_overlap
    validation_results["val_test_overlap"] = val_test_overlap
    
    # Check 6: Verify y values match complex IDs in X
    y_train_ids = set(y_train_df["complex_filename"])
    y_val_ids = set(y_val_df["complex_filename"])
    y_test_ids = set(y_test_df["complex_filename"])
    
    validation_results["y_train_matches_x_train"] = y_train_ids == set(X_train.index)
    validation_results["y_val_matches_x_val"] = y_val_ids == set(X_val.index)
    validation_results["y_test_matches_x_test"] = y_test_ids == set(X_test.index)
    
    # Print validation results
    print("\n--- DATA VALIDATION RESULTS ---\n")
    
    total_checks = 0
    passed_checks = 0
    
    for check, result in validation_results.items():
        if check.endswith("_integrity") or check.startswith("no_") or check.startswith("all_") or check.startswith("y_"):
            total_checks += 1
            if result:
                passed_checks += 1
                print(f"✅ PASS: {check}")
            else:
                print(f"❌ FAIL: {check}")
    
    if validation_results["feature_value_mismatches"]:
        print(f"\nFeature value mismatches: {len(validation_results['feature_value_mismatches'])}")
        for key, mismatch in list(validation_results["feature_value_mismatches"].items())[:5]:
            print(f"  {key}: processed={mismatch['processed']}, original={mismatch['original']}, diff={mismatch['diff']}")
        if len(validation_results["feature_value_mismatches"]) > 5:
            print(f"  ... and {len(validation_results['feature_value_mismatches']) - 5} more")
    
    if validation_results["pkd_value_mismatches"]:
        print(f"\npKd value mismatches: {len(validation_results['pkd_value_mismatches'])}")
        for key, mismatch in list(validation_results["pkd_value_mismatches"].items())[:5]:
            print(f"  {key}: processed={mismatch['processed']}, original={mismatch['original']}, diff={mismatch['diff']}")
        if len(validation_results["pkd_value_mismatches"]) > 5:
            print(f"  ... and {len(validation_results['pkd_value_mismatches']) - 5} more")
    
    if validation_results["decoy_value_mismatches"]:
        print(f"\nDecoy value mismatches: {len(validation_results['decoy_value_mismatches'])}")
        for key, mismatch in list(validation_results["decoy_value_mismatches"].items())[:5]:
            print(f"  {key}: processed={mismatch['processed']}, original={mismatch['original']}, diff={mismatch['diff']}")
        if len(validation_results["decoy_value_mismatches"]) > 5:
            print(f"  ... and {len(validation_results['decoy_value_mismatches']) - 5} more")
    
    # Check overlaps
    for overlap_type in ["train_val_overlap", "train_test_overlap", "val_test_overlap"]:
        if validation_results[overlap_type]:
            print(f"\n{overlap_type}: {validation_results[overlap_type]}")
    
    print(f"\nValidation checks passed: {passed_checks} / {total_checks}")
    
    # Assert all critical validation checks passed
    assert validation_results["feature_integrity"], "Feature values do not match original data"
    assert validation_results["pkd_integrity"], "pKd values do not match original data"
    assert validation_results["decoy_integrity"], "Decoy values do not match original data" 
    assert validation_results["no_train_val_overlap"], "Train and validation sets overlap"
    assert validation_results["no_train_test_overlap"], "Train and test sets overlap"
    assert validation_results["no_val_test_overlap"], "Validation and test sets overlap"
    assert validation_results["y_train_matches_x_train"], "y_train complex IDs don't match X_train"
    assert validation_results["y_val_matches_x_val"], "y_val complex IDs don't match X_val"
    assert validation_results["y_test_matches_x_test"], "y_test complex IDs don't match X_test"
    
    print("\nAll validation checks passed! ✅")
    
    return validation_results


def validate_processed_files_from_disk(base_dir):
    """
    Validate the processed files saved on disk.
    
    Args:
        base_dir: Path to the directory containing the processed files
    """
    # Load original data
    original_real = pd.read_csv(base_dir / "3_scores" / "scores.csv")
    original_decoys = pd.read_csv(base_dir / "3_scores" / "decoy_scores.csv")
    
    # Load processed data
    X_train = pd.read_csv(base_dir / "4_processed_scores" / "real_X_train.csv").set_index("complex_filename")
    X_val = pd.read_csv(base_dir / "4_processed_scores" / "real_X_val.csv").set_index("complex_filename")
    X_test = pd.read_csv(base_dir / "4_processed_scores" / "real_X_test.csv").set_index("complex_filename")
    
    y_train_df = pd.read_csv(base_dir / "4_processed_scores" / "real_y_train.csv")
    y_val_df = pd.read_csv(base_dir / "4_processed_scores" / "real_y_val.csv")
    y_test_df = pd.read_csv(base_dir / "4_processed_scores" / "real_y_test.csv")
    
    # Load decoy data
    decoy_splits = {
        "shuffle_X_train": pd.read_csv(base_dir / "4_processed_scores" / "shuffle_X_train.csv").set_index("complex_filename"),
        "shuffle_X_val": pd.read_csv(base_dir / "4_processed_scores" / "shuffle_X_val.csv").set_index("complex_filename"),
        "shuffle_X_test": pd.read_csv(base_dir / "4_processed_scores" / "shuffle_X_test.csv").set_index("complex_filename"),
        "random_X_train": pd.read_csv(base_dir / "4_processed_scores" / "random_X_train.csv").set_index("complex_filename"),
        "random_X_val": pd.read_csv(base_dir / "4_processed_scores" / "random_X_val.csv").set_index("complex_filename"),
        "random_X_test": pd.read_csv(base_dir / "4_processed_scores" / "random_X_test.csv").set_index("complex_filename"),
    }
    
    # Run validation
    return validate_processed_files(
        original_real=original_real,
        original_decoys=original_decoys,
        X_train=X_train,
        X_val=X_val, 
        X_test=X_test,
        y_train_df=y_train_df,
        y_val_df=y_val_df,
        y_test_df=y_test_df,
        decoy_splits=decoy_splits
    )


def remove_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove outliers from the dataframe based on predefined thresholds.
    Also removes rows containing infinite values.
    """
    initial_rows = len(df)
    
    # First, remove rows containing infinite values
    print("\nChecking for infinite values...")
    
    # Find rows with infinite values across numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    infinite_mask = np.isinf(df[numeric_cols]).any(axis=1)
    infinite_count = infinite_mask.sum()
    
    if infinite_count > 0:
        print(f"Found {infinite_count} rows with infinite values ({infinite_count/initial_rows:.1%} of data)")
        # Show which columns have infinite values
        for col in numeric_cols:
            col_inf_count = np.isinf(df[col]).sum()
            if col_inf_count > 0:
                print(f"  - {col}: {col_inf_count} infinite values")
        
        # Remove rows with infinite values
        df = df[~infinite_mask].copy()
        print(f"Removed {infinite_count} rows with infinite values")
    else:
        print("No infinite values found")
    
    # Define outlier thresholds for specific columns
    outlier_thresholds = {
        "alphafold_interface_dG": (-100, 25),  # interface_dG between -100 and 100
        "alphafold_rosetta_score": (-1200, 1500),        # rosetta_score less than 1500
        "alphafold_interface_sasa": 4000         # interface_sasa less than 4000
    }
    
    current_rows = len(df)
    
    # Create a combined mask to keep rows that pass all threshold checks
    mask = pd.Series(True, index=df.index)
    
    # Apply each threshold
    for col, threshold in outlier_thresholds.items():
        if col in df.columns:
            if isinstance(threshold, tuple):
                # Range threshold (min, max)
                col_mask = (df[col] >= threshold[0]) & (df[col] <= threshold[1])
            else:
                # Upper bound threshold
                col_mask = df[col] < threshold
                
            # Update the mask
            mask = mask & col_mask
            
            # Report the outliers found in this column
            outlier_count = (~col_mask).sum()
            if outlier_count > 0:
                if isinstance(threshold, tuple):
                    print(f"Found {outlier_count} in '{col}' (outside range {threshold})")
                else:
                    print(f"Found {outlier_count} in '{col}' (>= {threshold})")
    
    # Apply the mask to get the filtered dataframe
    filtered_df = df.loc[mask].copy()
    
    # Report total outliers removed
    outlier_removed_count = current_rows - len(filtered_df)
    total_removed_count = initial_rows - len(filtered_df)
    
    if outlier_removed_count > 0:
        print(f"Removed {outlier_removed_count} outliers ({outlier_removed_count/current_rows:.1%} of remaining data)")
    
    if total_removed_count > 0:
        print(f"Total removed: {total_removed_count} rows ({total_removed_count/initial_rows:.1%} of original data)")
        
    return filtered_df


def main():
    pdbs = pd.read_csv(DIR_COMPLEXES / "processed_pdbs.csv")
    pdbs["complex_filename"] = pdbs.pdb_code + '_' + pdbs.peptide_sequence

    real = pd.read_csv(DIR_SCORES / "scores.csv")
    real = real[real["boltz_in_binding_site"]]
    real = real[real["alphafold_in_binding_site"]]

    real_columns_to_drop = [
        "alphafold_in_binding_site", "boltz_in_binding_site", "is_decoy_x", "is_decoy_y", "pKd", "receptor_contacts", "alphafold_template_rmsd",
          "boltz_template_rmsd", "peptide_pde",
        "alphafold_in_binding_site_score", "boltz_in_binding_site_score", "alphafold_receptor_contacts", "boltz_receptor_contacts"
    ]
    decoy_columns_to_drop = ["rosetta_score","interface_sasa","interface_dG","interface_delta_hbond_unsat","packstat","distance_score","in_binding_site","n_contacts","in_binding_site_score","template_rmsd","receptor_contacts", 
                             "peptide_plddt", "interface_peptide_plddt", "peptide_pae", "ipsae_max", "ipsae_min", "iptm", "receptor_contacts", "alphafold_receptor_contacts", "boltz_receptor_contacts",
                             "boltz_template_rmsd", "alphafold_template_rmsd", "is_decoy", "peptide_pde"]

    dec_meta = pd.read_csv(DIR_COMPLEXES / "decoys.csv")
    dec_meta["complex_filename"] = dec_meta.pdb_code + '_' + dec_meta.peptide_sequence
    dec_scores = pd.read_csv(DIR_SCORES / "decoy_scores.csv")

    dec_scores.drop(columns=real_columns_to_drop + decoy_columns_to_drop, inplace=True, errors='ignore')
    dec_scores.dropna(inplace=True)

    real = real.drop(columns=real_columns_to_drop + decoy_columns_to_drop, errors='ignore')
    
    decoys = dec_meta[["complex_filename", "decoy_type"]].merge(
        dec_scores, on="complex_filename", how="inner"
    )

    bind_map = pd.Series(pdbs.binding_data.values, index=pdbs.complex_filename)
    real["pKd"] = process_binding_data(real.complex_filename.map(bind_map))
    real.dropna(inplace=True)
    
    
    # Store original data for validation
    original_real = real.copy()
    original_decoys = decoys.copy()

    print(original_decoys)


    print("\nPlotting unfiltered feature–pKd correlations...")
    real_full = real.set_index("complex_filename")

    X_all = real_full.copy()
    y_all = real_full["pKd"].astype(float)
    print(X_all.dtypes)

    # Remove outliers from real data before splitting
    print("\nRemoving outliers from real data:")
    real_full = remove_outliers(real_full)
    print(f"Real data now has {len(real_full)} rows")
    
    # Remove outliers from decoy data
    print("\nRemoving outliers from decoy data:")
    decoys = remove_outliers(decoys)
    print(f"Decoy data now has {len(decoys)} rows")
    decoy_splits = split_and_save_decoys(decoys)

    X_train, X_val, X_test, y_train_df, y_val_df, y_test_df = split_and_save_real(real_full)

    print("Number of rows in the different files: ")
    print(f"Real train: {len(X_train)}, Real val: {len(X_val)}, Real test: {len(X_test)}")
    print(f"Shuffled train: {len(decoy_splits['shuffle_X_train'])}, Shuffled val: {len(decoy_splits['shuffle_X_val'])}, Shuffled test: {len(decoy_splits['shuffle_X_test'])}")
    print(f"Random train: {len(decoy_splits['random_X_train'])}, Random val: {len(decoy_splits['random_X_val'])}, Random test: {len(decoy_splits['random_X_test'])}")
    
    save_if_not_exists(X_train.reset_index(), DIR_PROCESSED / 'real_X_train.csv')
    save_if_not_exists(X_val.reset_index(), DIR_PROCESSED / 'real_X_val.csv')
    save_if_not_exists(X_test.reset_index(), DIR_PROCESSED / 'real_X_test.csv')
    
    save_if_not_exists(y_train_df, DIR_PROCESSED / 'real_y_train.csv')
    save_if_not_exists(y_val_df, DIR_PROCESSED / 'real_y_val.csv')
    save_if_not_exists(y_test_df, DIR_PROCESSED / 'real_y_test.csv')

    #set_permissions_to_777(DIR_PROCESSED)

    for name, df in decoy_splits.items():
        if '_with_index' in name:
            save_if_not_exists(df, DIR_PROCESSED / f"{name.replace('_with_index', '')}.csv")

    plot_matrix(
        X_train,
        y_train_df.set_index('complex_filename')['pKd'],
        kind="correlation",
        fname=PLOTS_DIR / "corr_real.png",
    )

    plot_matrix(
        X_train,
        y_train_df.set_index('complex_filename')['pKd'],
        kind='distribution',
        fname=PLOTS_DIR / 'dist_real_vs_decoys.png',
        shuffled=decoy_splits['shuffle_X_train'],
        random=decoy_splits['random_X_train'],
    )

    #plot_feature_correlation_matrix(
    #    real_df=X_train,
    #    shuffled_df=decoy_splits['shuffle_X_train'], 
    #    random_df=decoy_splits['random_X_train'],
    #    fname=PLOTS_DIR / 'feature_correlation_matrix.png',
    #    sample_size=1000  # Limit points for performance
    #)
    
    #print("\nValidating data integrity...\n")
    # validate_processed_files(
    #     original_real=original_real,
    #     original_decoys=original_decoys,
    #     X_train=X_train,
    #     X_val=X_val, 
    #     X_test=X_test,
    #     y_train_df=y_train_df,
    #     y_val_df=y_val_df,
    #     y_test_df=y_test_df,
    #     decoy_splits=decoy_splits
    # )


if __name__ == "__main__":

    
    parser = argparse.ArgumentParser(description="Process immunopeptide data and validate processed files")
    parser.add_argument("--validate-only", action="store_true", 
                      help="Only validate existing processed files without regenerating them")
    args = parser.parse_args()
    
    if args.validate_only:
        print("Running validation on existing files...")
        try:
            validate_processed_files_from_disk(BASE_DIR)
            print("Validation complete!")
        except Exception as e:
            print(f"Validation failed: {e}")
            exit(1)
    else:
        main()
