import os
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from utils import create_model_comparison

from classification import (
    train_logistic_regression,
    train_random_forest_classifier,
    train_svm_classifier,
    perform_symbolic_classification,
)

from plotting import (
    plot_classification_metrics,
    plot_feature_importances,
    plot_model_comparison,
    plot_symbolic_complexity_tradeoff,
    plot_probability_histograms,
    plot_pkd_probability_correlation,
)

if __name__ == "__main__":
    # Data paths setup
    scores_path = "/home/er8813ha/immunopeptides/data/x_y_v2"
    output_dir = "./plots_v2/classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load data from pre-split files
    # Training data
    X_real_train_df = pd.read_csv(os.path.join(scores_path, "real_X_train.csv"))
    X_shuffle_train_df = pd.read_csv(os.path.join(scores_path, "shuffle_X_train.csv"))
    X_random_train_df = pd.read_csv(os.path.join(scores_path, "random_X_train.csv"))
    y_real_train_df = pd.read_csv(os.path.join(scores_path, "real_y_train.csv"))
    
    # Validation data
    X_real_val_df = pd.read_csv(os.path.join(scores_path, "real_X_val.csv"))
    X_shuffle_val_df = pd.read_csv(os.path.join(scores_path, "shuffle_X_val.csv"))
    X_random_val_df = pd.read_csv(os.path.join(scores_path, "random_X_val.csv"))
    y_real_val_df = pd.read_csv(os.path.join(scores_path, "real_y_val.csv"))
    
    # Remove any 'Unnamed:_0' columns that might have been created during saving/loading
    for df in [X_real_train_df, X_shuffle_train_df, X_random_train_df, 
               X_real_val_df, X_shuffle_val_df, X_random_val_df]:
        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        columns_to_drop.append("receptor_contacts")
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True, errors='ignore')

    # Print columns that contain NaNs
    for df, name in zip([X_real_train_df, X_shuffle_train_df, X_random_train_df,
                         X_real_val_df, X_shuffle_val_df, X_random_val_df],
                        ["X_real_train", "X_shuffle_train", "X_random_train",
                         "X_real_val", "X_shuffle_val", "X_random_val"]):
        nan_columns = df.columns[df.isna().any()].tolist()
        if nan_columns:
            print(f"Columns with NaNs in {name}: {nan_columns}")
        else:
            print(f"No NaN columns in {name}")

    
    # Set complex_filename as index for the feature DataFrames
    X_real_train = X_real_train_df.set_index("complex_filename")
    X_shuffle_train = X_shuffle_train_df.set_index("complex_filename")
    X_random_train = X_random_train_df.set_index("complex_filename")
    
    X_real_val = X_real_val_df.set_index("complex_filename")
    X_shuffle_val = X_shuffle_val_df.set_index("complex_filename")
    X_random_val = X_random_val_df.set_index("complex_filename")
    
    # Create separate DataFrames for mapping complex_filename to pKd for later use
    y_real_train = y_real_train_df.copy()
    y_real_val = y_real_val_df.copy()

    # Combine real and fake data for classification - training set
    X_fake_train = pd.concat([X_shuffle_train, X_random_train])
    
    # Combine real and fake data for classification - validation set
    X_fake_val = pd.concat([X_shuffle_val, X_random_val])

    print(f"X_fake_train shape: {X_fake_train.shape}")
    print(f"X_real_train shape: {X_real_train.shape}")
    print(f"X_fake_val shape: {X_fake_val.shape}")
    print(f"X_real_val shape: {X_real_val.shape}")

    # Create copies to avoid modifying the original DataFrames
    X_fake_train = X_fake_train.copy()
    X_real_train = X_real_train.copy()
    X_fake_val = X_fake_val.copy()
    X_real_val = X_real_val.copy()
    
    # Add binary labels: 0 for fake (decoys), 1 for real
    X_fake_train['label'] = 0
    X_real_train['label'] = 1
    X_fake_val['label'] = 0
    X_real_val['label'] = 1

    # Combine all data for training and validation
    X_train = pd.concat([X_fake_train, X_real_train], axis=0)
    y_train = X_train.pop('label').values
    
    X_val = pd.concat([X_fake_val, X_real_val], axis=0)
    y_val = X_val.pop('label').values

    # Save scaling parameters for later use
    scaling_params = X_train.describe().T[["mean", "std", "min", "max"]]
    scaling_params.to_csv(
        os.path.join(output_dir, "scaling_params.csv"), index=True)

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # Train models
    logreg_results = train_logistic_regression(X_train, y_train, X_val, y_val)
    rf_results = train_random_forest_classifier(X_train, y_train, X_val, y_val)
    svc_results = train_svm_classifier(X_train, y_train, X_val, y_val)
    symb_results = perform_symbolic_classification(
        X_train,
        y_train,
        X_val,
        y_val,
        niterations=100,
        populations=50,
        population_size=20,
        model_selection="best",
        select_k_features=25,
        scale_features=True
    )

    # Save training predictions
    train_preds_df = pd.DataFrame(
        {
            "complex_filename": X_train.index,
            "y_train": y_train,
            "logreg_pred": logreg_results["train_pred"],
            "rf_pred": rf_results["train_pred"],
            "svc_pred": svc_results["train_pred"],
            "symbolic_pred": symb_results["train_pred"],
        }
    )
    train_preds_df.to_csv(
        os.path.join(output_dir, "train_predictions.csv"), index=False
    )

    # Save validation predictions
    val_preds_df = pd.DataFrame(
        {
            "complex_filename": X_val.index,
            "y_val": y_val,
            "logreg_pred": logreg_results["val_pred"],
            "rf_pred": rf_results["val_pred"],
            "svc_pred": svc_results["val_pred"],
            "symbolic_pred": symb_results["val_pred"],
        }
    )
    val_preds_df.to_csv(
        os.path.join(output_dir, "val_predictions.csv"), index=False
    )

    # Create probability predictions dataframe for ROC analysis
    train_proba_df = pd.DataFrame(
        {
            "complex_filename": X_train.index,
            "y_train": y_train,
            "logreg_proba": logreg_results["train_proba"],
            "rf_proba": rf_results["train_proba"],
            "svc_proba": svc_results["train_proba"],
            "symbolic_proba": symb_results["train_proba"],
        }
    )
    train_proba_df.to_csv(
        os.path.join(output_dir, "train_probabilities.csv"), index=False
    )

    val_proba_df = pd.DataFrame(
        {
            "complex_filename": X_val.index,
            "y_val": y_val,
            "logreg_proba": logreg_results["val_proba"],
            "rf_proba": rf_results["val_proba"],
            "svc_proba": svc_results["val_proba"],
            "symbolic_proba": symb_results["val_proba"],
        }
    )
    val_proba_df.to_csv(
        os.path.join(output_dir, "val_probabilities.csv"), index=False
    )

    # Plot metrics for each model
    plot_classification_metrics(
        y_val,
        logreg_results["val_pred"],
        logreg_results["val_proba"],
        y_train,
        logreg_results["train_proba"],
        model_name="Logistic Regression",
        output_path=os.path.join(output_dir, "logreg_metrics.svg"),
    )

    print("Plotting classification metrics")
    plot_classification_metrics(
        y_val,
        rf_results["val_pred"],
        rf_results["val_proba"],
        y_train,
        rf_results["train_proba"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_metrics.svg"),
    )

    plot_classification_metrics(
        y_val,
        svc_results["val_pred"],
        svc_results["val_proba"],
        y_train,
        svc_results["train_proba"],
        model_name="SVC",
        output_path=os.path.join(output_dir, "svc_metrics.svg"),
    )

    plot_classification_metrics(
        y_val, 
        symb_results["val_pred"], 
        symb_results["val_proba"],
        y_train,
        symb_results["train_proba"],
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_metrics.svg")
    )

    print("Plotting probability histograms")
    plot_probability_histograms(
        y_val,
        logreg_results["val_proba"],
        model_name="Logistic Regression",
        output_path=os.path.join(output_dir, "logreg_proba_hist.svg"),
    )

    plot_probability_histograms(
        y_val,
        rf_results["val_proba"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_proba_hist.svg"),
    )

    plot_probability_histograms(
        y_val,
        svc_results["val_proba"],
        model_name="SVC",
        output_path=os.path.join(output_dir, "svc_proba_hist.svg"),
    )

    plot_probability_histograms(
        y_val,
        symb_results["val_proba"],
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_proba_hist.svg"),
    )

    print("Plotting feature importances")
    plot_feature_importances(
        logreg_results["coefficients"],
        output_path=os.path.join(output_dir, "logreg_coef.svg"),
    )

    plot_feature_importances(
        rf_results["feature_importance"],
        output_path=os.path.join(output_dir, "rf_importance.svg"),
    )

    rf_results["feature_importance"].to_csv(os.path.join(output_dir, "rf_feature_importance.csv"), index=False)

    # Save top symbolic equations to CSV
    if 'top_equations' in symb_results:
        symb_results['top_equations'].to_csv(
            os.path.join(output_dir, "symbolic_classification_equations.csv"), index=False
        )
        print(f"Saved top symbolic classification equations to {output_dir}/symbolic_classification_equations.csv")
    
    # Save all symbolic equations with metrics
    if 'all_equations' in symb_results:
        symb_results['all_equations'].to_csv(
            os.path.join(output_dir, "symbolic_classification_all_equations.csv"), index=False
        )
        print(f"Saved all symbolic classification equations to {output_dir}/symbolic_classification_all_equations.csv")
        
        # Plot complexity vs accuracy tradeoff
        plot_symbolic_complexity_tradeoff(
            symb_results['all_equations'],
            metric='val_auc', 
            model_type='classification',
            output_path=os.path.join(output_dir, "symbolic_classification_complexity_tradeoff.svg"),
            lower_is_better=False
        )

    # Create comparison dataframe
    results_dict = {
        "LogReg": logreg_results,
        "RF": rf_results,
        "SVC": svc_results,
        "Symbolic": symb_results,
    }
    
    # Calculate ROC curves for each model
    roc_data = {}
    for model_name, result in results_dict.items():
        fpr, tpr, _ = roc_curve(y_val, result["val_proba"])
        roc_auc = result.get("val_auc", roc_auc_score(y_val, result["val_proba"]))
        roc_data[model_name] = (fpr, tpr, roc_auc)

    comparison_df = create_model_comparison(results_dict)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    print("Plotting model comparison")
    plot_model_comparison(
        comparison_df, 
        roc_data=roc_data,
        output_path=os.path.join(output_dir, "model_comparison.svg")
    )

    # Extract real val samples and correlate pKd with classifier probabilities
    real_val_indices = val_preds_df["y_val"] == 1
    real_val_complexes = val_preds_df.loc[real_val_indices, "complex_filename"].values
    
    # Get pKd values for real val samples
    real_val_pkd = pd.DataFrame({
        "complex_filename": real_val_complexes,
    })
    
    # Extract real training samples and their complexes
    real_train_indices = train_preds_df["y_train"] == 1
    real_train_complexes = train_preds_df.loc[real_train_indices, "complex_filename"].values
    
    # Get pKd values for real train samples
    real_train_pkd = pd.DataFrame({
        "complex_filename": real_train_complexes,
    })
    
    # Map complex filenames to pKd values using pandas merge
    real_val_pkd = real_val_pkd.merge(y_real_val, on="complex_filename", how="left")
    real_train_pkd = real_train_pkd.merge(y_real_train, on="complex_filename", how="left")
    
    # Save the real val and train pKd values for future use
    real_val_pkd.to_csv(os.path.join(output_dir, "real_val_pkd.csv"), index=False)
    real_train_pkd.to_csv(os.path.join(output_dir, "real_train_pkd.csv"), index=False)
    
    # Create dictionaries of model probabilities for real val and train samples
    val_probabilities = {}
    train_probabilities = {}
    model_names = []
    
    for model in ["logreg", "rf", "svc", "symbolic"]:
        if f"{model}_proba" in val_proba_df.columns:
            model_upper = model.upper()
            model_names.append(model_upper)
            
            # Extract val probabilities for real samples
            val_probabilities[model_upper] = val_proba_df.loc[real_val_indices, f"{model}_proba"].values
            
            # Extract train probabilities for real samples
            train_probabilities[model_upper] = train_proba_df.loc[real_train_indices, f"{model}_proba"].values
    
    print("Plotting pKd probability correlation")
    # Plot correlation between pKd and model probabilities (both val and train)
    if not real_val_pkd.empty and "pKd" in real_val_pkd.columns:
        plot_pkd_probability_correlation(
            real_val_pkd["pKd"].values,
            val_probabilities,
            model_names,
            output_path=os.path.join(output_dir, "pkd_probability_correlation.svg"),
        )
        print(f"Generated pKd-probability correlation plot at {output_dir}/pkd_probability_correlation.svg")

    print("Classification pipeline complete!")