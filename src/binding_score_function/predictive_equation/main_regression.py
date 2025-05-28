import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from regression import (
    train_lasso,
    train_random_forest as train_rf,
    train_svr,
    perform_symbolic_regression as train_symbolic,
)

from plotting import (
    plot_regression_scatter,
    plot_feature_importances,
    plot_model_comparison,
    plot_symbolic_complexity_tradeoff,
    plot_predictions_by_data_type,
)

if __name__ == "__main__":
    # Data paths setup
    base_path = "/srv/data1/general/immunopeptides_data/"
    scores_path = os.path.join(
        base_path, "outputs/binding_score_function_prod/4_processed_scores/"
    )
    output_dir = "./plots/regression"
    os.makedirs(output_dir, exist_ok=True)

    # Load pre-split training and validation data files
    X_train_df = pd.read_csv(os.path.join(scores_path, "real_X_train.csv"))
    y_train_df = pd.read_csv(os.path.join(scores_path, "real_y_train.csv"))
    
    X_val_df = pd.read_csv(os.path.join(scores_path, "real_X_val.csv"))
    y_val_df = pd.read_csv(os.path.join(scores_path, "real_y_val.csv"))
    
    # Remove any 'Unnamed:_0' columns that might have been created during saving/loading
    for df in [X_train_df, X_val_df]:
        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
    
    # Set complex_filename as index for X DataFrames
    X_train = X_train_df.set_index("complex_filename")
    X_val = X_val_df.set_index("complex_filename")
    
    # Extract pKd values as arrays for model training
    y_train = y_train_df["pKd"].values
    y_val = y_val_df["pKd"].values

    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")

    # Save scaling parameters for later use
    scaling_params = X_train.describe().T[["mean", "std"]]
    scaling_params.to_csv(
        os.path.join(output_dir, "scaling_params.csv"), index=True)

    # Train models
    lasso_results = train_lasso(X_train, y_train, X_val, y_val)
    rf_results = train_rf(X_train, y_train, X_val, y_val)
    svr_results = train_svr(X_train, y_train, X_val, y_val)
    symb_results = train_symbolic(
        X_train,
        y_train,
        X_val,
        y_val,
        niterations=100,
        populations=50,
        population_size=50,
        model_selection="best",
        select_k_features=15,
        scale_features=True,
    )

    # Save predictions with complex filenames
    train_preds_df = pd.DataFrame(
        {
            "complex_filename": X_train.index,
            "y_train": y_train,
            "lasso_pred": lasso_results["train_pred"],
            "rf_pred": rf_results["train_pred"],
            "svr_pred": svr_results["train_pred"],
            "symbolic_pred": symb_results["train_pred"],
        }
    )
    train_preds_df.to_csv(
        os.path.join(output_dir, "train_predictions.csv"), index=False
    )

    val_preds_df = pd.DataFrame(
        {
            "complex_filename": X_val.index,
            "y_val": y_val,
            "lasso_pred": lasso_results["val_pred"],
            "rf_pred": rf_results["val_pred"],
            "svr_pred": svr_results["val_pred"],
            "symbolic_pred": symb_results["val_pred"],
        }
    )
    val_preds_df.to_csv(
        os.path.join(output_dir, "val_predictions.csv"), index=False
    )

    plot_regression_scatter(
        y_train,
        lasso_results["train_pred"],
        y_val,
        lasso_results.get("val_pred", None),
        model_name="Lasso",
        output_path=os.path.join(output_dir, "lasso_scatter.png"),
    )

    plot_regression_scatter(
        y_train,
        rf_results["train_pred"],
        y_val,
        rf_results.get("val_pred", None),
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_scatter.png"),
    )

    plot_regression_scatter(
        y_train,
        svr_results["train_pred"],
        y_val,
        svr_results.get("val_pred", None),
        model_name="SVR",
        output_path=os.path.join(output_dir, "svr_scatter.png"),
    )

    plot_regression_scatter(
        y_train,
        symb_results["train_pred"],
        y_val,
        symb_results.get("val_pred", None),
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_scatter.png"),
    )

    plot_feature_importances(
        lasso_results["coefficients"],
        model_name="Lasso",
        output_path=os.path.join(output_dir, "lasso_coef.png"),
    )

    plot_feature_importances(
        rf_results["feature_importance"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_importance.png"),
    )

    # Save top symbolic equations to CSV
    if 'top_equations' in symb_results:
        symb_results['top_equations'].to_csv(
            os.path.join(output_dir, "symbolic_regression_equations.csv"), index=False
        )
        print(f"Saved top symbolic regression equations to {output_dir}/symbolic_regression_equations.csv")
        
    # Save all symbolic equations with metrics
    if 'all_equations' in symb_results:
        symb_results['all_equations'].to_csv(
            os.path.join(output_dir, "symbolic_regression_all_equations.csv"), index=False
        )
        print(f"Saved all symbolic regression equations to {output_dir}/symbolic_regression_all_equations.csv")
        
        # Plot complexity vs accuracy tradeoff
        plot_symbolic_complexity_tradeoff(
            symb_results['all_equations'],
            metric='val_r2',
            model_type='regression',
            output_path=os.path.join(output_dir, "symbolic_regression_complexity_tradeoff.png"),
            lower_is_better=False
        )

    results_dict = {
        "Lasso": lasso_results,
        "RF": rf_results,
        "SVR": svr_results,
        "Symbolic": symb_results,
    }
    comparison = {
        "Model": [],
        "Train RMSE": [],
        "Train R²": [],
        "Val RMSE": [],
        "Val R²": [],
        "Val MAE": [],
    }

    for model_name, result in results_dict.items():
        comparison["Model"].append(model_name)
        comparison["Train RMSE"].append(result.get("train_rmse", np.nan))
        comparison["Train R²"].append(result.get("train_r2", np.nan))
        comparison["Val RMSE"].append(result.get("val_rmse", np.nan))
        comparison["Val R²"].append(result.get("val_r2", np.nan))
        comparison["Val MAE"].append(result.get("val_mae", np.nan))

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    plot_model_comparison(
        comparison_df, output_path=os.path.join(output_dir, "model_comparison.png")
    )

    print("\nPerforming analysis on shuffle and random data...")
    
    # Load shuffle and random data with proper index handling
    X_shuffle_df = pd.read_csv(os.path.join(scores_path, "shuffle_X_val.csv"))
    X_random_df = pd.read_csv(os.path.join(scores_path, "random_X_val.csv"))
    
    # Remove any 'Unnamed:_0' columns that might have been created during saving/loading
    for df in [X_shuffle_df, X_random_df]:
        columns_to_drop = [col for col in df.columns if col.startswith('Unnamed:')]
        if columns_to_drop:
            df.drop(columns=columns_to_drop, inplace=True)
    
    # Set complex_filename as index for consistency
    X_shuffle = X_shuffle_df.set_index("complex_filename")
    X_random = X_random_df.set_index("complex_filename")
    
    print(f"Loaded shuffle data: {X_shuffle.shape} samples")
    print(f"Loaded random data: {X_random.shape} samples")
    
    # Create a dictionary to store predictions
    model_predictions = {
        "data_type": [],
        "model": [],
        "prediction": []
    }
    
    # Get predictions for each model on real, shuffle, and random data
    data_types = {
        "Real": X_val,  # Use the validation data for real samples
        "Shuffled": X_shuffle,
        "Random": X_random
    }
    
    models = {
        "LASSO": (lasso_results["model"], lasso_results["scaler"]),
        "RF": (rf_results["model"], None),  # RF doesn't use a scaler
        "SVR": (svr_results["model"], svr_results["scaler"]),
        "Symbolic": (symb_results["model"], symb_results.get("scaler", None))
    }
    
    for data_name, X_data in data_types.items():
        for model_name, (model, scaler) in models.items():
            # For symbolic regression, we need special handling
            if model_name == "Symbolic":
                # Get the equation index to use
                all_eqs = symb_results["all_equations"]
                best_eq_idx = all_eqs.loc[all_eqs["val_rmse"].idxmin(), "equation_index"]
                
                # Scale data if needed
                if scaler is not None:
                    X_data_scaled = scaler.transform(X_data)
                else:
                    X_data_scaled = X_data
                
                # Get predictions using the best equation
                X_data_array = X_data_scaled if isinstance(X_data_scaled, np.ndarray) else X_data_scaled.values
                predictions = model.predict(X_data_array, index=best_eq_idx)
            else:
                # For regular models
                if scaler is not None:
                    X_data_scaled = scaler.transform(X_data)
                    predictions = model.predict(X_data_scaled)
                else:
                    predictions = model.predict(X_data)
            
            # Store all predictions
            for pred in predictions:
                model_predictions["data_type"].append(data_name)
                model_predictions["model"].append(model_name)
                model_predictions["prediction"].append(pred)
    
    # Convert to DataFrame
    predictions_df = pd.DataFrame(model_predictions)
    predictions_df.to_csv(os.path.join(output_dir, "shuffle_random_predictions.csv"), index=False)
    plot_paths = plot_predictions_by_data_type(predictions_df, output_dir)
    
    print("\nAnalysis on shuffle and random data complete!")
    print("Regression pipeline complete!")
