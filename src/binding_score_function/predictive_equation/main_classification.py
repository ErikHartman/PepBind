import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score

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
)

if __name__ == "__main__":
    # Example data
    base_path = "/srv/data1/general/immunopeptides_data/"
    scores_path = os.path.join(base_path, "outputs/binding_score_function/4_processed_scores/")
    output_dir = "./plots/classification"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    X_real = pd.read_csv(os.path.join(scores_path, "real_X_train.csv")).set_index("complex_filename")
    X_shuffled = pd.read_csv(os.path.join(scores_path, "shuffled_X_train.csv")).set_index("complex_filename")
    X_random = pd.read_csv(os.path.join(scores_path, "random_X_train.csv")).set_index("complex_filename")

    X_fake = pd.concat([X_shuffled, X_random])

    print(f"X_fake shape: {X_fake.shape}")
    print(f"X_real shape: {X_real.shape}")  

    X_fake = X_fake.copy()
    X_real = X_real.copy()
    X_fake['label'] = 0
    X_real['label'] = 1

    X_all = pd.concat([X_fake, X_real], axis=0)
    y_all = X_all.pop('label').values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    logreg_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_results = train_random_forest_classifier(X_train, y_train, X_test, y_test)
    svc_results = train_svm_classifier(X_train, y_train, X_test, y_test)
    symb_results = perform_symbolic_classification(
        X_train,
        y_train,
        X_test,
        y_test,
        niterations=500,
        populations=50,
        population_size=100,
        model_selection="accuracy",
        select_k_features=15,
        scale_features=True
    )

    train_preds_df = pd.DataFrame(
        {
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

    test_preds_df = pd.DataFrame(
        {
            "y_test": y_test,
            "logreg_pred": logreg_results["test_pred"],
            "rf_pred": rf_results["test_pred"],
            "svc_pred": svc_results["test_pred"],
            "symbolic_pred": symb_results["test_pred"],
        }
    )
    test_preds_df.to_csv(
        os.path.join(output_dir, "test_predictions.csv"), index=False
    )

    # Create probability predictions dataframe for ROC analysis
    train_proba_df = pd.DataFrame(
        {
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

    test_proba_df = pd.DataFrame(
        {
            "y_test": y_test,
            "logreg_proba": logreg_results["test_proba"],
            "rf_proba": rf_results["test_proba"],
            "svc_proba": svc_results["test_proba"],
            "symbolic_proba": symb_results["test_proba"],
        }
    )
    test_proba_df.to_csv(
        os.path.join(output_dir, "test_probabilities.csv"), index=False
    )

    # Plot metrics for each model
    plot_classification_metrics(
        y_test,
        logreg_results["test_pred"],
        logreg_results["test_proba"],
        y_train,
        logreg_results["train_proba"],
        model_name="Logistic Regression",
        output_path=os.path.join(output_dir, "logreg_metrics.png"),
    )

    plot_classification_metrics(
        y_test,
        rf_results["test_pred"],
        rf_results["test_proba"],
        y_train,
        rf_results["train_proba"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_metrics.png"),
    )

    plot_classification_metrics(
        y_test,
        svc_results["test_pred"],
        svc_results["test_proba"],
        y_train,
        svc_results["train_proba"],
        model_name="SVC",
        output_path=os.path.join(output_dir, "svc_metrics.png"),
    )

    plot_classification_metrics(
        y_test, 
        symb_results["test_pred"], 
        symb_results["test_proba"],
        y_train,
        symb_results["train_proba"],
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_metrics.png")
    )

    # Plot probability histograms for each model
    plot_probability_histograms(
        y_test,
        logreg_results["test_proba"],
        model_name="Logistic Regression",
        output_path=os.path.join(output_dir, "logreg_proba_hist.png"),
    )

    plot_probability_histograms(
        y_test,
        rf_results["test_proba"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_proba_hist.png"),
    )

    plot_probability_histograms(
        y_test,
        svc_results["test_proba"],
        model_name="SVC",
        output_path=os.path.join(output_dir, "svc_proba_hist.png"),
    )

    plot_probability_histograms(
        y_test,
        symb_results["test_proba"],
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_proba_hist.png"),
    )

    plot_feature_importances(
        logreg_results["coefficients"],
        model_name="LogReg",
        output_path=os.path.join(output_dir, "logreg_coef.png"),
    )

    plot_feature_importances(
        rf_results["feature_importance"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_importance.png"),
    )

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
            metric='test_auc', 
            model_type='classification',
            output_path=os.path.join(output_dir, "symbolic_classification_complexity_tradeoff.png"),
            lower_is_better=False
        )

    # Create comparison dataframe
    results_dict = {
        "LogReg": logreg_results,
        "RF": rf_results,
        "SVC": svc_results,
        "Symbolic": symb_results,
    }
    comparison = {
        "Model": [],
        "Train Accuracy": [],
        "Train F1": [],
        "Train AUC": [],
        "Test Accuracy": [],
        "Test F1": [],
        "Test AUC": [],
    }

    # Calculate ROC curves for each model
    roc_data = {}
    for model_name, result in results_dict.items():
        comparison["Model"].append(model_name)
        comparison["Train Accuracy"].append(result.get("train_acc", np.nan))
        comparison["Train F1"].append(result.get("train_f1", np.nan))
        comparison["Train AUC"].append(result.get("train_auc", np.nan))
        comparison["Test Accuracy"].append(result.get("test_acc", result.get("test_accuracy", np.nan)))
        comparison["Test F1"].append(result.get("test_f1", np.nan))
        comparison["Test AUC"].append(result.get("test_auc", np.nan))
        
        # Calculate ROC curve data for each model
        if "test_proba" in result and result["test_proba"] is not None:
            fpr, tpr, _ = roc_curve(y_test, result["test_proba"])
            roc_auc = result.get("test_auc", roc_auc_score(y_test, result["test_proba"]))
            roc_data[model_name] = (fpr, tpr, roc_auc)

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    # Plot model comparison with ROC curves
    plot_model_comparison(
        comparison_df, 
        roc_data=roc_data,
        output_path=os.path.join(output_dir, "model_comparison.png")
    )

    print("Classification pipeline complete!")