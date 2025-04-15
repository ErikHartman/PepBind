import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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

    X_fake = X_fake.copy()
    X_real = X_real.copy()
    X_fake['label'] = 0
    X_real['label'] = 1

    X_all = pd.concat([X_fake, X_real], axis=0)
    y_all = X_all.pop('label').values

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.2, random_state=42, stratify=y_all
    )

    logreg_results = train_logistic_regression(X_train, y_train, X_test, y_test)
    rf_results = train_random_forest_classifier(X_train, y_train, X_test, y_test)
    svc_results = train_svm_classifier(X_train, y_train, X_test, y_test)
    symb_results = perform_symbolic_classification(
        X_train,
        y_train,
        X_test,
        y_test,
        niterations=200,
        populations=50,
        population_size=100,
        model_selection="best",
        select_k_features=15,
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
        model_name="Logistic Regression",
        output_path=os.path.join(output_dir, "logreg_metrics.png"),
    )

    plot_classification_metrics(
        y_test,
        rf_results["test_pred"],
        rf_results["test_proba"],
        model_name="Random Forest",
        output_path=os.path.join(output_dir, "rf_metrics.png"),
    )

    plot_classification_metrics(
        y_test,
        svc_results["test_pred"],
        svc_results["test_proba"],
        model_name="SVC",
        output_path=os.path.join(output_dir, "svc_metrics.png"),
    )

    plot_classification_metrics(y_test, symb_results["test_pred"], symb_results["test_proba"],
        model_name="Symbolic",
        output_path=os.path.join(output_dir, "symbolic_metrics.png"))

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

    for model_name, result in results_dict.items():
        comparison["Model"].append(model_name)
        comparison["Train Accuracy"].append(result.get("train_acc", np.nan))
        comparison["Train F1"].append(result.get("train_f1", np.nan))
        comparison["Train AUC"].append(result.get("train_auc", np.nan))
        comparison["Test Accuracy"].append(result.get("test_acc", result.get("test_accuracy", np.nan)))
        comparison["Test F1"].append(result.get("test_f1", np.nan))
        comparison["Test AUC"].append(result.get("test_auc", np.nan))

    comparison_df = pd.DataFrame(comparison)
    comparison_df.to_csv(os.path.join(output_dir, "model_comparison.csv"), index=False)

    plot_model_comparison(
        comparison_df, output_path=os.path.join(output_dir, "model_comparison.png")
    )

    print("Classification pipeline complete!")