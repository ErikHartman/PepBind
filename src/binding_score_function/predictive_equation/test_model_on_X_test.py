import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import sympy
import logging
import seaborn as sns

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

sns.set_context("paper")

def load_test_data(base_path):
    """Load X_test and y_test data."""
    scores_path = os.path.join(base_path, "outputs/binding_score_function/4_processed_scores/")
    
    logger.info(f"Loading test data from {scores_path}")
    #random_X_test = pd.read_csv(os.path.join(scores_path, "random_X_test.csv"))
    #shuffle_X_test = pd.read_csv(os.path.join(scores_path, "shuffle_X_test.csv"))
    X_test = pd.read_csv(os.path.join(scores_path, "real_X_test.csv"))
    complex_filenames = X_test["complex_filename"].copy()
    X_test = X_test.set_index("complex_filename")
    
    y_test = pd.read_csv(os.path.join(scores_path, "real_y_test.csv"))
    
    if "complex_filename" in y_test.columns:
        y_test = y_test.set_index("complex_filename")["pKd"].values
    else:
        y_test = y_test["pKd"].values
    
    logger.info(f"Loaded X_test with shape {X_test.shape} and y_test with shape {y_test.shape}")
    
    return X_test, y_test, complex_filenames


def get_scaling_parameters(output_dir):
    """Load scaling parameters from the saved file."""
    scaling_path = os.path.join(output_dir, "scaling_params.csv")
    
    if not os.path.exists(scaling_path):
        logger.warning(f"Scaling parameters not found at {scaling_path}")
        return None
    
    scaling_params = pd.read_csv(scaling_path, index_col=0)
    logger.info(f"Loaded scaling parameters for {len(scaling_params)} features")
    
    return scaling_params


def scale_features_with_params(X, scaling_params):
    """Scale features using pre-computed mean and std values."""
    X_scaled = X.copy()
    
    for col in X.columns:
        if col in scaling_params.index:
            mean = scaling_params.loc[col, "mean"]
            std = scaling_params.loc[col, "std"]
            X_scaled[col] = (X[col] - mean) / std
        else:
            logger.warning(f"Feature {col} not found in scaling parameters")
    
    return X_scaled


def load_best_symbolic_equation(output_dir):
    """Load the best symbolic regression equation from the saved file."""
    equations_path = os.path.join(output_dir, "symbolic_regression_equations.csv")
    all_equations_path = os.path.join(output_dir, "symbolic_regression_all_equations.csv")
    
    if os.path.exists(all_equations_path):
        logger.info(f"Loading all symbolic equations from {all_equations_path}")
        equations = pd.read_csv(all_equations_path)
        # Sort by validation RMSE (lower is better)
        valid_eqs = equations[equations["val_r2"].notna()].sort_values(by="val_r2")
        
        if len(valid_eqs) > 0:
            best_eq = valid_eqs.iloc[0]
            logger.info(f"Best equation (by val_r2): {best_eq['equation']}")
            return best_eq["equation"], equations
    
    if os.path.exists(equations_path):
        logger.info(f"Loading top symbolic equations from {equations_path}")
        equations = pd.read_csv(equations_path)
        # Use the first equation (typically sorted by score)
        best_eq = equations.iloc[0]
        logger.info(f"Best equation (first in list): {best_eq['equation']}")
        return best_eq["equation"], equations
    
    logger.error("No symbolic regression equations found")
    return None, None


def evaluate_equation(equation_str, X, y, is_scaled=True):
    """Evaluate a symbolic equation on X and compare with y."""
    # Parse equation with sympy
    expr = sympy.sympify(equation_str)

    print(expr)
    
    # Get feature names from the equation
    features = [str(symbol) for symbol in expr.free_symbols]
    logger.info(f"Equation uses {len(features)} features: {features}")
    
    # Make sure all needed features are in X
    for feature in features:
        if feature not in X.columns:
            logger.error(f"Feature {feature} not found in X_test")
            return None
    
    # Create lambda function from sympy expression
    func = sympy.lambdify(features, expr)
    
    # Prepare inputs for function
    inputs = [X[feature].values for feature in features]
    
    # Evaluate function
    try:
        y_pred = func(*inputs)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        logger.info(f"Test metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
        
        return {
            'y_pred': y_pred,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    except Exception as e:
        logger.error(f"Error evaluating equation: {e}")
        return None


def evaluate_equation_ensemble(equations, X, y, top_n=5, weights=None):
    """
    Evaluate an ensemble of symbolic equations on X and combine predictions.
    """
    # Keep only equations with valid validation metrics
    valid_eqs = equations[equations["val_r2"].notna()].sort_values(by="val_r2")
    
    if len(valid_eqs) == 0:
        logger.error("No valid equations found for ensemble")
        return None
    
    # Select top N equations
    top_eqs = valid_eqs.head(min(top_n, len(valid_eqs)))
    logger.info(f"Creating ensemble with {len(top_eqs)} equations")
    
    # Store individual predictions
    all_predictions = []
    failed_equations = 0
    
    # Get predictions from each equation
    for i, row in top_eqs.iterrows():
        equation_str = row["equation"]
        try:
            # Parse equation with sympy
            expr = sympy.sympify(equation_str)
            features = [str(symbol) for symbol in expr.free_symbols]
            
            # Check features
            for feature in features:
                if feature not in X.columns:
                    logger.warning(f"Feature {feature} not found in X_test, skipping equation")
                    failed_equations += 1
                    continue
            
            # Create lambda function
            func = sympy.lambdify(features, expr)
            
            # Prepare inputs
            inputs = [X[feature].values for feature in features]
            
            # Get predictions
            y_pred = func(*inputs)
            all_predictions.append(y_pred)
            
            logger.info(f"Added equation to ensemble: {equation_str[:60]}{'...' if len(equation_str) > 60 else ''}")
        
        except Exception as e:
            logger.warning(f"Error evaluating equation for ensemble: {e}")
            failed_equations += 1
    
    if not all_predictions:
        logger.error("No valid predictions for ensemble")
        return None
    
    if failed_equations > 0:
        logger.warning(f"{failed_equations} equations failed and were excluded from ensemble")
    
    # Calculate weights if not provided
    if weights is None and len(all_predictions) > 1:
        weights = top_eqs["val_r2"].values
        weights = weights / np.sum(weights)  # Normalize to sum to 1
    elif weights is None:
        weights = [1.0]
    
    # Combine predictions
    all_predictions = np.array(all_predictions)
    ensemble_pred = np.average(all_predictions, axis=0, weights=weights)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y, ensemble_pred))
    mae = mean_absolute_error(y, ensemble_pred)
    r2 = r2_score(y, ensemble_pred)
    
    logger.info(f"Ensemble test metrics - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    return {
        'y_pred': ensemble_pred,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'individual_predictions': all_predictions,
        'weights': weights,
        'equations': top_eqs["equation"].values
    }


def plot_results(y_true, y_pred, output_path=None):
    """Create a scatter plot of actual vs. predicted values."""
    plt.figure(figsize=(4,4))
    
    # Scatter plot
    sns.regplot(x=y_true, y=y_pred)
    
    # Perfect prediction line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8)
    
    plt.xlabel('Actual pKd')
    plt.ylabel('Predicted pKd')
    plt.title('Symbolic Regression: Actual vs. Predicted on Test Set')
    
    # Add metrics as text
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.text(
        0.05, 0.95, 
        f'RMSE: {rmse:.3f}\nR²: {r2:.3f}\nMAE: {mae:.3f}', 
        transform=plt.gca().transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )
    
    plt.grid(alpha=0.3)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300)
        logger.info(f"Saved plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def compare_with_validation_results(output_dir, test_metrics):
    """Compare test results with validation results from model comparison."""
    model_comparison_path = os.path.join(output_dir, "model_comparison.csv")
    
    if not os.path.exists(model_comparison_path):
        logger.warning(f"Model comparison file not found at {model_comparison_path}")
        return
    
    model_comparison = pd.read_csv(model_comparison_path)
    symbolic_row = model_comparison[model_comparison["Model"] == "Symbolic"].iloc[0]
    
    validation_metrics = {
        "RMSE": symbolic_row["Val RMSE"],
        "R²": symbolic_row["Val R²"],
        "MAE": symbolic_row["Val MAE"]
    }
    
    test_metrics_dict = {
        "RMSE": test_metrics["rmse"],
        "R²": test_metrics["r2"],
        "MAE": test_metrics["mae"]
    }
    
    # Create comparison dataframe
    comparison = pd.DataFrame({
        "Metric": list(validation_metrics.keys()),
        "Validation": list(validation_metrics.values()),
        "Test": [test_metrics_dict[m] for m in validation_metrics.keys()]
    })
    
    logger.info("Comparison with validation results:")
    logger.info("\n" + comparison.to_string())
    
    return comparison


def save_predictions(complex_filenames, y_true, y_pred, output_path):
    """Save predictions with complex filenames to a CSV file."""
    predictions_df = pd.DataFrame({
        "complex_filename": complex_filenames,
        "y_true": y_true,
        "y_pred": y_pred
    })
    
    predictions_df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def main():
    """Main function to load data, evaluate the best equation, and visualize results."""
    # Configuration
    base_path = "/srv/data1/general/immunopeptides_data/"
    output_dir = "./plots/test"
    regression_dir = "./plots/regression"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load test data
    X_test, y_test, complex_filenames = load_test_data(base_path)
    
    # Load best symbolic equation
    best_equation, all_equations = load_best_symbolic_equation(regression_dir)
    
    if best_equation is None:
        logger.error("Could not find the best symbolic equation, exiting.")
        sys.exit(1)
    
    # Load and apply scaling if needed
    scaling_params = get_scaling_parameters(regression_dir)
    if scaling_params is not None:
        X_test_scaled = scale_features_with_params(X_test, scaling_params)
        logger.info("Applied feature scaling to X_test")
    else:
        X_test_scaled = X_test
    
    # Evaluate individual best equation on test data
    single_results = evaluate_equation(best_equation, X_test_scaled, y_test)
    
    if single_results is None:
        logger.error("Error evaluating single equation on test data, exiting.")
        sys.exit(1)
    
    # Plot single model results
    test_plot_path = os.path.join(output_dir, "symbolic_regression_test_predictions.png")
    plot_results(y_test, single_results["y_pred"], test_plot_path)
    
    # Compare with validation results
    comparison = compare_with_validation_results(output_dir, single_results)
    if comparison is not None:
        comparison.to_csv(os.path.join(output_dir, "test_vs_validation.csv"), index=False)
    
    # Save predictions for single model
    save_predictions(
        complex_filenames, y_test, single_results["y_pred"],
        os.path.join(output_dir, "test_predictions.csv")
    )
    
    # If we have multiple equations, try an ensemble approach
    if all_equations is not None and len(all_equations) > 1:
        logger.info("Creating ensemble model from top equations")
        # Try with top 5 equations
        ensemble_results = evaluate_equation_ensemble(all_equations, X_test_scaled, y_test, top_n=5)
        
        if ensemble_results is not None:
            # Plot ensemble results
            ensemble_plot_path = os.path.join(output_dir, "symbolic_regression_ensemble_test_predictions.png")
            plot_results(y_test, ensemble_results["y_pred"], ensemble_plot_path)
            
            # Save ensemble predictions
            save_predictions(
                complex_filenames, y_test, ensemble_results["y_pred"],
                os.path.join(output_dir, "ensemble_test_predictions.csv")
            )
            
            # Compare single model vs ensemble
            comparison_df = pd.DataFrame({
                "Metric": ["RMSE", "MAE", "R²"],
                "Single Model": [
                    single_results["rmse"], 
                    single_results["mae"], 
                    single_results["r2"]
                ],
                "Ensemble": [
                    ensemble_results["rmse"], 
                    ensemble_results["mae"], 
                    ensemble_results["r2"]
                ]
            })
            
            logger.info("Comparison between single model and ensemble:")
            logger.info("\n" + comparison_df.to_string())
            comparison_df.to_csv(os.path.join(output_dir, "single_vs_ensemble.csv"), index=False)
    
    logger.info("Evaluation on test set complete!")


if __name__ == "__main__":
    main()