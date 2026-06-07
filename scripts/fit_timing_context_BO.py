import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

GLOBAL_OUT_PATH = "/Users/pabloherrero/sabat/RaTagging/artifacts/timing_BO/"

def load_training_data(json_pattern: str = "timing_config_*.json"):
    """Scrapes the directory for BO output files and builds the arrays."""
    files = glob.glob(json_pattern)
    if not files:
        raise FileNotFoundError(f"No files matching {json_pattern} found.")
        
    X_list, Y_list = [], []
    param_names = ["N_s1", "N_s2", "window_size", "threshold_bs", "t_drift_margin"]
    
    for f in files:
        with open(f, "r") as file:
            data = json.load(file)
            
        el_field = data.get("el_field")
        if el_field is None:
            continue
            
        X_list.append([el_field])
        
        p = data["parameters"]
        Y_list.append([
            p["N_s1"], 
            p["N_s2"], 
            p["window_size"], 
            p["threshold_bs"], 
            p["t_drift_margin"]
        ])
        
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    # Sort by E_EL
    sort_idx = np.argsort(X[:, 0])
    return X[sort_idx], Y[sort_idx], param_names

def fit_contextual_models(X: np.ndarray, Y: np.ndarray, param_names: list) -> dict:
    """Fits an independent Gaussian Process for each parameter."""
    models = {}
    
    # We remove WhiteKernel and give the RBF wider bounds to handle the V/cm scale
    kernel = C(1.0, (1e-2, 1e2)) * RBF(length_scale=1000.0, length_scale_bounds=(100.0, 10000.0))
    
    for i, name in enumerate(param_names):
        # alpha=0.05 provides stable matrix regularization (assuming ~5% noise variance)
        gpr = GaussianProcessRegressor(
            kernel=kernel, 
            alpha=0.05, 
            n_restarts_optimizer=15, 
            normalize_y=True
        )
        # Fit this specific parameter (Y[:, i])
        gpr.fit(X, Y[:, i])
        models[name] = gpr
        print(f"Model for {name} fitted. Kernel: {gpr.kernel_}")
        
    return models

def plot_response_surfaces(models: dict, X: np.ndarray, Y: np.ndarray, param_names: list):
    """Generates the validation plot for all independent models."""
    X_dense = np.linspace(X.min() - 200, X.max() + 200, 500).reshape(-1, 1)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Contextual Parameter Scaling vs. EL Field", fontsize=16)
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        gpr = models[name]
        
        # Predict purely for this parameter
        y_pred, sigma = gpr.predict(X_dense, return_std=True)
        
        ax.scatter(X, Y[:, i], c='red', zorder=10, label="BO Points")
        ax.plot(X_dense, y_pred, 'b-', label="GPR Mean")
        ax.fill_between(
            X_dense.ravel(), 
            y_pred - 1.96 * sigma, 
            y_pred + 1.96 * sigma, 
            alpha=0.2, color='blue', label="95% CI"
        )
        
        ax.set_title(name)
        ax.set_xlabel("E_EL (V/cm)")
        ax.set_ylabel("Optimal Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    fig.delaxes(axes[5])
    plt.tight_layout()
    plt.savefig(GLOBAL_OUT_PATH + "contextual_scaling_model.png")
    plt.close()
    print("Response surface plot saved to 'contextual_scaling_model.png'.")

def main():
    
    try:
        X, Y, param_names = load_training_data(json_pattern=GLOBAL_OUT_PATH + "timing_config_*.json")
        print(f"Loaded {len(X)} configurations.")
        
        print("\nFitting Independent Gaussian Process Regressors...")
        models = fit_contextual_models(X, Y, param_names)
        
        print("\nGenerating response surfaces...")
        plot_response_surfaces(models, X, Y, param_names)
        
        
        model_file = GLOBAL_OUT_PATH + "timing_context_model.joblib"
        
        joblib.dump({"models": models, "param_names": param_names}, model_file)
        print(f"\nModels saved successfully to {model_file}")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()