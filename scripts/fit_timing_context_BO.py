import glob
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel

def load_training_data(json_pattern: str = "timing_config_*.json"):
    """
    Scrapes the directory for BO output files and builds the X and Y arrays.
    """
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
            print(f"Skipping {f}: No 'el_field' found.")
            continue
            
        X_list.append([el_field])
        
        # Ensure parameters are appended in the exact order of param_names
        p = data["parameters"]
        Y_list.append([
            p["N_s1"], 
            p["N_s2"], 
            p["window_size"], 
            p["threshold_bs"], 
            p["t_drift_margin"]
        ])
        
    # Sort by E_EL to make plotting and reading easier
    X = np.array(X_list)
    Y = np.array(Y_list)
    
    sort_idx = np.argsort(X[:, 0])
    return X[sort_idx], Y[sort_idx], param_names

def fit_contextual_model(X: np.ndarray, Y: np.ndarray) -> GaussianProcessRegressor:
    """
    Fits a Multi-Output Gaussian Process Regressor.
    """
    # The Kernel: 
    # C(1.0) scales the amplitude.
    # RBF models the smooth, non-linear physical scaling.
    # WhiteKernel accounts for the slight randomness/noise in the BO's convergence.
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=500.0, length_scale_bounds=(100.0, 5000.0)) \
             + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-3, 1.0))
             
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True)
    gpr.fit(X, Y)
    
    print(f"Model fitted. Learned Kernel: {gpr.kernel_}")
    return gpr

def plot_response_surfaces(gpr: GaussianProcessRegressor, X: np.ndarray, Y: np.ndarray, param_names: list):
    """
    Generates a 5-panel plot showing the interpolation and uncertainty for each parameter.
    """
    # Create a dense axis for smooth plotting (spanning from slightly below min E_EL to slightly above max)
    X_dense = np.linspace(X.min() - 200, X.max() + 200, 500).reshape(-1, 1)
    
    # Predict the parameters and their uncertainties (standard deviations)
    Y_pred, sigma = gpr.predict(X_dense, return_std=True)
    
    fig, axes = plt.subplots(3, 2, figsize=(12, 12))
    fig.suptitle("Contextual Parameter Scaling vs. EL Field", fontsize=16)
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        
        # Plot training data points
        ax.scatter(X, Y[:, i], c='red', zorder=10, label="BO Optimal Points")
        
        # Plot GPR mean prediction
        ax.plot(X_dense, Y_pred[:, i], 'b-', label="GPR Prediction")
        
        # Plot 95% Confidence Interval (±1.96 sigma)
        # Note: scikit-learn's multi-output GPR returns a single sigma for all outputs unless explicitly wrapped,
        # but typically this is acceptable for normalized Y. We apply it to the scaled space.
        ax.fill_between(
            X_dense.ravel(), 
            Y_pred[:, i] - 1.96 * sigma, 
            Y_pred[:, i] + 1.96 * sigma, 
            alpha=0.2, color='blue', label="95% CI"
        )
        
        ax.set_title(name)
        ax.set_xlabel("E_EL (V/cm)")
        ax.set_ylabel("Optimal Value")
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.6)
        
    # Remove the empty 6th subplot
    fig.delaxes(axes[5])
    
    plt.tight_layout()
    plt.savefig("/Users/pabloherrero/sabat/RaTagging/artifacts/timing_BO/contextual_scaling_model.png")
    plt.close()
    print("Response surface plot saved to 'contextual_scaling_model.png'.")

def main():
    try:
        print("Loading BO configurations...")
        X, Y, param_names = load_training_data(json_pattern="/Users/pabloherrero/sabat/RaTagging/artifacts/timing_BO/timing_config_*.json")
        print(f"Loaded {len(X)} configurations.")
        
        print("\nFitting Gaussian Process Regressor...")
        gpr = fit_contextual_model(X, Y)
        
        print("\nGenerating response surfaces...")
        plot_response_surfaces(gpr, X, Y, param_names)
        
        model_file = "/Users/pabloherrero/sabat/RaTagging/artifacts/timing_BO/timing_context_model.joblib"
        joblib.dump({"model": gpr, "param_names": param_names}, model_file)
        print(f"\nModel saved successfully to {model_file}")
        print("You can now use this model to predict parameters for new runs.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()