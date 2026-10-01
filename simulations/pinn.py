from typing import NamedTuple, Tuple, List, Dict
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PhysicalConstants(NamedTuple):
    v_drift: float = 0.185        # Drift velocity in cm/us
    D_L: float = 0.012            # Longitudinal diffusion coefficient in cm^2/us
    sigma_0: float = 0.082        # Intrinsic THGEM avalanche transit dispersion in us
    c_threshold: float = 5.5941   # 2 * sqrt(2 * ln(1 / 0.02)) for 2% peak threshold

class ModelHyperparameters(NamedTuple):
    input_dim: int = 2            # [E_PIN, Delta_t]
    hidden_dim: int = 32
    output_dim: int = 1           # S2_predicted
    learning_rate: float = 1e-3
    lambda_diff: float = 0.05     # Weighting factor for transport regularization
    epochs: int = 500
    batch_size: int = 64

class NormalizationStats(NamedTuple):
    feat_mean: torch.Tensor       
    feat_std: torch.Tensor        
    target_mean: float
    target_std: float

class NetworkWeights(NamedTuple):
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor

# ----------------------------------------------------------------------

def compute_normalization(features: torch.Tensor, targets: torch.Tensor) -> NormalizationStats:
    """Calculates scaling statistics strictly from training data."""
    f_mean = torch.mean(features, dim=0)
    f_std = torch.std(features, dim=0).clamp(min=1e-6)
    t_mean = torch.mean(targets).item()
    t_std = torch.std(targets).clamp(min=1e-6).item()
    return NormalizationStats(f_mean, f_std, t_mean, t_std)

def normalize_inputs(features: torch.Tensor, stats: NormalizationStats) -> torch.Tensor:
    return (features - stats.feat_mean) / stats.feat_std

def init_network_weights(hyperparams: ModelHyperparameters, 
                         rng: torch.Generator) -> NetworkWeights:
    """Pure functional Xavier weight initialization."""
    def kaiming_uniform(in_f: int, out_f: int) -> Tuple[torch.Tensor, torch.Tensor]:
        bound = (6.0 / (in_f + out_f)) ** 0.5
        w = torch.empty(out_f, in_f).uniform_(-bound, bound, generator=rng)
        b = torch.zeros(out_f)
        w.requires_grad_(True)
        b.requires_grad_(True)
        return w, b

    w1, b1 = kaiming_uniform(hyperparams.input_dim, hyperparams.hidden_dim)
    w2, b2 = kaiming_uniform(hyperparams.hidden_dim, hyperparams.hidden_dim)
    w3, b3 = kaiming_uniform(hyperparams.hidden_dim, hyperparams.output_dim)
    
    return NetworkWeights(w1=w1, b1=b1, w2=w2, b2=b2, w3=w3, b3=b3)

def forward(features: torch.Tensor, weights: NetworkWeights) -> torch.Tensor:
    """Pure feedforward pass: maps [E_PIN, Delta_t] -> S2_pred via SiLU activations."""
    h1 = torch.nn.functional.silu(torch.addmm(weights.b1, features, weights.w1.t()))
    h2 = torch.nn.functional.silu(torch.addmm(weights.b2, h1, weights.w2.t()))
    out = torch.addmm(weights.b3, h2, weights.w3.t())
    return out.squeeze(-1)

def compute_expected_width(delta_t: torch.Tensor, constants: PhysicalConstants) -> torch.Tensor:
    """Evaluates expected S2 temporal width at 2% dynamic threshold."""
    dispersion_term = (2.0 * constants.D_L * delta_t) / (constants.v_drift ** 2)
    sigma_t = torch.sqrt(constants.sigma_0 ** 2 + dispersion_term)
    return constants.c_threshold * sigma_t

def compute_loss(weights: NetworkWeights, features_raw: torch.Tensor, features_norm: torch.Tensor,
                 s2_obs_norm: torch.Tensor, s2_width_raw: torch.Tensor,
                 hyperparams: ModelHyperparameters, constants: PhysicalConstants) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluates empirical MSE and 1D transport loss."""
    s2_pred_norm = forward(features_norm, weights)
    l_mse = torch.mean((s2_pred_norm - s2_obs_norm) ** 2)

    delta_t_raw = features_raw[:, 1]
    expected_width = compute_expected_width(delta_t_raw, constants)
    l_diff = torch.mean((s2_width_raw - expected_width) ** 2)

    total_loss = l_mse + (hyperparams.lambda_diff * l_diff)
    return total_loss, l_mse, l_diff

# ----------------------------------------------------------------------
def step_optimizer( weights: NetworkWeights, features_raw: torch.Tensor, 
                   features_norm: torch.Tensor, s2_obs_norm: torch.Tensor, 
                   s2_width_raw: torch.Tensor, hyperparams: ModelHyperparameters,
                   constants: PhysicalConstants) -> Tuple[NetworkWeights, float, float]:
    """Applies a functional gradient step and returns an updated parameter tuple."""
    total_loss, l_mse, l_diff = compute_loss(
        weights, features_raw, features_norm, s2_obs_norm, s2_width_raw, hyperparams, constants
    )
    total_loss.backward()

    with torch.no_grad():
        updated = [
            (param - hyperparams.learning_rate * param.grad).clone().detach().requires_grad_(True)
            for param in weights
        ]
    return NetworkWeights(*updated), l_mse.item(), l_diff.item()

# -----------------------------------------------------------------------------
# Execution Pipeline
# -----------------------------------------------------------------------------
def train_phase0(features_raw: torch.Tensor, 
                 s2_obs_raw: torch.Tensor, s2_width_raw: torch.Tensor,
                hyperparams: ModelHyperparameters, seed: int = 42) -> Tuple[NetworkWeights, NormalizationStats, List[float]]:
    rng = torch.Generator().manual_seed(seed)
    
    # 1. Transform target to log-space (avalanche multiplicative domain)
    log_s2 = torch.log(s2_obs_raw.clamp(min=1e-3))
    
    # 2. Compute normalization stats in log-space
    stats = compute_normalization(features_raw, log_s2)
    features_norm = normalize_inputs(features_raw, stats)
    targets_norm = (log_s2 - stats.target_mean) / stats.target_std

    def init_tensor(in_f: int, out_f: int, rng: torch.Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pure functional tensor initializer with valid autograd leaves."""
        bound = (6.0 / (in_f + out_f)) ** 0.5
        
        # 1. Allocate tensor and sample from uniform generator
        w = torch.empty((out_f, in_f), dtype=torch.float32)
        w.uniform_(-bound, bound, generator=rng)
        
        # 2. Establish as autograd leaf parameter
        w_leaf = w.clone().detach().requires_grad_(True)
        b_leaf = torch.zeros(out_f, dtype=torch.float32, requires_grad=True)
        
        return w_leaf, b_leaf
    
    # 3. Initialize weights
    w1, b1 = init_tensor(hyperparams.input_dim, hyperparams.hidden_dim, rng)
    w2, b2 = init_tensor(hyperparams.hidden_dim, hyperparams.hidden_dim, rng)
    w3, b3 = init_tensor(hyperparams.hidden_dim, hyperparams.output_dim, rng)
    weights = NetworkWeights(w1, b1, w2, b2, w3, b3)

    n_samples = features_raw.shape[0]
    history: List[float] = []

    for epoch in range(hyperparams.epochs):
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{hyperparams.epochs} -- Training Phase 0...")
        perm = torch.randperm(n_samples, generator=rng)
        epoch_loss, batches = 0.0, 0

        for i in range(0, n_samples, hyperparams.batch_size):
            idx = perm[i : i + hyperparams.batch_size]
            
            # Forward pass
            pred_norm = forward(features_norm[idx], weights)
            loss = torch.mean((pred_norm - targets_norm[idx]) ** 2)
            
            loss.backward()
            with torch.no_grad():
                updated = [
                    (param - hyperparams.learning_rate * param.grad).clone().detach().requires_grad_(True)
                    for param in weights
                ]
            weights = NetworkWeights(*updated)
            
            epoch_loss += loss.item()
            batches += 1

        history.append(epoch_loss / batches)

    return weights, stats, history

# ------------------------------------------------------------------------------
def evaluate_phase0( weights: NetworkWeights, stats: NormalizationStats, 
                    features_raw: torch.Tensor, s2_obs_raw: torch.Tensor,
                    s2_width_raw: torch.Tensor, constants: PhysicalConstants ) -> Dict[str, float]:
    """Evaluates the physical model and prints true empirical distributions."""
    with torch.no_grad():
        features_norm = normalize_inputs(features_raw, stats)
        log_pred_norm = forward(features_norm, weights)

        # Denormalize log prediction
        log_pred = (log_pred_norm * stats.target_std) + stats.target_mean
        s2_pred = torch.exp(log_pred)
        # Transport verification
        delta_t = features_raw[:, 1]
        expected_width = compute_expected_width(delta_t, constants)
        
        # Residuals
        residual = s2_obs_raw - s2_pred
        diff_error = s2_width_raw - expected_width

    metrics = {
        "n_events": float(features_raw.shape[0]),
        "e_pin_mean": float(torch.mean(features_raw[:, 0])),
        "e_pin_std": float(torch.std(features_raw[:, 0])),
        "delta_t_mean": float(torch.mean(delta_t)),
        "delta_t_std": float(torch.std(delta_t)),
        "s2_obs_mean": float(torch.mean(s2_obs_raw)),
        "s2_obs_std": float(torch.std(s2_obs_raw)),
        "s2_pred_mean": float(torch.mean(s2_pred)),
        "s2_mse": float(torch.mean(residual ** 2)),
        "s2_width_mean": float(torch.mean(s2_width_raw)),
        "s2_width_std": float(torch.std(s2_width_raw)),
        "diff_mse": float(torch.mean(diff_error ** 2)),
    }
    return metrics


def extract_evaluation_tensors(
    weights,
    stats,
    features_raw: torch.Tensor,
    s2_obs_raw: torch.Tensor,
    s2_width_raw: torch.Tensor,
    constants
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Pure transformation from torch tensors to numpy arrays for visualization."""
    with torch.no_grad():
        features_norm = normalize_inputs(features_raw, stats)
        s2_pred_norm = forward(features_norm, weights)
        s2_pred = (s2_pred_norm * stats.target_std) + stats.target_mean
        expected_width = compute_expected_width(features_raw[:, 1], constants)
        
    e_pin_np = features_raw[:, 0]
    delta_t_np = features_raw[:, 1]
    s2_obs_np = s2_obs_raw
    s2_pred_np = s2_pred
    s2_width_np = s2_width_raw
    expected_w_np = expected_width
    
    return e_pin_np, delta_t_np, s2_obs_np, s2_pred_np, s2_width_np, expected_w_np

def plot_phase0_diagnostics(
    e_pin: np.ndarray,
    delta_t: np.ndarray,
    s2_obs: np.ndarray,
    s2_pred: np.ndarray,
    s2_width: np.ndarray,
    expected_width: np.ndarray,
    output_path: str = "phase0_diagnostics.png"
) -> None:
    """Renders 4-panel PRX-style kinematic & transport diagnostics."""
    residuals = s2_obs - s2_pred

    fig, axs = plt.subplots(2, 2, figsize=(13, 10), dpi=300)
    plt.subplots_adjust(hspace=0.28, wspace=0.24)

    # -------------------------------------------------------------------------
    # Panel 1: S2 Distribution Overlay (Observed vs. Predicted)
    # -------------------------------------------------------------------------
    ax1 = axs[0, 0]
    bins_s2 = np.linspace(0, np.percentile(s2_obs, 99.5), 80)
    ax1.hist(s2_obs, bins=bins_s2, histtype="step", linewidth=1.8, color="#1f77b4", label=r"Observed $S2_{\text{obs}}$")
    ax1.hist(s2_pred, bins=bins_s2, histtype="stepfilled", alpha=0.45, color="#ff7f0e", label=r"PINN $S2_{\text{pred}}$")
    ax1.set_xlabel(r"$S2$ Area [$\mathrm{mV}\cdot\mu\mathrm{s}$]", fontsize=11)
    ax1.set_ylabel("Events / Bin", fontsize=11)
    ax1.set_title("Secondary Scintillation Yield Reconstruction", fontsize=12, fontweight="semibold")
    ax1.legend(frameon=True, loc="upper right")
    ax1.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel 2: Kinematic Coupling (E_PIN vs S2_obs with PINN Surface)
    # -------------------------------------------------------------------------
    ax2 = axs[0, 1]
    h = ax2.hexbin(e_pin, s2_obs, gridsize=60, cmap="viridis", mincnt=1, bins="log")
    
    # Sort by E_PIN to plot mean profile line
    sort_idx = np.argsort(e_pin)
    e_pin_sorted = e_pin[sort_idx]
    s2_pred_sorted = s2_pred[sort_idx]
    
    # Running average of prediction across E_PIN
    bin_edges = np.linspace(e_pin.min(), e_pin.max(), 30)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    digitized = np.digitize(e_pin, bin_edges)
    profile = [s2_pred[digitized == i].mean() if np.sum(digitized == i) > 5 else np.nan for i in range(1, len(bin_edges))]
    
    ax2.plot(bin_centers, profile, color="crimson", linewidth=2.2, label="PINN Mean Surface")
    cb = fig.colorbar(h, ax=ax2)
    cb.set_label(r"$\log_{10}(\mathrm{Counts})$", fontsize=10)
    ax2.set_xlabel(r"Silicon Deposit $E_{\mathrm{PIN}}$ [$\mathrm{keV}_{\mathrm{ee}}$]", fontsize=11)
    ax2.set_ylabel(r"$S2_{\mathrm{obs}}$ [$\mathrm{mV}\cdot\mu\mathrm{s}$]", fontsize=11)
    ax2.set_title(r"Coupled Attenuation: $E_{\mathrm{PIN}}$ vs $S2$", fontsize=12, fontweight="semibold")
    ax2.legend(frameon=True, loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel 3: Kinematic Residuals (R = S2_obs - S2_pred)
    # -------------------------------------------------------------------------
    ax3 = axs[1, 0]
    r_lim = np.percentile(np.abs(residuals), 99)
    bins_r = np.linspace(-r_lim, r_lim, 70)
    ax3.hist(residuals, bins=bins_r, histtype="stepfilled", color="#2ca02c", alpha=0.6, edgecolor="black")
    ax3.axvline(0, color="black", linestyle="--", linewidth=1.2)
    ax3.set_xlabel(r"Residual $R = S2_{\mathrm{obs}} - S2_{\mathrm{pred}}$ [$\mathrm{mV}\cdot\mu\mathrm{s}$]", fontsize=11)
    ax3.set_ylabel("Events / Bin", fontsize=11)
    ax3.set_title("Phase 0 Residual Anomaly Spectrum", fontsize=12, fontweight="semibold")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # -------------------------------------------------------------------------
    # Panel 4: Longitudinal Diffusion Bound (Delta t vs S2_width)
    # -------------------------------------------------------------------------
    ax4 = axs[1, 1]
    ax4.scatter(delta_t, s2_width, color="#4c72b0", s=2, alpha=0.15, rasterized=True, label="Data Events")
    
    # Sort for analytical curve
    dt_order = np.argsort(delta_t)
    ax4.plot(delta_t[dt_order], expected_width[dt_order], color="darkred", linewidth=2.0, label="Analytical 1D Diffusion")
    ax4.set_xlabel(r"Drift Time $\Delta t$ [$\mu\mathrm{s}$]", fontsize=11)
    ax4.set_ylabel(r"$S2_{\mathrm{width}}$ [$\mu\mathrm{s}$]", fontsize=11)
    ax4.set_title(r"Temporal Dispersion: $\Delta t$ vs $S2_{\mathrm{width}}$", fontsize=12, fontweight="semibold")
    ax4.legend(frameon=True, loc="upper right")
    ax4.grid(True, linestyle="--", alpha=0.4)

    plt.savefig(output_path, bbox_inches="tight")
    plt.close(fig)