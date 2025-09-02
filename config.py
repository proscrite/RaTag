# -------------------------------
# General analysis thresholds
# -------------------------------
BASELINE_RMS_MAX = 2.0
AMPLITUDE_MIN = 10.0
AMPLITUDE_MAX = 500.0
PEAK_TIME_WINDOW = (100, 200)

# -------------------------------
# Gas / transport parameters
# -------------------------------
# Fit parameters for drift velocity model (Xe @ 2 bar, say)
DRIFT_VELOCITY_PARAMS = {
    "p0": 0.92809704,
    "p1": 17.17333489,
    "p2": 0.51193002,
    "p3": 0.30107278,
}

# Optionally: define common drift fields to evaluate
DRIFT_FIELDS = [35, 50, 70, 107, 142, 178, 214, 250, 285, 321, 357, 428]  # V/cm
