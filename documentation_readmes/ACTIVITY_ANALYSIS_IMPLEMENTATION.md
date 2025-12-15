# Activity Analysis Module - Implementation Summary

## Overview

Created comprehensive **activity analysis** module for time-resolved alpha spectroscopy. Tracks isotope decay (particularly **Ra-224** with T½ = 3.6 days) over multiple measurement sets.

---

## New Files Created

### 1. `RaTag/alphas/activity_analysis.py` (500 lines)
**Core functionality for activity measurements:**

#### Data Types
- `TimeStampedSpectrum`: Spectrum + acquisition timestamp from .wfm files
- `ActivityMeasurement`: Activity [Bq] + count rate + uncertainties for single time point
- `DecayFitResult`: Exponential decay fit parameters (A₀, λ, T½)

#### Key Functions
- `get_wfm_timestamps()`: Extract timestamps from .wfm files (uses **mtime**, not btime!)
- `load_timestamped_spectrum()`: Load single set with timestamp
- `load_all_timestamped_spectra()`: Load all sets, sorted by time
- `measure_activity()`: Convert counts → activity [Bq] with efficiency/branching corrections
- `measure_activity_timeseries()`: Measure activity for all time points
- `fit_exponential_decay()`: Fit A(t) = A₀ × exp(-λt) to extract half-life

#### Reference Values
- `HALF_LIVES`: Literature half-lives for Th-232 chain isotopes [hours]
- `BRANCHING_RATIOS`: Alpha branching ratios for main lines

---

### 2. `RaTag/alphas/activity_plotting.py` (260 lines)
**Visualization for activity time series:**

#### Plotting Functions
- `plot_activity_timeseries()`: Activity vs time with fitted decay curve
- `plot_count_rate_timeseries()`: Count rate (no efficiency correction)
- `plot_multi_isotope_activity()`: Compare multiple isotopes (normalized)
- `plot_activity_diagnostic()`: 4-panel comprehensive figure (activity + residuals + count rate + statistics)

---

### 3. `RaTag/notebooks/activity_analysis_demo.ipynb`
**Step-by-step demonstration notebook:**

1. **Initialize run** (RUN18, 12 sets)
2. **Derive isotope ranges** (reuse existing spectrum_fitting pipeline!)
3. **Load time-stamped spectra** (per-set, not aggregated)
4. **Measure Ra-224 activity** for each time point
5. **Fit exponential decay** to extract half-life
6. **Visualize** activity evolution with diagnostic plots

---

## Key Design Decisions

### ✅ Reuses Existing Functions
**Did NOT rewrite:**
- Spectrum loading (`load_spectrum_from_run`, `load_spectrum_from_energy_maps`)
- Peak fitting (`fit_multi_crystalball_progressive`)
- Energy calibration (`derive_energy_calibration`)
- Isotope range derivation (`derive_isotope_ranges`)

**New code only for:**
- Timestamp extraction from .wfm files
- Activity calculation (counts → Bq)
- Decay curve fitting
- Time-series visualization

### ✅ Timestamp Handling
**Critical finding:** When copying files between disks on macOS:
- ❌ `Birth time (btime)` = copy time (NOT original!)
- ✅ `Modification time (mtime)` = **preserved** original creation time

**Solution:** Use `.stat().st_mtime` for acquisition timestamps.

### ✅ Ra-224 as Target Isotope
**Why Ra-224 (not Th-228)?**
- Th-228: T½ = 1.9 years → constant over 2 days
- **Ra-224: T½ = 3.6 days → ~40% decay over 2 days** ← Observable!
- Ra-224 is the **second peak** (5.69 MeV) in spectrum

---

## Usage Example

```python
from RaTag.alphas import (
    load_all_timestamped_spectra,
    measure_activity_timeseries,
    fit_exponential_decay,
    plot_activity_diagnostic,
    HALF_LIVES, BRANCHING_RATIOS
)

# Load time-stamped spectra
spectra = load_all_timestamped_spectra(run, energy_range=(4000, 8200))

# Measure Ra-224 activity
ra224_measurements = measure_activity_timeseries(
    spectra, 
    ra224_range,
    efficiency=1.0,  # Relative activity
    branching_ratio=BRANCHING_RATIOS['Ra224']  # 0.949
)

# Fit decay curve
decay_fit = fit_exponential_decay(ra224_measurements, HALF_LIVES['Ra224'])

# Visualize
fig, axes = plot_activity_diagnostic(ra224_measurements, decay_fit)
plt.show()

print(f"Fitted T½: {decay_fit.half_life/24:.2f} ± {decay_fit.half_life_err/24:.2f} days")
print(f"Literature: {decay_fit.half_life_literature/24:.2f} days")
```

---

## Physical Interpretation

### What You Can Measure

1. **Decay curve validation**: Test exponential decay model A(t) = A₀ × exp(-λt)
2. **Half-life measurement**: Compare fitted T½ with literature (3.66 days)
3. **Source characterization**: Initial activity tells you source age/strength
4. **Secular equilibrium**: Compare Ra-224 with downstream isotopes (Rn-220, Po-216)

### RUN18 Dataset
- **12 measurement sets** over ~2 days (Nov 4-6, 2025)
- Each set: **~4-5 hours** acquisition
- Expected Ra-224 decay: **~40%** over this period
- Chi-squared test for goodness-of-fit

---

## Next Steps for Absolute Calibration

Current implementation uses `efficiency=1.0` → **relative activity** [counts/BR/time]

For **absolute activity** [Bq], need to measure:
1. **Geometric efficiency**: Solid angle from source to detector
2. **Intrinsic efficiency**: Energy-dependent detection probability
3. **Dead time corrections**: High count rate losses
4. **Energy window efficiency**: Fraction of peak captured in range
5. **Background subtraction**: Environmental/cosmic background

---

## Testing Recommendations

1. **Run demo notebook** (`activity_analysis_demo.ipynb`)
2. **Check timestamps**: Verify .wfm mtime extraction is correct
3. **Compare isotopes**: Measure Th-228, Rn-220, Po-216 (test equilibrium)
4. **Vary n_sigma**: Test sensitivity to isotope range definition
5. **Bootstrap analysis**: Estimate systematic uncertainties

---

## File Structure

```
RaTag/
├── alphas/
│   ├── __init__.py                    # Updated with new exports
│   ├── spectrum_fitting.py            # Existing (unchanged)
│   ├── spectrum_plotting.py           # Existing (unchanged)
│   ├── activity_analysis.py           # NEW: Activity calculation & decay fitting
│   └── activity_plotting.py           # NEW: Activity visualization
└── notebooks/
    ├── spectrum_fitting_demo.ipynb    # Existing
    └── activity_analysis_demo.ipynb   # NEW: Activity analysis demo
```

---

## Summary

✅ **Reused existing pipeline** (spectrum loading, fitting, calibration)  
✅ **Correct timestamp extraction** (mtime, not btime)  
✅ **Ra-224 as target** (T½ = 3.6 days, observable decay)  
✅ **Full workflow** (loading → activity → fitting → visualization)  
✅ **Comprehensive demo notebook** with step-by-step guide  

Ready for activity measurements! 🚀
