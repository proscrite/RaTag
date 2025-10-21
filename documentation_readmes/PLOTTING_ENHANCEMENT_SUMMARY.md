# Data Persistence and Plotting Enhancement Summary

## Overview

This document summarizes the comprehensive enhancements made to the RaTag pipeline for data persistence and diagnostic plotting.

## Changes Made

### 1. Enhanced Data Storage (`dataIO.py`)

#### Modified Functions:
- **`store_s2area()`**: Enhanced to save complete set metadata
  - Added optional `set_pmt` parameter
  - Now saves timing parameters (t_s1, t_s2_start_mean, t_s2_end_mean, s2_duration, etc.)
  - Saves field parameters (drift_field, EL_field, time_drift, speed_drift)
  - All data saved to `s2_results.json` alongside `s2_areas.npy`

#### New Functions:
- **`store_xray_areas_combined()`**: Save combined X-ray areas from all sets
  - Creates `{run_id}_xray_areas_combined.npy`
  - Creates `{run_id}_xray_metadata.json` with statistics
  
- **`save_figure()`**: Utility to save matplotlib figures consistently
  - Default 150 DPI for publication quality
  - Creates parent directories automatically
  - Prints confirmation message

### 2. Enhanced Plotting (`plotting.py`)

#### Modified Functions:
- **`plot_hist_fit()`**: Now returns `(fig, ax)` tuple instead of just `ax`
  - Enables saving figures after creation
  
- **`plot_s2_vs_drift()`**: Enhanced with normalization option
  - Added `normalized` parameter
  - Normalizes by X-ray reference when `normalized=True`
  - Returns `(fig, ax)` tuple
  - Better figure size and formatting

#### New Functions:
- **`plot_xray_histogram()`**: Plot combined X-ray histogram with fit
  - Accepts fit results and statistics
  - Consistent styling with other histograms
  - Returns `(fig, ax)` for saving

### 3. Enhanced Pipeline (`pipeline.py`)

#### Modified Function: `run_ion_integration()`

**Before:**
- Integrated S2 areas
- Fitted distributions
- Saved NPY files only
- Optional plot_s2_vs_drift display

**After:**
- Integrates S2 areas
- Fits distributions
- **Creates `plots/` directory**
- **For each set:**
  - Saves S2 areas with complete metadata (`store_s2area` with `set_pmt`)
  - Generates and saves histogram plot (`{set_name}_s2_histogram.png`)
  - Generates and saves waveform validation plot (`{set_name}_waveform_validation.png`)
    - Shows 10 random waveforms (individual frames, not averaged)
    - Overlays S1, S2 start, S2 end timing with CI bands
- **For the run:**
  - Saves S2 vs drift field plot (unnormalized)

#### Modified Function: `run_calibration_analysis()`

**Before:**
- Ran calibration
- Optional interactive plots

**After:**
- Runs calibration
- **Creates `plots/` directory**
- **Saves combined X-ray data:**
  - `{run_id}_xray_areas_combined.npy`
  - `{run_id}_xray_metadata.json`
  - `{run_id}_xray_histogram.png` (from xray_calibration.py)
- **Generates and saves:**
  - Normalized S2 vs drift plot (`{run_id}_s2_vs_drift_normalized.png`)
  - Diffusion analysis plot (`{run_id}_diffusion_analysis.png`)
    - Three panels: σ² vs t_d, σ² vs t_d/v_d², σ² vs E/p
- Added `save_plots` parameter (default: `True`)

### 4. Enhanced X-ray Calibration (`xray_calibration.py`)

#### Modified Function: `calibrate_and_analyze()`

**Added:**
- Saves combined X-ray areas after loading
- Generates and saves X-ray histogram plot
- Uses `plot_xray_histogram()` from plotting module
- Consistent with other plotting functions

### 5. Type Fixes

**Fixed Optional parameters:**
- `store_s2area(s2, set_pmt: Optional[SetPmt] = None)`
- `store_xray_results(xr, path: Optional[PathLike] = None)`
- `store_xrayset(xrays, outdir: Optional[Path] = None)`

**Added import:**
- `from typing import Optional` in `dataIO.py`

## Data Logged

### Per-Set JSON Metadata (`s2_results.json`)

All parameters needed to reproduce histograms:

```json
{
  "method": "integration_method",
  "params": {...},
  "mean": 1.234,
  "sigma": 0.123,
  "ci95": 0.241,
  "fit_success": true,
  "set_metadata": {
    "t_s1": 10.5,
    "t_s1_std": 0.3,
    "t_s2_start_mean": 45.2,
    "t_s2_start_std": 1.1,
    "t_s2_end_mean": 46.3,
    "t_s2_end_std": 1.2,
    "s2_duration_mean": 1.1,
    "s2_duration_std": 0.15,
    "drift_field": 250.0,
    "EL_field": 3000.0,
    "time_drift": 35.0,
    "speed_drift": 0.04,
    "red_drift_field": 125.0
  }
}
```

## Plots Generated

### Per-Set Plots (in `plots/` directory)

1. **`{set_name}_s2_histogram.png`**
   - Ion recoil S2 area histogram
   - Gaussian fit overlay
   - Mean and CI markers

2. **`{set_name}_waveform_validation.png`**
   - 10 random waveforms (individual, not averaged)
   - Each in separate subplot
   - S1 time mean + CI (red)
   - S2 start mean + CI (blue)
   - S2 end mean + CI (purple)

### Run-Level Plots (in `plots/` directory)

3. **`{run_id}_xray_histogram.png`**
   - Combined X-ray S2 areas from all sets
   - Gaussian fit for calibration
   - Used to extract g_S2 gain factor

4. **`{run_id}_s2_vs_drift.png`**
   - Unnormalized S2 area vs drift field
   - Raw mV·µs values

5. **`{run_id}_s2_vs_drift_normalized.png`**
   - Normalized S2 area (A_ion / A_xray)
   - Shows recombination effects

6. **`{run_id}_diffusion_analysis.png`**
   - Three-panel plot:
     - σ² vs t_drift (linear fit)
     - σ² vs t_drift/v_drift²
     - σ² vs reduced field (E/p)
   - Extracts transverse diffusion coefficient

## Usage Example

```python
from RaTag.pipeline import prepare_run, run_ion_integration, run_calibration_analysis

# Complete pipeline with all plotting and storage
run = prepare_run(run, estimate_s2_windows=True)

fitted = run_ion_integration(
    run,
    use_estimated_s2_windows=True,
    flag_plot=True  # Generates all per-set plots
)

calib, recomb = run_calibration_analysis(
    run,
    ion_fitted_areas=fitted,
    flag_plot=True,      # Generate plots
    save_plots=True      # Save to disk
)
```

## Output Directory Structure

```
RUN8/
├── set1/
│   ├── s2_areas.npy              # Raw arrays
│   ├── s2_results.json           # Complete metadata
│   ├── xray_areas.npy
│   └── xray_results.json
├── plots/
│   ├── set1_s2_histogram.png           # Per-set
│   ├── set1_waveform_validation.png
│   ├── set2_s2_histogram.png
│   ├── set2_waveform_validation.png
│   ├── RUN8_xray_histogram.png         # Run-level
│   ├── RUN8_xray_areas_combined.npy
│   ├── RUN8_xray_metadata.json
│   ├── RUN8_s2_vs_drift.png
│   ├── RUN8_s2_vs_drift_normalized.png
│   └── RUN8_diffusion_analysis.png
```

## Key Features

✅ **Complete metadata persistence** - All analysis parameters saved to JSON
✅ **NPY arrays** for fast numerical processing
✅ **Publication-quality plots** (150 DPI PNG)
✅ **Waveform validation** - Visual check of S2 window placement
✅ **Comprehensive logging** - t_s1, s2_start, s2_end, drift parameters
✅ **Modular workflow** - Can load from disk and skip pipeline steps
✅ **Automatic directory creation** - plots/ created automatically
✅ **Consistent naming** - All files follow `{identifier}_{type}.{ext}` pattern

## Documentation

- **DATA_PERSISTENCE_GUIDE.md**: Complete reference for all stored data and plots
- **PIPELINE_SUMMARY.md**: Updated to reflect new features
- **QUICKSTART.md**: Updated with plotting examples

## Backward Compatibility

- Old `s2_areas.npy` files can still be loaded
- `load_s2area()` handles both old (NPY only) and new (NPY + JSON) formats
- If `s2_results.json` missing, falls back to NPY-only loading

## Benefits

1. **Reproducibility**: All parameters logged
2. **Validation**: Waveform plots enable visual QA
3. **Publication**: High-quality plots ready for papers
4. **Debugging**: Easy to identify timing window issues
5. **Archiving**: Complete analysis state saved
6. **Collaboration**: JSON files are human-readable

## Testing Recommendation

Run the pipeline on a small dataset (nfiles=10) to verify:
1. All plots are generated
2. JSON files contain expected fields
3. Waveform validation plots show correct timing markers
4. Directory structure is clean and organized
