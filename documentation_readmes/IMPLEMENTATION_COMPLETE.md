# Summary: Comprehensive Data Persistence and Plotting Implementation

## ✅ Implementation Complete

All requested features for data persistence and plot logging have been successfully implemented in the RaTag analysis pipeline.

## Changes Summary

### 1. **dataIO.py** - Enhanced Data Storage
- ✅ `store_s2area()` enhanced to save complete set metadata
- ✅ `store_xray_areas_combined()` added for run-level X-ray data
- ✅ `save_figure()` utility for consistent plot saving
- ✅ All timing parameters logged: t_s1, s2_start, s2_end, s2_duration (with std)
- ✅ All field parameters logged: drift_field, EL_field, time_drift, speed_drift
- ✅ Storage format: NPY (arrays) + JSON (metadata)

### 2. **plotting.py** - Enhanced Plotting Functions
- ✅ `plot_hist_fit()` returns (fig, ax) for saving
- ✅ `plot_s2_vs_drift()` enhanced with normalization option
- ✅ `plot_xray_histogram()` added for combined X-ray data
- ✅ All plotting functions return figure objects for saving

### 3. **pipeline.py** - Comprehensive Plotting Integration

#### `run_ion_integration()`
**Per-Set Outputs:**
- ✅ S2 histogram with Gaussian fit (`{set}_s2_histogram.png`)
- ✅ Waveform validation plot (`{set}_waveform_validation.png`)
  - 10 random waveforms (individual, not averaged)
  - S1, S2 start, S2 end timing markers with CI bands
- ✅ Complete metadata saved to JSON

**Run-Level Outputs:**
- ✅ S2 vs drift field (unnormalized) (`{run_id}_s2_vs_drift.png`)

#### `run_calibration_analysis()`
**Run-Level Outputs:**
- ✅ Combined X-ray histogram (`{run_id}_xray_histogram.png`)
- ✅ S2 vs drift field normalized (`{run_id}_s2_vs_drift_normalized.png`)
- ✅ Diffusion analysis (`{run_id}_diffusion_analysis.png`)
  - Panel 1: σ² vs t_drift
  - Panel 2: σ² vs t_drift/v_drift²
  - Panel 3: σ² vs reduced field E/p
- ✅ Combined X-ray areas saved (NPY + JSON)

### 4. **xray_calibration.py** - X-ray Plot Saving
- ✅ X-ray histogram generation and saving integrated
- ✅ Uses `plot_xray_histogram()` from plotting module

## Data Logged (per set)

### Timing Parameters (in s2_results.json)
```json
{
  "t_s1": 10.5,               // S1 peak time (µs)
  "t_s1_std": 0.3,            // S1 timing uncertainty
  "t_s2_start_mean": 45.2,    // S2 start time mean (µs)
  "t_s2_start_std": 1.1,      // S2 start timing spread
  "t_s2_end_mean": 46.3,      // S2 end time mean (µs)
  "t_s2_end_std": 1.2,        // S2 end timing spread
  "s2_duration_mean": 1.1,    // S2 duration mean (µs)
  "s2_duration_std": 0.15     // S2 duration variance
}
```

### Field & Transport Parameters
```json
{
  "drift_field": 250.0,       // V/cm
  "EL_field": 3000.0,         // V/cm
  "time_drift": 35.0,         // µs
  "speed_drift": 0.04,        // mm/µs
  "red_drift_field": 125.0    // Td
}
```

### Fit Results
```json
{
  "mean": 1.234,              // Gaussian fit mean (mV·µs)
  "sigma": 0.123,             // Gaussian width
  "ci95": 0.241,              // 95% confidence interval
  "fit_success": true
}
```

## Plots Generated

### Per-Set (in `plots/` directory)
1. **`{set_name}_s2_histogram.png`**
   - Ion recoil S2 area histogram
   - Gaussian fit overlay
   - Mean with 95% CI

2. **`{set_name}_waveform_validation.png`**
   - 10 random individual waveforms (NOT averaged)
   - Each in separate subplot
   - S1 time: red line + CI band
   - S2 start: blue line + CI band
   - S2 end: purple line + CI band

### Run-Level (in `plots/` directory)
3. **`{run_id}_xray_histogram.png`**
   - Combined X-ray areas from all sets
   - Gaussian fit for calibration

4. **`{run_id}_s2_vs_drift.png`**
   - Unnormalized S2 area vs drift field

5. **`{run_id}_s2_vs_drift_normalized.png`**
   - Normalized (A_ion / A_xray) vs drift field
   - Shows recombination effects

6. **`{run_id}_diffusion_analysis.png`**
   - Three panels showing S2 duration variance
   - Extracts transverse diffusion coefficient

## Usage Example

```python
from RaTag.pipeline import prepare_run, run_ion_integration, run_calibration_analysis

# Prepare with S2 window estimation
run = prepare_run(
    run,
    estimate_s2_windows=True,
    nfiles=None  # Process all files
)

# Ion integration: saves per-set histograms + waveform validation
fitted = run_ion_integration(
    run,
    use_estimated_s2_windows=True,
    flag_plot=True  # Generates and saves all plots
)

# Calibration: saves X-ray histogram + normalized results + diffusion
calib, recomb = run_calibration_analysis(
    run,
    ion_fitted_areas=fitted,
    flag_plot=True,
    save_plots=True
)
```

## Output Directory Structure

```
RUN8/
├── set1/
│   ├── *.wfm                              # Raw waveforms
│   ├── s2_areas.npy                       # Raw S2 areas
│   ├── s2_results.json                    # Complete metadata
│   ├── xray_areas.npy                     # X-ray areas
│   └── xray_results.json                  # X-ray classification
├── set2/
│   └── ...
└── plots/                                 # All diagnostic plots
    ├── set1_s2_histogram.png
    ├── set1_waveform_validation.png
    ├── set2_s2_histogram.png
    ├── set2_waveform_validation.png
    ├── RUN8_xray_histogram.png
    ├── RUN8_xray_areas_combined.npy
    ├── RUN8_xray_metadata.json
    ├── RUN8_s2_vs_drift.png
    ├── RUN8_s2_vs_drift_normalized.png
    └── RUN8_diffusion_analysis.png
```

## Key Features ✅

- ✅ **Complete metadata persistence** - All parameters logged to JSON
- ✅ **NPY arrays** for fast numerical processing
- ✅ **Publication-quality plots** (150 DPI PNG)
- ✅ **Waveform validation** - Individual waveforms with timing markers
- ✅ **Comprehensive logging** - t_s1, s2_start, s2_end, drift parameters
- ✅ **Modular workflow** - Can reload data without re-running pipeline
- ✅ **Automatic directory creation** - plots/ created automatically
- ✅ **Histogram and fits logged** - All per-set and run-level fits saved
- ✅ **Diffusion analysis** - Extracts transverse diffusion coefficient
- ✅ **Normalized results** - A_ion/A_xray for recombination analysis

## Documentation Created

1. **DATA_PERSISTENCE_GUIDE.md** - Complete reference for all data formats and plots
2. **PLOTTING_ENHANCEMENT_SUMMARY.md** - Summary of all changes made

## Validation Checklist

Before running on full dataset, test with `nfiles=10`:

- [ ] All JSON files contain expected fields
- [ ] Waveform validation plots show correct timing markers
- [ ] S2 histograms have Gaussian fits
- [ ] X-ray histogram combines all sets
- [ ] Diffusion analysis plots show three panels
- [ ] Normalized S2 plot shows recombination trend
- [ ] Directory structure is clean and organized

## Next Steps

1. Run pipeline on small test dataset (nfiles=10)
2. Inspect waveform validation plots to verify S2 windows
3. Check JSON files for completeness
4. Run full analysis and generate publication figures

## Notes

- All plots saved at 150 DPI (publication quality)
- Waveform validation shows **individual** frames (not averaged) for FastFrame data
- JSON files are human-readable for easy inspection
- NPY files provide fast array access for post-processing
- Complete parameter logging ensures reproducibility
