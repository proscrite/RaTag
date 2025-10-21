# Quick Guide: New Plotting & Modular Features

## 🎨 New Feature 1: Automatic Plotting

### In `run_ion_integration()`
Add `flag_plot=True` to see:
1. Individual histograms with Gaussian fits for each set
2. Summary plot: Mean S2 area vs drift field

```python
ion_fitted = pipeline.run_ion_integration(
    run8,
    ts2_tol=-2.7,
    integration_config=ion_int_config,
    fit_config=ion_fit_config,
    flag_plot=True  # ← Enable plotting!
)
```

**Output**:
- Multiple matplotlib figures showing histograms + fits
- Final figure: S2 vs Drift Field with error bars

---

## 💾 New Feature 2: Modular Calibration (Load from Disk)

### Previously (All in Memory)
```python
# Had to keep everything in memory
ion_fitted = pipeline.run_ion_integration(run8, ...)
calib, recomb = pipeline.run_calibration_analysis(run8, ion_fitted)
```

### Now (Can Load from Disk)
```python
# Option 1: Provide fitted areas (original behavior)
calib, recomb = pipeline.run_calibration_analysis(
    run8, 
    ion_fitted_areas=ion_fitted  # Use in-memory data
)

# Option 2: Load from disk (NEW!)
calib, recomb = pipeline.run_calibration_analysis(
    run8,
    ion_fitted_areas=None  # Loads from disk automatically
)

# Option 3: Even simpler (None is default now)
calib, recomb = pipeline.run_calibration_analysis(run8)
```

---

## 📁 New Files Saved

After running `run_ion_integration()`, each set directory now contains:

```
FieldScan_Gate050_Anode1950/
├── *.wfm                    # Raw waveform files
├── s2_areas.npy             # Raw S2 areas (as before)
└── s2_results.json          # NEW! Fit results (mean, sigma, ci95, etc.)
```

**`s2_results.json` example**:
```json
{
  "method": "s2_area_pipeline",
  "params": {...},
  "mean": 2.456,
  "sigma": 0.823,
  "ci95": 0.15,
  "fit_success": true
}
```

---

## 🔄 Typical Workflows

### Workflow A: Quick Interactive Analysis
```python
# All in one go, with plots
run8 = pipeline.prepare_run(run8)
xray_results = pipeline.run_xray_classification(run8)
ion_fitted = pipeline.run_ion_integration(run8, flag_plot=True)
calib, recomb = pipeline.run_calibration_analysis(run8, ion_fitted)
```

### Workflow B: Process Data → Analyze Later
**Day 1: Data Processing**
```python
run8 = pipeline.prepare_run(run8)
xray_results = pipeline.run_xray_classification(run8)
ion_fitted = pipeline.run_ion_integration(run8)
# Close notebook, go home 🏠
```

**Day 2: Analysis & Plotting**
```python
# Reload run structure
run8 = pipeline.prepare_run(run8)

# Calibration loads fitted areas from disk
calib, recomb = pipeline.run_calibration_analysis(run8, flag_plot=True)

# Or manually load for custom plotting
from RaTag.dataIO import load_s2area
ion_fitted = {s.source_dir.name: load_s2area(s) for s in run8.sets}
plotting.plot_s2_vs_drift(run8, ion_fitted)
```

### Workflow C: Batch Processing with Plots
```python
# Process in batches, plot each batch
for i in range(0, len(run8.sets), 3):
    batch_slice = slice(i, i+3)
    
    ion_fitted = pipeline.run_ion_integration(
        run8, 
        range_sets=batch_slice,
        flag_plot=True  # Plot after each batch
    )
```

---

## 🎯 Use Cases

### Use Case 1: Check Fit Quality Visually
```python
# Add flag_plot=True to see if fits look good
ion_fitted = pipeline.run_ion_integration(
    run8, 
    fit_config=FitConfig(bin_cuts=(0, 10), nbins=100),
    flag_plot=True  # Visual QA
)
```

### Use Case 2: Run Calibration from Different Notebook
```python
# In a new notebook (different session):
from pathlib import Path
from RaTag.datatypes import Run
import RaTag.pipeline as pipeline

base_dir = Path('/Volumes/KINGSTON/RaTag_data/RUN8_EL2350Vcm_5GSsec')
run8 = Run(root_directory=base_dir, run_id="RUN8", ...)
run8 = pipeline.prepare_run(run8)

# No need to have ion_fitted in memory!
calib, recomb = pipeline.run_calibration_analysis(run8)
```

### Use Case 3: Custom Plotting After Processing
```python
# Load all fitted results
from RaTag.dataIO import load_s2area

ion_fitted = {}
for s in run8.sets:
    s2 = load_s2area(s)
    if s2.fit_success:
        ion_fitted[s.source_dir.name] = s2

# Custom plotting
import matplotlib.pyplot as plt
for set_id, fit in ion_fitted.items():
    plotting.plot_hist_fit(fit, bin_cuts=(0, 15))
    plt.savefig(f'figures/{set_id}_fit.png')
    plt.close()
```

---

## 🐛 Troubleshooting

### "FileNotFoundError: s2_results.json"
**Cause**: Trying to load results before running integration  
**Solution**: Run `pipeline.run_ion_integration()` first

### "No ion S2 fitted areas found"
**Cause**: JSON files don't exist or fit_success=False for all sets  
**Solution**: Check that integration completed successfully

### "KeyError: source_dir.name"
**Cause**: Mismatch between run.sets and saved directories  
**Solution**: Ensure `prepare_run()` loads same sets as integration

### Plots not showing in notebook
**Cause**: Matplotlib backend issue  
**Solution**: Add `%matplotlib inline` at top of notebook

---

## 📊 Plot Customization

### Change S2 vs Drift plot appearance
Edit `plotting.py`:
```python
def plot_s2_vs_drift(run: Run, fitted: dict[str, S2Areas]):
    # Customize marker style, colors, etc.
    plt.errorbar(..., 
                 fmt='s',           # square markers
                 markersize=10,     # bigger
                 capsize=5,
                 color='crimson')   # red color
```

### Save plots to file
```python
ion_fitted = pipeline.run_ion_integration(run8, flag_plot=False)

# Manual plotting with save
plotting.plot_s2_vs_drift(run8, ion_fitted)
plt.savefig('figures/s2_vs_drift.png', dpi=300, bbox_inches='tight')
```

---

## Summary of New Parameters

| Function | Parameter | Default | Effect |
|----------|-----------|---------|--------|
| `fit_run_s2()` | `flag_plot` | `False` | Show histogram+fit for each set |
| `run_ion_integration()` | `flag_plot` | `False` | Show histograms + S2 vs drift |
| `calibrate_and_analyze()` | `ion_fitted_areas` | `None` | If None, loads from disk |
| `run_calibration_analysis()` | `ion_fitted_areas` | `None` | If None, loads from disk |

---

Enjoy the enhanced pipeline! 🎉
