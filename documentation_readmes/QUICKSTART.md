# Quick Start Guide - RaTag Analysis Pipeline

## What Was Created

Three files that streamline your analysis workflow:

1. **`calibration.py`** - Calibration functions (X-ray fitting, g_S2 calculation, recombination)
2. **`pipeline.py`** - High-level pipeline orchestrators (prepare, classify, integrate, calibrate)
3. **`run_analysis.ipynb`** - Clean notebook that uses the pipeline functions

## How to Use

### Step 1: Open the New Notebook
```bash
# Open the unified analysis notebook
jupyter notebook notebooks/run_analysis.ipynb
```

### Step 2: Test with Small Dataset First
The notebook includes examples for testing with limited data. Look for sections marked "Testing mode":

```python
# Example: Process only 2 sets with 5 files each
run8 = pipeline.prepare_run(run8, nfiles=5)

xray_results = pipeline.run_xray_classification(
    run8, 
    range_sets=slice(0, 2),  # Only first 2 sets
    nfiles=5                  # Only 5 files per set
)
```

### Step 3: Run Full Analysis
Once you verify the pipeline works:
- Remove `nfiles` parameters
- Remove `range_sets` restrictions
- Run all cells

## Key Features

### Testing Mode (`nfiles` parameter)
Every pipeline function accepts `nfiles` to limit data processing:
- **`prepare_run(..., nfiles=5)`** - Limit each set to 5 files during preparation
- **`run_xray_classification(..., nfiles=5)`** - Process only 5 files per set
- **`run_ion_integration(..., nfiles=10)`** - Process only 10 files per set

This lets you quickly test the pipeline without waiting for full processing.

### Range Selection (`range_sets` parameter)
Process specific subsets of sets:
```python
# Process only sets 0-2
range_sets = slice(0, 2)

# Process sets 4-7
range_sets = slice(4, 7)

# Process all sets
range_sets = None  # or omit the parameter
```

## Typical Workflow

```python
# 1. Define run
run8 = Run(root_directory=base_dir, run_id="RUN8", ...)

# 2. Prepare (estimate S1, compute fields, transport)
run8 = pipeline.prepare_run(run8)

# 3. Classify X-rays for calibration
xray_results = pipeline.run_xray_classification(run8, config=xray_config)

# 4. Integrate ion S2 signals
ion_fitted = pipeline.run_ion_integration(run8, integration_config=ion_int_config)

# 5. Calibrate and compute recombination
calib, recomb = pipeline.run_calibration_analysis(run8, ion_fitted)
```

## Comparison with Old Notebooks

| Old Approach | New Approach |
|--------------|--------------|
| 3 notebooks with test sections | 1 notebook with pipeline calls |
| ~500 lines of mixed code | ~200 lines of workflow + 560 lines of reusable modules |
| Manual result tracking | Automatic saving/loading |
| Copy-paste between notebooks | Function calls with clear parameters |
| Hard to test subsets | `nfiles` parameter for easy testing |

## Output Files

Results are automatically saved to each set's directory:
- **`xray_results.npy`** - X-ray classification results
- **`s2areas.npy`** - Ion S2 integration results

These are automatically loaded by `run_calibration_analysis()`.

## Troubleshooting

### S1 Time Estimation Issues
If automatic S1 estimation is off:
```python
# After prepare_run(), manually adjust:
run8.sets[2].metadata['t_s1'] = -3.32
run8.sets[4].metadata['t_s1'] = -3.32
```

### Verify S2 Windows
Use the plotting function to check windows are correct:
```python
plotting.plot_run_winS2(run8, ts2_tol=-2.7)
```

### Memory Issues
Use `nfiles` and `range_sets` to process in batches:
```python
# Process sets 0-3 with 20 files each
results1 = pipeline.run_ion_integration(run8, range_sets=slice(0, 3), nfiles=20)

# Process sets 3-6 with 20 files each
results2 = pipeline.run_ion_integration(run8, range_sets=slice(3, 6), nfiles=20)

# Combine results
results = {**results1, **results2}
```

## Next Steps

1. Open `run_analysis.ipynb`
2. Update `base_dir` to your data location
3. Run with `nfiles=5` for quick test
4. Verify plots look correct
5. Remove `nfiles` and run full analysis
6. Use results for publication!

For detailed documentation, see `PIPELINE_SUMMARY.md`.
