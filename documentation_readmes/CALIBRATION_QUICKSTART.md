# Quick Reference: Energy Calibration Workflow

## TL;DR

Raw alpha spectrum is in **SCA scale** (instrumental units), not MeV.

**Solution**: Fit peaks in SCA scale → Derive calibration using literature → Apply to get true energy.

## Minimal Working Example

```python
from RaTag.alphas.spectrum_fitting import (
    load_spectrum_from_run,
    fit_multi_crystalball_progressive,
    derive_energy_calibration,
    EnergyCalibration
)

# 1. Load spectrum (SCA scale)
spectrum = load_spectrum_from_run(run, energy_range=(4, 6), aggregate=True)
energies, counts = spectrum.select_roi()

# 2. Define peaks in SCA scale (as observed in raw data)
peak_definitions = [
    {'name': 'Th228', 'position': 4.6, 'window': (4.3, 4.8), 'sigma_init': 0.08},
    {'name': 'Ra224', 'position': 4.75, 'window': (4.5, 5.0), 'sigma_init': 0.08},
    {'name': 'Rn220', 'position': 5.3, 'window': (5.0, 5.6), 'sigma_init': 0.10},
    {'name': 'Po216', 'position': 5.75, 'window': (5.5, 6.1), 'sigma_init': 0.12}
]

# 3. Fit peaks in SCA scale
fit_results = fit_multi_crystalball_progressive(energies, counts, peak_definitions)

# 4. Derive energy calibration
literature_energies = {
    'Th228': 5.423,  # MeV
    'Ra224': 5.686,
    'Rn220': 6.405,
    'Po216': 6.906
}

calibration = derive_energy_calibration(fit_results, literature_energies)

# 5. Apply calibration
energies_calibrated = calibration.apply(energies)

# 6. Use calibrated energies for isotope separation
print(f"Calibration: E_true = {calibration.a:.5f} * E_SCA + {calibration.b:.5f}")
print(f"RMS residual: {calibration.residuals*1000:.1f} keV")
```

## Visual Summary

```
Raw Data (SCA scale)
        ↓
    [Fit peaks]  ← Progressive multi-peak fitting
        ↓
    Fitted positions (SCA)
        ↓
    [Derive calibration]  ← Compare with literature
        ↓
    E_true = a * E_SCA + b
        ↓
    [Apply to all energies]
        ↓
    Calibrated spectrum (MeV)
        ↓
    [Derive isotope ranges]
        ↓
    Ranges in true energy (MeV)
```

## Key Objects

### `EnergyCalibration`
```python
@dataclass(frozen=True)
class EnergyCalibration:
    a: float                                # Slope
    b: float                                # Intercept
    anchors: Dict[str, Tuple[float, float]] # {isotope: (E_SCA, E_true)}
    residuals: float                        # RMS error [MeV]
    
    def apply(self, E_SCA) -> E_true        # Forward transform
    def inverse(self, E_true) -> E_SCA      # Inverse transform
```

## Common Patterns

### Check Calibration Quality
```python
print(calibration)  # Shows equation, anchors, RMS

# Expected output:
# E_true = 1.14352 * E_SCA + -0.26831
# Anchors: ['Th228', 'Ra224', 'Rn220', 'Po216']
# RMS residual: 12.3 keV  ← Should be < 50 keV
```

### Convert SCA Windows to True Energy
```python
# Have: window in SCA scale (4.5, 5.0)
# Want: window in MeV

E_SCA_window = np.array([4.5, 5.0])
E_true_window = calibration.apply(E_SCA_window)
# → [4.89, 5.46] MeV
```

### Derive Calibrated Isotope Ranges
```python
from RaTag.alphas.spectrum_fitting import IsotopeRange

ranges_calibrated = {}
for name, fit_result in fit_results.items():
    # Get fitted params (SCA scale)
    x0_SCA = fit_result.params['cb_x0'].value
    sigma_SCA = fit_result.params['cb_sigma'].value
    
    # Convert to true energy
    x0_true = calibration.apply(np.array([x0_SCA]))[0]
    sigma_true = calibration.a * sigma_SCA
    
    # Define range (±2.5σ)
    E_min = x0_true - 2.5 * sigma_true
    E_max = x0_true + 2.5 * sigma_true
    
    ranges_calibrated[name] = IsotopeRange(
        name=name, E_min=E_min, E_max=E_max,
        E_peak=x0_true, sigma=sigma_true, purity=1.0
    )
```

## Literature Values (Quick Reference)

| Isotope | Energy (MeV) | Use for Calibration? |
|---------|--------------|---------------------|
| Th-228  | 5.423        | ✅ Yes (strong peak) |
| Ra-224  | 5.686        | ✅ Yes (clean peak)  |
| Rn-220  | 6.405        | ✅ Yes (good stats)  |
| Po-216  | 6.906        | ✅ Yes (well-separated) |
| Bi-212  | β decay      | ❌ No (β continuum, not α) |
| Po-212  | 8.785        | ⚠️  Maybe (needs α+β model) |

## Troubleshooting

### Problem: Large RMS residual (> 100 keV)

**Causes**:
- Incorrect peak identification
- Nonlinear energy response (rare for alphas)
- Poor fit quality

**Solutions**:
- Check fit visually (plot individual peaks)
- Use only 2-3 strongest peaks
- Verify literature values

### Problem: Calibration seems inverted (a < 0)

**Causes**:
- Swapped E_SCA ↔ E_true
- Incorrectly identified peaks

**Solutions**:
- Check peak_definitions match actual data
- Verify `position` values are in SCA scale (not MeV)

### Problem: Peaks don't align after calibration

**Causes**:
- Used wrong literature values
- Detector has nonlinear response

**Solutions**:
- Double-check literature values (NNDC database)
- Plot calibration residuals to check for systematic trends
- Consider position-dependent calibration (advanced)

## See Also

- **Full Documentation**: `ENERGY_CALIBRATION_GUIDE.md`
- **Demo Notebook**: `notebooks/spectrum_fitting_demo.ipynb`
- **Module**: `RaTag/alphas/spectrum_fitting.py`
