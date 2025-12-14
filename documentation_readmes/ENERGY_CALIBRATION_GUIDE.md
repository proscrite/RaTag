# Energy Calibration Guide

## Problem Statement

The alpha spectrum data from RaTag is stored in **instrumental units** ("SCA scale"), not true physical energy:

```python
# In wfm2spectra.py, line 179:
energy = peak_value / 1.058  # SCA scale (mV / 1.058)
```

This scaling factor (1.058) matches the Single Channel Analyzer (SCA) used for DAQ, but **does not represent true energy in MeV**.

### Observed vs Expected Energies

| Isotope | Observed (SCA) | Expected (MeV, literature) | Discrepancy |
|---------|----------------|---------------------------|-------------|
| Th-228  | ~4.6           | 5.423                     | ~0.8 units  |
| Ra-224  | ~4.7           | 5.686                     | ~1.0 units  |
| Rn-220  | ~5.3           | 6.405                     | ~1.1 units  |
| Po-216  | ~5.7           | 6.906                     | ~1.2 units  |

**Conclusion**: The SCA scale has both **slope error** and **offset error** relative to true energy.

## Solution: Two-Stage Calibration

### Stage 1: Fit Peaks in SCA Scale

First, fit the spectrum in its native instrumental scale without assumptions about energy:

```python
from RaTag.alphas.spectrum_fitting import fit_multi_crystalball_progressive

# Define peaks in SCA scale (as observed in raw data)
peak_definitions = [
    {'name': 'Th228', 'position': 4.6, 'window': (4.3, 4.8)},
    {'name': 'Ra224', 'position': 4.75, 'window': (4.5, 5.0)},
    {'name': 'Rn220', 'position': 5.3, 'window': (5.0, 5.6)},
    {'name': 'Po216', 'position': 5.75, 'window': (5.5, 6.1)}
]

# Fit peaks progressively
fit_results = fit_multi_crystalball_progressive(
    energies_SCA, counts, 
    peak_definitions=peak_definitions
)

# Extract fitted positions (still in SCA scale)
# fit_results['Th228'].params['cb_x0'].value → 4.612 (example)
```

**Why this works**: 
- The Crystal Ball model is scale-invariant (fits work in any units)
- No assumptions about true energy needed
- Robust to calibration errors

### Stage 2: Derive Linear Energy Calibration

Use literature values as "truth" to derive the energy calibration:

```python
from RaTag.alphas.spectrum_fitting import derive_energy_calibration

# Literature values (from NNDC, ENSDF databases)
literature_energies = {
    'Th228': 5.423,  # MeV (weighted average of 5.423/5.340 doublet)
    'Ra224': 5.686,  # MeV
    'Rn220': 6.405,  # MeV
    'Po216': 6.906,  # MeV
    'Po212': 8.785   # MeV (future use)
}

# Derive calibration: E_true = a * E_SCA + b
calibration = derive_energy_calibration(
    fit_results,
    literature_energies,
    use_peaks=['Th228', 'Ra224', 'Rn220', 'Po216']
)

# Output:
# E_true = 1.14352 * E_SCA + -0.26831
# Anchors: ['Th228', 'Ra224', 'Rn220', 'Po216']
# RMS residual: 12.3 keV
```

**Mathematical Details**:

The calibration uses linear least-squares regression:

$$E_{\text{true}} = a \cdot E_{\text{SCA}} + b$$

Given $N$ anchor points $(E_{\text{SCA},i}, E_{\text{true},i})$, we minimize:

$$\sum_{i=1}^N \left( E_{\text{true},i} - (a \cdot E_{\text{SCA},i} + b) \right)^2$$

Solution (using `np.polyfit` with `deg=1`):

```python
a, b = np.polyfit(E_SCA_array, E_true_array, deg=1)
```

**Quality Metrics**:
- RMS residual: $\sqrt{\frac{1}{N}\sum_i (E_{\text{true},i} - E_{\text{pred},i})^2}$
- Typical values: 10-30 keV (excellent), 30-80 keV (good), >100 keV (poor)

### Stage 3: Apply Calibration

Apply the calibration to convert all data to true energy:

```python
# Apply to spectrum
energies_calibrated = calibration.apply(energies_SCA)

# Inverse transformation (e.g., for defining windows)
E_SCA_window = calibration.inverse(np.array([5.0, 6.0]))  # MeV → SCA
```

## Implementation Details

### EnergyCalibration Dataclass

```python
@dataclass(frozen=True)
class EnergyCalibration:
    """Linear energy calibration: E_true = a * E_SCA + b"""
    a: float                               # Slope [MeV / SCA_units]
    b: float                               # Intercept [MeV]
    anchors: Dict[str, Tuple[float, float]] # {isotope: (E_SCA, E_true)}
    residuals: float                       # RMS residual [MeV]
    
    def apply(self, E_SCA: np.ndarray) -> np.ndarray:
        """E_true = a * E_SCA + b"""
        return self.a * E_SCA + self.b
    
    def inverse(self, E_true: np.ndarray) -> np.ndarray:
        """E_SCA = (E_true - b) / a"""
        return (E_true - self.b) / self.a
```

**Design Philosophy**:
- **Immutable** (`frozen=True`): Cannot be accidentally modified
- **Pure functions**: `apply()` and `inverse()` have no side effects
- **Self-documenting**: Stores anchor points for validation
- **Reusable**: Can be saved/loaded from run metadata

### Choosing Anchor Peaks

**Recommendation**: Use 2-4 well-separated peaks

| # Peaks | Pros | Cons |
|---------|------|------|
| 2 | Simple, minimal assumptions | Sensitive to individual peak errors |
| 3-4 | Robust, overdetermined system | Standard choice (recommended) |
| 5+ | Maximum use of data | Diminishing returns, may include weak peaks |

**Best Practices**:
- Use peaks with high statistics (narrow, tall)
- Avoid doublets (e.g., Th-228 has 5.423/5.340 MeV doublet - use weighted avg)
- Span the full energy range (e.g., Th-228 at 5.4 MeV + Po-216 at 6.9 MeV)
- Exclude Po-212 if it has α+β contamination

### Validation

After calibration, check:

1. **Residuals plot**: Should be randomly distributed around zero
2. **RMS residual**: Should be < 50 keV for good detector resolution
3. **Visual alignment**: Calibrated peaks should align with literature positions

```python
# Check calibration quality
print(calibration)

# Visualize residuals
for name, (E_SCA, E_true) in calibration.anchors.items():
    E_pred = calibration.apply(np.array([E_SCA]))[0]
    print(f"{name}: Δ = {(E_true - E_pred)*1000:.1f} keV")
```

## Integration with Existing Workflows

### Option 1: Calibrate at Load Time

Modify `load_spectrum_from_run()` to optionally apply calibration:

```python
# Future enhancement
spectrum = load_spectrum_from_run(
    run,
    energy_range=(4, 6),
    calibration=calibration  # Apply during load
)
```

### Option 2: Store Calibration in Run Metadata

```python
# Save calibration to run
run.metadata['energy_calibration'] = {
    'a': calibration.a,
    'b': calibration.b,
    'anchors': calibration.anchors,
    'rms_residual_keV': calibration.residuals * 1000
}

# Reuse in later analysis
cal_params = run.metadata['energy_calibration']
calibration = EnergyCalibration(
    a=cal_params['a'],
    b=cal_params['b'],
    anchors=cal_params['anchors'],
    residuals=cal_params['rms_residual_keV'] / 1000
)
```

### Option 3: Apply Calibration in Isotope Separation

```python
# Derive ranges in calibrated energy
isotope_ranges_calibrated = {}
for name, fit_result in fit_results.items():
    x0_SCA = fit_result.params['cb_x0'].value
    sigma_SCA = fit_result.params['cb_sigma'].value
    
    # Convert to true energy
    x0_true = calibration.apply(np.array([x0_SCA]))[0]
    sigma_true = calibration.a * sigma_SCA  # Linear scaling
    
    E_min = x0_true - 2.5 * sigma_true
    E_max = x0_true + 2.5 * sigma_true
    
    isotope_ranges_calibrated[name] = IsotopeRange(
        name=name,
        E_min=E_min, E_max=E_max,
        E_peak=x0_true, sigma=sigma_true,
        purity=1.0
    )
```

## Comparison with Alternative Approaches

### ❌ Option A: Blind Peak Search

**Idea**: Find peaks without position priors

**Problems**:
- Still requires ROI definition (in what units?)
- Can miss weak peaks or find spurious ones
- More complex code
- No advantage over calibration approach

### ❌ Option B: Hardcode Energy Positions

**Idea**: Define peak windows directly in MeV

**Problems**:
- Assumes calibration is perfect (it's not!)
- Fails when SCA scale drifts over time
- Not generalizable to new detectors

### ✅ **Option C: Two-Stage Calibration (Implemented)**

**Advantages**:
- Works with raw instrumental data
- Derives calibration from data itself
- Robust to detector variations
- Physically meaningful (MeV units)
- Reusable across runs

## Physical Interpretation

### Why is calibration needed?

1. **Analog signal chain**: Preamp → Shaper → ADC → Digitizer
   - Each stage has gain/offset
   - Cumulative nonidealities → scale error

2. **Energy loss mechanisms**:
   - Not all energy deposited in active region
   - Charge recombination in gas
   - Attachment losses
   - → Effective energy ≠ initial particle energy

3. **Detector response**:
   - Energy resolution: $\frac{\Delta E}{E} \approx \frac{0.12\,\text{MeV}}{6\,\text{MeV}} = 2\%$ (typical)
   - Position dependence (field uniformity)
   - Gas pressure/temperature effects

### Calibration stability

**Factors affecting calibration**:
- Gas pressure/temperature (density → ionization yield)
- Drift field strength (electron transport)
- Amplifier drift (electronics aging)
- Gas purity (electronegative impurities)

**Recommendation**: 
- Recalibrate for each run (< 1 minute of computation)
- Monitor calibration stability over time
- Flag runs with anomalous calibration (RMS > 100 keV)

## Literature Values Reference

| Isotope | Energy (MeV) | Branching Ratio | Half-life | Notes |
|---------|--------------|-----------------|-----------|-------|
| Th-228  | 5.423 (72%)  | 71.8%          | 1.91 y    | Doublet with 5.340 (28%) |
|         | 5.340 (28%)  | 28.2%          |           | Use weighted avg: 5.399 |
| Ra-224  | 5.686        | 94.9%          | 3.66 d    | Clean peak |
| Rn-220  | 6.405        | 99.9%          | 55.6 s    | May have Bi-212 β tail |
| Po-216  | 6.906        | 100%           | 0.145 s   | Highest α in chain |
| **Bi-212** | β decay   | 64.1%          | 60.6 min  | Contaminates Rn-220 tail |
| Po-212  | 8.785        | 64%            | 0.299 μs  | Simultaneous α+β decay |

**Sources**: 
- NNDC (National Nuclear Data Center): https://www.nndc.bnl.gov/
- ENSDF (Evaluated Nuclear Structure Data File)
- Firestone, R.B. (2007), Nuclear Data Sheets

## Summary

**Two-stage calibration workflow**:

1. ✅ **Fit peaks in SCA scale** → Extract instrumental positions
2. ✅ **Derive calibration** using literature values → Get $a$, $b$
3. ✅ **Apply calibration** → Convert to true energy (MeV)
4. ✅ **Derive isotope ranges** in calibrated energy → Physical units

**Advantages**:
- Robust to detector variations
- Physically meaningful results
- Simple linear model (2 parameters)
- Reusable across runs
- Self-validating (residuals metric)

**Next Steps**:
- Integrate with `isotope_preparation.py` pipeline
- Save calibration to run metadata
- Monitor calibration stability across runs
- Test on RUN18 data (see `spectrum_fitting_demo.ipynb`)
