## ⚠️ Known Issue: High-Energy Shoulders on All Peaks

### Observation

All major alpha peaks (Th228, Ra224, Rn220, Po216) exhibit **systematic right-side shoulders** at approximately 70-100 keV higher than the main peak position:

- **Th228**: Main peak at 5.42 MeV (~40,000 counts), shoulder at ~5.5 MeV (~30,000 counts)
- **Ra224**: Main peak at 5.69 MeV, shoulder at ~5.8 MeV
- **Rn220**: Main peak at 6.40 MeV, shoulder at ~6.45 MeV  
- **Po216**: Main peak at 6.91 MeV, visible shoulder on right side

**Key characteristics:**
- Shoulder intensity: **~70-80% of main peak** (not a minor tail)
- Energy shift: **Constant ~80-100 keV** across all peaks
- Shoulders appear on **right side** (higher energy) only

### Ruled Out Causes

Through systematic investigation, the following potential causes have been **eliminated**:

1. ✅ **Savitzky-Golay filter artifacts**: Window size tests (201 to 5001 samples) show no change in shoulder structure
2. ✅ **Pulse shape variation**: Rise and decay times are constant within 0.1-0.2 μs  
3. ✅ **Calibration non-linearity**: Factor 1.058 is constant, verified across energy range
4. ✅ **S1/S2 pile-up**: Trigger is on alpha channel, events are isolated
5. ✅ **Detector geometry** (Si vs SiO2 layers): Commercial MCA does not show these shoulders
6. ✅ **Spatial dependence**: No correlation with detector position
7. ✅ **Charge collection effects**: Surface charging mitigation (gate voltage inversion) was applied

### Most Likely Cause: Baseline Subtraction Artifact

The prime suspect is **bimodal baseline distribution**:

**Hypothesis**: If the threshold-based baseline subtraction (`V[V < 0.3]`) produces two populations:
- **Population 1** (~30%): Correct baseline → main peak
- **Population 2** (~70%): Systematically offset baseline by ~0.1V → shoulder at +95 keV

This would explain:
- ✅ 70-80% relative intensity matches population fraction
- ✅ Constant ~100 keV shift = 0.1V / 1.058
- ✅ Affects all peaks equally (systematic baseline issue)
- ✅ Not seen in MCA (which has AC coupling and analog baseline restoration)

### Diagnostic Plan (Future Work)

To confirm baseline hypothesis:

1. **Raw voltage peak histogram** (no S-G filter, just `V_corrected.max()`)
   - If shoulders persist → baseline issue confirmed
   - If shoulders disappear → filter artifact (unlikely)

2. **Baseline distribution analysis**:
   ```python
   baselines = [np.mean(V[V < 0.3]) for V in all_waveforms]
   plt.hist(baselines)  # Look for bimodality
   ```

3. **Baseline vs Energy correlation**:
   - Scatter plot to identify two populations
   - Check if shoulder events have systematically different baselines

4. **Individual waveform inspection**:
   - Compare "main peak" vs "shoulder" waveforms
   - Examine pre-pulse region for baseline differences

### Impact on Analysis

**Current status**: Hierarchical fitting proceeds with shoulders treated as **unresolved structure**. The Crystal Ball model captures the overall peak shape including shoulders, so:

- ✅ Peak positions are still accurately determined (main peak dominates)
- ✅ Isotope ranges remain valid (capture full distribution)
- ⚠️ Energy resolution appears worse than intrinsic detector resolution
- ⚠️ Fitted σ(E) includes systematic broadening from shoulders

**Recommendation**: Resolve baseline issue before final energy calibration and resolution studies. For isotope tagging and activity determination, current approach is adequate.

---

## Appendix: Q-value vs Alpha Particle Energy

### Physical Distinction

When nuclear databases (e.g., NNDC) list **two different energies** for the same α decay (e.g., Bi-212 from NDS 108,1583 2007):

1. **Q-value = 6207.26 keV**: Total energy released in the decay
2. **α-particle energy = 6050.78 keV**: Kinetic energy of the emitted α particle
3. **Difference ≈ 157 keV**: Recoil energy of the daughter nucleus (Tl-208)

### Energy Conservation

$$E_Q = E_\alpha + E_\text{recoil}$$

The recoil energy is given by momentum conservation:

$$E_\text{recoil} = \frac{m_\alpha}{m_\alpha + m_\text{daughter}} \cdot E_Q \approx \frac{4}{212} \cdot 6207 \approx 117 \text{ keV}$$

### Why Thick Silicon Detectors Measure Q-value

In **thick silicon detectors** (like ours):
- Both the α particle **and** the recoiling daughter nucleus stop within the detector volume
- Total deposited energy ≈ $E_\alpha + E_\text{recoil} \approx E_Q$
- This is why we observe Bi-212 peaks at **~6.2 MeV** (Q-value) rather than 6.05 MeV (α-only)

### Detector Type Comparison

| Detector Type | What Stops | Measured Energy | Example (Bi-212) |
|---------------|------------|-----------------|------------------|
| **Thin detectors** | α escapes, recoil absorbed | $E_\text{recoil} \approx 150$ keV | 150 keV |
| **Thick detectors** | Both α and recoil stop | $E_Q$ | 6.2 MeV ✓ (our case) |
| **Spectrometers** | α selected, recoil ignored | $E_\alpha$ | 6.05 MeV |

This explains why our fitted Bi-212 energies (6.207 MeV main, 6.090 MeV satellite) correspond to Q-values rather than literature α-particle kinetic energies.

---