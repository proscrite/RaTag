# RaTag — Radium Tagging analysis

Short summary
---------------
RaTag is a modular Python analysis package developed to process waveform data from the RITA setup (PMT + silicon alpha detector). It provides reproducible pipelines to: prepare runs (S1/S2 timing and transport properties), build alpha energy maps and calibrations, extract and fit recoil S2 areas (single- and multi-isotope modes), identify and fit X-ray S2 peaks for detector calibration (g_S2), and compute electron recombination fractions vs drift field.

Key ideas
---------
- Data model: `Run`, `SetPmt`, `FrameProxy`, and waveform dataclasses track provenance and enable lazy loading of large waveform files.
- Pipelines: independent, composable stages (alpha calibration, preparation, recoil integration, X-ray classification, unified pipeline, recombination analysis) orchestrated by `scripts/run_analysis.py`.
- Multi-isotope support: alpha tagging correlates silicon-detector frames with PMT frames via UIDs and computed isotope ranges.

Dependencies
------------
See `pyproject.toml` for exact pins. The project requires Python 3.8+ and common scientific libs: numpy, scipy, matplotlib, pandas, lmfit, pytest.

Quick start / common commands
-----------------------------
1. Compute alpha energy maps (multi-isotope recommended first):

```bash
python RaTag/scripts/run_analysis.py configs/runXX_analysis.yaml --alphas-only
```

2. Run the full pipeline (uses computed isotope ranges if available):

```bash
python RaTag/scripts/run_analysis.py configs/runXX_analysis.yaml
```

3. Run only integration (assumes preparation done):

```bash
python RaTag/scripts/run_analysis.py configs/runXX_analysis.yaml --recoil-only
```

4. Force refit of calibration/fits without regenerating energy maps:

```bash
python RaTag/scripts/run_analysis.py configs/runXX_analysis.yaml --alphas-only --force-refit
```

Useful flags
------------
- `--prepare-only`: estimate S1/S2 windows and transport properties only.
- `--xray-only`: run X-ray classification stage only.
- `--only-unified`: run the combined X-ray + S2 unified pipeline for efficiency.
- `--use-yaml-ranges`: override computed isotope ranges with YAML-provided ranges.

Outputs and locations
---------------------
- Processed results and plots are saved under each run's `processed_data/` directory (examples: `spectrum_calibration/`, `{run_id}_s2_vs_drift.csv`, `{run_id}_recomb_factors.csv`, PNG diagnostic plots).
- Computed isotope ranges are stored as `{run_id}_isotope_ranges.npz` in `processed_data/spectrum_calibration/` and reused by downstream stages.

Where to look next
------------------
- `RaTag/core/` for datatypes and default configs (integration, fitting, alpha peak definitions).
- `RaTag/scripts/run_analysis.py` for the CLI and pipeline orchestration.

Configuration template
----------------------
A ready-to-use configuration template is provided at `RaTag/configs/run_analysis_template.yaml`. Copy it and adapt the paths and parameters for your run (for example: `cp RaTag/configs/run_analysis_template.yaml configs/run23_analysis.yaml`).

Raw data layout (required)
-------------------------
The raw data directory for a run must follow the set organization expected by the pipelines. In short:

- The root of a run contains one subdirectory per measurement set. Each set directory must be named with the pattern `FieldScan_GateXXX_AnodeYYY` (for example: `FieldScan_Gate200_Anode350`).
- Inside each set directory place the waveform files for that set. For single-isotope runs the silicon detector files use the `.ch4` naming conventions such as `Ch4_noSCA` and `Ch4_SCA_ZZZ` (these are silicon waveform files used by the alpha-analysis pipeline). PMT channel files live in the same set directory and follow your oscilloscope naming (e.g. `...Ch1.wfm` or FastFrame files).

Important: the following directories are produced by the pipelines and should NOT be required in the raw data layout before running the analysis — they will be generated under the run root as the pipelines execute:

- `energy_maps/` (binary maps of alpha energies)
- `plots/` (collection of diagnostic PNGs)
- `processed_data/` (CSV/NPZ/json outputs such as `{run_id}_isotope_ranges.npz`, `{run_id}_s2_vs_drift.csv`, `{run_id}_recomb_factors.csv`)

Example raw run root (what your disk should look like before running pipelines):

```
/Volumes/KINGSTON/RaTag_data/RUN23_Ra224/
├─ FieldScan_Gate200_Anode350/
│  ├─ PMT_Ch1_file1.wfm
│  ├─ PMT_Ch1_file2.wfm
│  ├─ Ch4_noSCA/           # directory containing raw silicon waveform files (no SCA)
│  │  ├─ ...
│  └─ Ch4_SCA_001/         # directory containing SCA-processed silicon frames
│     ├─ ...
├─ FieldScan_Gate250_Anode400/
│  ├─ ...
└─ FieldScan_Gate.../
```

If your raw data is organized differently, the `initialize_run` logic (used by `scripts/run_analysis.py`) may be adapted, but the recommended convention above is what the pipelines expect by default.


