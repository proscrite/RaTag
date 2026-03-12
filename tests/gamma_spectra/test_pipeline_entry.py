"""
Integration test for the top-level pipeline entrypoint `execute_gamma_pipeline`.

This test writes a minimal `gamma_measurements.yaml` pointing at the real
Hidex data directory, runs `execute_gamma_pipeline` with a temporary output
folder, and asserts that CSV artifacts were written. The test also exercises
our recently-added `suffix` parameter by ensuring exported filenames contain
`_SpectraFits.csv` (as `gamma_pipeline.py` requests).
"""

from pathlib import Path
import yaml
import pytest
import pandas as pd

from RaTag.gamma_spectra.gamma_pipeline import execute_gamma_pipeline


def test_execute_gamma_pipeline_writes_artifacts(hidex_data_dir, tmp_path):
    # Prepare a small config YAML in the temp directory
    cfg = {
        "data_dir": str(hidex_data_dir),
        "accumulation_params": {}
    }
    cfg_path = tmp_path / "gamma_measurements.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    out_dir = tmp_path / "artifacts"
    execute_gamma_pipeline(str(cfg_path), str(out_dir))

    # Verify CSV artifacts exist
    csv_files = list(out_dir.glob("*.csv"))
    assert len(csv_files) >= 1, "No CSV artifacts were created by the pipeline"

    # gamma_pipeline calls export_batch_artifacts(..., suffix="_SpectraFits")
    assert any(p.name.endswith("_SpectraFits.csv") for p in csv_files), (
        "Expected at least one CSV to end with _SpectraFits.csv"
    )
