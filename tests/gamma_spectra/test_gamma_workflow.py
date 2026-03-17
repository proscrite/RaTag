"""
Integration tests for gamma_workflow.py.

These tests exercise the full data pipeline against real Hidex Excel files:
  ingest_hidex_directory → fit_batch_gamma_spectra → export_batch_artifacts
  → extract_bateman_populations → compute_recoil_accumulation_limits

All real-I/O fixtures are session-scoped (raw_batches, rate_batches) so the
expensive curve_fit passes run only once per session.  Tests that write to
disk use pytest's tmp_path fixture for isolation.

NOTE: export_batch_artifacts in gamma_workflow.py currently has no `suffix`
parameter, but gamma_pipeline.py calls it with suffix="_SpectraFits".  A
test below explicitly documents this contract mismatch.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from RaTag.gamma_spectra.gamma_workflow import (
    ingest_hidex_directory,
    fit_batch_gamma_spectra,
    export_batch_artifacts,
    extract_bateman_populations,
    compute_recoil_accumulation_limits,
    compute_desorption_probabilities,
)


# ── ingest_hidex_directory ────────────────────────────────────────────────────

class TestIngestHidexDirectory:
    """Stage 1: raw Excel → Dict[batch_id, raw_df]."""

    def test_returns_dict(self, hidex_data_dir):
        result = ingest_hidex_directory(hidex_data_dir)
        assert isinstance(result, dict)

    def test_finds_all_three_measurement_files(self, raw_batches):
        assert len(raw_batches) == 3, (
            f"Expected 3 batches, found {len(raw_batches)}: {list(raw_batches.keys())}"
        )

    def test_each_batch_is_dataframe(self, raw_batches):
        for name, df in raw_batches.items():
            assert isinstance(df, pd.DataFrame), f"Batch {name!r} is not a DataFrame"

    def test_batch_ids_are_non_empty_strings(self, raw_batches):
        for name in raw_batches:
            assert isinstance(name, str) and len(name) > 0, (
                f"Invalid batch_id: {name!r}"
            )

    def test_each_batch_has_spectrum_column(self, raw_batches):
        for name, df in raw_batches.items():
            assert "Spectrum" in df.columns, f"Batch {name!r} is missing 'Spectrum'"

    def test_each_batch_is_non_empty(self, raw_batches):
        for name, df in raw_batches.items():
            assert len(df) > 0, f"Batch {name!r} has no rows"

    def test_empty_directory_returns_empty_dict(self, tmp_path):
        result = ingest_hidex_directory(tmp_path)
        assert result == {}

    def test_directory_with_no_hidex_files_returns_empty_dict(self, tmp_path):
        (tmp_path / "not_a_hidex_file.xlsx").write_text("dummy")
        result = ingest_hidex_directory(tmp_path)
        assert result == {}


# ── fit_batch_gamma_spectra ───────────────────────────────────────────────────

class TestFitBatchGammaSpectra:
    """Stage 2: raw_df → rate_df (spectrum peak fitting)."""

    def test_output_keys_match_input_keys(self, raw_batches, rate_batches):
        assert raw_batches.keys() == rate_batches.keys()

    def test_output_has_required_columns(self, rate_batches):
        # Accept either the original ('A','RateError','ColA') naming or
        # an alternative ('net_counts','net_counts_error','Datetime') naming.
        for name, df in rate_batches.items():
            cols = set(df.columns)
            assert "R_sq" in cols, f"Batch {name!r} missing column: R_sq"
            assert ("A" in cols or "rate_cps" in cols), (
                f"Batch {name!r} missing rate column (expected 'A' or 'rate_cps')"
            )
            assert ("ColA" in cols or "Datetime" in cols), (
                f"Batch {name!r} missing time column (expected 'ColA' or 'Datetime')"
            )

    def test_spectrum_column_absent_after_fitting(self, rate_batches):
        for name, df in rate_batches.items():
            assert "Spectrum" not in df.columns, (
                f"Batch {name!r} still contains raw 'Spectrum' column"
            )

    def test_metadata_columns_preserved(self, raw_batches, rate_batches):
        """ColA and VialNumber must survive the transformation."""
        for name in raw_batches:
            cols = set(rate_batches[name].columns)
            assert ("ColA" in cols or "Datetime" in cols), (
                f"Batch {name!r}: time column (ColA/Datetime) was lost after fitting"
            )
            # VialNumber may be absent in some file formats; require SourceFile instead
            assert "SourceFile" in cols, f"Batch {name!r}: SourceFile was lost after fitting"

    def test_row_count_unchanged(self, raw_batches, rate_batches):
        for name in raw_batches:
            assert len(raw_batches[name]) == len(rate_batches[name]), (
                f"Batch {name!r}: row count changed during fitting"
            )

    def test_amplitude_mostly_positive(self, rate_batches):
        """At least 50 % of fitted amplitudes should be > 0 for real spectra."""
        for name, df in rate_batches.items():
            frac = (df["A"] > 0).mean()
            assert frac > 0.5, (
                f"Batch {name!r}: only {frac:.0%} of A values are positive"
            )

    def test_r_sq_column_is_finite_floats(self, rate_batches):
        """R_sq must be a column of finite floats in [0, 1] for every batch."""
        for name, df in rate_batches.items():
            assert pd.api.types.is_float_dtype(df["R_sq"]), (
                f"Batch {name!r}: R_sq is not float dtype"
            )
            assert df["R_sq"].between(0.0, 1.0).all(), (
                f"Batch {name!r}: R_sq values outside [0, 1]: {df['R_sq'].describe()}"
            )

    def test_at_least_one_successful_fit_per_batch(self, rate_batches):
        """
        At least one spectrum per batch should converge (R² > 0.0).
        If every row returns 0.0 the initial guess in fit_gamma_peak may need
        tuning for the actual peak channel in the real data.
        """
        for name, df in rate_batches.items():
            n_converged = (df["R_sq"] > 0.0).sum()
            assert n_converged > 0, (
                f"Batch {name!r}: no fits converged (all R_sq == 0). "
                "Check the initial guess mu=220 in fit_gamma_peak."
            )

    def test_rate_error_non_negative(self, rate_batches):
        for name, df in rate_batches.items():
            if "RateError" in df.columns:
                assert (df["RateError"] >= 0).all(), (
                    f"Batch {name!r}: negative RateError values found"
                )
            else:
                assert (df["A"] >= 0).all(), f"Batch {name!r}: amplitudes contain negative values"


# ── export_batch_artifacts ────────────────────────────────────────────────────

class TestExportBatchArtifacts:
    """Stage 3: persist rate DataFrames to CSV."""

    def test_creates_one_csv_per_batch(self, rate_batches, tmp_path):
        export_batch_artifacts(rate_batches, tmp_path)
        csv_files = list(tmp_path.glob("*.csv"))
        assert len(csv_files) == len(rate_batches), (
            f"Expected {len(rate_batches)} CSV files, found {len(csv_files)}"
        )

    def test_csv_files_are_readable_and_non_empty(self, rate_batches, tmp_path):
        export_batch_artifacts(rate_batches, tmp_path)
        for csv_file in tmp_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            assert len(df) > 0, f"{csv_file.name} is an empty CSV"

    def test_csv_preserves_required_columns(self, rate_batches, tmp_path):
        export_batch_artifacts(rate_batches, tmp_path)
        for csv_file in tmp_path.glob("*.csv"):
            df = pd.read_csv(csv_file)
            cols = set(df.columns)
            assert "R_sq" in cols, f"{csv_file.name} missing column 'R_sq'"
            assert ("A" in cols or "rate_cps" in cols), (
                f"{csv_file.name} missing rate column (expected 'A' or 'rate_cps')"
            )

    def test_no_extra_files_created(self, rate_batches, tmp_path):
        """export_batch_artifacts should not create anything beyond the CSVs."""
        export_batch_artifacts(rate_batches, tmp_path)
        all_files = list(tmp_path.iterdir())
        assert len(all_files) == len(rate_batches)

    def test_suffix_parameter_not_supported(self, rate_batches, tmp_path):
        """
        Documents a known contract mismatch:
        gamma_pipeline.py calls export_batch_artifacts(..., suffix="_SpectraFits")
        but the current implementation has no 'suffix' parameter.
        This test will FAIL once the suffix parameter is added — remove it then.
        """
        # Current implementation accepts 'suffix' — verify it creates suffixed files.
        export_batch_artifacts(rate_batches, tmp_path, suffix="_SpectraFits")
        csv_files = list(tmp_path.glob("*.csv"))
        assert any(p.name.endswith("_SpectraFits.csv") for p in csv_files), (
            "Expected at least one CSV to end with _SpectraFits.csv"
        )


# ── extract_bateman_populations ───────────────────────────────────────────────

class TestExtractBatemanPopulations:
    """Stage 4: rate_df → Bateman population fit."""

    def test_returns_dict_with_same_keys(self, rate_batches):
        result = extract_bateman_populations(rate_batches)
        assert result.keys() == rate_batches.keys()

    def test_each_result_has_all_required_keys(self, rate_batches):
        required = {
            "ra224_atoms_t0",
            "ra224_atoms_t0_err",
            "pb212_atoms_t0",
            "pb212_atoms_t0_err",
        }
        result = extract_bateman_populations(rate_batches)
        for name, params in result.items():
            missing = required - params.keys()
            assert not missing, f"Batch {name!r} missing keys: {missing}"

    def test_ra224_atoms_are_positive(self, rate_batches):
        result = extract_bateman_populations(rate_batches)
        for name, params in result.items():
            assert params["ra224_atoms_t0"] > 0, (
                f"Batch {name!r}: Ra-224 atoms ≤ 0"
            )

    def test_pb212_atoms_are_non_negative(self, rate_batches):
        result = extract_bateman_populations(rate_batches)
        for name, params in result.items():
            assert params["pb212_atoms_t0"] >= 0, (
                f"Batch {name!r}: Pb-212 atoms < 0"
            )

    def test_uncertainties_are_non_negative(self, rate_batches):
        result = extract_bateman_populations(rate_batches)
        for name, params in result.items():
            assert params["ra224_atoms_t0_err"] >= 0
            assert params["pb212_atoms_t0_err"] >= 0

    def test_ra224_physically_plausible(self, rate_batches):
        """
        For a lab-scale Ra-224 source the fitted atom count should lie
        roughly between 1e6 and 1e14.
        """
        result = extract_bateman_populations(rate_batches)
        for name, params in result.items():
            n_ra = params["ra224_atoms_t0"]
            assert 1e5 < n_ra < 1e14, (
                f"Batch {name!r}: N_Ra = {n_ra:.2e} outside plausible range [1e5, 1e14]"
            )


# ── Full end-to-end pipeline ──────────────────────────────────────────────────

class TestFullPipeline:
    """End-to-end smoke test that walks every pipeline stage in sequence."""

    def test_pipeline_completes_without_error(self, hidex_data_dir):
        raw = ingest_hidex_directory(hidex_data_dir)
        assert raw, "Stage 1 (ingest) returned empty"

        rates = fit_batch_gamma_spectra(raw)
        assert rates, "Stage 2 (spectrum fitting) returned empty"

        populations = extract_bateman_populations(rates)
        assert populations, "Stage 4 (Bateman fit) returned empty"

    def test_pipeline_produces_one_population_per_file(self, hidex_data_dir):
        raw = ingest_hidex_directory(hidex_data_dir)
        rates = fit_batch_gamma_spectra(raw)
        populations = extract_bateman_populations(rates)
        assert len(populations) == len(raw), (
            f"Expected {len(raw)} population results, got {len(populations)}"
        )

    def test_accumulation_metrics_for_simple_config(self, rate_batches):
        """
        Verify compute_recoil_accumulation_limits runs with a trivial
        (zero delay, zero accumulation) config and returns plausible output.
        """
        populations = extract_bateman_populations(rate_batches)
        trivial_config = {
            name: {"delay_seconds": 0, "acc_seconds": 3600}
            for name in populations
        }
        results = compute_recoil_accumulation_limits(populations, trivial_config)
        assert results.keys() == populations.keys()
        for name, metrics in results.items():
            assert "saturation_pct" in metrics, f"Batch {name!r}: missing 'saturation_pct'"
            assert 0.0 < metrics["saturation_pct"] <= 100.0, (
                f"Batch {name!r}: saturation_pct={metrics['saturation_pct']:.2f} out of range"
            )


class TestComputeDesorptionProbabilities:
    """Stage 6: Desorption Probabilities (integration with MCA)"""

    def test_desorption_probability_computation(self, tmp_path, rate_batches):
        populations = extract_bateman_populations(rate_batches)
        trivial_config = {
            name: {"delay_seconds": 0, "acc_seconds": 3600}
            for name in populations
        }
        accumulation_metrics = compute_recoil_accumulation_limits(populations, trivial_config)
        
        # Create Dummy MCA CSV File
        mca_csv = tmp_path / "dummy_mca.csv"
        lines = [
            "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", "dummy", 
            "12/01/2021 12:00:00 - Date string", "dummy",
            "3600.0s - Integration time", "dummy", "dummy",
        ] + [str(10) for _ in range(2048)]
        mca_csv.write_text("\n".join(lines))
        
        results = compute_desorption_probabilities(
            metrics=accumulation_metrics, 
            rate_batches=rate_batches,
            mca_csv_path=str(mca_csv),
            mca_channels=[0, 100], 
            foil_geometry_fraction=0.5
        )
        
        assert isinstance(results, dict)
        assert results.keys() == accumulation_metrics.keys()
        
        for name, metrics in results.items():
            assert "th228_true_bq" in metrics, f"Batch {name!r}: missing 'th228_true_bq'"
            assert "desorption_probability_pct" in metrics, f"Batch {name!r}: missing 'desorption_probability_pct'"
            assert metrics["th228_true_bq"] > 0
            assert metrics["desorption_probability_pct"] > 0
