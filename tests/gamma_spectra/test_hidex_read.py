"""
Unit tests for hidex_read.py — the first stage of the pipeline.

These tests read each of the three real Hidex Excel files and verify the
shape, dtypes, and content constraints that downstream functions depend on.

All fixtures are session-scoped (see conftest.py), so the Excel files are
read only once per test session.
"""

import numpy as np
import pandas as pd
import pytest

from RaTag.gamma_spectra.hidex_read import extract_hidex_raw_data

N_CHANNELS = 2048
# The reader now provides a 'Datetime' column and may not include a numeric
# VialNumber depending on firmware/Excel export. Require ColA/Datetime,
# Spectrum and SourceFile instead.
REQUIRED_COLUMNS = {"ColA", "Datetime", "Spectrum", "SourceFile"}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _check_dataframe_invariants(df: pd.DataFrame, filename: str) -> None:
    """Shared assertions that every valid raw DataFrame must satisfy."""
    assert isinstance(df, pd.DataFrame), f"{filename}: result is not a DataFrame"
    assert len(df) > 0, f"{filename}: DataFrame is empty"
    assert REQUIRED_COLUMNS <= set(df.columns), (
        f"{filename}: missing columns {REQUIRED_COLUMNS - set(df.columns)}"
    )
    for spec in df["Spectrum"]:
        assert isinstance(spec, np.ndarray), f"{filename}: Spectrum is not ndarray"
        assert len(spec) == N_CHANNELS, (
            f"{filename}: spectrum has {len(spec)} channels, expected {N_CHANNELS}"
        )
        assert not np.any(np.isnan(spec)), f"{filename}: spectrum contains NaN"


# ── Tests against each individual file ───────────────────────────────────────

class TestExtractHidexRawDataM1:
    """Tests for the first measurement file (Pablo_measurement, 2026-02-03)."""

    def test_returns_non_empty_dataframe(self, raw_df_m1):
        assert isinstance(raw_df_m1, pd.DataFrame)
        assert len(raw_df_m1) > 0

    def test_has_required_columns(self, raw_df_m1):
        cols = set(raw_df_m1.columns)
        assert ("ColA" in cols or "Datetime" in cols), (
            f"{raw_df_m1['SourceFile'].iloc[0]}: missing time column (ColA/Datetime)"
        )
        assert "Spectrum" in cols, f"{raw_df_m1['SourceFile'].iloc[0]}: missing Spectrum"
        assert "SourceFile" in cols, f"{raw_df_m1['SourceFile'].iloc[0]}: missing SourceFile"

    def test_spectrum_is_numpy_array(self, raw_df_m1):
        for spec in raw_df_m1["Spectrum"]:
            assert isinstance(spec, np.ndarray)

    def test_spectrum_has_2048_channels(self, raw_df_m1):
        for spec in raw_df_m1["Spectrum"]:
            assert len(spec) == N_CHANNELS

    def test_no_nan_values_in_spectra(self, raw_df_m1):
        for spec in raw_df_m1["Spectrum"]:
            assert not np.any(np.isnan(spec))

    def test_source_file_matches_filename(self, raw_df_m1, hidex_file_m1):
        assert (raw_df_m1["SourceFile"] == hidex_file_m1.name).all()

    def test_vial_number_is_non_null(self, raw_df_m1):
        """VialNumber may be absent; if present it must be non-null."""
        if "VialNumber" in raw_df_m1.columns:
            assert raw_df_m1["VialNumber"].notna().all(), "VialNumber contains null values"

    def test_col_a_is_datetime_parseable(self, raw_df_m1):
        """ColA is later parsed by parse_timestamps; it must be datetime-like."""
        parsed = pd.to_datetime(raw_df_m1["ColA"], errors="coerce")
        assert not parsed.isna().any(), "ColA contains values that cannot be parsed as datetime"

    def test_spectra_are_non_negative(self, raw_df_m1):
        """Gamma counts should never be negative."""
        for spec in raw_df_m1["Spectrum"]:
            assert np.all(spec >= 0), "Some spectrum channels are negative"

    def test_full_invariants(self, raw_df_m1, hidex_file_m1):
        _check_dataframe_invariants(raw_df_m1, hidex_file_m1.name)


class TestExtractHidexRawDataM2:
    """Tests for the second measurement file (Pablo_measurement_2, 2026-02-08)."""

    def test_full_invariants(self, raw_df_m2, hidex_file_m2):
        _check_dataframe_invariants(raw_df_m2, hidex_file_m2.name)

    def test_source_file_matches_filename(self, raw_df_m2, hidex_file_m2):
        assert (raw_df_m2["SourceFile"] == hidex_file_m2.name).all()

    def test_col_a_is_datetime_parseable(self, raw_df_m2):
        parsed = pd.to_datetime(raw_df_m2["ColA"], errors="coerce")
        assert not parsed.isna().any()


class TestExtractHidexRawDataM3:
    """Tests for the third measurement file (Pablo_measurement_3, 2026-02-17)."""

    def test_full_invariants(self, raw_df_m3, hidex_file_m3):
        _check_dataframe_invariants(raw_df_m3, hidex_file_m3.name)

    def test_source_file_matches_filename(self, raw_df_m3, hidex_file_m3):
        assert (raw_df_m3["SourceFile"] == hidex_file_m3.name).all()

    def test_col_a_is_datetime_parseable(self, raw_df_m3):
        parsed = pd.to_datetime(raw_df_m3["ColA"], errors="coerce")
        assert not parsed.isna().any()


# ── Cross-file consistency checks ─────────────────────────────────────────────

class TestCrossFileconsistency:
    """Verify that all three files produce structurally compatible DataFrames."""

    def test_all_files_have_same_spectrum_length(self, raw_df_m1, raw_df_m2, raw_df_m3):
        for df in (raw_df_m1, raw_df_m2, raw_df_m3):
            for spec in df["Spectrum"]:
                assert len(spec) == N_CHANNELS

    def test_all_files_have_same_columns(self, raw_df_m1, raw_df_m2, raw_df_m3):
        cols1 = set(raw_df_m1.columns)
        cols2 = set(raw_df_m2.columns)
        cols3 = set(raw_df_m3.columns)
        assert cols1 == cols2 == cols3, (
            f"Column sets differ: {cols1} vs {cols2} vs {cols3}"
        )

    def test_source_files_are_all_distinct(self, raw_df_m1, raw_df_m2, raw_df_m3):
        names = {
            raw_df_m1["SourceFile"].iloc[0],
            raw_df_m2["SourceFile"].iloc[0],
            raw_df_m3["SourceFile"].iloc[0],
        }
        assert len(names) == 3, f"Expected 3 distinct source files, got: {names}"


# ── Edge-case / robustness ────────────────────────────────────────────────────

class TestExtractHidexEdgeCases:
    def test_missing_file_raises_file_not_found(self, tmp_path):
        missing = tmp_path / "nonexistent_file.xlsx"
        with pytest.raises(Exception):   # openpyxl raises FileNotFoundError or similar
            extract_hidex_raw_data(missing)
