"""
Shared pytest fixtures for the gamma_spectra test suite.

Fixture hierarchy:
  - Path fixtures (session): hidex_data_dir, hidex_file_m{1,2,3}
  - Real DataFrame fixtures (session): raw_df_m{1,2,3}, rate_df_m{1,2,3}
  - Batch fixtures (session, cover all 3 files): raw_batches, rate_batches
  - Synthetic fixtures (function-scoped, no I/O): synthetic_spectrum,
    noisy_spectrum, synthetic_raw_df, synthetic_bateman_data

Session scope is used for expensive I/O and curve_fit operations so they
run only once per test session.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from RaTag.gamma_spectra.hidex_read import extract_hidex_raw_data
from RaTag.gamma_spectra.etl_hidex import (
    transform_spectra_to_rates,
    bateman_count_rate,
    LAMBDA_RA,
    LAMBDA_PB,
)
from RaTag.gamma_spectra.gamma_workflow import (
    ingest_hidex_directory,
    fit_batch_gamma_spectra,
)

# ── Absolute paths to raw test data ──────────────────────────────────────────

_HIDEX_DATA_DIR = Path("/Users/pabloherrero/sabat/RaTagging/scope_data/hidex_data")

_FILE_M1 = _HIDEX_DATA_DIR / "HidexAMG-Pablo_measurement-001-20260203-153747.xlsx"
_FILE_M2 = _HIDEX_DATA_DIR / "HidexAMG-Pablo_measurement_2-001-20260208-180513.xlsx"
_FILE_M3 = _HIDEX_DATA_DIR / "HidexAMG-Pablo_measurement_3-001-20260217-234347.xlsx"


# ── Path fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def hidex_data_dir() -> Path:
    """Path to the Hidex data directory. Skips if not found."""
    if not _HIDEX_DATA_DIR.exists():
        pytest.skip(f"Hidex data directory not found: {_HIDEX_DATA_DIR}")
    return _HIDEX_DATA_DIR


@pytest.fixture(scope="session")
def hidex_file_m1() -> Path:
    """Path to the first measurement file (Pablo_measurement, 2026-02-03)."""
    if not _FILE_M1.exists():
        pytest.skip(f"Test file not found: {_FILE_M1}")
    return _FILE_M1


@pytest.fixture(scope="session")
def hidex_file_m2() -> Path:
    """Path to the second measurement file (Pablo_measurement_2, 2026-02-08)."""
    if not _FILE_M2.exists():
        pytest.skip(f"Test file not found: {_FILE_M2}")
    return _FILE_M2


@pytest.fixture(scope="session")
def hidex_file_m3() -> Path:
    """Path to the third measurement file (Pablo_measurement_3, 2026-02-17)."""
    if not _FILE_M3.exists():
        pytest.skip(f"Test file not found: {_FILE_M3}")
    return _FILE_M3


# ── Raw DataFrame fixtures (one per Excel file) ───────────────────────────────

@pytest.fixture(scope="session")
def raw_df_m1(hidex_file_m1) -> pd.DataFrame:
    """Raw DataFrame extracted from measurement 1 (no fitting applied)."""
    return extract_hidex_raw_data(hidex_file_m1)


@pytest.fixture(scope="session")
def raw_df_m2(hidex_file_m2) -> pd.DataFrame:
    """Raw DataFrame extracted from measurement 2."""
    return extract_hidex_raw_data(hidex_file_m2)


@pytest.fixture(scope="session")
def raw_df_m3(hidex_file_m3) -> pd.DataFrame:
    """Raw DataFrame extracted from measurement 3."""
    return extract_hidex_raw_data(hidex_file_m3)


# ── Rate DataFrame fixtures (spectrum fitting applied) ────────────────────────

@pytest.fixture(scope="session")
def rate_df_m1(raw_df_m1) -> pd.DataFrame:
    """Fitted rate DataFrame for measurement 1 (expensive: runs curve_fit)."""
    return transform_spectra_to_rates(raw_df_m1)


@pytest.fixture(scope="session")
def rate_df_m2(raw_df_m2) -> pd.DataFrame:
    """Fitted rate DataFrame for measurement 2."""
    return transform_spectra_to_rates(raw_df_m2)


@pytest.fixture(scope="session")
def rate_df_m3(raw_df_m3) -> pd.DataFrame:
    """Fitted rate DataFrame for measurement 3."""
    return transform_spectra_to_rates(raw_df_m3)


# ── Batch fixtures (all 3 files together, matching workflow API) ──────────────

@pytest.fixture(scope="session")
def raw_batches(hidex_data_dir) -> dict:
    """
    Dict[batch_id, raw_df] — output of ingest_hidex_directory.
    Covers all three Excel files found in the data directory.
    """
    return ingest_hidex_directory(hidex_data_dir)


@pytest.fixture(scope="session")
def rate_batches(raw_batches) -> dict:
    """
    Dict[batch_id, rate_df] — output of fit_batch_gamma_spectra.
    Most expensive fixture: runs curve_fit on every spectrum in every batch.
    """
    return fit_batch_gamma_spectra(raw_batches)


# ── Synthetic (in-memory) fixtures — no file I/O, always available ────────────

# Parameters for the synthetic Gaussian spectrum
_N_CHANNELS = 2048
_TRUE_A     = 500.0
_TRUE_MU    = 220.0
_TRUE_SIGMA = 15.0
_TRUE_B     = 0.1
_TRUE_C     = 50.0

# Ra/Pb atom counts used for the Bateman fixture
_TRUE_N_RA = 5.0e9   # Ra-224 atoms at t = 0
_TRUE_N_PB = 1.0e8   # Pb-212 atoms at t = 0


@pytest.fixture
def synthetic_spectrum() -> np.ndarray:
    """
    Noiseless Gaussian peak on a linear background.
    True parameters: A=500, mu=220, sigma=15, b=0.1, c=50.
    Ideal for testing that mathematical functions recover exact parameters.
    """
    x = np.arange(_N_CHANNELS, dtype=float)
    return (
        _TRUE_A * np.exp(-0.5 * ((x - _TRUE_MU) / _TRUE_SIGMA) ** 2)
        + _TRUE_B * x + _TRUE_C
    )


@pytest.fixture
def noisy_spectrum(synthetic_spectrum) -> np.ndarray:
    """
    Gaussian peak with Poisson noise applied (seed=42 for reproducibility).
    Used to test fitter performance under realistic counting statistics.
    """
    rng = np.random.default_rng(42)
    return rng.poisson(synthetic_spectrum.clip(0)).astype(float)


@pytest.fixture
def synthetic_raw_df(synthetic_spectrum) -> pd.DataFrame:
    """
    Small DataFrame (5 rows) that mimics the output of extract_hidex_raw_data.
    ColA contains consecutive 12-hourly timestamps; all spectra are identical
    clean Gaussians. Suitable for testing transform_spectra_to_rates.
    """
    n_rows = 5
    base_time = pd.Timestamp("2026-02-03 15:37:47")
    timestamps = [base_time + pd.Timedelta(hours=i * 12) for i in range(n_rows)]
    return pd.DataFrame({
        "ColA":       timestamps,
        "VialNumber": list(range(1, n_rows + 1)),
        "ParamC":     [None] * n_rows,
        "ParamD":     [None] * n_rows,
        "ParamE":     [None] * n_rows,
        "Spectrum":   [synthetic_spectrum.copy() for _ in range(n_rows)],
        "SourceFile": ["synthetic_test.xlsx"] * n_rows,
    })


@pytest.fixture
def synthetic_bateman_data() -> tuple:
    """
    Returns (times_sec, count_rates, errors, true_n_ra, true_n_pb).

    Count rates are generated with the known forward Bateman model so that
    fit_initial_populations should recover the injected values within ~1% for
    Ra-224 and ~5% for Pb-212.
    """
    t = np.linspace(0, 7 * 24 * 3600, 20)   # 0 … 7 days, 20 evenly-spaced points
    rates = bateman_count_rate(t, _TRUE_N_RA, _TRUE_N_PB)
    errors = np.full_like(rates, rates.mean() * 0.05)  # 5 % flat uncertainty
    return t, rates, errors, _TRUE_N_RA, _TRUE_N_PB
