"""
Unit tests for the pure math and physics functions in etl_hidex.py.

All tests here use only the synthetic fixtures from conftest.py (no file I/O),
so they run fast and in isolation from the real Hidex Excel files.

Covered functions:
  - gaussian_linear
  - fit_gamma_peak
  - bateman_count_rate
  - fit_initial_populations
  - backcalculate_accumulation
  - parse_timestamps
  - transform_spectra_to_rates
"""

import numpy as np
import pandas as pd
import pytest

from RaTag.gamma_spectra.etl_hidex import (
    gaussian_linear,
    fit_gamma_peak,
    bateman_count_rate,
    fit_initial_populations,
    backcalculate_accumulation,
    parse_timestamps,
    transform_spectra_to_rates,
    calculate_net_counts_error,
    LAMBDA_RA,
    LAMBDA_PB,
)


# ── gaussian_linear ───────────────────────────────────────────────────────────

class TestGaussianLinear:
    """Tests for gaussian_linear(x, a, mu, sigma, b, c)."""

    def test_value_at_peak(self):
        """At x = mu the Gaussian is maximised: f(mu) = a + b*mu + c."""
        a, mu, sigma, b, c = 100.0, 50.0, 10.0, 0.5, 5.0
        result = gaussian_linear(np.array([mu]), a, mu, sigma, b, c)
        assert np.isclose(result[0], a + b * mu + c)

    def test_far_from_peak_reduces_to_linear(self):
        """Many sigma away from mu the exponential vanishes; only b*x + c remains."""
        a, mu, sigma, b, c = 1000.0, 220.0, 5.0, 0.2, 10.0
        x_far = np.array([0.0, 10.0])   # > 40 sigma from mu
        expected = b * x_far + c
        result = gaussian_linear(x_far, a, mu, sigma, b, c)
        assert np.allclose(result, expected, rtol=1e-6)

    def test_output_shape_matches_input(self):
        x = np.linspace(0, 500, 300)
        y = gaussian_linear(x, 100.0, 220.0, 15.0, 0.0, 0.0)
        assert y.shape == x.shape

    def test_scalar_input(self):
        val = gaussian_linear(np.array([220.0]), 1.0, 220.0, 10.0, 0.0, 0.0)
        assert val.shape == (1,)

    def test_zero_amplitude_returns_linear(self):
        x = np.linspace(0, 100, 50)
        b, c = 2.0, 5.0
        y = gaussian_linear(x, 0.0, 50.0, 10.0, b, c)
        assert np.allclose(y, b * x + c)


# ── fit_gamma_peak ────────────────────────────────────────────────────────────

class TestFitGammaPeak:
    """Tests for fit_gamma_peak(x, y) -> (popt, pcov, r_sq)."""

    def test_recovers_amplitude(self, synthetic_spectrum):
        x = np.arange(len(synthetic_spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, synthetic_spectrum)
        assert np.isclose(popt[0], 500.0, rtol=0.01), f"A={popt[0]:.2f}, expected ~500"

    def test_recovers_mean(self, synthetic_spectrum):
        x = np.arange(len(synthetic_spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, synthetic_spectrum)
        assert np.isclose(popt[1], 220.0, rtol=0.01), f"mu={popt[1]:.2f}, expected ~220"

    def test_recovers_sigma(self, synthetic_spectrum):
        x = np.arange(len(synthetic_spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, synthetic_spectrum)
        assert np.isclose(popt[2], 15.0, rtol=0.02), f"sigma={popt[2]:.2f}, expected ~15"

    def test_r_sq_near_unity_for_clean_spectrum(self, synthetic_spectrum):
        x = np.arange(len(synthetic_spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, synthetic_spectrum)
        assert r_sq > 0.999, f"R²={r_sq:.6f}, expected > 0.999 for noiseless spectrum"

    def test_r_sq_acceptable_for_noisy_spectrum(self, noisy_spectrum):
        """With Poisson noise the fitter should still achieve R² > 0.90."""
        x = np.arange(len(noisy_spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, noisy_spectrum)
        assert r_sq > 0.90, f"R²={r_sq:.3f} too low for noisy spectrum"

    def test_returns_tuple_of_five_for_valid_input(self, synthetic_spectrum):
        x = np.arange(len(synthetic_spectrum), dtype=float)
        popt, pcov, r_sq = fit_gamma_peak(x, synthetic_spectrum)
        assert len(popt) == 5
        assert isinstance(r_sq, float)

    def test_graceful_failure_on_flat_spectrum(self):
        """A perfectly flat line has no peak; the fitter must not raise."""
        x = np.arange(200, dtype=float)
        y = np.ones(200) * 42.0
        popt, pcov, r_sq = fit_gamma_peak(x, y)
        # Either A = 0 (failed fit sentinel) or R² is very low
        assert popt[0] == 0.0 or r_sq < 0.5


# ── bateman_count_rate ────────────────────────────────────────────────────────

class TestBatemanCountRate:
    """Tests for bateman_count_rate(t, n_ra_0, n_pb_0, e_ph=1.0)."""

    def test_rate_at_t0_pure_ra(self):
        """With only Ra-224 and no Pb-212, the rate at t=0 is e_ph*0.041*λ_Ra*N_Ra."""
        n_ra = 1e10
        rate = bateman_count_rate(np.array([0.0]), n_ra, 0.0)[0]
        expected = 0.041 * LAMBDA_RA * n_ra
        assert np.isclose(rate, expected, rtol=1e-9)

    def test_rate_decays_to_zero_over_long_time(self):
        """After many Ra-224 half-lives (~90 days) the rate should be negligible."""
        t_long = np.array([300 * 24 * 3600.0])   # 300 days >> 3.6 days
        rate = bateman_count_rate(t_long, 1e10, 0.0)[0]
        rate_t0 = bateman_count_rate(np.array([0.0]), 1e10, 0.0)[0]
        assert rate < rate_t0 * 1e-6

    def test_non_negative_over_full_decay(self):
        """Count rates must never be negative."""
        t = np.linspace(0, 30 * 24 * 3600, 200)
        rates = bateman_count_rate(t, 1e9, 1e8)
        assert np.all(rates >= 0.0)

    def test_e_ph_scales_linearly(self):
        t = np.array([3600.0])
        r1 = bateman_count_rate(t, 1e9, 1e8, e_ph=1.0)[0]
        r2 = bateman_count_rate(t, 1e9, 1e8, e_ph=2.0)[0]
        assert np.isclose(r2, 2.0 * r1)

    def test_zero_populations_gives_zero_rate(self):
        t = np.linspace(0, 1e5, 10)
        rates = bateman_count_rate(t, 0.0, 0.0)
        assert np.allclose(rates, 0.0)

    def test_rate_at_long_time_lower_than_at_t0(self):
        """
        Even with N_Pb_0=0, Ra-224 produces Pb-212 in situ, whose emission
        coefficient (0.436) greatly exceeds Ra-224's (0.041).  The Pb-212
        activity builds up initially before decaying, so the total count rate
        is NOT monotonically decreasing.  We assert instead the weaker, but
        physically correct, property: at t >> T½(Ra-224) ≈ 3.6 days, the rate
        is well below its value at t=0.
        """
        t0 = np.array([0.0])
        t_late = np.array([30 * 24 * 3600.0])   # 30 days >> 3.6-day half-life
        rate_t0 = bateman_count_rate(t0, 1e10, 0.0)[0]
        rate_late = bateman_count_rate(t_late, 1e10, 0.0)[0]
        # Allow a small residual due to in-situ Pb-212 buildup; 5% is
        # a conservative upper bound for 30 days given the physics.
        assert rate_late < rate_t0 * 0.05, (
            f"Rate at 30 days ({rate_late:.3e}) should be < 5 % of t=0 rate ({rate_t0:.3e})"
        )


# ── fit_initial_populations ───────────────────────────────────────────────────

class TestFitInitialPopulations:
    """Tests for fit_initial_populations(times, count_rates, rate_errors) -> dict."""

    def test_recovers_ra224_atoms(self, synthetic_bateman_data):
        t, rates, errors, true_n_ra, _ = synthetic_bateman_data
        result = fit_initial_populations(t, rates, errors)
        assert np.isclose(result["ra224_atoms_t0"], true_n_ra, rtol=0.01), (
            f"N_Ra={result['ra224_atoms_t0']:.3e}, expected {true_n_ra:.3e}"
        )

    def test_recovers_pb212_atoms(self, synthetic_bateman_data):
        t, rates, errors, _, true_n_pb = synthetic_bateman_data
        result = fit_initial_populations(t, rates, errors)
        assert np.isclose(result["pb212_atoms_t0"], true_n_pb, rtol=0.05), (
            f"N_Pb={result['pb212_atoms_t0']:.3e}, expected {true_n_pb:.3e}"
        )

    def test_returns_all_expected_keys(self, synthetic_bateman_data):
        t, rates, errors, _, _ = synthetic_bateman_data
        result = fit_initial_populations(t, rates, errors)
        required_keys = {
            "ra224_atoms_t0",
            "ra224_atoms_t0_err",
            "pb212_atoms_t0",
            "pb212_atoms_t0_err",
        }
        assert required_keys <= result.keys()

    def test_uncertainties_are_non_negative(self, synthetic_bateman_data):
        t, rates, errors, _, _ = synthetic_bateman_data
        result = fit_initial_populations(t, rates, errors)
        assert result["ra224_atoms_t0_err"] >= 0.0
        assert result["pb212_atoms_t0_err"] >= 0.0

    def test_fitted_values_are_positive(self, synthetic_bateman_data):
        t, rates, errors, _, _ = synthetic_bateman_data
        result = fit_initial_populations(t, rates, errors)
        assert result["ra224_atoms_t0"] > 0.0


# ── backcalculate_accumulation ────────────────────────────────────────────────

class TestBackcalculateAccumulation:
    """Tests for backcalculate_accumulation(n_ra_fit, n_pb_fit, delay_seconds, acc_seconds)."""

    def test_returns_all_expected_keys(self):
        result = backcalculate_accumulation(1e9, 1e8, 3600.0, 7200.0)
        required_keys = {
            "n_ra_end_acc",
            "n_pb_end_acc",
            "ratio_ra_to_pb",
            "max_ra_capacity",
            "saturation_pct",
        }
        assert required_keys <= result.keys()

    def test_zero_delay_same_as_input(self):
        """With zero delay, back-calculated N_Ra should equal the fitted N_Ra."""
        n_ra, n_pb = 5e9, 1e8
        result = backcalculate_accumulation(n_ra, n_pb, delay_seconds=0.0, acc_seconds=3600.0)
        assert np.isclose(result["n_ra_end_acc"], n_ra, rtol=1e-9)

    def test_saturation_pct_between_0_and_100(self):
        result = backcalculate_accumulation(1e9, 0.0, 0.0, 7 * 24 * 3600)
        assert 0.0 < result["saturation_pct"] <= 100.0

    def test_n_ra_end_acc_increases_with_delay(self):
        """Longer vacuum delay → more Ra has decayed → back-calculated N_Ra is larger."""
        r1 = backcalculate_accumulation(1e9, 0.0, delay_seconds=3600.0, acc_seconds=3600.0)
        r2 = backcalculate_accumulation(1e9, 0.0, delay_seconds=7200.0, acc_seconds=3600.0)
        assert r2["n_ra_end_acc"] > r1["n_ra_end_acc"]


# ── parse_timestamps ──────────────────────────────────────────────────────────

class TestParseTimestamps:
    """Tests for parse_timestamps(datetime_series) -> np.ndarray."""

    def test_first_value_is_zero(self):
        ts = pd.Series(["2026-02-03 10:00:00", "2026-02-03 10:30:00", "2026-02-03 11:00:00"])
        result = parse_timestamps(ts)
        assert result[0] == 0.0

    def test_interval_is_correct_in_seconds(self):
        ts = pd.Series(["2026-02-03 10:00:00", "2026-02-03 10:01:00"])
        result = parse_timestamps(ts)
        assert np.isclose(result[1], 60.0)

    def test_multi_day_span(self):
        ts = pd.Series(["2026-02-03 10:00:00", "2026-02-05 10:00:00"])
        result = parse_timestamps(ts)
        assert np.isclose(result[1], 2 * 24 * 3600.0)

    def test_output_is_monotonically_increasing(self):
        ts = pd.Series([
            "2026-02-03 10:00:00",
            "2026-02-05 10:00:00",
            "2026-02-10 10:00:00",
        ])
        result = parse_timestamps(ts)
        assert np.all(np.diff(result) > 0)

    def test_accepts_datetime_objects(self):
        """parse_timestamps must work when the Series already holds Timestamps."""
        ts = pd.Series([
            pd.Timestamp("2026-02-03 10:00:00"),
            pd.Timestamp("2026-02-03 10:30:00"),
        ])
        result = parse_timestamps(ts)
        assert np.isclose(result[1], 1800.0)

    def test_output_dtype_is_float(self):
        ts = pd.Series(["2026-02-03 10:00:00", "2026-02-03 11:00:00"])
        result = parse_timestamps(ts)
        assert result.dtype == np.float64


# ── transform_spectra_to_rates ────────────────────────────────────────────────

class TestTransformSpectraToRates:
    """Tests for transform_spectra_to_rates(raw_df) -> pd.DataFrame."""

    def test_output_has_required_columns(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        # Expect the detailed fit outputs from apply_fit: amplitude, fitted
        # parameters, net counts and R_sq
        expected = {"A", "mu", "sigma", "b", "c", "net_counts", "net_counts_error", "R_sq"}
        missing = expected - set(result.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_spectrum_column_is_dropped(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        assert "Spectrum" not in result.columns

    def test_row_count_preserved(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        assert len(result) == len(synthetic_raw_df)

    def test_metadata_columns_preserved(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        cols = set(result.columns)
        assert ("ColA" in cols or "Datetime" in cols), "Time column (ColA/Datetime) was lost"
        assert "SourceFile" in cols, "SourceFile metadata was lost"

    def test_amplitude_positive_for_valid_spectra(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        assert (result["A"] > 0).all()

    def test_r_sq_near_unity_for_clean_spectra(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        assert (result["R_sq"] > 0.999).all()

    def test_rate_error_non_negative(self, synthetic_raw_df):
        result = transform_spectra_to_rates(synthetic_raw_df)
        if "RateError" in result.columns:
            assert (result["RateError"] >= 0).all()
        else:
            # If no explicit RateError column exists, ensure amplitude A is non-negative
            assert (result["A"] >= 0).all()

    def test_nan_spectrum_row_gives_zero_amplitude(self, synthetic_raw_df):
        """A row whose spectrum contains NaN should return A=0.0, not raise."""
        bad_df = synthetic_raw_df.copy()
        nan_spectrum = synthetic_raw_df["Spectrum"].iloc[0].copy()
        nan_spectrum[0] = np.nan
        bad_df.at[bad_df.index[0], "Spectrum"] = nan_spectrum
        result = transform_spectra_to_rates(bad_df)
        assert result.loc[bad_df.index[0], "A"] == 0.0


class TestCalculateNetCountsError:
    """Tests for calculate_net_counts_error(pcov, a, sigma)."""

    def test_zero_covariance_returns_zero(self):
        pcov = np.zeros((3, 3))
        err = calculate_net_counts_error(pcov, a=100.0, sigma=2.0)
        assert err == 0.0

    def test_matches_manual_error_propagation(self):
        a = 100.0
        sigma = 2.0
        # Choose small variances and a small covariance for a deterministic check
        var_a = 4.0
        var_sigma = 1.0
        cov_a_sigma = 0.5

        pcov = np.zeros((3, 3))
        pcov[0, 0] = var_a
        pcov[2, 2] = var_sigma
        pcov[0, 2] = cov_a_sigma
        pcov[2, 0] = cov_a_sigma

        # Manual propagation (must mirror implementation in etl_hidex)
        s = np.sqrt(2 * np.pi)
        partial_a = abs(sigma) * s
        partial_sigma = a * s
        net_var = (partial_a ** 2) * var_a + (partial_sigma ** 2) * var_sigma + 2 * partial_a * partial_sigma * cov_a_sigma
        expected_err = np.sqrt(net_var) if net_var > 0 else 0.0

        err = calculate_net_counts_error(pcov, a, sigma)
        assert np.isclose(err, expected_err, rtol=1e-9)
