import numpy as np
import pytest

from RaTag.el_tpc.waveform_features import (
    compute_waveform_baseline,
    compute_timing_statistics,
    _compute_left_half_std,
    find_s1,
    find_s2,
)


def test_compute_waveform_baseline_matches_manual_slice(sample_waveform):
    n_points = 25

    baseline_median, baseline_std = compute_waveform_baseline(sample_waveform, n_points=n_points)
    pre_trigger = sample_waveform.v[:, :n_points] if sample_waveform.ff else sample_waveform.v[:n_points]

    assert baseline_median == pytest.approx(float(np.median(pre_trigger)))
    assert baseline_std == pytest.approx(float(np.std(pre_trigger)))


def test_compute_waveform_baseline_respects_available_samples(sample_waveform):
    n_points = sample_waveform.v.shape[-1] + 100

    baseline_median, baseline_std = compute_waveform_baseline(sample_waveform, n_points=n_points)
    pre_trigger = sample_waveform.v[: sample_waveform.v.shape[-1]]

    assert baseline_median == pytest.approx(float(np.median(pre_trigger)))
    assert baseline_std == pytest.approx(float(np.std(pre_trigger)))


def test_compute_timing_statistics_rejects_outlier():
    times = [1.0] * 50 + [100.0]

    out = compute_timing_statistics(times, name="t_s1")

    assert out["t_s1"] == pytest.approx(1.0, abs=0.1)
    assert out["t_s1_std"] == pytest.approx(0.0, abs=1e-6)


def test_compute_timing_statistics_handles_empty_input():
    out = compute_timing_statistics([], name="t_s2_start")

    assert out == {"t_s2_start": None, "t_s2_start_std": 0.0}


def test_compute_left_half_std_uses_left_side_only():
    times = np.array([0.8, 0.9, 1.0, 1.0, 1.2, 1.4], dtype=float)
    std = _compute_left_half_std(times, mode=1.0)

    expected = float(np.sqrt(np.mean((times[times <= 1.0] - 1.0) ** 2)))
    assert std == pytest.approx(expected)


def test_find_s1_and_s2_return_expected_shapes(sample_waveform):
    s1 = find_s1(sample_waveform, threshold=1.0)
    s2_start, s2_end = find_s2(sample_waveform, threshold_s2=0.8, t_min=0.0)

    assert s1.shape == (sample_waveform.nframes,)
    assert s2_start.shape == (sample_waveform.nframes,)
    assert s2_end.shape == (sample_waveform.nframes,)
