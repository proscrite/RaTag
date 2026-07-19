import numpy as np
import pytest

from RaTag.core.dataIO import load_wfm
from RaTag.waveform.preprocessing import (
    subtract_pedestal,
    moving_average,
    threshold_clip,
    standard_preprocessing,
    average_waveform,
)


def test_subtract_pedestal_centers_pretrigger(sample_waveform):
    n_points = 20
    out = subtract_pedestal(sample_waveform, n_points=n_points)

    assert out is not sample_waveform
    assert np.mean(out.v[:n_points]) == pytest.approx(0.0, abs=1e-10)


def test_moving_average_preserves_length(sample_waveform):
    out = moving_average(sample_waveform, window=7)

    assert out.v.shape == sample_waveform.v.shape
    assert out.t.shape == sample_waveform.t.shape


def test_threshold_clip_zeroes_negative_values(sample_waveform):
    out = threshold_clip(sample_waveform, threshold=0.0)

    assert out.v.shape == sample_waveform.v.shape
    assert np.all(out.v >= 0.0)


def test_standard_preprocessing_composes_operations(sample_waveform):
    out = standard_preprocessing(sample_waveform, n_pedestal=20, ma_window=5, threshold=0.0)

    manual = threshold_clip(
        moving_average(
            subtract_pedestal(sample_waveform, n_points=20),
            window=5,
        ),
        threshold=0.0,
    )

    assert np.allclose(out.v, manual.v)
    assert np.allclose(out.t, manual.t)

def test_average_waveform_uses_real_files(sample_waveform_paths):
    t, v_avg = average_waveform(sample_waveform_paths)

    waveforms = [load_wfm(path) for path in sample_waveform_paths]
    if waveforms[0].ff:
        expected = waveforms[0].v.mean(axis=0)
    else:
        expected = np.stack([wf.v for wf in waveforms]).mean(axis=0)

    assert t.shape == waveforms[0].t.shape
    assert v_avg.shape == expected.shape
    assert np.allclose(v_avg, expected)
