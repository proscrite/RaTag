import pytest
from dataclasses import replace
from pathlib import Path

from RaTag.el_tpc.baseline_workflow import (
    resolve_set_baseline,
    map_baseline,
)
from RaTag.core.datatypes import SetPmt


def _noop_load_cache(x):
    return None


def _noop_save_cache(x):
    return None


def test_resolve_set_baseline_populates(sample_set, monkeypatch):
    # Use real waveform files from the sample_set (from conftest)
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    out = resolve_set_baseline(sample_set, max_frames=1, n_points=100)

    assert getattr(out, 'baseline_median', None) is not None
    assert getattr(out, 'baseline_std', None) is not None
    assert isinstance(out.baseline_median, float)
    assert isinstance(out.baseline_std, float)


def test_resolve_set_baseline_empty_set_returns_zero(tmp_path, monkeypatch):
    # Create an empty set (no waveform files) and ensure defaults are used
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    set_dir = tmp_path / "FieldScan_Gate050_Anode1950"
    set_dir.mkdir()
    empty = SetPmt(source_dir=set_dir, filenames=[], gate=50.0, anode=1950.0)

    out = resolve_set_baseline(empty, max_frames=1, n_points=50)

    assert out.baseline_median == 0.0
    assert out.baseline_std == 0.0


def test_map_baseline_applies_to_run(run8, monkeypatch):
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    out_run = map_baseline(run8, max_frames=1, n_points=50)

    # At least one set should have baseline_std populated
    assert any(getattr(s, 'baseline_std', None) is not None for s in out_run.sets)
