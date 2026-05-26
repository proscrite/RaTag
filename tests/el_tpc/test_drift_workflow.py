import pytest
from dataclasses import replace
from pathlib import Path

from RaTag.el_tpc import physics
from RaTag.el_tpc.drift_workflow import (
    with_gas_density,
    set_fields,
    set_reduced_fields,
    set_transport,
    map_drift_physics,
    resolve_set_drift,
)
from RaTag.core.datatypes import SetPmt, Run
from RaTag.core import units


def _noop_load_cache(x): 
    return None

def _noop_save_cache(x):
    return None

def test_with_gas_density_calculation(run8):
    expected = physics.gas_density_cm3(run8.pressure, run8.temperature)
    out = with_gas_density(run8)
    assert out.gas_density == expected


def test_with_gas_density_missing_attributes_raises_error(run8):
    bad = replace(run8, temperature=None)
    with pytest.raises(ValueError):
        with_gas_density(bad)


def test_set_fields_calculation(fresh_set, monkeypatch):
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    s = replace(fresh_set, gate=50.0, anode=1950.0)
    out = set_fields(s, el_gap_cm=0.8, drift_gap_cm=1.4)

    assert pytest.approx(out.drift_field, rel=1e-6) == 50.0 / 1.4
    assert pytest.approx(out.EL_field, rel=1e-6) == (1950.0 - 50.0) / 0.8


def test_set_fields_missing_voltages_raises_error(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)
    
    # Create a minimal SetPmt with no gate/anode
    set_dir = tmp_path / "FieldScan_Gate050_Anode1950"
    set_dir.mkdir()
    bad = SetPmt(source_dir=set_dir, filenames=[], gate=None, anode=None)
    
    with pytest.raises(ValueError, match="gate|anode"):
        set_fields(bad, el_gap_cm=0.8, drift_gap_cm=1.4)


def test_set_reduced_fields_calculation(fresh_set, monkeypatch):
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    # Provide numeric fields expected by the decorator
    s = replace(fresh_set, drift_field=150.0, EL_field=1200.0)
    gas_density = 1e19
    out = set_reduced_fields(s, gas_density)

    expected_red_drift = physics.compute_reduced_field(150.0, gas_density)
    expected_red_EL = physics.compute_reduced_field(1200.0, gas_density)

    assert pytest.approx(out.red_drift_field, rel=1e-12) == units.to_Td(expected_red_drift)
    assert pytest.approx(out.red_EL_field, rel=1e-12) == units.to_Td(expected_red_EL)


def test_set_transport_calculation(fresh_set, monkeypatch):
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    # Provide red_drift_field required by decorator
    s = replace(fresh_set, red_drift_field=0.5)
    drift_gap_cm = 1.4

    out = set_transport(s, drift_gap_cm=drift_gap_cm)

    expected_speed = physics.redfield_to_speed(0.5)
    expected_time = units.cm_to_mm(drift_gap_cm) / expected_speed

    assert pytest.approx(out.speed_drift, rel=1e-8) == expected_speed
    assert pytest.approx(out.time_drift, rel=1e-8) == expected_time


def test_map_drift_physics_success(run8, monkeypatch):
    # Prevent disk cache IO during tests
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    out_run = map_drift_physics(run8)

    # At least one set should have time_drift populated after full mapping
    assert any(getattr(s, "time_drift", None) is not None for s in out_run.sets)


def test_map_drift_physics_isolates_set_errors(run8, monkeypatch):
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    if len(run8.sets) < 1:
        pytest.skip("run8 fixture has no sets to test with")

    good = run8.sets[0]
    bad = replace(good, gate=None)  # will fail require_attributes in set_fields
    mixed_run = replace(run8, sets=[good, bad])

    out_run = map_drift_physics(mixed_run)

    # Good set should be enriched, bad set should be left in the original (no time_drift)
    assert getattr(out_run.sets[0], "time_drift", None) is not None
    assert getattr(out_run.sets[1], "time_drift", None) is None


def test_resolve_set_drift_basic(run8, fresh_set, monkeypatch):
    # Ensure the per-set pipeline (fields -> reduced -> transport) runs
    monkeypatch.setattr("RaTag.io.file_ops.load_cache", _noop_load_cache)
    monkeypatch.setattr("RaTag.io.file_ops.save_cache", _noop_save_cache)

    s = replace(fresh_set, gate=50.0, anode=1950.0)
    run_with_density = with_gas_density(run8)

    out = resolve_set_drift(s, run_with_density)

    # Fields and transport should be populated
    assert getattr(out, 'drift_field', None) is not None
    assert getattr(out, 'EL_field', None) is not None
    assert getattr(out, 'time_drift', None) is not None