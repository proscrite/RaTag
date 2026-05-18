import pytest
from pathlib import Path

from RaTag.io.bootstrap import (bootstrap_from_path,
                            bootstrap_from_config,
                            _resolve_bootstrapping_params,
                            bootstrap_bare_set)


class DummyWf:
    def __init__(self, ff=False, nframes=1):
        self.ff = ff
        self.nframes = nframes


def test_bootstrap_bare_set_parsing(tmp_path: Path, monkeypatch):
    # Create a FieldScan dir with one .wfm file
    set_dir = tmp_path / "FieldScan_Gate0050_Anode1950"
    set_dir.mkdir()
    (set_dir / "RUNX_20250902_Gate50_Anode1950_P1_1Wfm.wfm").write_text("dummy")

    # Monkeypatch WHERE load_wfm is USED (in file_ops), not where it's defined
    monkeypatch.setattr("RaTag.io.file_ops.load_wfm", lambda p: DummyWf(ff=False, nframes=1))

    s = bootstrap_bare_set(set_dir)

    assert s.gate == 50
    assert s.anode == 1950
    assert isinstance(s.filenames, list)
    assert len(s.filenames) == 1


def test_bootstrap_bare_set_detects_multiiso(tmp_path: Path, monkeypatch):
    # Create a set dir with Ch1 and Ch4 files (multi-isotope)
    set_dir = tmp_path / "FieldScan_Gate0100_Anode2000"
    set_dir.mkdir()
    (set_dir / "RUN8_20250902_Gate100_Anode2000_P1_Ch1.wfm").write_text("ch1")
    (set_dir / "RUN8_20250902_Gate100_Anode2000_P1_Ch4.wfm").write_text("ch4")

    monkeypatch.setattr("RaTag.io.file_ops.load_wfm", lambda p: DummyWf(ff=False, nframes=1))

    s = bootstrap_bare_set(set_dir)

    assert s.multiiso is True
    # filenames come from find_set_files() (all .wfm names)
    assert any(name.endswith("_Ch1.wfm") for name in s.filenames)


def test__resolve_bootstrapping_params_parsing(tmp_path: Path):
    # Directory name encodes run/isotope/field
    run_dir = tmp_path / "Run8_Th228_2350Vcm"
    run_dir.mkdir()

    run_id, el_field, isotope = _resolve_bootstrapping_params(run_dir, None, None, None)

    assert run_id == "RUN8"
    assert el_field == 2350
    assert isotope == "Th228"


def test__resolve_bootstrapping_params_fails(tmp_path: Path):
    # Missing run id
    run_dir = tmp_path / "Th228_2350Vcm"
    run_dir.mkdir()
    with pytest.raises(ValueError, match="run_id"):
        _resolve_bootstrapping_params(run_dir, None, None, None)

    # Missing el_field
    run_dir2 = tmp_path / "Run8_Th228"
    run_dir2.mkdir()
    with pytest.raises(ValueError, match="el_field"):
        _resolve_bootstrapping_params(run_dir2, None, None, None)


def test_bootstrap_from_path_success(run8_directory: Path, monkeypatch):
    # Patch to avoid real .wfm parsing on fast-frame detection
    monkeypatch.setattr("RaTag.io.file_ops.load_wfm", lambda p: DummyWf(ff=False, nframes=1))
    
    # Integration: use existing test fixture for RUN8; pass mandatory params explicitly
    run = bootstrap_from_path(run8_directory, run_id="RUN8", el_field=2350.0)

    assert run.run_id == "RUN8"
    assert run.el_field == 2350.0
    # There should be at least one FieldScan -> Set
    assert len(run.sets) >= 1
    # Each set must point to a real directory under the run root
    assert all(s.source_dir.exists() and s.source_dir.parent == run.root_directory for s in run.sets)


def test_bootstrap_from_config_success(tmp_path: Path):
    # Build a small run dir and write a YAML config (fix indentation)
    run_dir = tmp_path / "Run8_Th228_2350Vcm"
    run_dir.mkdir()

    cfg = tmp_path / "run8.yaml"
    cfg.write_text(f"""run_id: RUN8
data:
    raw_data_path: "{run_dir}"
experiment:
    el_field: 2350.0
    target_isotope: "Th228"
    pressure: 2.0
    temperature: 297.0
    sampling_rate: 5000000000.0
    el_gap: 0.8
    drift_gap: 1.4
        """)

    run = bootstrap_from_config(cfg)

    assert run.run_id == "RUN8"
    assert run.root_directory == run_dir
    assert run.el_field == 2350.0
    assert run.target_isotope == "Th228"
    assert run.pressure == 2.0
    assert run.temperature == 297.0
    assert run.sampling_rate == 5.0e9
    assert run.el_gap == 0.8
    assert run.drift_gap == 1.4
    assert run.recoil_energy == 96.8