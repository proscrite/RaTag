import json
from pathlib import Path



from RaTag.io import file_ops
from RaTag.core.datatypes import SetPmt
from RaTag.tests.conftest import DummyWf


def test_load_yaml(tmp_path: Path):
    p = tmp_path / "conf.yaml"
    p.write_text("a: 1\nb: hello\n")
    d = file_ops.load_yaml(p)
    assert d["a"] == 1
    assert d["b"] == "hello"


def test_scan_for_set_directories(tmp_path: Path):
    root = tmp_path / "run"
    root.mkdir()
    (root / "FieldScan_Gate050_Anode1950").mkdir()
    (root / "other_dir").mkdir()
    dirs = file_ops.scan_for_set_directories(root)
    assert all("FieldScan" in d.name for d in dirs)
    assert len(dirs) == 1


def test_parse_run_id_variants():
    assert file_ops.parse_run_id(Path("Run8_Th228_2350Vcm")) == "RUN8"
    assert file_ops.parse_run_id(Path("RUN_9_extra")) == "RUN9"
    assert file_ops.parse_run_id(Path("no_match")) is None


def test_parse_target_isotope_and_el_field():
    p = Path("Run8_Th228_2375Vcm")
    assert file_ops.parse_target_isotope(p) == "Th228"
    assert file_ops.parse_el_field(p) == 2375


def test_parse_subdir_name_sampling_gate_anode():
    name = "FieldScan_Gate050_Anode1950_5GSsec"
    out = file_ops.parse_subdir_name(name)
    assert int(out["gate"]) == 50
    assert int(out["anode"]) == 1950
    assert out["sampling_rate"] == 5e9


def test_parse_filename_components():
    fname = "RUN8_20250902_Gate50_Anode1950_P1_123_ch2.wfm"
    parsed = file_ops.parse_filename(fname)
    assert parsed["run"] == 8
    assert parsed["date"] == "20250902"
    assert parsed["gate"] == 50
    assert parsed["anode"] == 1950
    assert parsed["position"] == 1
    assert parsed["event_id"] == 123
    assert parsed["channel"] == 2


def test_find_set_files_and_limit(tmp_path: Path):
    d = tmp_path / "FieldScan_Gate050_Anode1950"
    d.mkdir()
    for i in range(5):
        (d / f"file_{i}.wfm").write_text("x")
    names = file_ops.find_set_files(d)
    assert len(names) == 5
    names_limited = file_ops.find_set_files(d, nfiles=2)
    assert len(names_limited) == 2


def test_detect_multiiso_set_true_false():
    assert file_ops.detect_multiiso_set(["a_Ch1.wfm", "b_Ch4.wfm"]) is True
    assert file_ops.detect_multiiso_set(["a.wfm", "b.wfm"]) is False


def test_detect_fastframe_properties(monkeypatch, tmp_path: Path):
    d = tmp_path / "FieldScan_Gate050_Anode1950"
    d.mkdir()
    fname = "RUNX_1Wfm.wfm"
    (d / fname).write_text("dummy")
    # Patch where load_wfm is USED (in file_ops)
    monkeypatch.setattr("RaTag.io.file_ops.load_wfm", lambda p: DummyWf(ff=True, nframes=64))
    ff, nframes = file_ops.detect_fastframe_properties(d, [fname])
    assert ff is True
    assert nframes == 64


def test_save_and_load_cache_roundtrip(tmp_path: Path):
    run_root = tmp_path / "run_root"
    set_dir = run_root / "FieldScan_Gate050_Anode1950"
    set_dir.mkdir(parents=True)
    # Create a SetPmt with some computed fields
    s = SetPmt(source_dir=set_dir, filenames=["f.wfm"])
    # dynamically assign some non-None compute fields via replace-like construction
    from dataclasses import replace
    s_with = replace(s, t_s1=-3.14, time_drift=12.34, speed_drift=0.56)
    # save
    file_ops.save_cache(s_with)
    # create a fresh set (same source_dir) and load cache
    fresh = SetPmt(source_dir=set_dir, filenames=["f.wfm"])
    loaded = file_ops.load_cache(fresh)
    assert loaded is not None
    assert loaded.t_s1 == -3.14
    assert loaded.time_drift == 12.34
    assert loaded.speed_drift == 0.56