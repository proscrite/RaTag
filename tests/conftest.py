"""
Shared pytest fixtures for all test modules.
"""

import pytest
from pathlib import Path
from dataclasses import replace

from RaTag.io.bootstrap import bootstrap_from_path
from RaTag.core.dataIO import load_wfm


@pytest.fixture(scope="session")
def run8_directory():
    project_root = Path(__file__).resolve().parents[2]
    run8_path = project_root / "test_data" / "raw_data" / "RUN8"

    if run8_path.exists():
        return run8_path

    pytest.skip(f"Test data not found: {run8_path}")


@pytest.fixture(scope="session")
def run8(run8_directory):
    return bootstrap_from_path(run8_directory,
                               run_id="RUN8", el_field=2350.0, target_isotope="Th228")


@pytest.fixture
def sample_set(run8):
    if not run8.sets:
        pytest.skip("No sets found in RUN8")
    return run8.sets[0]


@pytest.fixture
def sample_waveform(sample_set):
    if not sample_set.filenames:
        pytest.skip("sample_set has no waveform files")
    return load_wfm(sample_set.source_dir / sample_set.filenames[0])


@pytest.fixture
def sample_waveform_paths(sample_set):
    if len(sample_set.filenames) < 2:
        pytest.skip("Need at least two waveform files")
    return [sample_set.source_dir / name for name in sample_set.filenames[:2]]


@pytest.fixture
def prepared_set(sample_set):
    from RaTag.core.dataIO import load_set_metadata

    loaded_set = load_set_metadata(sample_set)
    return loaded_set or sample_set


@pytest.fixture
def fresh_set(sample_set):
    from dataclasses import fields

    exclude = {"source_dir", "filenames", "ff", "nframes", "multiiso", "drift_field", "EL_field"}
    clear_kwargs = {
        f.name: None
        for f in fields(sample_set)
        if f.name not in exclude
    }
    return replace(sample_set, **clear_kwargs)


@pytest.fixture
def all_sets(run8):
    if len(run8.sets) < 3:
        pytest.skip(f"Need at least 3 sets, found {len(run8.sets)}")
    return run8.sets


@pytest.fixture
def monkey_dummy_wf(monkeypatch):
    def _set(ff=False, nframes=1):
        monkeypatch.setattr("RaTag.core.dataIO.load_wfm", lambda p: DummyWf(ff=ff, nframes=nframes))
    return _set


class DummyWf:
    def __init__(self, ff=False, nframes=1):
        self.ff = ff
        self.nframes = nframes