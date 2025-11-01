"""
Shared pytest fixtures for all test modules.
"""

import pytest
from pathlib import Path


def _initialize_run8(run8_directory, max_files=10):
    """Helper to initialize RUN8 with all sets."""
    from workflows.run_construction import initialize_run
    from core.datatypes import Run
    
    run = Run(
        run_id="RUN8",
        root_directory=run8_directory,
        target_isotope="Th228",
        pressure=2.0,
        temperature=297.0,
        sampling_rate=5.0,
        drift_gap=1.4,
        el_gap=0.8,
        el_field=2375.0,
        sets=[]
    )
    
    return initialize_run(run, max_files=max_files)


@pytest.fixture(scope="session")
def run8_directory():
    """Path to RUN8 test data directory."""
    test_data = Path(__file__).parent.parent.parent / "scope_data" / "waveforms" / "RUN8"
    
    if not test_data.exists():
        pytest.skip(f"Test data not found at {test_data}")
    
    return test_data


@pytest.fixture
def test_run(run8_directory):
    """Create a fresh Run object for RUN8 (no initialization)."""
    from core.datatypes import Run
    
    return Run(
        run_id="RUN8",
        root_directory=run8_directory,
        target_isotope="Th228",
        pressure=2.0,
        temperature=297.0,
        sampling_rate=5.0,
        drift_gap=1.4,
        el_gap=0.8,
        el_field=2375.0,
        sets=[]
    )


@pytest.fixture
def sample_set(run8_directory):
    """Get first populated set from RUN8 for testing."""
    run = _initialize_run8(run8_directory, max_files=10)
    
    if not run.sets:
        pytest.skip("No sets found in RUN8")
    
    return run.sets[0]


@pytest.fixture
def fresh_set(sample_set):
    """Get a fresh set without cached metadata (for testing validation)."""
    from dataclasses import replace
    return replace(sample_set, metadata={})


@pytest.fixture
def all_sets(run8_directory):
    """Get all sets from RUN8, fully initialized with fields and transport."""
    run = _initialize_run8(run8_directory, max_files=10)
    
    if len(run.sets) < 3:
        pytest.skip(f"Need at least 3 sets, found {len(run.sets)}")
    
    return run.sets