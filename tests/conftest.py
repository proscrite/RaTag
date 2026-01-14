"""
Shared pytest fixtures for all test modules.
"""

import pytest
from pathlib import Path
import sys


def _initialize_run8(run8_directory, max_files=10):
    """Helper to initialize RUN8 with all sets."""
    from RaTag.workflows.run_construction import initialize_run
    from RaTag.core.datatypes import Run
    from dataclasses import replace
    import traceback

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

    # Primary initialization using the library workflow
    run = initialize_run(run, max_files=max_files)

    # Test-only permissive fallback: if initialize_run found no sets (because
    # directory naming differs), try to construct sets from any subdirectory
    # that contains .wfm files (but skip alpha-monitoring Ch4_* dirs).
    if not run.sets:
        print("DEBUG: initialize_run found 0 sets, attempting permissive fallback (tests only)...")
        try:
            from RaTag.core.constructors import set_from_dir

            sets = []
            for subdir in sorted(run8_directory.iterdir()):
                if not subdir.is_dir():
                    continue
                # Skip alpha-monitoring directories handled elsewhere
                if subdir.name.startswith('Ch4'):
                    continue
                try:
                    s = set_from_dir(subdir, nfiles=max_files)
                    sets.append(s)
                    print(f"Fallback: added set from {subdir.name}")
                except Exception as e:
                    print(f"Fallback: failed to add set from {subdir.name}: {e}")
                    traceback.print_exc()
                    continue

            if sets:
                run = replace(run, sets=sets)
                print(f"Fallback: populated {len(sets)} sets from RUN8 directory")
            else:
                print("Fallback: no valid sets found in RUN8 during permissive scan")
        except Exception as e:
            print(f"DEBUG: fallback population failed: {e}")
            traceback.print_exc()

    return run


@pytest.fixture(scope="session")
def run8_directory():
    """Path to RUN8 test data directory."""
    # Original project-relative path (used on macOS dev machine)
    project_path = Path(__file__).parent.parent.parent / "scope_data" / "waveforms" / "RUN8"

    # Windows-specific path on this machine
    windows_path = Path("E:/Pablos_Mighty_measurements/RUN8")

    # Prefer the Windows path when running on Windows, otherwise prefer the project-relative path.
    if sys.platform.startswith("win"):
        candidates = [windows_path, project_path]
    else:
        candidates = [project_path, windows_path]

    for p in candidates:
        if p.exists():
            return p

    # If none of the expected locations exist, skip the tests with a clear message.
    pytest.skip("Test data not found in any of the expected locations: " + ", ".join(str(p) for p in candidates))


@pytest.fixture
def test_run(run8_directory):
    """Create a fresh Run object for RUN8 (no initialization)."""
    from RaTag.core.datatypes import Run
    
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
def prepared_set(sample_set):
    """Get a set with cached metadata loaded (for integration tests)."""
    from RaTag.core.dataIO import load_set_metadata
    
    # Try to load existing metadata from disk
    loaded_set = load_set_metadata(sample_set)
    
    # Use loaded metadata if it exists, otherwise use fresh set
    return loaded_set # if loaded_set is not None else sample_set


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


@pytest.fixture(autouse=True)
def cleanup_matplotlib():
    """Close all matplotlib figures after each test."""
    import matplotlib.pyplot as plt
    yield
    plt.close('all')