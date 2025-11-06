"""
Tests for recoil integration workflow.
"""

import pytest
import numpy as np
from pathlib import Path

from RaTag.workflows.recoil_integration import (
    workflow_s2_integration,
    integrate_s2_in_run,
    summarize_s2_vs_field,
    _collect_s2_data
)
from RaTag.core.config import IntegrationConfig, FitConfig
from RaTag.core.dataIO import load_s2area


def test_workflow_s2_integration_single_set(prepared_set):
    """Test complete S2 integration workflow on single set."""
    # prepared_set should have timing metadata from previous test runs
    if 't_s2_start' not in prepared_set.metadata:
        pytest.skip("Set must have timing metadata - run preparation tests first")
    
    result = workflow_s2_integration(prepared_set, max_files=2)
    
    # Check metadata was updated with fit results
    assert 'area_s2_mean' in result.metadata
    assert 'area_s2_ci95' in result.metadata
    assert 'area_s2_sigma' in result.metadata
    assert 'area_s2_fit_success' in result.metadata
    
    # Check files were created in default locations
    processed_dir = prepared_set.source_dir.parent / "processed_data"
    plots_dir = prepared_set.source_dir.parent / "plots" / "s2_histograms"
    
    set_name = prepared_set.source_dir.name
    assert (processed_dir / f"{set_name}_s2_areas.npy").exists()
    assert (processed_dir / f"{set_name}_s2_results.json").exists()
    assert (plots_dir / f"{set_name}_s2_histogram.png").exists()
    
    # Verify we can load the raw areas back
    s2_loaded = load_s2area(prepared_set)
    assert s2_loaded is not None
    assert len(s2_loaded.areas) > 0


def test_workflow_s2_integration_caching(prepared_set):
    """Test that workflow results are cached in metadata."""
    if 't_s2_start' not in prepared_set.metadata:
        pytest.skip("Set must have timing metadata")
    
    # First run
    result1 = workflow_s2_integration(prepared_set, max_files=2)
    
    # Load raw areas from disk to verify they were saved
    s2_areas_1 = load_s2area(prepared_set)
    
    # Second run should skip integration (use metadata cache)
    result2 = workflow_s2_integration(prepared_set, max_files=2)
    
    # Metadata should be identical
    assert result2.metadata['area_s2_mean'] == result1.metadata['area_s2_mean']
    assert result2.metadata['area_s2_ci95'] == result1.metadata['area_s2_ci95']
    
    # Raw areas on disk should still be accessible
    s2_areas_2 = load_s2area(prepared_set)
    np.testing.assert_array_equal(s2_areas_2.areas, s2_areas_1.areas)


def test_integrate_s2_in_run(test_run):
    """Test run-level integration with multiple sets."""
    from RaTag.workflows.timing_estimation import estimate_s1_in_run, estimate_s2_in_run
    from RaTag.workflows.run_construction import initialize_run
    
    # Prepare run
    run = initialize_run(test_run, max_files=2)
    run = estimate_s1_in_run(run, max_frames=20)
    run = estimate_s2_in_run(run, max_frames=50)
    
    # Integrate S2
    run_with_results = integrate_s2_in_run(run, max_files=2)
    
    # Check that sets have updated metadata
    sets_with_results = [s for s in run_with_results.sets 
                         if 'area_s2_mean' in s.metadata]
    assert len(sets_with_results) > 0
    
    # Check that at least one set has successful fit
    successful_fits = [s for s in run_with_results.sets
                       if s.metadata.get('area_s2_fit_success', False)]
    assert len(successful_fits) > 0
    
    # Verify files were created in default location
    processed_dir = run.root_directory / "processed_data"
    assert processed_dir.exists()
    
    # Check at least one set's files exist
    for set_pmt in run_with_results.sets[:3]:
        set_name = set_pmt.source_dir.name
        areas_file = processed_dir / f"{set_name}_s2_areas.npy"
        if areas_file.exists():
            break
    else:
        pytest.fail("No S2 areas files found in processed_data")


def test_collect_s2_data(test_run):
    """Test data collection for plotting."""
    from RaTag.workflows.timing_estimation import estimate_s1_in_run, estimate_s2_in_run
    from RaTag.workflows.run_construction import initialize_run
    
    # Prepare and integrate
    run = initialize_run(test_run, max_files=2)
    run = estimate_s1_in_run(run, max_frames=20)
    run = estimate_s2_in_run(run, max_frames=50)
    run = integrate_s2_in_run(run, max_files=2)
    
    df = _collect_s2_data(run)
    
    assert len(df) > 0
    assert 'set_name' in df.columns
    assert 'drift_field' in df.columns
    assert 's2_mean' in df.columns
    assert 's2_ci95' in df.columns
    assert 's2_sigma' in df.columns


def test_summarize_s2_vs_field(test_run):
    """Test field-dependent summary plotting and CSV export."""
    from RaTag.workflows.timing_estimation import estimate_s1_in_run, estimate_s2_in_run
    from RaTag.workflows.run_construction import initialize_run
    
    # Prepare and integrate
    run = initialize_run(test_run, max_files=2)
    run = estimate_s1_in_run(run, max_frames=20)
    run = estimate_s2_in_run(run, max_frames=50)
    run = integrate_s2_in_run(run, max_files=2)
    
    # Create summary
    run = summarize_s2_vs_field(run)
    
    # Check files created in default locations
    csv_file = run.root_directory / "processed_data" / f"{run.run_id}_s2_vs_drift.csv"
    plot_file = run.root_directory / "plots" / f"{run.run_id}_s2_vs_drift.png"
    
    assert csv_file.exists()
    assert plot_file.exists()
    
    # Verify CSV contents
    import pandas as pd
    df = pd.read_csv(csv_file)
    assert len(df) > 0
    assert 'drift_field' in df.columns
    assert 's2_mean' in df.columns
    assert 's2_ci95' in df.columns


def test_integration_error_handling(sample_set):
    """Test that integration handles errors gracefully."""
    from dataclasses import replace
    
    # Missing metadata should raise ValueError
    bad_set = replace(sample_set, metadata={})
    
    with pytest.raises(ValueError, match="missing S2 window metadata"):
        workflow_s2_integration(bad_set, max_files=1)


def test_fit_failure_handling(prepared_set):
    """Test handling of fit failures."""
    if 't_s2_start' not in prepared_set.metadata:
        pytest.skip("Set must have timing metadata")
    
    # Use existing timing but expect integration to work regardless of fit
    result = workflow_s2_integration(prepared_set, max_files=2)
    
    # Metadata should be updated (even if fit failed)
    assert 'area_s2_fit_success' in result.metadata
    
    # Raw areas should still be saved on disk
    processed_dir = prepared_set.source_dir.parent / "processed_data"
    set_name = prepared_set.source_dir.name
    assert (processed_dir / f"{set_name}_s2_areas.npy").exists()
    
    # Should be able to load areas
    s2_loaded = load_s2area(prepared_set)
    assert len(s2_loaded.areas) > 0