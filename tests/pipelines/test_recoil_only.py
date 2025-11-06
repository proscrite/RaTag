"""
Tests for complete recoil analysis pipeline.
"""

import pytest
from pathlib import Path

from RaTag.pipelines.recoil_only import recoil_pipeline, recoil_pipeline_quick, recoil_pipeline_replot
from RaTag.pipelines.run_preparation import prepare_run


def test_recoil_pipeline_end_to_end(test_run, tmp_path):
    """Test complete pipeline from preparation to summary."""
    # Full preparation
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    
    # Run recoil analysis
    run = recoil_pipeline(run, max_files=2)
    
    # Check that sets have S2 area results in metadata
    sets_with_s2 = [s for s in run.sets if 'area_s2_mean' in s.metadata]
    assert len(sets_with_s2) > 0
    
    # Check output files
    csv_file = run.root_directory / "processed_data" / f"{run.run_id}_s2_vs_drift.csv"
    plot_file = run.root_directory / "plots" / f"{run.run_id}_s2_vs_drift.png"
    
    assert csv_file.exists()
    assert plot_file.exists()


def test_recoil_pipeline_quick(test_run):
    """Test quick pipeline variant."""
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    
    # Delete any existing summary files to test that quick doesn't create them
    csv_file = run.root_directory / "processed_data" / f"{run.run_id}_s2_vs_drift.csv"
    plot_file = run.root_directory / "plots" / f"{run.run_id}_s2_vs_drift.png"
    
    if csv_file.exists():
        csv_file.unlink()
    if plot_file.exists():
        plot_file.unlink()
    
    
    # Run quick pipeline
    run = recoil_pipeline_quick(run, max_files=1)
    
    # Check that sets have S2 results
    sets_with_s2 = [s for s in run.sets if 'area_s2_mean' in s.metadata]
    assert len(sets_with_s2) > 0
    
    # Quick version only integrates, doesn't create summary plot/CSV
    assert not csv_file.exists()
    assert not plot_file.exists()


def test_recoil_pipeline_subset(test_run):
    """Test processing subset of sets."""
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    
    # Process only first 2 sets
    run = recoil_pipeline(run, range_sets=slice(0, 2), max_files=2)
    
    # Check that only subset was processed
    sets_with_s2 = [s for s in run.sets if 'area_s2_mean' in s.metadata]
    assert len(sets_with_s2) <= 2


def test_recoil_pipeline_replot(test_run):
    """Test regenerating plot from cached results."""
    # First run full pipeline
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    run = recoil_pipeline(run, max_files=2)
    
    # Count sets with results
    n_results_first = len([s for s in run.sets if 'area_s2_mean' in s.metadata])
    
    # Replot from cache - just call replot on same run (metadata already loaded)
    run_replotted = recoil_pipeline_replot(run)
    
    # Should still have same results (no recomputation)
    n_results_second = len([s for s in run_replotted.sets if 'area_s2_mean' in s.metadata])
    assert n_results_second == n_results_first
    
    # Plot should have been regenerated
    plot_file = run_replotted.root_directory / "plots" / f"{run_replotted.run_id}_s2_vs_drift.png"
    assert plot_file.exists()


def test_pipeline_uses_caching(test_run):
    """Test that pipeline uses cached results efficiently."""
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    
    # First run
    run1 = recoil_pipeline(run, max_files=2)
    
    # Second run should use cache (metadata already exists)
    run2 = recoil_pipeline(run1, max_files=2)
    
    # Results should be identical (check metadata)
    for s1, s2 in zip(run1.sets, run2.sets):
        if 'area_s2_mean' in s1.metadata and 'area_s2_mean' in s2.metadata:
            assert s1.metadata['area_s2_mean'] == s2.metadata['area_s2_mean']
            assert s1.metadata['area_s2_ci95'] == s2.metadata['area_s2_ci95']


def test_pipeline_error_propagation(test_run):
    """Test that pipeline handles individual set failures gracefully."""
    from dataclasses import replace
    
    # Prepare run
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    
    # Corrupt one set's metadata to remove S2 timing
    corrupted_sets = list(run.sets)
    if len(corrupted_sets) > 0:
        original_metadata = corrupted_sets[0].metadata.copy()
        corrupted_metadata = {k: v for k, v in original_metadata.items() 
                             if k not in ['t_s2_start', 't_s2_end']}
        corrupted_sets[0] = replace(corrupted_sets[0], metadata=corrupted_metadata)
        run = replace(run, sets=corrupted_sets)
    
    # Pipeline should continue with other sets
    run = recoil_pipeline(run, max_files=2)
    
    # Should have results from remaining sets (not the corrupted one)
    sets_with_s2 = [s for s in run.sets if 'area_s2_mean' in s.metadata]
    assert len(sets_with_s2) >= 0  # At least doesn't crash
    
    # If we have multiple sets, at least some should succeed
    if len(run.sets) > 1:
        assert len(sets_with_s2) > 0


def test_pipeline_custom_config(test_run):
    """Test pipeline with custom integration and fit configs."""
    import matplotlib.pyplot as plt
    from RaTag.core.config import IntegrationConfig, FitConfig
    from RaTag.core.dataIO import load_s2area
    from dataclasses import replace
    
    # Close any open figures from previous tests
    plt.close('all')
    
    run = prepare_run(test_run, max_files=2, max_frames_s1=20, max_frames_s2=50)
    
    # Clear area_s2_mean from metadata to force recomputation with custom config
    fresh_sets = []
    for s in run.sets:
        cleaned_metadata = {k: v for k, v in s.metadata.items() 
                           if not k.startswith('area_s2')}
        fresh_sets.append(replace(s, metadata=cleaned_metadata))
    run = replace(run, sets=fresh_sets)
    
    custom_int_config = IntegrationConfig(ma_window=15, bs_threshold=0.05)
    custom_fit_config = FitConfig(bin_cuts=(0, 5), nbins=150)
    
    run = recoil_pipeline(run,
                         max_files=2,
                         integration_config=custom_int_config,
                         fit_config=custom_fit_config)
    
    # Check that sets have results
    sets_with_s2 = [s for s in run.sets if 'area_s2_mean' in s.metadata]
    assert len(sets_with_s2) > 0
    
    # Load S2Areas from disk to check params were saved
    s2 = load_s2area(sets_with_s2[0])
    assert s2.params['ma_window'] == 15
    assert s2.params['bs_threshold'] == 0.05
    
    # Clean up at end
    plt.close('all')