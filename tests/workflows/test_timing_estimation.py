# tests/workflows/test_timing_estimation.py

"""
Tests for timing estimation workflows at the set level.
"""

import pytest
from pathlib import Path
import numpy as np
from dataclasses import replace

from workflows.timing_estimation import (
    compute_s1,
    compute_s2,
    workflow_s1_set,
    workflow_s2_set,
    save_timing_results,
    validate_timing_windows,
    summarize_timing_vs_field
)

class TestS1Computation:
    """Test pure S1 computation (no side effects)."""
    
    def test_compute_s1_returns_data(self, sample_set):
        """Test that compute_s1 returns set and timing data."""
        updated_set, s1_times = compute_s1(sample_set, max_frames=100, threshold_s1=1.0)
        
        # Should return both set and data
        assert updated_set is not None
        assert s1_times is not None
        assert isinstance(s1_times, np.ndarray)
        assert len(s1_times) > 0
    
    
    def test_compute_s1_updates_metadata(self, sample_set):
        """Test that compute_s1 adds metadata."""
        updated_set, s1_times = compute_s1(sample_set, max_frames=100)
        
        # Metadata should be populated
        assert "t_s1" in updated_set.metadata
        assert "t_s1_std" in updated_set.metadata
        
        # Values should be reasonable
        assert updated_set.metadata["t_s1"] < 0  # Before trigger
        assert updated_set.metadata["t_s1_std"] > 0
    
    
    def test_compute_s1_with_different_thresholds(self, sample_set):
        """Test S1 computation with different thresholds."""
        _, s1_low = compute_s1(sample_set, max_frames=100, threshold_s1=0.5)
        _, s1_high = compute_s1(sample_set, max_frames=100, threshold_s1=2.0)
        
        # Both should find at least some peaks
        assert len(s1_low) > 0
        assert len(s1_high) > 0
        
        # High threshold should give more consistent (lower std) results
        # because it filters out noise
        std_low = np.std(s1_low)
        std_high = np.std(s1_high)
        
        # High threshold data should be more tightly clustered
        assert std_high < std_low, f"High threshold std ({std_high:.2f}) should be < low threshold std ({std_low:.2f})"
        
        # High threshold mean should be in reasonable range
        assert -5.0 < np.mean(s1_high) < -1.0, f"S1 mean should be -5 to -1 µs, got {np.mean(s1_high):.2f}"
    
    
    def test_s1_consistent_across_sets(self, all_sets):
        """Test that S1 timing is consistent across all sets (same source position)."""
        print("\n" + "="*60)
        print("S1 TIMING CONSISTENCY")
        print("="*60)
        
        s1_times = []
        for i, set_pmt in enumerate(all_sets, 1):
            set_pmt, _ = compute_s1(set_pmt, max_frames=100)
            t_s1 = set_pmt.metadata["t_s1"]
            t_s1_std = set_pmt.metadata["t_s1_std"]
            
            print(f"\nSet {i}: t_s1 = {t_s1:.2f} ± {t_s1_std:.2f} µs")
            s1_times.append(t_s1)
        
        # S1 should be roughly the same across sets (source position doesn't change)
        # Allow 2 µs variation
        s1_mean = np.mean(s1_times)
        s1_std = np.std(s1_times)
        
        print(f"\nOverall: {s1_mean:.2f} ± {s1_std:.2f} µs")
        print(f"Max variation: {max(s1_times) - min(s1_times):.2f} µs")
        
        assert s1_std < 2.0, f"S1 varies too much across sets: std={s1_std:.2f} µs"


class TestS2Computation:
    """Test pure S2 computation (no side effects)."""
    
    def test_compute_s2_requires_s1(self, fresh_set):
        """Test that compute_s2 fails without S1."""
        with pytest.raises(ValueError, match="t_s1 must be estimated first"):
            compute_s2(fresh_set, max_frames=100)
    
    
    def test_compute_s2_returns_data(self, sample_set):
        """Test that compute_s2 returns set and timing dict."""
        # First compute S1
        sample_set, _ = compute_s1(sample_set, max_frames=100)
        
        # No need to manually set time_drift - it's already computed!
        
        # Compute S2 - will use the correct time_drift from transport calculation
        updated_set, s2_data = compute_s2(sample_set, max_frames=200)
        
        # Should return dict with three arrays
        assert isinstance(s2_data, dict)
        assert "t_s2_start" in s2_data
        assert "t_s2_end" in s2_data
        assert "s2_duration" in s2_data
        
        # Arrays should have same length
        assert len(s2_data["t_s2_start"]) == len(s2_data["t_s2_end"])
        assert len(s2_data["t_s2_start"]) == len(s2_data["s2_duration"])
        
        # Now it should find many more events!
        print(f"\n  Found {len(s2_data['t_s2_start'])} S2 events (with correct time_drift)")
        assert len(s2_data["t_s2_start"]) > 100, "Should find many S2 events with correct drift time"
    
    
    def test_s2_timing_ordering_across_sets(self, all_sets):
        """Test that S2 timing follows drift time ordering across sets."""
        print("\n" + "="*60)
        print("S2 TIMING ORDERING")
        print("="*60)
        
        # Compute S1 and S2 for all sets
        s2_starts = []
        for i, set_pmt in enumerate(all_sets, 1):
            # S1 first
            set_pmt, _ = compute_s1(set_pmt, max_frames=100)
            
            # S2
            set_pmt, s2_data = compute_s2(set_pmt, max_frames=200)
            
            t_s1 = set_pmt.metadata["t_s1"]
            t_s2 = set_pmt.metadata["t_s2_start"]
            t_drift = set_pmt.time_drift
            
            print(f"\nSet {i}: {set_pmt.source_dir.name}")
            print(f"  t_s1: {t_s1:.2f} µs")
            print(f"  t_drift: {t_drift:.2f} µs")
            print(f"  t_s2_start: {t_s2:.2f} µs")
            print(f"  Expected: {t_s1 + t_drift:.2f} µs")
            
            s2_starts.append(t_s2)
        
        # Physics: Higher drift field → shorter drift time → earlier S2
        print("\n" + "="*60)
        print("EXPECTED ORDERING:")
        print("  t_drift: Set1 > Set2 > Set3")
        print("  t_s2_start: Set1 > Set2 > Set3")
        print("="*60)
        
        set1_s2, set2_s2, set3_s2 = s2_starts
        
        # S2 should arrive earlier with higher drift fields
        assert set1_s2 > set2_s2 > set3_s2, \
            f"S2 timing not decreasing: {set1_s2:.2f} > {set2_s2:.2f} > {set3_s2:.2f}"
        
        print("\n✓ S2 timing follows expected physical ordering")
    
    
    def test_s2_duration_consistency_across_sets(self, all_sets):
        """Test that S2 duration is roughly consistent across sets."""
        print("\n" + "="*60)
        print("S2 DURATION CONSISTENCY")
        print("="*60)
        
        durations = []
        for i, set_pmt in enumerate(all_sets, 1):
            # S1 first
            set_pmt, _ = compute_s1(set_pmt, max_frames=100)
            
            # S2
            set_pmt, s2_data = compute_s2(set_pmt, max_frames=200)
            
            duration = set_pmt.metadata["s2_duration"]
            duration_std = set_pmt.metadata["s2_duration_std"]
            
            print(f"\nSet {i}: duration = {duration:.2f} ± {duration_std:.2f} µs")
            durations.append(duration)
        
        # S2 duration should be reasonably consistent
        # (EL gap is constant, so light emission time shouldn't vary much)
        duration_mean = np.mean(durations)
        duration_std = np.std(durations)
        
        print(f"\nOverall: {duration_mean:.2f} ± {duration_std:.2f} µs")
        
        # Allow moderate variation (EL field does change)
        assert duration_std < 5.0, f"S2 duration varies too much: std={duration_std:.2f} µs"


class TestS2Expected:
    """Test S2 detection with computed drift times."""
    
    def test_s2_windows_align_with_histogram(self, all_sets, run8_directory):
        """Test that S2 search windows align with actual S2 distribution."""
        data_dir = run8_directory / "processed_data"
        data_dir.mkdir(exist_ok=True)
        
        for i, set_pmt in enumerate(all_sets, 1):
            print(f"\n{'='*60}")
            print(f"Set {i}: {set_pmt.source_dir.name}")
            print(f"{'='*60}")
            
            # Run S1 first
            set_pmt, _ = compute_s1(set_pmt, max_frames=100)
            
            t_s1 = set_pmt.metadata["t_s1"]
            expected_s2 = t_s1 + set_pmt.time_drift
            
            print(f"  t_s1: {t_s1:.2f} µs")
            print(f"  time_drift: {set_pmt.time_drift:.2f} µs")
            print(f"  Expected S2 start: {expected_s2:.2f} µs")
            
            # Run S2
            set_pmt, s2_data = compute_s2(set_pmt, max_frames=200)
            
            if len(s2_data['t_s2_start']) > 0:
                actual_s2 = set_pmt.metadata["t_s2_start"]
                print(f"  Actual S2 start: {actual_s2:.2f} µs")
                print(f"  Difference: {actual_s2 - expected_s2:.2f} µs")
                
                # The actual S2 should be reasonably close to expected
                # (within 20% now that we have correct drift time)
                relative_error = abs(actual_s2 - expected_s2) / actual_s2
                print(f"  Relative error: {relative_error*100:.1f}%")
                
                # If this fails, time_drift calculation is wrong
                assert relative_error < 0.2, \
                    f"S2 prediction off by {relative_error*100:.1f}% - check time_drift calculation"
            else:
                print(f"  ⚠ No S2 events found (search window may be wrong)")


class TestWorkflows:
    """Test complete set-level workflows (with side effects)."""
    
    def test_workflow_s1_set_creates_files(self, sample_set, tmp_path):
        """Test that S1 workflow creates all expected files."""
        plots_dir = tmp_path / "plots"
        data_dir = tmp_path / "data"
        
        updated_set = workflow_s1_set(sample_set,
                                      max_frames=100,
                                      plots_dir=plots_dir,
                                      data_dir=data_dir)
        
        # Metadata should exist
        assert "t_s1" in updated_set.metadata
        
        # Data file should exist
        data_file = data_dir / f"{sample_set.source_dir.name}_s1.npz"
        assert data_file.exists()
        
        # Plot should exist
        plot_file = plots_dir / f"{sample_set.source_dir.name}_s1.png"
        assert plot_file.exists()
        assert plot_file.stat().st_size > 0
    
    
    def test_workflow_s2_set_creates_files(self, sample_set, tmp_path):
        """Test that S2 workflow creates all expected files."""
        # First run S1
        plots_dir = tmp_path / "plots"
        data_dir = tmp_path / "data"
        
        sample_set = workflow_s1_set(sample_set, 
                                    max_frames=100, 
                                    plots_dir=plots_dir,
                                    data_dir=data_dir)
        
        # No need to manually set time_drift - it's already computed!
        
        # Run S2 - uses computed time_drift
        updated_set = workflow_s2_set(sample_set,
                                      max_frames=200,
                                      plots_dir=plots_dir,
                                      data_dir=data_dir)
        
        # Metadata should exist
        assert "t_s2_start" in updated_set.metadata
        
        # Data file should exist
        data_file = data_dir / f"{sample_set.source_dir.name}_s2.npz"
        assert data_file.exists()
        
        # Verify data structure
        s2_data = np.load(data_file)
        assert "t_s2_start" in s2_data
        assert "t_s2_end" in s2_data
        assert "s2_duration" in s2_data
        
        # Should find many events now!
        assert len(s2_data["t_s2_start"]) > 100, "Should find many S2 events"
    
    
    def test_workflow_s1_creates_files_in_run8(self, sample_set, run8_directory):
        """Test S1 workflow with output to actual RUN8 directory."""
        plots_dir = run8_directory / "plots"
        data_dir = run8_directory / "processed_data"
        
        # This will create files you can inspect!
        updated_set = workflow_s1_set(sample_set,
                                      max_frames=100,
                                      plots_dir=plots_dir,
                                      data_dir=data_dir)
        
        # Files will persist in RUN8/ after test runs
        data_file = data_dir / f"{sample_set.source_dir.name}_s1.npz"
        print(f"\n📁 Output saved to: {data_file}")
        assert data_file.exists()


    def test_workflow_s2_creates_files_in_run8(self, sample_set, run8_directory):
        """Test S2 workflow with output to actual RUN8 directory."""
        plots_dir = run8_directory / "plots" / "s2_timing"
        data_dir = run8_directory / "processed_data"
        
        # First run S1 (needed for S2)
        s1_plots_dir = run8_directory / "plots" / "s1_timing"
        sample_set = workflow_s1_set(sample_set,
                                    max_frames=100,
                                    plots_dir=s1_plots_dir,
                                    data_dir=data_dir)
        
        # No need to manually set time_drift - it's already computed!
        
        # Run S2 workflow - uses computed time_drift
        updated_set = workflow_s2_set(sample_set,
                                      max_frames=200,
                                      plots_dir=plots_dir,
                                      data_dir=data_dir)
        
        # Files will persist in RUN8/ after test runs
        data_file = data_dir / f"{sample_set.source_dir.name}_s2.npz"
        plot_file = plots_dir / f"{sample_set.source_dir.name}_s2.png"
        
        print(f"\n📁 S2 Data saved to: {data_file}")
        print(f"📊 S2 Plot saved to: {plot_file}")
        
        # Check files exist
        assert data_file.exists(), f"S2 data file not created at {data_file}"
        assert plot_file.exists(), f"S2 plot file not created at {plot_file}"
        
        # Check data content
        s2_data = np.load(data_file)
        assert "t_s2_start" in s2_data
        assert "t_s2_end" in s2_data
        assert "s2_duration" in s2_data
        
        print(f"✓ S2 data contains {len(s2_data['t_s2_start'])} events")
    
    
    def test_workflow_creates_outputs_for_all_sets(self, all_sets, run8_directory):
        """Test that workflows create outputs for all sets."""
        plots_dir = run8_directory / "plots"
        data_dir = run8_directory / "processed_data"
        
        for i, set_pmt in enumerate(all_sets, 1):
            set_name = set_pmt.source_dir.name
            print(f"\nProcessing set {i}/3: {set_name}")
            
            # S1 workflow
            set_pmt = workflow_s1_set(set_pmt,
                                     max_frames=100,
                                     plots_dir=plots_dir / "s1_timing",
                                     data_dir=data_dir)
            
            # S2 workflow
            set_pmt = workflow_s2_set(set_pmt,
                                     max_frames=200,
                                     plots_dir=plots_dir / "s2_timing",
                                     data_dir=data_dir)
            
            # Verify outputs exist
            s1_data = data_dir / f"{set_name}_s1.npz"
            s2_data = data_dir / f"{set_name}_s2.npz"
            s1_plot = plots_dir / "s1_timing" / f"{set_name}_s1.png"
            s2_plot = plots_dir / "s2_timing" / f"{set_name}_s2.png"
            
            assert s1_data.exists(), f"Missing S1 data for {set_name}"
            assert s2_data.exists(), f"Missing S2 data for {set_name}"
            assert s1_plot.exists(), f"Missing S1 plot for {set_name}"
            assert s2_plot.exists(), f"Missing S2 plot for {set_name}"
            
            print(f"  ✓ All outputs created")


class TestDataPersistence:
    """Test data saving/loading."""
    
    def test_save_timing_results_s1(self, sample_set, tmp_path):
        """Test saving S1 data."""
        s1_times = np.array([-2.0, -2.1, -1.9, -2.2])
        
        save_timing_results(sample_set, s1_times, tmp_path, "s1")
        
        # File should exist
        data_file = tmp_path / f"{sample_set.source_dir.name}_s1.npz"
        assert data_file.exists()
        
        # Data should be loadable
        loaded = np.load(data_file)
        assert "times" in loaded
        np.testing.assert_array_equal(loaded["times"], s1_times)
    
    
    def test_save_timing_results_s2(self, sample_set, tmp_path):
        """Test saving S2 data."""
        s2_data = {
            't_s2_start': np.array([8.0, 8.1, 7.9]),
            't_s2_end': np.array([18.0, 18.1, 17.9]),
            's2_duration': np.array([10.0, 10.0, 10.0])
        }
        
        save_timing_results(sample_set, s2_data, tmp_path, "s2")
        
        # File should exist
        data_file = tmp_path / f"{sample_set.source_dir.name}_s2.npz"
        assert data_file.exists()
        
        # Data should be loadable
        loaded = np.load(data_file)
        assert "t_s2_start" in loaded 
        assert "t_s2_end" in loaded   
        assert "s2_duration" in loaded


class TestValidationWorkflow:
    """Test the validation workflow specifically."""
    
    def test_validation_requires_timing_estimates(self, test_run):
        """Test that validation requires S1/S2 timing to be estimated first."""
        
        # Create a run with sets that have NO timing (empty metadata)
        sets_no_timing = [replace(s, metadata={}) for s in test_run.sets]
        run_no_timing = replace(test_run, sets=sets_no_timing)
        
        # Validation should skip all sets
        result = validate_timing_windows(run_no_timing, n_waveforms=3)
        
        # Run unchanged
        assert result.run_id == run_no_timing.run_id
        assert len(result.sets) == len(run_no_timing.sets)
    
    
    def test_validation_creates_plots_with_timing(self, all_sets, test_run):
        """Test that validation creates plots when timing is available."""
        
        # First compute S1 and S2 for one set
        sample_set = all_sets[0]
        sample_set = workflow_s1_set(sample_set, max_frames=100)
        sample_set = workflow_s2_set(sample_set, max_frames=200)
        
        # Create run with completed timing
        test_run = replace(test_run, sets=[sample_set])
        
        # Validation should create plot
        result = validate_timing_windows(test_run, n_waveforms=3)

        validation_dir = test_run.root_directory / "plots" / "validation"
        plot_file = validation_dir / f"{sample_set.source_dir.name}_sample_window_wfm.png"
        
        assert plot_file.exists(), "Validation plot should be created"
        assert plot_file.stat().st_size > 1000, "Validation plot should not be empty"
    
    
    def test_validation_plots_show_timing_windows(self, all_sets, test_run):
        """Test that validation plots include S1/S2 window overlays."""
        
        # Compute timing for all sets
        processed_sets = []
        for set_pmt in all_sets:
            set_pmt = workflow_s1_set(set_pmt, max_frames=100)
            set_pmt = workflow_s2_set(set_pmt, max_frames=200)
            processed_sets.append(set_pmt)
        
        # Create run with timing
        test_run = replace(test_run, sets=processed_sets)
        
        # Run validation
        validate_timing_windows(test_run, n_waveforms=3)

        validation_dir = test_run.root_directory / "plots" / "validation"

        # Check plots exist and are substantial
        for set_pmt in processed_sets:
            plot_file = validation_dir / f"{set_pmt.source_dir.name}_sample_window_wfm.png"
            assert plot_file.exists(), f"Missing validation plot for {set_pmt.source_dir.name}"
            assert plot_file.stat().st_size > 1000, f"Validation plot too small for {set_pmt.source_dir.name}"
        
        print(f"\n✓ Created {len(processed_sets)} validation plots")
    
    
    def test_validation_returns_unchanged_run(self, all_sets, test_run):
        """Test that validation is a pure QA step (doesn't modify run)."""
        
        # Setup: compute timing for one set
        sample_set = all_sets[0]
        sample_set = workflow_s1_set(sample_set, max_frames=100)
        sample_set = workflow_s2_set(sample_set, max_frames=200)
        
        test_run = replace(test_run, sets=[sample_set])

        # Validate
        result = validate_timing_windows(test_run, n_waveforms=3)
        
        # Run should be unchanged (same object, not a copy)
        assert result is test_run, "Validation should return the same Run object"
        assert result.sets[0].metadata == sample_set.metadata, "Set metadata should be unchanged"


class TestSummaryPlots:
    """Test run-level summary visualizations."""
    
    def test_summarize_timing_vs_field(self, all_sets, test_run):
        """Test timing vs field summary plot creation."""
        from core.datatypes import Run
        
        # Compute timing for all sets
        processed_sets = []
        for set_pmt in all_sets:
            print(f"\n  Processing {set_pmt.source_dir.name}...")
            set_pmt = workflow_s1_set(set_pmt, max_frames=100)
            set_pmt = workflow_s2_set(set_pmt, max_frames=200)
            processed_sets.append(set_pmt)
        
        print(f"\n✓ Processed {len(processed_sets)} sets")
        test_run = replace(test_run, sets=processed_sets)
        
        
        # Create summary plot
        result = summarize_timing_vs_field(test_run)
        
        # Check plot exists
        summary_dir = test_run.root_directory / "plots" / "summary_preparation"
        plot_file = summary_dir / f"RUN8_timing_vs_field.png"
        
        assert plot_file.exists(), f"Summary plot should be created at {plot_file}"
        assert plot_file.stat().st_size > 10000, "Summary plot should not be empty"
        
        # Run should be unchanged
        assert result is test_run


    def test_summary_shows_field_dependence(self, all_sets, test_run):
        """Test that summary plot shows expected field dependence."""
        from core.datatypes import Run
        
        # Compute timing for all sets
        processed_sets = []
        for set_pmt in all_sets:
            set_pmt = workflow_s1_set(set_pmt, max_frames=100)
            set_pmt = workflow_s2_set(set_pmt, max_frames=200)
            processed_sets.append(set_pmt)
        test_run = replace(test_run, sets=processed_sets)
        
        # Create summary
        summarize_timing_vs_field(test_run)
        
        # Verify data makes physical sense
        drift_fields = [s.drift_field for s in processed_sets]
        t_s2_starts = [s.metadata["t_s2_start"] for s in processed_sets]
        
        # Should have data
        assert len(drift_fields) >= 3, f"Should have at least 3 sets with data, got {len(drift_fields)}"
        
        # Higher drift field → shorter drift time → earlier S2
        assert drift_fields[0] < drift_fields[-1], "Drift fields should increase"
        assert t_s2_starts[0] > t_s2_starts[-1], "S2 arrival should decrease with field"
        
        print(f"\n✓ Physical ordering verified:")
        for s in processed_sets:
            print(f"  {s.drift_field:.1f} V/cm → S2 at {s.metadata['t_s2_start']:.2f} µs")
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])