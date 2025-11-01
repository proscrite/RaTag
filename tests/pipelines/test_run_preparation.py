# tests/pipelines/test_run_preparation.py

"""
Integration tests for the complete run preparation pipeline.

Tests orchestration, not individual workflow correctness.
"""

import pytest
import json
from pathlib import Path

from pipelines.run_preparation import prepare_run


class TestPipelineOrchestration:
    """Test that pipeline coordinates all workflows correctly."""
    
    def test_pipeline_initializes_all_sets(self, test_run):
        """Test that pipeline initializes all sets with transport properties."""
        result = prepare_run(test_run, max_files=10)
        
        # Should have sets
        assert len(result.sets) >= 3, "Should load at least 3 sets"
        
        # All sets should have transport properties
        for set_pmt in result.sets:
            assert set_pmt.speed_drift is not None
            assert set_pmt.time_drift is not None
            assert set_pmt.drift_field > 0
    
    
    def test_pipeline_processes_s1_then_s2(self, test_run):
        """Test that pipeline processes S1 before S2 for all sets."""
        result = prepare_run(test_run, max_files=10)
        
        # All sets should have both S1 and S2 results
        for set_pmt in result.sets:
            assert "t_s1" in set_pmt.metadata, f"Missing S1 for {set_pmt.source_dir.name}"
            assert "t_s2_start" in set_pmt.metadata, f"Missing S2 for {set_pmt.source_dir.name}"
            
            # Get values
            t_s1 = set_pmt.metadata["t_s1"]
            t_s2 = set_pmt.metadata["t_s2_start"]
            
            # Both should be computed (not None)
            assert t_s1 is not None, f"t_s1 is None for {set_pmt.source_dir.name}"
            assert t_s2 is not None, f"t_s2_start is None for {set_pmt.source_dir.name}"
            
            # S2 should use S1 timing
            expected_s2 = t_s1 + set_pmt.time_drift
            
            # S2 should be reasonably close to expected
            assert abs(t_s2 - expected_s2) < 5.0, \
                f"S2 timing inconsistent with S1+drift: {t_s2:.2f} vs {expected_s2:.2f}"
    
    def test_pipeline_computes_gas_density(self, test_run):
        """Test that pipeline computes gas density."""
        result = prepare_run(test_run, max_files=10)
        
        assert result.gas_density is not None
        assert result.gas_density > 0, "Gas density should be positive"
        
        # For Xe at 2 bar, 297K, should be ~4.8e19 cm^-3
        assert 4e19 < result.gas_density < 5e19, \
            f"Gas density {result.gas_density:.2e} outside expected range"


class TestPipelineOutputs:
    """Test that pipeline creates organized output structure."""
    
    def test_pipeline_creates_directory_structure(self, test_run):
        """Test that pipeline creates organized output directories."""
        prepare_run(test_run, max_files=10)
        
        # Check directory structure
        assert (test_run.root_directory / "processed_data").exists()
        assert (test_run.root_directory / "plots" / "s1_timing").exists()
        assert (test_run.root_directory / "plots" / "s2_timing").exists()
        # assert (test_run.root_directory / "metadata").exists()
    
    
    def test_pipeline_creates_all_set_outputs(self, test_run):
        """Test that pipeline creates outputs for every set."""
        result = prepare_run(test_run, max_files=10)
        
        data_dir = test_run.root_directory / "processed_data"
        s1_plots = test_run.root_directory / "plots" / "s1_timing"
        s2_plots = test_run.root_directory / "plots" / "s2_timing"
        
        for set_pmt in result.sets:
            set_name = set_pmt.source_dir.name
            
            # Check data files
            assert (data_dir / f"{set_name}_s1.npz").exists(), f"Missing S1 data for {set_name}"
            assert (data_dir / f"{set_name}_s2.npz").exists(), f"Missing S2 data for {set_name}"
            
            # Check plots
            assert (s1_plots / f"{set_name}_s1.png").exists(), f"Missing S1 plot for {set_name}"
            assert (s2_plots / f"{set_name}_s2.png").exists(), f"Missing S2 plot for {set_name}"
    
    
    """def test_pipeline_saves_run_metadata(self, test_run):
        "Test that pipeline saves run-level metadata JSON."
        prepare_run(test_run, max_files=10)
        
        # Check run metadata exists
        metadata_file = test_run.root_directory / "metadata" / "run_info.json"
        assert metadata_file.exists(), "Run metadata not saved"
        
        # Verify content
        with open(metadata_file) as f:
            metadata = json.load(f)
        
        assert metadata["run_id"] == "RUN8"
        assert "gas_density" in metadata
        assert "sets" in metadata
        assert len(metadata["sets"]) >= 3"""
    
    
    def test_pipeline_output_data_format(self, test_run):
        """Test that output data files have correct format."""
        import numpy as np
        
        result = prepare_run(test_run, max_files=10)
        
        data_dir = test_run.root_directory / "processed_data"
        set_pmt = result.sets[0]
        set_name = set_pmt.source_dir.name
        
        # Check S1 data format
        s1_data = np.load(data_dir / f"{set_name}_s1.npz")
        assert "times" in s1_data
        assert len(s1_data["times"]) > 0
        
        # Check S2 data format
        s2_data = np.load(data_dir / f"{set_name}_s2.npz")
        assert "t_s2_start" in s2_data
        assert "t_s2_end" in s2_data
        assert "s2_duration" in s2_data
        assert len(s2_data["t_s2_start"]) > 0


class TestPipelineCaching:
    """Test that pipeline uses caching efficiently."""
    
    def test_pipeline_uses_cached_transport(self, test_run):
        """Test that pipeline loads transport properties from cache on second run."""
        # First run
        result1 = prepare_run(test_run, max_files=10)
        
        # Second run should use cache
        result2 = prepare_run(test_run, max_files=10)
        
        # Results should be identical
        assert len(result1.sets) == len(result2.sets)
        
        for set1, set2 in zip(result1.sets, result2.sets):
            assert set1.time_drift == set2.time_drift
            assert set1.speed_drift == set2.speed_drift
    
    
    def test_pipeline_uses_cached_s1_s2(self, test_run):
        """Test that pipeline uses cached S1/S2 results."""
        # First run
        result1 = prepare_run(test_run, max_files=10)
        
        # Second run
        result2 = prepare_run(test_run, max_files=10)
        
        # S1/S2 metadata should match
        for set1, set2 in zip(result1.sets, result2.sets):
            assert set1.metadata["t_s1"] == set2.metadata["t_s1"]
            assert set1.metadata["t_s2_start"] == set2.metadata["t_s2_start"]


class TestPipelineRobustness:
    """Test pipeline error handling and edge cases."""
    
    def test_pipeline_handles_partial_sets(self, test_run):
        """Test that pipeline continues if some sets have fewer files."""
        # Run with very low max_files
        result = prepare_run(test_run, max_files=2)
        
        # Should still process all sets
        assert len(result.sets) >= 3
        
        # All sets should have at least some results
        for set_pmt in result.sets:
            assert "t_s1" in set_pmt.metadata
            # t_s1 might be None if too few files, that's ok
    
    
    def test_pipeline_handles_different_max_files(self, test_run):
        """Test that pipeline works with different file limits."""
        result_small = prepare_run(test_run, max_files=5)
        result_large = prepare_run(test_run, max_files=10)
        
        # Both should complete
        assert len(result_small.sets) >= 3
        assert len(result_large.sets) >= 3
        
        # Results should be similar (within reason)
        for set_small, set_large in zip(result_small.sets, result_large.sets):
            t_s1_small = set_small.metadata.get("t_s1")
            t_s1_large = set_large.metadata.get("t_s1")
            
            # Skip if either is None (not enough events)
            if t_s1_small is None or t_s1_large is None:
                continue
            
            # S1 timing should be stable regardless of file count
            assert abs(t_s1_small - t_s1_large) < 1.0, \
                "S1 timing should be consistent across different file counts"
    
    
    def test_pipeline_all_sets_have_minimum_events(self, test_run):
        """Test that all sets find minimum number of events."""
        result = prepare_run(test_run, max_files=10)
        
        for set_pmt in result.sets:
            # Load S2 data to check event count
            data_dir = test_run.root_directory / "processed_data"
            set_name = set_pmt.source_dir.name
            
            import numpy as np
            s2_data = np.load(data_dir / f"{set_name}_s2.npz")
            
            # Should find reasonable number of S2 events
            assert len(s2_data["t_s2_start"]) > 50, \
                f"Too few S2 events in {set_name}: {len(s2_data['t_s2_start'])}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])