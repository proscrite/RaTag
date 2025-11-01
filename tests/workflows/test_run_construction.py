"""
Tests for run construction and transport property computation.
"""

import pytest
import numpy as np


class TestTransportProperties:
    """Test that transport properties are correctly computed."""
    
    def test_all_sets_have_transport(self, all_sets):
        """Verify all sets have transport properties computed."""
        for i, set_pmt in enumerate(all_sets, 1):
            print(f"\nSet {i}: {set_pmt.source_dir.name}")
            
            # Check transport properties exist
            assert set_pmt.speed_drift is not None, f"speed_drift not set for set {i}"
            assert set_pmt.time_drift is not None, f"time_drift not set for set {i}"
            
            # Print for inspection
            print(f"  Drift field: {set_pmt.drift_field:.2f} V/cm")
            print(f"  Reduced field: {set_pmt.red_drift_field:.2f} Td")
            print(f"  Drift speed: {set_pmt.speed_drift:.3f} mm/µs")
            print(f"  Drift time: {set_pmt.time_drift:.2f} µs")
            
            # Sanity checks
            assert 0 < set_pmt.time_drift < 100, f"Unreasonable drift time: {set_pmt.time_drift} µs"
    
    
    def test_transport_ordering_across_sets(self, all_sets):
        """Test that transport properties follow expected physical ordering."""
        set1, set2, set3 = all_sets[:3]
        
        print("\n" + "="*60)
        print("TRANSPORT PROPERTIES ORDERING")
        print("="*60)
        
        # Print values for inspection
        for i, s in enumerate([set1, set2, set3], 1):
            print(f"\nSet {i}: {s.source_dir.name}")
            print(f"  E_drift: {s.drift_field:.2f} V/cm")
            print(f"  v_drift: {s.speed_drift:.3f} mm/µs")
            print(f"  t_drift: {s.time_drift:.2f} µs")
        
        # Physics: Higher drift field → Higher drift speed → Lower drift time
        print("\n" + "="*60)
        print("EXPECTED ORDERING:")
        print("  E_drift: Set1 < Set2 < Set3")
        print("  v_drift: Set1 < Set2 < Set3")
        print("  t_drift: Set1 > Set2 > Set3")
        print("="*60)
        
        # Test drift field ordering
        assert set1.drift_field < set2.drift_field < set3.drift_field, \
            f"Drift fields not increasing: {set1.drift_field:.2f} < {set2.drift_field:.2f} < {set3.drift_field:.2f}"
        
        # Test drift speed ordering (higher field → faster drift)
        assert set1.speed_drift < set2.speed_drift < set3.speed_drift, \
            f"Drift speeds not increasing: {set1.speed_drift:.3f} < {set2.speed_drift:.3f} < {set3.speed_drift:.3f}"
        
        # Test drift time ordering (faster drift → shorter time)
        assert set1.time_drift > set2.time_drift > set3.time_drift, \
            f"Drift times not decreasing: {set1.time_drift:.2f} > {set2.time_drift:.2f} > {set3.time_drift:.2f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])