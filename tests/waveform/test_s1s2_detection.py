# tests/waveform/test_s1s2_detection.py

"""
Tests for low-level S1/S2 detection in individual waveforms.
"""

import pytest
import numpy as np

from core.datatypes import PMTWaveform
from waveform.s1s2_detection import detect_s1_in_frame, detect_s2_in_frame


@pytest.fixture
def mock_waveform():
    """Create a mock waveform for testing."""
    # Time array: -10 to +30 µs
    t = np.linspace(-10, 30, 1000)
    
    # Voltage: baseline + S1 peak at -2µs + S2 signal at 8-18µs
    v = np.zeros_like(t)
    
    # Add S1 peak (negative time)
    s1_mask = (t > -2.5) & (t < -1.5)
    v[s1_mask] = 2.0  # 2 mV peak
    
    # Add S2 signal (positive time)
    s2_mask = (t > 8) & (t < 18)
    v[s2_mask] = 1.5  # 1.5 mV signal
    
    return PMTWaveform(t=t, v=v, ff=False, source='mock')


class TestS1Detection:
    """Test S1 peak detection."""
    
    def test_detect_s1_finds_peak(self, mock_waveform):
        """Test that S1 is detected in mock waveform."""
        t_s1 = detect_s1_in_frame(mock_waveform, threshold_s1=1.0)
        
        assert t_s1 is not None
        assert t_s1 < 0  # Should be negative time
        assert -3 < t_s1 < -1  # Should be near -2µs
    
    
    def test_detect_s1_threshold_dependency(self, mock_waveform):
        """Test S1 detection with different thresholds."""
        # Low threshold should find peak
        t_low = detect_s1_in_frame(mock_waveform, threshold_s1=0.5)
        assert t_low is not None
        
        # High threshold should miss peak
        t_high = detect_s1_in_frame(mock_waveform, threshold_s1=5.0)
        assert t_high is None
    
    
    def test_detect_s1_no_signal(self):
        """Test S1 detection with no signal."""
        # Flat waveform (noise only)
        t = np.linspace(-10, 30, 1000)
        v = np.random.normal(0, 0.1, 1000)
        wf = PMTWaveform(t=t, v=v, ff=False, source='flat')
        
        t_s1 = detect_s1_in_frame(wf, threshold_s1=1.0)
        assert t_s1 is None


class TestS2Detection:
    """Test S2 boundary detection."""
    
    def test_detect_s2_finds_boundaries(self, mock_waveform):
        """Test that S2 boundaries are detected."""
        result = detect_s2_in_frame(mock_waveform,
                                   t_s1=-2.0,
                                   t_drift=10.0,
                                   threshold_s2=0.8)
        
        assert result is not None
        t_start, t_end = result
        
        # Should find boundaries around 8-18 µs
        assert 7 < t_start < 9
        assert 17 < t_end < 19
        assert t_end > t_start  # End should be after start
    
    
    def test_detect_s2_duration(self, mock_waveform):
        """Test S2 duration is reasonable."""
        t_start, t_end = detect_s2_in_frame(mock_waveform,
                                           t_s1=-2.0,
                                           t_drift=10.0,
                                           threshold_s2=0.8)
        
        duration = t_end - t_start
        assert 9 < duration < 11  # Should be ~10 µs
    
    
    def test_detect_s2_no_signal(self):
        """Test S2 detection with no signal."""
        t = np.linspace(-10, 30, 1000)
        v = np.random.normal(0, 0.1, 1000)
        wf = PMTWaveform(t=t, v=v, ff=False, source='flat')
        
        result = detect_s2_in_frame(wf,
                                   t_s1=-2.0,
                                   t_drift=10.0,
                                   threshold_s2=0.8)
        
        assert result is None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])