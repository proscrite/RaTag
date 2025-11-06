"""
Tests for waveform integration module.
"""

import pytest
import numpy as np
from pathlib import Path

from waveform.integration import integrate_s2_in_frame
from core.config import IntegrationConfig
from core.dataIO import load_wfm, iter_frames


def test_integrate_s2_in_frame_single(sample_set):
    """Test S2 integration on a single frame."""
    # Get first frame
    frames = list(iter_frames(sample_set, max_files=1))
    frame = frames[0]
    
    # Define S2 window (using typical values)
    s2_start = 5.0
    s2_end = 20.0
    
    config = IntegrationConfig()
    
    area = integrate_s2_in_frame(frame, s2_start, s2_end, config)
    
    assert isinstance(area, float)
    assert area > 0, "S2 area should be positive"
    assert area < 1000, "S2 area seems unreasonably large"


def test_integrate_s2_different_windows(sample_set):
    """Test that different S2 windows give different areas."""
    frames = list(iter_frames(sample_set, max_files=1))
    frame = frames[0]
    
    config = IntegrationConfig()
    
    # Narrow window
    area1 = integrate_s2_in_frame(frame, 5.0, 10.0, config)
    
    # Wide window
    area2 = integrate_s2_in_frame(frame, 5.0, 25.0, config)
    
    assert area2 > area1, "Wider window should capture more area"


def test_integrate_s2_config_parameters(sample_set):
    """Test that config parameters affect integration."""
    frames = list(iter_frames(sample_set, max_files=1))
    frame = frames[0]
    
    # Default config
    config1 = IntegrationConfig()
    area1 = integrate_s2_in_frame(frame, 5.0, 20.0, config1)
    
    # Different moving average window
    config2 = IntegrationConfig(ma_window=15)
    area2 = integrate_s2_in_frame(frame, 5.0, 20.0, config2)
    
    # Results should differ due to different preprocessing
    assert area1 != area2


def test_integrate_s2_requires_metadata(sample_set):
    """Test that integration requires S2 window metadata."""
    from workflows.recoil_integration import workflow_s2_integration
    from dataclasses import replace
    
    # Remove S2 metadata
    set_without_s2 = replace(sample_set, metadata={})
    
    with pytest.raises(ValueError, match="missing S2 window metadata"):
        workflow_s2_integration(set_without_s2, max_files=1)