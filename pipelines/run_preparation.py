# pipelines/run_preparation.py
import matplotlib.pyplot as plt
from functools import partial
from typing import Optional
from pathlib import Path

from core.datatypes import Run
from core.functional import pipe_run
from workflows.run_construction import initialize_run
from workflows.timing_estimation import estimate_s1_in_run, estimate_s2_in_run, validate_timing_windows, summarize_timing_vs_field

# ============================================================================
# MAIN PIPELINE - FUNCTIONAL COMPOSITION
# ============================================================================

def prepare_run(run: Run,
                max_files: Optional[int] = None,
                max_frames_s1: int = 200,
                max_frames_s2: int = 500,
                threshold_s1: float = 1.0,
                threshold_s2: float = 0.8,
                s2_duration_cuts: tuple = (3, 35)) -> Run:
    """
    Complete run preparation pipeline.
    
    Pipeline stages:
    1. Initialize run (gas density, populate sets, compute fields/transport)
    2. Estimate S1 timing
    3. Estimate S2 timing
    4. Validate timing windows
    5. Summarize timing vs field
    
    Args:
        run: Run with root_directory and experiment parameters
        max_files: Limit files per set (testing only)
        max_frames_s1: Target frames for S1 estimation
        max_frames_s2: Target frames for S2 estimation
        threshold_s1: S1 detection threshold (mV)
        threshold_s2: S2 detection threshold (mV)
        s2_duration_cuts: (min, max) duration cuts for S2 (µs)
        validate: If True, generate validation plots
        
    Returns:
        Fully prepared Run ready for integration
    """
    print("=" * 60)
    print(f"PREPARING RUN: {run.run_id}")
    print("=" * 60)
    
    steps = [
        partial(initialize_run, max_files=max_files),
        partial(estimate_s1_in_run,
                max_frames=max_frames_s1,
                threshold_s1=threshold_s1),
        partial(estimate_s2_in_run,
                max_frames=max_frames_s2,
                threshold_s2=threshold_s2,
                s2_duration_cuts=s2_duration_cuts),
        partial(validate_timing_windows, n_waveforms=5),
        summarize_timing_vs_field
    ]
    
    return pipe_run(run, *steps)