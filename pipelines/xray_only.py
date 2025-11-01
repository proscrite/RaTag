
def run_xray_classification(
    run: Run,
    ts2_tol: float = -2.7,
    range_sets: Optional[slice] = None,
    config: IntegrationConfig = IntegrationConfig(),
    nfiles: Optional[int] = None
) -> Dict[str, any]:
    """
    Execute X-ray event classification across all sets in a run.

    This identifies X-ray-like signals in the drift region between S1 and S2,
    classifying events as accepted/rejected based on signal quality criteria.

    Args:
        run: Prepared Run object with sets populated
        ts2_tol: Time tolerance before S2 window start (µs)
        range_sets: Optional slice to process subset of sets
        config: IntegrationConfig with analysis parameters
        nfiles: If provided, limit each set to this many files (for testing)

    Returns:
        Dictionary mapping set_id -> XRayResults
    """
    print("\n" + "=" * 60)
    print("X-RAY CLASSIFICATION PIPELINE")
    print("=" * 60)
    
    # If nfiles is specified, create limited version of run
    if nfiles is not None:
        print(f"\nLimiting sets to {nfiles} files each (test mode)")
        limited_sets = []
        for s in run.sets:
            limited_set = set_from_dir(s.source_dir, nfiles=nfiles)
            # Copy metadata from prepared set
            limited_set.metadata = s.metadata
            limited_set.drift_field = s.drift_field
            limited_set.EL_field = s.EL_field
            limited_set.time_drift = s.time_drift
            limited_sets.append(limited_set)
        run = replace(run, sets=limited_sets)
    
    results = classify_xrays_run(run, ts2_tol=ts2_tol, range_sets=range_sets, config=config)
    
    print("\n" + "=" * 60)
    print("X-RAY CLASSIFICATION COMPLETE")
    print("=" * 60)
    
    return results