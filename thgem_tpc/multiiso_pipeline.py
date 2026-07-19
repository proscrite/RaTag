from RaTag.thgem_tpc.multiiso_workflow import map_multiiso_separation
from RaTag.thgem_tpc.recoil_workflow import map_recoil_fits, map_recoil_plots
from core.datatypes import Run

def pipeline_multi_isotope(run: Run, config: dict = None) -> Run:
    
    # 1. Separate: Replaces run.sets with a flattened list of isotope-specific SetPmts
    run = map_multiiso_separation(run)
    
    # 2. Fit: same as single-isotope recoil workflow
    run = map_recoil_fits(run)
    
    # 3. Plot: same as in single-isotope recoil workflow
    run = map_recoil_plots(run)
    
    return run