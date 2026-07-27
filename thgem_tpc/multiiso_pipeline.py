from RaTag.thgem_tpc.multiiso_workflow import map_multiiso_s2_vs_field, map_multiiso_separation
from RaTag.thgem_tpc.recoil_workflow import map_recoil_fits, map_recoil_plots
from RaTag.core.datatypes import Run
from RaTag.core.config import FitConfig

def pipeline_multi_isotope(run: Run, config: dict = None) -> Run:
    if config is None:
        config = {}
    fit_config = config.get('fit_config', FitConfig())
    fit_config = FitConfig(**{k: v for k, v in fit_config.items() if hasattr(FitConfig, k)})
    print(f"Force fitting: {fit_config.force}")
    # 1. Separate: Replaces run.sets with a flattened list of isotope-specific SetPmts
    multi_run_dict = map_multiiso_separation(run)
    
    for iso, iso_run in multi_run_dict.items():
        print(f"\n" + "="*60 + f"\nFITTING ISOTOPE: {iso}\n" + "="*60)
        fitted_run = map_recoil_fits(iso_run, config=fit_config, force=fit_config.force)
        multi_run_dict[iso] = fitted_run  # Update the dict with the fitted run

        # 3. Plot: same as in single-isotope recoil workflow
        plotted_run = map_recoil_plots(fitted_run, config=fit_config, force=fit_config.force)
    
    run = map_multiiso_s2_vs_field(run, spawned_runs=multi_run_dict, force=fit_config.force)
    return run