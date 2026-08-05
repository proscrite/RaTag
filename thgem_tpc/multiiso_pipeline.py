from RaTag.thgem_tpc.multiiso_workflow import (map_multiiso_s2_vs_field, map_multiiso_separation,
                                               map_multiiso_hist_grid)
from RaTag.thgem_tpc.recoil_workflow import (map_finetuned_plots, map_finetune_fits,
                                              map_recoil_fits, map_recoil_plots)
from RaTag.core.datatypes import Run
from RaTag.core.config import FitConfig, FinetuneConfig

def pipeline_multi_isotope(run: Run, config: dict) -> Run:
    
    exec_cfg = config.get('execution', {})
    fit_config = config.get('fit_config', FitConfig())
    fit_config = FitConfig(**{k: v for k, v in fit_config.items() if hasattr(FitConfig, k)})
    

    multiiso_state = exec_cfg.get('run_multiiso', False)
    force_multiiso = (multiiso_state == 'force')
    finetune_state = exec_cfg.get('run_finetune', False)
    force_finetune = (finetune_state == 'force')
    # 1. Separate: Replaces run.sets with a flattened list of isotope-specific SetPmts
    multi_run_dict = map_multiiso_separation(run, force=force_multiiso)
    
    for iso, iso_run in multi_run_dict.items():
        print(f"\n" + "="*60 + f"\nFITTING ISOTOPE: {iso}\n" + "="*60)
        fitted_run = map_recoil_fits(iso_run, config=fit_config, force=force_multiiso)
        multi_run_dict[iso] = fitted_run  # Update the dict with the fitted run

        # 3. Plot: same as in single-isotope recoil workflow
        plotted_run = map_recoil_plots(fitted_run, config=fit_config, force=force_multiiso)

        if exec_cfg.get('run_finetune', False):
            print(f"\n" + "="*60 + f"\nFINETUNING ISOTOPE: {iso}\n" + "="*60)
            finetune_dict = config.get('finetuning', {})

            fitted_run = map_finetune_fits(fitted_run, finetune_dict=finetune_dict, force=force_finetune)
            multi_run_dict[iso] = fitted_run  # Update the dict with the finetuned run
            plotted_run = map_finetuned_plots(fitted_run, finetune_dict=finetune_dict, force=force_finetune)

    run = map_multiiso_hist_grid(run, spawned_runs=multi_run_dict, config=fit_config, force=force_multiiso)
    run = map_multiiso_s2_vs_field(run, spawned_runs=multi_run_dict, force=force_multiiso)
    return run
