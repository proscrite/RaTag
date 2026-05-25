from dataclasses import replace
from pathlib import Path
from typing import Union, Optional, Tuple

from RaTag.io import file_ops
from RaTag.core.datatypes import Run, SetPmt


def _resolve_bootstrapping_params(run_dir: Path,
                                 run_id: Optional[str],
                                 el_field: Optional[float],
                                 target_isotope: Optional[str]) -> Tuple[str, float, Optional[str]]:
    """Quarantine the logic for resolving bootstrapping parameters from explicit args or directory parsing."""
    # 1. Resolve Run ID
    final_run_id = run_id if run_id is not None else file_ops.parse_run_id(run_dir)
    if final_run_id is None:
        raise ValueError(f"Mandatory parameter 'run_id' missing. Provide it explicitly or ensure the directory name '{run_dir.name}' follows the parsing format.")
        
    # 2. Resolve EL Field
    final_el_field = el_field if el_field is not None else file_ops.parse_el_field(run_dir)
    if final_el_field is None:
        raise ValueError(f"Mandatory parameter 'el_field' missing. Provide it explicitly or ensure the directory name '{run_dir.name}' follows the parsing format.")
    
    # 3. Resolve Target Isotope (Allowed to be None if Dataclass allows it)
    final_isotope = target_isotope if target_isotope is not None else file_ops.parse_target_isotope(run_dir)

    return final_run_id, final_el_field, final_isotope

def bootstrap_bare_set(dir_path: Path) -> SetPmt:
    """
    PURE: Maps raw strings to the base SetPmt structure.
    """
    voltage_dict = file_ops.parse_subdir_name(str(dir_path.name))
    
    filenames = file_ops.find_set_files(dir_path)

    multiiso = file_ops.detect_multiiso_set(filenames)

    ff, nframes = file_ops.detect_fastframe_properties(dir_path, filenames)

    return SetPmt(source_dir=dir_path,
                  filenames=filenames, 
                  gate=voltage_dict.get('gate'),
                  anode=voltage_dict.get('anode'),
                  sampling_rate=voltage_dict.get('sampling_rate'),
                  multiiso=multiiso,
                  ff=ff, nframes=nframes)


def bootstrap_from_path(run_dir: Union[str, Path], run_id: Optional[str] = None, el_field: Optional[float] = None, target_isotope: Optional[str] = None) -> Run:
    """
    DECLARATIVE PIPELINE: Assembles a Run object from a raw data directory.
    
    Args:
        run_dir: Path to the raw data directory.
    """
    run_dir = Path(run_dir)
    
    run_id, el_field, target_isotope = _resolve_bootstrapping_params(run_dir, run_id, el_field, target_isotope)

    # 4. IO: Scan the disk for what exists
    raw_set_dirs = file_ops.scan_for_set_directories(run_dir)
    print(f"Found {len(raw_set_dirs)} set directories in {run_dir.name}: {[d.name for d in raw_set_dirs]}")
    # 5. ORCHESTRATE: Map the resolution logic over the directories
    bare_sets = [bootstrap_bare_set(path) for path in raw_set_dirs]
    
    return Run(run_id=run_id, 
               root_directory=run_dir, 
               el_field=el_field,
               target_isotope=target_isotope,
               sets=bare_sets)


def bootstrap_from_config(config_path: Union[str, Path]) -> Run:
    """
    WRAPPER: Reads a YAML config and passes the physical parameters 
    down to the path builder.
    """
    # 1. IO: Load the dictionary
    config = file_ops.load_yaml(Path(config_path))
    
    run_dir = Path(config['data']['raw_data_path'])
    exp_params = config['experiment']
    # 2. Pass the explicitly unpacked dictionary as kwargs
    bare_run = bootstrap_from_path(run_dir=run_dir,
                                   run_id=config['run_id'],
                                   el_field=exp_params['el_field'],
                                   target_isotope=exp_params['target_isotope'])
    
    # 3. Enrich the Run with physical parameters from the config
    enriched_run = replace(bare_run,
                           pressure=exp_params['pressure'],
                           temperature=exp_params['temperature'],
                           sampling_rate=exp_params['sampling_rate'],
                           el_gap=exp_params['el_gap'],
                           drift_gap=exp_params['drift_gap'],
                           recoil_energy=exp_params.get('recoil_energy', 96.8))
    return enriched_run
