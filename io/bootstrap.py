from dataclasses import replace
import fnmatch
from pathlib import Path
import time
from typing import Union, Optional, Tuple, List
import json
from RaTag.io import file_ops

from RaTag.core.datatypes import Run, SetPmt, SetAlpha
from RaTag.core.paths import get_output_root

def _sniff_fastframe_cache(dir_path: Path) -> tuple[Optional[bool], Optional[int]]:
    """
    Attempts to read fastframe properties from the metadata JSON.
    This is needed because of the chickend and egg problem in bootstrapping: we need to know if the set is fastframe to avoid 
    peeking a wfm file, but the fastframe properties are only known after reading a wfm file. Unless a cache JSON file exists.
    """
    summaries_dir = get_output_root(dir_path.parent) / "set_summaries"

    # Check for either the PMT or Alpha cache (whichever pipeline ran first)
    possible_caches = [
        summaries_dir / f"{dir_path.name}_metadata.json",
        summaries_dir / f"{dir_path.name}_alphas_metadata.json"
    ]
    
    for cache_path in possible_caches:
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                    if 'ff' in data and 'nframes' in data:
                        return data['ff'], data['nframes']
            except Exception:
                continue # If one file is corrupted, try the other
                
    return None, None


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

def _detect_sensor_sets(dir_path: Path, pmt_pattern: str = "*_CH3.wfm", alpha_pattern: str = "*_CH4.wfm") -> dict:
    """
    Determines the types of sets present in a physical directory based on naming conventions.
    Returns: {'pmt': list[Path], 'alpha': list[Path], 'multiiso': bool}
    """
    all_files = [f for f in dir_path.iterdir() if f.is_file() and f.suffix.lower() == '.wfm']
    
    if not all_files:
        return {'pmt': [], 'alpha': [], 'multiiso': False}
    
    # RULE 1: Directory-name override for Single-Isotope Alpha sets
    if dir_path.name.startswith('Ch4_'):
        return {'pmt': [], 'alpha': all_files, 'multiiso': False}
        return {'pmt': [], 'alpha': [], 'multiiso': False}

    pmt_files = sorted([f for f in all_files if fnmatch.fnmatch(f.name, pmt_pattern)])
    alpha_files = sorted([f for f in all_files if fnmatch.fnmatch(f.name, alpha_pattern)])

    all_files_sorted = sorted(all_files)

    if set(pmt_files) == set(alpha_files):
        return {'pmt': all_files, 'alpha': [], 'multiiso': False}

    # RULE 3: Multiplexed Multi-Isotope resolution
    if pmt_files and alpha_files:
        return {'pmt': pmt_files, 'alpha': alpha_files, 'multiiso': True}
        
    # RULE 4: Standard fallbacks
    elif alpha_files and not pmt_files:
        return {'pmt': [], 'alpha': alpha_files, 'multiiso': False}
    else: 
        return {'pmt': pmt_files or all_files, 'alpha': [], 'multiiso': False}
    
def bootstrap_bare_pmt_set(dir_path: Path, filenames: List[Path], multiiso: bool) -> SetPmt:
    """Explicitly builds a SetPmt from a directory and a list of PMT waveform files."""
    voltage_dict = file_ops.parse_subdir_name(str(dir_path.name))

    ff, nframes = _sniff_fastframe_cache(dir_path) # Try to fetch from JSON cache first

    if ff is None or nframes is None:
        ff, nframes = file_ops.detect_fastframe_properties(dir_path, filenames)  # Fallback for first time bootstrapping (cache missing)
    
    return SetPmt(source_dir=dir_path,
                  filenames=[f.name for f in filenames], 
                  gate=voltage_dict.get('gate'),
                  anode=voltage_dict.get('anode'),
                  sampling_rate=voltage_dict.get('sampling_rate'),
                  multiiso=multiiso,
                  ff=ff, nframes=nframes)


def bootstrap_bare_alpha_set(dir_path: Path, filenames: List[Path], multiiso: bool) -> SetAlpha:
    """Explicitly builds a SetAlpha from a directory and a list of Silicon waveform files."""
    
    print(f"  🔹 Bootstrapping Alpha Set from: {dir_path.name} with {len(filenames)} waveforms.")
    voltage_dict = file_ops.parse_subdir_name(str(dir_path.name)) # Try to fetch from JSON cache first
    ff, nframes = _sniff_fastframe_cache(dir_path)

    if ff is None or nframes is None:
        ff, nframes = file_ops.detect_fastframe_properties(dir_path, filenames)  # Fallback for first time bootstrapping (cache missing)

    return SetAlpha(source_dir=dir_path,
                    filenames=[f.name for f in filenames], 
                    gate=voltage_dict.get('gate'),
                    anode=voltage_dict.get('anode'),
                    sampling_rate=voltage_dict.get('sampling_rate'),
                    multiiso=multiiso,
                    ff=ff, nframes=nframes)


def bootstrap_from_path(run_dir: Union[str, Path], run_id: Optional[str] = None, 
                        el_field: Optional[float] = None, target_isotope: Optional[str] = None,
                        pmt_pattern: str = "*_CH3.wfm", 
                        alpha_pattern: str = "*_CH4.wfm") -> Run:
    """
    DECLARATIVE PIPELINE: Assembles a Run object from a raw data directory.
    
    Args:
        run_dir: Path to the raw data directory.
    """
    run_dir = Path(run_dir)
    
    run_id, el_field, target_isotope = _resolve_bootstrapping_params(run_dir, run_id, el_field, target_isotope)
    
    # Scan for valid subdirectories (ignoring hidden folders, etc.)
    raw_set_dirs = [d for d in run_dir.iterdir() if d.is_dir() and not d.name.startswith('.')]
    
    print(f"Found {len(raw_set_dirs)} directories in {run_dir.name}")

    pmt_sets = []
    alpha_sets = []
    
    for dir_path in sorted(raw_set_dirs):
        set_type = _detect_sensor_sets(dir_path, pmt_pattern, alpha_pattern)   # This is a dict with keys 'pmt', 'alpha', and 'multiiso'
        
        if set_type['pmt']:
            pmt_sets.append(bootstrap_bare_pmt_set(dir_path, set_type['pmt'], set_type['multiiso']))
        if set_type['alpha']:
            alpha_sets.append(bootstrap_bare_alpha_set(dir_path, set_type['alpha'], set_type['multiiso']))
    
    is_multiiso = any(s['multiiso'] for s in [set_type])  # If any set is multi-isotope, flag the whole run
    print(f"  → Bootstrapped {len(pmt_sets)} PMT Sets and {len(alpha_sets)} Alpha Sets. (Multi-Isotope: {is_multiiso})")

    return Run(run_id=run_id, 
               root_directory=run_dir, 
               el_field=el_field,
               target_isotope=target_isotope,
               multiiso=is_multiiso,  # If any set is multi-isotope, flag the whole run
               sets=pmt_sets,         # Keeping 'sets' mapped to PMT for legacy compatibility
               alpha_sets=alpha_sets)


def bootstrap_from_config(config_path: Union[str, Path]) -> Run:
    """
    WRAPPER: Reads a YAML config and passes the physical parameters 
    down to the path builder.
    """
    # 1. IO: Load the dictionary
    config = file_ops.load_yaml(Path(config_path))
    
    run_dir = Path(config['data']['raw_data_path'])
    exp_params = config['experiment']

    data_config = config.get('data', {})
    pmt_pattern = data_config.get('pmt_pattern', '*_Ch1.wfm')
    alpha_pattern = data_config.get('alpha_pattern', '*_Ch4.wfm')
    # 2. Pass the explicitly unpacked dictionary as kwargs
    bare_run = bootstrap_from_path(run_dir=run_dir,
                                   run_id=config['run_id'],
                                   el_field=exp_params['el_field'],
                                   target_isotope=exp_params['target_isotope'],
                                   pmt_pattern=pmt_pattern,
                                   alpha_pattern=alpha_pattern)
    
    # 3. Enrich the Run with physical parameters from the config
    enriched_run = replace(bare_run,
                           pressure=exp_params['pressure'],
                           temperature=exp_params['temperature'],
                           sampling_rate=exp_params['sampling_rate'],
                           el_gap=exp_params['el_gap'],
                           drift_gap=exp_params['drift_gap'],
                           recoil_energy=exp_params.get('recoil_energy', 96.8))
    return enriched_run
