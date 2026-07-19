from functools import wraps
from typing import Callable, Any, TypeVar
from pathlib import Path
import numpy as np
import time
import sys
from functools import wraps
from dataclasses import replace

from RaTag.io import file_ops
from RaTag.core.paths import get_output_root
from RaTag.core.datatypes import SetAlpha, SetPmt, Run

# Type variable for type hinting in decorators, strictly for SetPmt or Run
T = TypeVar('T', Run, SetPmt, SetAlpha)

def require_attributes(*target_attrs: str):
    """
    Decorator to check if a SetPmt has a required attribute before executing the function.
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(set_or_run: T, *args, **kwargs) -> T:
            # Find any required fields that are None or missing
            missing = [x for x in target_attrs if getattr(set_or_run, x, None) is None]
            
            if missing:
                obj_name = getattr(set_or_run, 'source_dir', None) or getattr(set_or_run, 'run_id', 'unknown')
                raise ValueError(
                    f"Cannot run '{func.__name__}' on {obj_name}. "
                    f"Missing required attributes: {missing}" )
            return func(set_or_run, *args, **kwargs)
        return wrapper
    return decorator 

def limit_frames(func: Callable[..., Any]):
    """
    Translates a requested `max_frames` limit into the actual `max_files` 
    required, based on the SetPmt's FastFrame geometry.
    """
    @wraps(func)
    def wrapper(set: T, max_frames: int = None, **kwargs) -> Any:
        
        if max_frames is None:
            max_files = None
        else:
            # Round up to ensure we process complete files
            max_files = int(np.ceil(max_frames / set.nframes))
        # Call the pure compute function, injecting max_files instead of max_frames
        return func(set, max_files=max_files, **kwargs)
        
    return wrapper

def track_iterator_progress(func):
    """
    Passive observer decorator. Adds an in-place ETA progress bar to file iterators.
    It does not manage data or slicing; it only sniffs kwargs to calculate the denominator.
    """
    @wraps(func)
    def wrapper(set_obj, *args, **kwargs):
        show_progress = kwargs.get('show_progress', False)
        
        # 1. Instantly call the original generator (it handles its own logic and slicing)
        generator = func(set_obj, *args, **kwargs)
        
        if not show_progress:
            yield from generator
            return

        # 2. Passively sniff the denominator for the progress bar UI
        max_files = kwargs.get('max_files', None)
        total = min(len(set_obj.filenames), max_files) if max_files else len(set_obj.filenames)
        prefix = f"Iterating {set_obj.source_dir.name}"
        
        if total == 0:
            yield from generator
            return

        # 3. The Interception Loop
        start_time = time.time()
        for i, item in enumerate(generator, 1):
            yield item 
            
            # Calculate metrics when control returns
            elapsed = time.time() - start_time
            percent = (i / total) * 100
            eta_seconds = (elapsed / i) * (total - i)
            
            m, s = divmod(int(eta_seconds), 60)
            h, m = divmod(m, 60)
            eta_str = f"{h:02d}:{m:02d}:{s:02d}" if h > 0 else f"{m:02d}:{s:02d}"
            
            sys.stdout.write(f"\r  {prefix}: {i}/{total} ({percent:.1f}%) | ETA: {eta_str}   ")
            sys.stdout.flush()
            
        sys.stdout.write("\n")
        
    return wrapper

def allow_force(func):
    """
    Intercepts 'force'. If True, bypasses all downstream loader decorators 
    by walking the __wrapped__ closure chain until it finds a non-loader.
    """
    @wraps(func)
    def wrapper(*args, force: bool = False, **kwargs):
        if force:
            target = func
            # Programmatically dig past any decorator marked as a loader
            while getattr(target, "is_loader", False):
                target = getattr(target, "__wrapped__")
            
            # Call the remaining stack (Writers + Core Function) directly
            return target(*args, **kwargs)
        
        # force=False: Proceed normally through the loaders
        return func(*args, **kwargs)
    return wrapper

def load_cached_metadata(target_attr: str):
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt: SetPmt, *args, **kwargs):
            cached_set = file_ops.load_cache(set_pmt)
            if cached_set and getattr(cached_set, target_attr, None) is not None:
                print(f"  📂 Cache Hit: Loaded metadata for '{target_attr} in {set_pmt.source_dir.name}'")
                return cached_set
                
            return func(set_pmt, *args, **kwargs)
        
        # Tag the wrapper so @allow_recompute knows it can be bypassed
        wrapper.is_loader = True
        return wrapper
    return decorator

def write_metadata(target_attr: str):
    """Writer: Saves the JSON metadata cache upon successful execution."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt: SetPmt, *args, **kwargs):
            # Execute downstream stack
            result = func(set_pmt, *args, **kwargs)
            
            # Unpack safely depending on what the downstream layer returned
            updated_set = result[0] if isinstance(result, tuple) else result
            
            # Disk save
            file_ops.save_cache(updated_set)
            print(f"  ✓ {updated_set.source_dir.name}: Computed '{target_attr}' and stored metadata")
            
            return updated_set
        return wrapper
    return decorator

def load_cached_npz(signal_type: str):
    """Loader: Checks for dense numpy arrays on disk. Returns early if hit."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt: SetPmt, *args, **kwargs):
            
            arrays = file_ops.load_npz_arrays(set_pmt, signal_type)
            if arrays is not None and len(arrays) > 0:
                print(f"  📂 {set_pmt.source_dir.name}: Loaded '{signal_type}' arrays from disk")
                return set_pmt
                
            return func(set_pmt, *args, **kwargs)
            
        wrapper.is_loader = True # Tag for @allow_recompute traversal
        return wrapper
    return decorator

def write_npz_arrays(signal_type: str):
    """Writer: Intercepts the arrays tuple and dumps to disk."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt: SetPmt, *args, **kwargs):
            result = func(set_pmt, *args, **kwargs)
            
            # Extract the numpy arrays tuple
            updated_set, arrays = result if isinstance(result, tuple) else (result, None)
            # Save to disk
            if arrays is not None and len(arrays) > 0:
                file_ops.save_npz_arrays(updated_set, signal_type, arrays)
                
            return updated_set
        return wrapper
    return decorator

def persist_run_results(signal_type: str):
    """Decorator to automatically save results at the Run level."""
    def decorator(func):
        @wraps(func)
        def wrapper(run: Run, *args, **kwargs):
            # 1. Run the compute logic
            updated_run, arrays = func(run, *args, **kwargs)

            # 2. Delegate to file_ops helper to save the dense arrays (e.g., timings, areas)
            file_ops.save_run_npz_arrays(updated_run, signal_type, arrays)
            return updated_run 
        return wrapper
    return decorator

# ---------------------------------------------------------
#  Multi-Isotope I/O Spawn Decorators
# ---------------------------------------------------------

def load_cached_isotope_arrays(signal_type: str):
    """Loader: Checks if ALL expected isotope clones already exist on disk."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt, set_alpha, *args, **kwargs):
            if not getattr(set_alpha, 'isotope_ranges_E', None):
                return []
            
            # Predict what the spawned sets will look like
            expected_clones = [
                replace(set_pmt, target_isotope=iso, multiiso=True) 
                for iso in set_alpha.isotope_ranges_E.keys()
            ]
            
            if all(file_ops.check_npz_exists(clone, signal_type) for clone in expected_clones):
                print(f"  ⏭ Skipping separation for {set_pmt.source_dir.name} (Multi-iso clones already exist)")
                return expected_clones
                
            return func(set_pmt, set_alpha, *args, **kwargs)
        wrapper.is_loader = True
        return wrapper
    return decorator

def write_isotope_arrays(signal_type: str):
    """Writer: Iterates over the spawned list and saves each array tuple."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt, set_alpha, *args, **kwargs):
            # Expects a list of tuples: [(cloned_set, arrays), ...]
            isotope_data_list = func(set_pmt, set_alpha, *args, **kwargs)
            
            completed_sets = []
            for cloned_set, arrays in isotope_data_list:
                file_ops.save_npz_arrays(cloned_set, signal_type, arrays)
                completed_sets.append(cloned_set)
                
            return completed_sets # The orchestrator flattens this into run.sets
        return wrapper
    return decorator

# ---------------------------------------------------------
# PLOTS: Split into Loader and Writer
# ---------------------------------------------------------

def load_cached_plots(subfolder: str, expected_suffixes: list[str]):
    """Loader: Checks if expected plot files exist. Returns early if hit."""
    def decorator(func):
        @wraps(func)
        def wrapper(obj, *args, **kwargs):
            is_run = hasattr(obj, 'run_id')
            name = obj.run_id if is_run else obj.source_dir.name
            root = get_output_root(obj) if is_run else get_output_root(obj.source_dir.parent)
            out_dir = root / "plots" / subfolder
            
            target_files = [out_dir / Path(s).parent / f"{name}_{Path(s).name}.png" for s in expected_suffixes]
            
            # If all plots exist, bypass the core function and return the immutable object immediately
            if all(f.exists() for f in target_files):
                print(f"  ⏭ Skipping plots for {name} (already exist)")
                return obj
                
            return func(obj, *args, **kwargs)
            
        wrapper.is_loader = True  # Tag for @allow_recompute traversal
        return wrapper
    return decorator

def write_plots(subfolder: str):
    """Writer: Handles matplotlib memory management and file I/O."""
    def decorator(func):
        @wraps(func)
        def wrapper(obj, *args, **kwargs):
            # Execute the downstream stack. 
            # Expects the core function to return (updated_obj, figures_dict)
            result = func(obj, *args, **kwargs)
            updated_obj, figures_dict = result if isinstance(result, tuple) else (result, {})
            
            is_run = hasattr(updated_obj, 'run_id')
            name = updated_obj.run_id if is_run else updated_obj.source_dir.name
            root = get_output_root(updated_obj) if is_run else get_output_root(updated_obj.source_dir.parent)
            out_dir = root / "plots" / subfolder
            
            for suffix_path, fig in figures_dict.items():
                if fig is not None:
                    p = Path(suffix_path)
                    file_path = out_dir / p.parent / f"{name}_{p.name}.png"
                    
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_ops.save_figure(fig, file_path) 
                    
                    import matplotlib.pyplot as plt
                    plt.close(fig) 
                    
            return updated_obj
        return wrapper
    return decorator

# ---------------------------------------------------------
# FITS: Split into Loader and Writer
# ---------------------------------------------------------

def load_cached_fit(suffix: str):
    """Loader: Checks if fit JSON exists. Returns early if hit."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt, *args, **kwargs):
            out_dir = get_output_root(set_pmt.source_dir.parent) / "fits"
            out_path = out_dir / f"{set_pmt.source_dir.name}_{suffix}.json"
            
            if out_path.exists():
                print(f"  ⏭ Skipping fit '{suffix}' for {set_pmt.source_dir.name} (already exists)")
                return set_pmt
                
            return func(set_pmt, *args, **kwargs)
            
        wrapper.is_loader = True # Tag for @allow_recompute traversal
        return wrapper
    return decorator

def write_fit(suffix: str):
    """Writer: Extracts lmfit result and writes JSON to disk."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt, *args, **kwargs):
            updated_set, model_result = func(set_pmt, *args, **kwargs)
            
            if model_result is not None:
                out_dir = get_output_root(updated_set.source_dir.parent) / "fits"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{updated_set.source_dir.name}_{suffix}.json"
                file_ops.save_fit_result(model_result, out_path)
                
            return updated_set 
        return wrapper
    return decorator

def load_cached_alpha_fits(suffix: str = "alpha_fits"):
    """Loader: Checks if alpha peak fits exist. Returns early if hit."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_alpha, *args, **kwargs):
            # We just check if the metadata knows the fit succeeded
            if getattr(set_alpha, 'alpha_fit_success', False):
                print(f"  ⏭ Skipping alpha fits for {set_alpha.source_dir.name} (already exist)")
                return set_alpha
            return func(set_alpha, *args, **kwargs)
        wrapper.is_loader = True 
        return wrapper
    return decorator

def write_alpha_fits(suffix: str = "alpha_fit"):
    """Writer: Iterates a dict of lmfit results and writes them to flat JSONs."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_alpha, *args, **kwargs):
            result = func(set_alpha, *args, **kwargs)
            updated_set, fit_dict = result if isinstance(result, tuple) else (result, None)
            
            if fit_dict is not None:
                out_dir = get_output_root(updated_set.source_dir.parent) / "fits" / "alpha_fits"
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # Save each peak as a flat file: e.g., 'Ch4_noSCA_Th228_alpha_fit.json'
                for peak_name, model_result in fit_dict.items():
                    out_path = out_dir / f"{updated_set.source_dir.name}_{peak_name}_{suffix}.json"
                    file_ops.save_fit_result(model_result, out_path)
                
            return updated_set 
        return wrapper
    return decorator