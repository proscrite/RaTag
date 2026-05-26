from functools import wraps
from typing import Callable, Any, List, TypeVar
from pathlib import Path
import numpy as np

from RaTag.io import file_ops
from RaTag.core.paths import get_output_root
from RaTag.core.datatypes import SetPmt, Run

# Type variable for type hinting in decorators, strictly for SetPmt or Run
T = TypeVar('T', Run, SetPmt)

def disk_cache(target_attr: str):
    """
    Decorator that memoizes a SetPmt operation to disk.
    target_attr: The attribute that signifies this step is complete (e.g., 'time_drift').
    """
    def decorator(func: Callable[..., SetPmt]):
        @wraps(func)
        def wrapper(set_pmt: SetPmt, *args, force: bool = False, **kwargs) -> SetPmt:
            #  1. Is the decorator being overridden with force=True? If so, skip all checks and recompute.
            if force:
                print(f"  ⚡ Force enabled: Recomputing {target_attr}")
            else:
                
                # 2. MEMORY CHECK: Is it already done?
                if getattr(set_pmt, target_attr, None) is not None:
                    return set_pmt
                    
                # 3. DISK CHECK: Did a previous run already compute this?
                cached_set = file_ops.load_cache(set_pmt)
                if cached_set and getattr(cached_set, target_attr, None) is not None:
                    print(f"  📂 {set_pmt.source_dir.name}: Loaded '{target_attr}' from cache")
                    return cached_set
                
            # 4. COMPUTE: Execute the pure workflow function
            enriched_set = func(set_pmt, *args, **kwargs)
            
            # 5. DISK SAVE: Update the JSON cache immediately
            file_ops.save_cache(enriched_set)
            print(f"  ✓ {set_pmt.source_dir.name}: Computed '{target_attr}' and cached")
            
            return enriched_set
            
        return wrapper
    return decorator

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
    def wrapper(set_pmt: SetPmt, max_frames: int = None, **kwargs) -> Any:
        
        if max_frames is None:
            max_files = None
        else:
            # Round up to ensure we process complete files
            max_files = int(np.ceil(max_frames / set_pmt.nframes))
            
        # Call the pure compute function, injecting max_files instead of max_frames
        return func(set_pmt, max_files=max_files, **kwargs)
        
    return wrapper

def persist_results(signal_type: str):
    """Decorator to automatically save results to npz and metadata."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 1. Run the compute logic
            updated_set, payload = func(*args, **kwargs)
            
            # 2. Extract Data Dir (standardized)
            data_dir = get_output_root(updated_set.source_dir.parent)
            data_dir.mkdir(parents=True, exist_ok=True)
            data_file = data_dir / f"{updated_set.source_dir.name}_{signal_type}.npz"
            
            # 3. Flattened persistence
            
            np.savez_compressed(data_file, **payload)
            print(f"    💾 Saved to {data_file.relative_to(data_dir.parent)}")
            return updated_set 
        return wrapper
    return decorator

def persist_plots(subfolder: str, expected_suffixes: list[str]):
    """
    Decorator to cache Matplotlib figures, auto-save them, and manage RAM.
    Checks for file existence before running to save time, with force override.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(obj, *args, force: bool = False, **kwargs):
            # 1. Determine if we are wrapping a Run or a SetPmt
            is_run = hasattr(obj, 'run_id')
            name = obj.run_id if is_run else obj.source_dir.name
            root = get_output_root(obj) if is_run else get_output_root(obj.source_dir.parent)
            out_dir = root / "plots" / subfolder
            
            # 2. Check Cache
            target_files = [out_dir / Path(s).parent / f"{name}_{Path(s).name}.png" for s in expected_suffixes]
            if not force and all(f.exists() for f in target_files):
                print(f"  ⏭ Skipping plots for {name} (already exist)")
                return obj
                
            if force:
                print(f"  ⚡ Force enabled: Regenerating plots for {name}")

            # 3. Execute the Plotting Function
            updated_obj, figures_dict = func(obj, *args, **kwargs)
            
            # 4. Save and Close (RAM Management)
            for suffix_path, fig in figures_dict.items():
                if fig is not None:
                    p = Path(suffix_path)
                    file_path = out_dir / p.parent / f"{name}_{p.name}.png"
                    
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    file_ops.save_figure(fig, file_path) 
                    
                    # Import locally to avoid global dependencies where unneeded
                    import matplotlib.pyplot as plt
                    plt.close(fig) 
                    
            return updated_obj
        return wrapper
    return decorator