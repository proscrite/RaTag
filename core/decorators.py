from functools import wraps
from typing import Callable, Any, List, TypeVar
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
        def wrapper(set_pmt: SetPmt, *args, **kwargs) -> SetPmt:
            
            # 1. MEMORY CHECK: Is it already done?
            if getattr(set_pmt, target_attr, None) is not None:
                return set_pmt
                
            # 2. DISK CHECK: Did a previous run already compute this?
            cached_set = file_ops.load_cache(set_pmt)
            if cached_set and getattr(cached_set, target_attr, None) is not None:
                print(f"  📂 {set_pmt.source_dir.name}: Loaded '{target_attr}' from cache")
                return cached_set
                
            # 3. COMPUTE: Execute the pure workflow function
            enriched_set = func(set_pmt, *args, **kwargs)
            
            # 4. DISK SAVE: Update the JSON cache immediately
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

def persist_plots(subfolder: str):
    """Decorator to automate saving Matplotlib figures and managing memory."""
    def decorator(func):
        @wraps(func)
        def wrapper(set_pmt: SetPmt, *args, **kwargs):
            # 1. Generate the figures
            updated_set, figures_dict = func(set_pmt, *args, **kwargs)
            
            # 2. Setup unified directory
            out_dir = get_output_root(set_pmt.source_dir.parent) / "plots" / subfolder
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # 3. Save and close
            for suffix, fig in figures_dict.items():
                if fig is not None:
                    file_path = out_dir / f"{set_pmt.source_dir.name}_{suffix}.png"
                    # save_figure handles the actual saving logic you already wrote
                    file_ops.save_figure(fig, file_path) 
                    import matplotlib.pyplot as plt
                    plt.close(fig) # Critical to prevent memory leaks in loops
                    
            return updated_set
        return wrapper
    return decorator