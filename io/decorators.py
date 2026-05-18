from functools import wraps
from typing import Callable, Any, List, TypeVar

from RaTag.io import file_ops
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