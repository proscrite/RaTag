"""Functional programming utilities for pipeline composition."""

from typing import Callable, TypeVar
from functools import reduce
from dataclasses import replace
from pathlib import Path

T = TypeVar('T')


def pipe_run(value: T, *funcs: Callable[[T], T]) -> T:
    """
    Apply functions sequentially to a value (left-to-right).
    
    Example:
        result = pipe(
            initial_value,
            func1,
            func2,
            func3
        )
    
    Equivalent to: func3(func2(func1(initial_value)))
    
    Args:
        value: Initial value to transform
        *funcs: Functions to apply sequentially
        
    Returns:
        Final transformed value
    """
    return reduce(lambda v, f: f(v), funcs, value)


def compose(*funcs: Callable) -> Callable:
    """
    Compose functions (right-to-left).
    
    Example:
        pipeline = compose(func3, func2, func1)
        result = pipeline(initial_value)
    
    Equivalent to: func3(func2(func1(initial_value)))
    
    Args:
        *funcs: Functions to compose
        
    Returns:
        Composed function
    """
    return lambda x: reduce(lambda v, f: f(v), reversed(funcs), x)



def map_over(items: list[T], fn: Callable[[T], T], 
             catch_errors: bool = False,
             error_msg: str = "Operation failed") -> list[T]:
    """
    Apply function to each item in a list.
    
    Args:
        items: List of items to process
        fn: Function to apply to each item
        catch_errors: If True, catch exceptions and keep original item
        error_msg: Error message prefix
        
    Returns:
        List of transformed items
    """
    if not catch_errors:
        return [fn(item) for item in items]
    
    results = []
    for i, item in enumerate(items, 1):
        try:
            results.append(fn(item))
        except Exception as e:
            print(f"  ⚠ Item {i} failed: {e}")
            results.append(item)  # Keep original on error
    return results

# -------------------------------
# Persistence: save decorator  --
# -------------------------------

def with_persistence(fn: Callable[[T], T], 
                     save_fn: Callable[[T], None]) -> Callable[[T], T]:
    """
    Wrap a function to automatically save its result.
    
    Args:
        fn: Function to wrap
        save_fn: Function to save the result
        
    Returns:
        Wrapped function that saves after execution
    """
    def wrapped(item: T) -> T:
        result = fn(item)
        save_fn(result)
        return result
    return wrapped


# ============================================================================
# RUN-LEVEL ORCHESTRATION
# ============================================================================

def apply_workflow_to_run(run,
                          workflow_func: Callable,
                          workflow_name: str,
                          cache_key: str,
                          data_file_suffix: str,
                          **workflow_kwargs):
    """
    Generic run-level workflow orchestration with caching.
    
    Applies a set-level workflow to all sets in a run, with automatic:
    - Directory setup (plots_dir, data_dir)
    - Metadata-based caching
    - Error handling
    
    Args:
        run: Run object to process
        workflow_func: Set-level workflow function
        workflow_name: Display name for logging (e.g., "S1 TIMING ESTIMATION")
        cache_key: Metadata key to check for caching (e.g., "t_s1")
        data_file_suffix: Suffix for data file (e.g., "s1.npz")
        **workflow_kwargs: Additional arguments passed to workflow_func
        
    Returns:
        Updated Run with processed sets
        
    Example:
        >>> run = apply_workflow_to_run(
        ...     run,
        ...     workflow_s1_timing,
        ...     "S1 TIMING ESTIMATION",
        ...     cache_key="t_s1",
        ...     data_file_suffix="s1.npz",
        ...     max_frames=200
        ... )
    """
    from RaTag.core.dataIO import load_set_metadata
    
    print("\n" + "="*60)
    print(workflow_name.upper())
    print("="*60)
    
    # Setup data directory (always needed)
    data_dir = run.root_directory / "processed_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup plots directory
    plots_dir = run.root_directory / "plots"
    
    updated_sets = []
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")
        
        # Check cache
        data_file = data_dir / f"{set_pmt.source_dir.name}_{data_file_suffix}"
        print(f"  Checking cache at {data_file}...")
        loaded = load_set_metadata(set_pmt)
        
        if loaded and cache_key in loaded.metadata and data_file.exists():
            print(f"  📂 Loaded from cache")
            updated_sets.append(loaded)
            continue
        # Run workflow
        try:
            updated_set = workflow_func(set_pmt, **workflow_kwargs)
            updated_sets.append(updated_set)
        except Exception as e:
            print(f"  ⚠ Failed: {e}")
            updated_sets.append(set_pmt)
    
    return replace(run, sets=updated_sets)


def map_isotopes_in_run(run,
                       workflow_func: Callable,
                       workflow_name: str,
                       isotope_ranges: dict,
                       **workflow_kwargs):
    """
    Generic run-level isotope mapping orchestration.
    
    Applies an isotope mapping workflow to all sets in a run.
    Typically used after initial processing to split results by isotope.
    
    Args:
        run: Run object to process
        workflow_func: Set-level isotope mapping function
        workflow_name: Display name for logging (e.g., "S1 ISOTOPE MAPPING")
        isotope_ranges: Dictionary of {isotope: (Emin, Emax)}
        **workflow_kwargs: Additional arguments passed to workflow_func
        
    Returns:
        Updated Run (sets typically unchanged, side effects are saved files)
        
    Example:
        >>> run = map_isotopes_in_run(
        ...     run,
        ...     workflow_s1_multiiso,
        ...     "S1 ISOTOPE MAPPING",
        ...     isotope_ranges={"Th228": (5000, 7000)}
        ... )
    """
    print("\n" + "="*60)
    print(workflow_name.upper())
    print("="*60)

    updated_sets = []
    for i, set_pmt in enumerate(run.sets, 1):
        print(f"\nSet {i}/{len(run.sets)}: {set_pmt.source_dir.name}")

        try:
            workflow_func(set_pmt,
                         isotope_ranges=isotope_ranges,
                         **workflow_kwargs)
        except Exception as e:
            import traceback
            print(f"  ⚠ Failed: {e}")
            # print(f"  Full traceback:")
            # traceback.print_exc()

        updated_sets.append(set_pmt)

    return replace(run, sets=updated_sets)
