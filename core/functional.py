"""Functional programming utilities for pipeline composition."""

from typing import Callable, TypeVar
from functools import reduce

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