"""
Utility functions for timing and profiling.
"""

from time import perf_counter
from contextlib import contextmanager
from typing import Dict


# Global timing dictionary (optional - use if you want to track across calls)
_timing_store: Dict[str, float] = {}


@contextmanager
def timer(name: str, store_dict: Dict[str, float] = None, verbose: bool = True):
    """
    Context manager for timing code blocks.
    
    Args:
        name: Name of the timed operation
        store_dict: Optional dictionary to store timings in (if None, uses global _timing_store)
        verbose: If True, prints timing result immediately
    
    Example:
        >>> with timer("Data loading"):
        ...     data = load_data()
        [2.34s] Data loading
    """
    start = perf_counter()
    yield
    elapsed = perf_counter() - start
    
    # Store in provided dict or global store
    if store_dict is not None:
        store_dict[name] = elapsed
    else:
        _timing_store[name] = elapsed
    
    if verbose:
        print(f"  [{elapsed:.2f}s] {name}")


def get_timings(store_dict: Dict[str, float] = None) -> Dict[str, float]:
    """
    Get collected timings.
    
    Args:
        store_dict: Dictionary to retrieve timings from (if None, returns global timings)
    
    Returns:
        Dictionary of timing results
    """
    if store_dict is not None:
        return store_dict
    return _timing_store.copy()


def print_timing_summary(store_dict: Dict[str, float] = None, title: str = "Timing Summary"):
    """
    Print a formatted summary of timings, sorted by duration.
    
    Args:
        store_dict: Dictionary of timings (if None, uses global timings)
        title: Title for the summary
    
    Example:
        >>> print_timing_summary()
        Timing Summary:
        ============================================================
          Advanced features         :  45.23s (60.1%)
          Data loading              :  12.34s (16.4%)
          Sequence creation         :   8.12s (10.8%)
          Basic features            :   5.67s ( 7.5%)
    """
    timings = get_timings(store_dict)
    
    if not timings:
        print("No timings collected.")
        return
    
    total = sum(timings.values())
    
    print(f"\n{title}:")
    print("=" * 60)
    for name, elapsed in sorted(timings.items(), key=lambda x: x[1], reverse=True):
        pct = elapsed / total * 100
        print(f"  {name:25s}: {elapsed:6.2f}s ({pct:5.1f}%)")
    print(f"{'Total':25s}: {total:6.2f}s")


def clear_timings(store_dict: Dict[str, float] = None):
    """
    Clear collected timings.
    
    Args:
        store_dict: Dictionary to clear (if None, clears global timings)
    """
    if store_dict is not None:
        store_dict.clear()
    else:
        _timing_store.clear()