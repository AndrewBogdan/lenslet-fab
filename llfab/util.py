"""util.py

Misc. Utilities
"""

import warnings
import functools


def depreciate(func):
    @functools.wraps(func)
    def depreciated_func(*args, **kwargs):
        warnings.warn(f'Function {func.__name__} is depreciated',
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    return depreciated_func
