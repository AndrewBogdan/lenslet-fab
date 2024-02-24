"""util.py

Misc. Utilities
"""

from typing import Optional, Tuple

import logging
import functools
import os
import sys
import warnings

from matplotlib import pyplot as plt
import numpy as np


# --- String Formatting -------------------------------------------------------
POSITION_FORMAT_CSV = '{0:>10.3f}, {1:>10.3f}, {2:>10.3f}, ' \
                      '{3:>10.3f}, {4:>10.3f}, {5:>10.3f}'
POSITION_FORMAT_PRINT = 'X: {:.1f} um\tY: {:.1f} um\tZ: {:.1f} um\t' \
                      'N: {:.2f}°\tP: {:.2f}°\tV: {:.2f}°'


def fmt_position(pos: Tuple, fmt: str = 'print') -> str:
    """Formats a 6-axis position.

    Supply 'fmt' for the following options:
    - 'csv': Comma-separated values.
    """
    match fmt:
        case 'csv':
            return POSITION_FORMAT_CSV.format(*pos)
        case 'print':
            return POSITION_FORMAT_PRINT.format(*pos)


# --- Logging -----------------------------------------------------------------
LOGFILE_LASES = os.path.join(os.path.split(sys.prefix)[0], 'lases.csv')
_formatter_fileout = logging.Formatter('%(asctime)s, %(message)s')
_handler_laselog = logging.FileHandler(LOGFILE_LASES)
_handler_laselog.setFormatter(_formatter_fileout)


def get_lase_logger(name: str):
    """Get a logger to log lases and related output. Pass it __name__."""
    logger = logging.getLogger(name + '.lases')
    logger.addHandler(_handler_laselog)
    logger.propagate = False
    return logger


# --- Plotting ----------------------------------------------------------------
def plot_histogram(data, ax=None, goal: Optional[float] = None):
    """Plots data in a histogram."""
    edge_dists = data

    mean_dist = np.mean(edge_dists)
    median_dist = np.median(edge_dists)

    ax = plt.gca() if ax is None else ax
    ax.hist(edge_dists, bins=40)
    ax.set_ylabel('Count')
    ax.set_xlabel('Center-to-center distance (nm)')
    ax.axvline(mean_dist, color='black')
    ax.axvline(median_dist, color='red')
    ax.text(mean_dist + 0.5, ax.get_ylim()[1] - 100, 'mean', color='black',
            rotation='vertical')
    ax.text(median_dist + 0.5, ax.get_ylim()[1] - 100, 'median',
            color='red', rotation='vertical')
    if goal is not None:
        ax.axvline(goal, color='green')
        ax.text(goal + 0.5, ax.get_ylim()[1] - 100, 'goal', color='green',
                rotation='vertical')
    return ax


# --- Other Utility Functions -------------------------------------------------
def depreciate(func):
    @functools.wraps(func)
    def depreciated_func(*args, **kwargs):
        warnings.warn(f'Function {func.__name__} is depreciated',
                      category=DeprecationWarning)
        return func(*args, **kwargs)
    return depreciated_func



