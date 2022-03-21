"""gas.py

A short module for controlling the gas.
"""

import contextlib
import logging


_logger = logging.getLogger(__name__)


@contextlib.contextmanager
def gas():
    try:
        open_gas()
        yield None
    finally:
        close_gas()


def open_gas():
    _logger.info('Opening the gas line.')
    pass


def close_gas():
    _logger.info('Closing the gas line.')
    pass
