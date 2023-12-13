"""
I'm gonna make small lenses with a 6-axis and a laser.
- Andrew
"""

import logging
import os
import yaml

_logger = logging.getLogger(__name__)

# --- Config ------------------------------------------------------------------
CONFIG_ROOT = os.path.abspath(os.path.join(
    os.path.abspath(__file__), '..', '..'
))
CONFIG_FILE = os.path.join(CONFIG_ROOT, 'config.yaml')

# Load the config file
config = None
try:
    with open(CONFIG_FILE, 'r') as config_file:
        config = yaml.safe_load(config_file.read())
    del config_file
except FileNotFoundError as err:
    _logger.error(f'Config file \'{CONFIG_FILE}\' not found')
    raise err

# --- Logging -----------------------------------------------------------------
LOGFILE = os.path.join(CONFIG_ROOT, f'{__name__}.log')
LOGFILE_FMT = '%(asctime)s: %(message)s'

_logger.setLevel(logging.DEBUG)
_handler_logfile = logging.FileHandler(LOGFILE)
_handler_logfile.setFormatter(logging.Formatter(LOGFILE_FMT))
_handler_logfile.setLevel(logging.DEBUG)
_logger.addHandler(_handler_logfile)
_logger.debug(
    f'Package {__name__} loaded with logfile {LOGFILE}.\n'
    f'To disable logging, use logging.getLogger({__name__}).setLevel or '
    f'set logging.getLogger({__name__}).propagate = False.'
)

# --- Package Scope -----------------------------------------------------------
__all__ = []
