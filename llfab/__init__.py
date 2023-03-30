"""
I'm gonna make small lenses with a 6-axis and a laser.
- Andrew
"""

import os
import yaml


# Global constants
dir_path = os.sep.split(__file__)[:-2]
CONFIG_FILE = os.path.join(*dir_path, 'config.yaml')
# r'C:\Users\LaserLab\Andrew\LensletFab\config.yaml'


# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file.read())


# Delete unnecessary variables
del yaml
del config_file
