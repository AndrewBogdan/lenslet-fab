"""
I'm gonna make small lenses with a 6-axis and a laser.
- Andrew
"""

import yaml


# Global constants
CONFIG_FILE = r'C:\Users\LaserLab\Andrew\LensletFab\config.yaml'


# Load the config file
with open(CONFIG_FILE, 'r') as config_file:
    config = yaml.safe_load(config_file.read())


# Delete unnecessary variables
del yaml
del config_file
