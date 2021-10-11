"""
Make a package wrapper to access pyximc.py easily, from anywhere.

Note: This requires copying the ximc/ directory into venv/Lib/
Credit: standa
"""

import os
import setuptools

# --- Run Setup Script --------------------------------------------------------
setuptools.setup(
    name='pyximc',
    version='2.13.1',
    packages=setuptools.find_packages(),
)
