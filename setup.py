"""
Setup.py for package llfab.
"""

import os
import setuptools

# --- Collect Setup Information -----------------------------------------------
# Get README
with open('README.md', 'r') as readme_file:
    readme = readme_file.read()

# --- Run Setup Script --------------------------------------------------------
setuptools.setup(
    name='llfab',
    version='0.1',
    author='Andrew Bogdan',
    description='Control Shawn\'s stuff',
    long_description=readme,
    packages=setuptools.find_packages(),
)
