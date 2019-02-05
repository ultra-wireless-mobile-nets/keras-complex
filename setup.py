#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md') as f:
    DESCRIPTION = f.read()


setup(
    name='DeepComplexNetworks',
    version='1',
    license='MIT',
    long_description=DESCRIPTION,
    packages=find_packages(),
    scripts=['scripts/run.py', 'scripts/training.py'],
    install_requires=[
        "numpy", "scipy", "sklearn", "tensorflow", "keras"]
)
