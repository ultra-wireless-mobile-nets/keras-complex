#!/usr/bin/env python
from setuptools import setup, find_packages

with open('README.md') as f:
    DESCRIPTION = f.read()


setup(
    name='keras-complex',
    version='0.1.1',
    license='MIT',
    long_description=DESCRIPTION,
    url='https://github.com/JesperDramsch/keras-complex',
    packages=find_packages(),
    scripts=['scripts/run.py', 'scripts/training.py'],
    install_requires=[
        "numpy", "scipy", "sklearn", "keras"]
    extras_require={
        "tf": ["tensorflow"],
        "tf_gpu": ["tensorflow-gpu"],
    }
    classifiers=[
          'Development Status :: 3 - Alpha',
          'Intended Audience :: Education',
          'Intended Audience :: Science/Research',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3 :: Only',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
          'Topic :: Software Development :: Libraries',
          'Topic :: Software Development :: Libraries :: Python Modules'
      ],
)
