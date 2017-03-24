from distutils.core import setup
from setuptools import find_packages

setup(name='deepmonster',
      version='1.0',
      author='Olivier Mastropietro',
      packages=find_packages(exclude=['tools']))
