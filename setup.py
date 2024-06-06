from setuptools import setup, find_packages
from naip_asr._version import __version__

setup(
    name = 'naip_asr.py',
    packages = find_packages(),
    author = 'The Mayo Clinic Neurology AI Program',
    version = __version__,
)