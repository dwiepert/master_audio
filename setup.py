from setuptools import setup, find_packages
from master_audio._version import __version__

setup(
    name = 'master_audio.py',
    packages = find_packages(),
    author = 'The Mayo Clinic Neurology AI Program',
    version = __version__,
)