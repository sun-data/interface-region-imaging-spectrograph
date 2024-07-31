"""
Python package for analyzing observations from the Interface Region Imaging Spectrograph
"""

from . import planning
from . import data
from ._spectrograph import SpectrographObservation

__all__ = [
    "planning",
    "data",
    "SpectrographObservation",
]
