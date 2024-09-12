"""
Represent and manipulate data captured using the IRIS spectrograph
"""

from . import background
from ._spectrograph import SpectrographObservation

__all__ = [
    "background",
    "SpectrographObservation",
]
