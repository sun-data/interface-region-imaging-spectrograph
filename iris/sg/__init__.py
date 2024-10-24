"""
Represent and manipulate data captured using the IRIS spectrograph
"""

from . import background
from ._effective_area import effective_area
from ._spectrograph import SpectrographObservation

__all__ = [
    "background",
    "effective_area",
    "SpectrographObservation",
]
