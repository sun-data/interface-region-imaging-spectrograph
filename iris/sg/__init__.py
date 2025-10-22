"""
Represent and manipulate data captured using the IRIS spectrograph
"""

from . import background
from ._effective_area import (
    gain,
    width_slit,
    effective_area,
)
from ._spectrograph import SpectrographObservation
from ._sg import open

__all__ = [
    "background",
    "gain",
    "width_slit",
    "effective_area",
    "SpectrographObservation",
    "open",
]
