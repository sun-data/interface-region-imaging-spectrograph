"""
Represent and manipulate data captured using the IRIS spectrograph
"""

from . import background
from ._effective_area import (
    dn_to_photons,
    width_slit,
    effective_area,
)
from ._spectrograph import SpectrographObservation
from ._sg import open

__all__ = [
    "background",
    "dn_to_photons",
    "width_slit",
    "effective_area",
    "SpectrographObservation",
    "open",
]
