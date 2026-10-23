"""
Represent and manipulate data captured using the IRIS spectrograph
"""

from . import models
from . import background
from ._effective_area import (
    dn_to_photons,
    read_noise,
    width_slit,
    effective_area,
)
from ._spectrograph import SpectrographObservation
from ._sg import open
from ._fit_line import fit_line

__all__ = [
    "models",
    "background",
    "dn_to_photons",
    "read_noise",
    "width_slit",
    "effective_area",
    "SpectrographObservation",
    "open",
    "fit_line",
]
