"""
Compiled models of the profiles of spectral lines, which
:func:`iris.sg.fit_line` fits to IRIS spectra with :mod:`ndoptimize`.

The common models of :mod:`ndoptimize.models`, such as
:func:`ndoptimize.models.gaussian_linear`, can be fit to IRIS spectra too.
"""

from ._models import kappa_linear

__all__ = [
    "kappa_linear",
]
