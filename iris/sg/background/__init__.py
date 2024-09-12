"""
Utilities for estimating the stray light background of FUV spectrograph images.
"""

from ._background import (
    model_background,
    model_spectral_line,
    model_total,
    fit,
    average,
    subtract_spectral_line,
    smooth,
    estimate,
)

__all__ = [
    "model_background",
    "model_spectral_line",
    "model_total",
    "fit",
    "average",
    "subtract_spectral_line",
    "smooth",
    "estimate",
]
