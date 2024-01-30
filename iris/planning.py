"""
Utilities for planning IRIS observations
"""

from __future__ import annotations
import numpy as np
import astropy.units as u
import astropy.time

__all__ = [
    "num_repeats",
]


def num_repeats(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    timedelta_raster: u.Quantity,
    timedelta_slew: u.Quantity = 10 * u.min,
) -> np.ndarray:
    """
    Calculate the number of times we can repeat a raster given the start and stop
    times and the length of the raster.

    Parameters
    ----------
    time_start
        The starting time of the observation
    time_stop
        The stop time of the observation
    timedelta_raster
        The amount of time needed to complete one raster
    timedelta_slew
        The amount of time needed to slew before the observation begins

    Examples
    --------
    Determine how many times we can repeat a 1-hour raster from 00:00 to 04:00

    .. jupyter-execute::

        import astropy.units as u
        import iris

        iris.planning.num_repeats(
            time_start="2024-01-01T00:00",
            time_stop="2024-01-01T04:00",
            timedelta_raster=1 * u.hr,
        )
    """
    if not isinstance(time_start, astropy.time.Time):
        time_start = astropy.time.Time(time_start)
    if not isinstance(time_stop, astropy.time.Time):
        time_stop = astropy.time.Time(time_stop)

    timedelta_full = time_stop - time_start
    result = (timedelta_full - timedelta_slew) / timedelta_raster
    result = result.to(u.dimensionless_unscaled).value

    return result
