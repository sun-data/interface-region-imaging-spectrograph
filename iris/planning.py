import astropy.units as u
import astropy.time

__all__ = [
    "num_repeats",
]


def num_repeats(
    time_start,
    time_stop,
    time_raster,
    time_slew=10 * u.min,
):
    if not isinstance(time_start, astropy.time.Time):
        time_start = astropy.time.Time(time_start)
    if not isinstance(time_stop, astropy.time.Time):
        time_stop = astropy.time.Time(time_stop)

    result = (time_stop - time_start - time_slew) / time_raster
    result = result.to(u.dimensionless_unscaled)

    return result
