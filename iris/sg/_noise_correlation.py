from typing import cast
import numpy as np
import named_arrays as na
from ._spectrograph import SpectrographObservation

__all__ = [
    "noise_correlation",
]


def noise_correlation(
    obs: SpectrographObservation,
    axis: str,
    num_lag: int = 2,
    where: bool | na.AbstractScalar = True,
) -> tuple[float, ...]:
    r"""
    Measure the correlation of the noise of two pixels of an IRIS
    spectrograph observation, by their separation along `axis`.

    Level 2 data are resampled from the pixels of the detector onto a grid of
    wavelength and position, which shares the noise of each pixel of the
    detector between neighboring pixels of the grid, so their noise is
    correlated.

    The correlation is measured from the differences between consecutive
    exposures, along :attr:`~iris.sg.SpectrographObservation.axis_detector_x`,
    which cancel the fixed pattern of the detector and what of the Sun does
    not change between them.
    The noise of a difference has the same correlation as that of one
    exposure, and a mean of zero, so the correlation is the mean product of
    two differences over their mean square.
    Only the fainter half of the pairs of pixels is used, since the Sun
    changes the least there, and only where `where` is true in both
    exposures and the outputs are not NaN, as
    :meth:`~iris.sg.SpectrographObservation.clean` makes them at the spikes.
    Pairs with a difference more than five standard deviations from zero,
    from the median absolute deviation of the differences, are left out, so
    that the spikes left and the largest changes of the Sun do not count.

    Parameters
    ----------
    obs
        The observation, with at least two exposures along
        :attr:`~iris.sg.SpectrographObservation.axis_detector_x`.
    axis
        The axis along which the pixels are separated, such as
        :attr:`~iris.sg.SpectrographObservation.axis_wavelength`.
    num_lag
        The number of separations to measure, from one pixel up.
    where
        The pixels to use.

    Returns
    -------
    :
        The correlation of the noise of two pixels separated by zero, one,
        two, and so on, up to `num_lag` pixels along `axis`, starting with 1,
        the correlation of a pixel with itself, as :func:`iris.sg.fit_line`
        takes its `correlation` along
        :attr:`~iris.sg.SpectrographObservation.axis_wavelength`.

    Raises
    ------
    ValueError
        If `axis` is an axis along which the exposures vary, if there are
        fewer than two exposures along
        :attr:`~iris.sg.SpectrographObservation.axis_detector_x`, or if there
        are no pixels to use.

    Examples
    --------

    Measure the correlation of the noise along wavelength and along the slit
    in a raster, and fit a Gaussian to every spectrum with the noise
    correlated along wavelength.

    .. jupyter-execute::

        import astropy.units as u
        import iris

        # Load the first 40 exposures of a 320-step raster, without the
        # spikes, hot pixels and dark rows
        obs = iris.sg.open("2021-09-23T06:13")
        obs = obs[{obs.axis_detector_x: slice(0, 40)}].clean()

        # Measure the correlation of the noise
        rho = iris.sg.noise_correlation(obs, axis=obs.axis_wavelength)
        rho_slit = iris.sg.noise_correlation(obs, axis=obs.axis_detector_y)
        print(f"along wavelength: {rho}")
        print(f"along the slit: {rho_slit}")

        # Fit every spectrum with that correlation
        fit = iris.sg.fit_line(
            obs,
            velocity_max=100 * u.km / u.s,
            correlation=rho,
        )
    """
    axis_x = obs.axis_detector_x
    if axis in (obs.axis_time, axis_x):
        raise ValueError(
            f"The correlation is measured along an axis of each exposure, "
            f"got `axis` {axis!r}, along which the exposures vary."
        )
    outputs = obs.outputs.value
    num = outputs.shape.get(axis_x, 1)
    if num < 2:
        raise ValueError(
            f"The correlation is measured from consecutive exposures along "
            f"{axis_x!r}, but there are {num}."
        )

    lower = {axis_x: slice(None, num - 1)}
    upper = {axis_x: slice(1, None)}
    difference = outputs[upper] - outputs[lower]
    level = (outputs[upper] + outputs[lower]) / 2

    where_pair = where
    if isinstance(where, na.AbstractArray) and axis_x in where.shape:
        where_pair = where[lower] & where[upper]
    used = np.isfinite(difference) & where_pair
    used = used & (level < _median(level, where=used))

    # the largest difference which is noise
    limit = 5 * 1.4826 * _mad(difference, where=used)

    num_axis = difference.shape[axis]
    result = [1.0]
    for lag in range(1, num_lag + 1):
        first = difference[{axis: slice(None, num_axis - lag)}]
        second = difference[{axis: slice(lag, None)}]
        pair = used[{axis: slice(None, num_axis - lag)}]
        pair = pair & used[{axis: slice(lag, None)}]
        pair = pair & (np.abs(first) < limit) & (np.abs(second) < limit)
        covariance = _sum(first * second, where=pair)
        variance = _sum(np.square(first), where=pair)
        variance = variance * _sum(np.square(second), where=pair)
        result.append(float(covariance / np.sqrt(variance)))
    return tuple(result)


def _sum(
    a: na.AbstractArray,
    where: na.AbstractArray,
) -> float:
    """
    The sum of `a` where `where` is true.

    Parameters
    ----------
    a
        The values.
    where
        The values to use, of the same shape as `a`.
    """
    values = cast(na.ScalarArray, a[where])
    return float(np.sum(values.ndarray_aligned(tuple(values.shape))))


def _median(
    a: na.AbstractArray,
    where: na.AbstractArray,
) -> float:
    """
    The median of `a` where `where` is true.

    Parameters
    ----------
    a
        The values.
    where
        The values to use, of the same shape as `a`.

    Raises
    ------
    ValueError
        If there are no values to use.
    """
    values = cast(na.ScalarArray, a[where])
    if not values.size:
        raise ValueError("There are no pixels to measure the correlation from.")
    return float(np.median(values.ndarray_aligned(tuple(values.shape))))


def _mad(
    a: na.AbstractArray,
    where: na.AbstractArray,
) -> float:
    """
    The median absolute deviation of `a` from its median, where `where` is
    true.

    Parameters
    ----------
    a
        The values.
    where
        The values to use, of the same shape as `a`.
    """
    return _median(np.abs(a - _median(a, where=where)), where=where)
