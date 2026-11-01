"""
Find the pixels of an IRIS spectrograph observation which do not show the
Sun as it is: the pixels hit by energetic particles, the hot pixels of the
detector, and the rows darkened along the slit.

These are the implementations of
:meth:`iris.sg.SpectrographObservation.spikes`,
:meth:`~iris.sg.SpectrographObservation.hot_pixels`, and
:meth:`~iris.sg.SpectrographObservation.dark_rows`, on the values of the
outputs in DN, without units.
"""

from typing import cast
import math
import warnings
import numpy as np
import named_arrays as na

__all__ = []


def spikes(
    data: na.AbstractScalarArray,
    axis_exposure: str,
    axis_image: tuple[str, str],
    gain: float,
    read_noise: float,
    sigclip: float,
    objlim: float,
    satlevel: float,
    limit: float,
    sigma: float,
) -> na.AbstractExplicitArray:
    """
    The pixels hit by energetic particles: those which L.A.Cosmic flags in
    each exposure on fainter Sun than `limit`, or which :func:`_clip` flags
    among the neighboring exposures, and the negative lobes around them,
    :func:`_lobes`.

    Parameters
    ----------
    data
        The outputs, in DN, which are NaN where they are missing.
    axis_exposure
        The axis along which consecutive exposures are neighbors.
    axis_image
        The axes of each exposure, along the slit and along wavelength.
    gain
        The number of photons per DN.
    read_noise
        The read noise, in DN.
    sigclip
        The detection limit of L.A.Cosmic, in standard deviations.
    objlim
        The contrast limit of L.A.Cosmic between a hit and the fine
        structure of the image.
    satlevel
        The level, in DN, at and above which pixels are saturated.
    limit
        The brightest Sun, in DN, on which the flags of L.A.Cosmic are kept.
    sigma
        The detection limit of :func:`_clip`, in standard deviations.
    """
    valid = np.isfinite(data)

    # the Sun around each pixel: the mean of the middle 5 of the 25 pixels
    # in the 5 x 5 box around it, which neither a hit nor the lobe of 8
    # pixels around it change
    background = na.ndfilters.trimmed_mean_filter(
        array=data,
        size={a: 5 for a in axis_image},
        where=valid,
        proportion=0.4,
    )

    # astroscrappy works in photons, which are the Poisson unit of the noise
    hits = na.despike_mask(
        array=data,
        axis=axis_image,
        sigclip=sigclip,
        objlim=objlim,
        gain=gain,
        readnoise=gain * read_noise,
        satlevel=gain * satlevel,
    )

    # L.A.Cosmic also flags the edges and the thin strands of bright Sun,
    # which are as sharp as hits
    hits = hits & (background < limit)

    axis_box = tuple(a for a in (axis_exposure, *axis_image) if a in data.shape)
    hits = hits | _clip(data, valid=valid, axis=axis_box, sigma=sigma)

    return hits | _lobes(
        data=data,
        valid=valid,
        hits=hits,
        background=background,
        axis=axis_image,
        gain=gain,
        read_noise=read_noise,
    )


def _clip(
    data: na.AbstractScalarArray,
    valid: na.AbstractArray,
    axis: tuple[str, ...],
    sigma: float,
    size: int = 5,
    std_min: float = 1,
    num_iteration: int = 10,
) -> na.AbstractExplicitArray:
    """
    The pixels more than `sigma` standard deviations above the mean of the
    good pixels in the box of `size` pixels around them along `axis`, as
    ``iris_prep_despike.pro`` of SolarSoft (C. Kankelborg) finds them, after
    ``DESPIK.PRO``.

    The standard deviation is that of the good pixels in the box about the
    mean of the box around each, at least `std_min`.
    The pixels found are no longer good, and the search is repeated until no
    more are found, up to `num_iteration` times.
    The edges of the box repeat the pixels at the edges of the data, as IDL's
    ``CONVOL(/EDGE_TRUNCATE)`` does.

    Parameters
    ----------
    data
        The outputs, in DN.
    valid
        The pixels which have data, the good pixels at the start.
    axis
        The axes of the box.
    sigma
        The detection limit, in standard deviations.
    size
        The width of the box along each axis.
    std_min
        The least standard deviation, in DN.
    num_iteration
        The largest number of searches.
    """
    shape = {a: size for a in axis}
    good = valid
    for _ in range(num_iteration):
        mean = na.ndfilters.mean_filter(data, size=shape, where=good, mode="nearest")
        deviation = data - mean
        variance = na.ndfilters.mean_filter(
            array=np.square(deviation),
            size=shape,
            where=good,
            mode="nearest",
        )
        std = np.maximum(np.sqrt(variance), std_min)
        bad = good & (deviation > sigma * std)
        if not bad.any(where=True):
            break
        good = good & ~bad
    return valid & ~good


def _lobes(
    data: na.AbstractScalarArray,
    valid: na.AbstractArray,
    hits: na.AbstractExplicitArray,
    background: na.AbstractExplicitArray,
    axis: tuple[str, str],
    gain: float,
    read_noise: float,
    threshold: float = 2,
) -> na.AbstractExplicitArray:
    """
    The pixels next to a hit, along `axis`, which are more than `threshold`
    standard deviations of the noise below the Sun around them.

    Level 2 data are resampled from the pixels of the detector, which leaves
    a negative lobe around each hit, a median of 6% of its peak.

    Parameters
    ----------
    data
        The outputs, in DN.
    valid
        The pixels which have data.
    hits
        The pixels hit by particles.
    background
        The Sun around each pixel, which the hits and their lobes do not
        change, in DN.
    axis
        The axes of each exposure.
    gain
        The number of photons per DN.
    read_noise
        The read noise, in DN.
    threshold
        The detection limit, in standard deviations.
    """
    # the photon noise of the background where it is positive
    noise = np.sqrt(np.square(read_noise) + background * (background > 0) / gain)
    z = (data - background) / noise
    return valid & _dilate(hits, axis=axis) & ~hits & (z < -threshold)


def _dilate(
    mask: na.AbstractExplicitArray,
    axis: tuple[str, ...],
) -> na.AbstractExplicitArray:
    """
    Grow a mask by one pixel along each of `axis`, which along two axes is
    the 3 x 3 box around each pixel.

    Parameters
    ----------
    mask
        The mask to grow.
    axis
        The axes to grow it along.
    """
    result = mask
    for a in axis:
        num = result.shape[a]
        lower = {a: slice(None, num - 1)}
        upper = {a: slice(1, None)}
        grown = result.copy()
        grown[upper] = grown[upper] | result[lower]
        grown[lower] = grown[lower] | result[upper]
        result = grown
    return result


def _nanpercentile(
    a: na.AbstractScalarArray,
    q: float,
    axis: tuple[str, ...],
) -> na.ScalarArray:
    """
    The `q` percentile of `a` along `axis`, ignoring NaN, which is NaN where
    `a` is NaN along all of `axis`.

    Parameters
    ----------
    a
        The array.
    q
        The percentile, from 0 to 100.
    axis
        The axes to take it along.
    """
    axes = tuple(ax for ax in a.shape if ax not in axis)
    ndarray = cast(na.ScalarArray, a).ndarray_aligned(axes + axis)
    ndarray = ndarray.reshape(ndarray.shape[: len(axes)] + (-1,))
    with warnings.catch_warnings():
        # the pixels outside the slit, which have no data
        warnings.filterwarnings("ignore", "All-NaN slice encountered")
        ndarray = np.nanpercentile(ndarray, q, axis=-1)
    return na.ScalarArray(ndarray, axes=axes)


def hot_pixels(
    data: na.AbstractScalarArray,
    variance: na.AbstractScalarArray,
    spikes: na.AbstractArray,
    axis_exposure: tuple[str, ...],
    axis_slit: str,
    fraction: float,
    threshold: float,
    limit: float,
    size: int = 15,
    significance: float = 5,
) -> na.AbstractExplicitArray:
    """
    The pixels which are spikes in more than `fraction` of the exposures in
    which they have data, or whose fixed pattern along the slit is more than
    `threshold`, and more than `significance` times its uncertainty, on
    fainter Sun than `limit`.

    The fixed pattern of each pixel is its interquartile mean over the
    exposures, without the spikes, minus the 25% trimmed mean of that over
    the `size` pixels around it along the slit.

    Parameters
    ----------
    data
        The outputs, in DN.
    variance
        The variance of the noise of the outputs, in DN squared.
    spikes
        The pixels hit by particles.
    axis_exposure
        The axes along which the exposures vary.
    axis_slit
        The axis along the slit.
    fraction
        The largest fraction of the exposures in which a pixel may be a spike.
    threshold
        The largest fixed pattern, in DN.
    limit
        The brightest smoothed Sun, in DN, on which the fixed pattern is
        tested, since the structure of a bright line along the slit is larger
        than the fixed pattern.
    size
        The number of pixels along the slit over which the pattern is
        smoothed.
    significance
        The number of standard deviations of its noise by which the fixed
        pattern of a hot pixel must differ from zero.
    """
    valid = np.isfinite(data)
    good = valid & ~spikes

    with np.errstate(divide="ignore", invalid="ignore"):
        flagged = (spikes & valid).sum(axis=axis_exposure, where=True) / valid.sum(
            axis=axis_exposure,
            where=True,
        )

    data_good = cast(na.ScalarArray, data.copy())
    data_good[~good] = np.nan
    lower = _nanpercentile(data_good, 25, axis=axis_exposure)
    upper = _nanpercentile(data_good, 75, axis=axis_exposure)
    central = good & (data >= lower) & (data <= upper)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        # the pixels outside the slit, which have no data
        warnings.filterwarnings("ignore", "Mean of empty slice")
        profile = data.mean(axis=axis_exposure, where=central)

    ok = np.isfinite(profile)
    smooth = na.ndfilters.trimmed_mean_filter(
        array=profile,
        size={axis_slit: size},
        where=ok,
        proportion=0.25,
    )
    pattern = profile - smooth

    # the uncertainty of the pattern, from the noise of the outputs, where the
    # interquartile mean has 1.1 times the uncertainty of the mean
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        warnings.filterwarnings("ignore", "Mean of empty slice")
        num = good.sum(axis=axis_exposure, where=True)
        error = 1.1 * np.sqrt(variance.mean(axis=axis_exposure, where=good) / num)

    fixed = np.abs(pattern) > threshold
    fixed = fixed & (np.abs(pattern) > significance * error) & (smooth < limit)
    return (flagged > fraction) | fixed


def dark_rows(
    data: na.AbstractScalarArray,
    axis_exposure: tuple[str, ...],
    axis_slit: str,
    axis_wavelength: str,
    ratio: float,
    size: int = 21,
    significance: float = 5,
    num_exposure: int = 20,
    num_iteration: int = 5,
) -> na.AbstractExplicitArray:
    """
    The rows along the slit whose signal is less than `ratio` times that of
    the rows around them, in the median over the exposures, by more than
    `significance` times its uncertainty, and the rows next to them.

    The signal of a row in an exposure is its mean over the brightest
    quarter of the wavelengths, where the stray light and the offsets of the
    rows hide a dark row the least, and that of the rows around it is the
    25% trimmed mean of the signal of the `size` rows around it which are
    not dark.
    The uncertainty of the median of their ratio is from its scatter over
    the exposures.
    The rows are found again without the dark rows among the rows around
    each, until no more are found, up to `num_iteration` times.
    With fewer than `num_exposure` exposures, none are found.

    Parameters
    ----------
    data
        The outputs, in DN.
    axis_exposure
        The axes along which the exposures vary.
    axis_slit
        The axis along the slit.
    axis_wavelength
        The axis of wavelength.
    ratio
        The largest ratio of the signal of a dark row to that of the rows
        around it.
    size
        The number of rows around each row.
    significance
        The number of standard deviations by which the ratio of a dark row
        must be less than one.
    num_exposure
        The least number of exposures in which dark rows are found.
    num_iteration
        The largest number of searches.
    """
    finite = np.isfinite(data)
    with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        # the rows outside the slit, which have no data
        warnings.filterwarnings("ignore", "Mean of empty slice")
        mean = data.mean(axis=(*axis_exposure, axis_slit), where=finite)
        bright = mean >= _nanpercentile(mean, 75, axis=(axis_wavelength,))
        signal = data.mean(axis=axis_wavelength, where=finite & bright)
    ok = np.isfinite(signal)

    dark = ok.any(axis=axis_exposure, where=True) & False
    if math.prod(data.shape[a] for a in axis_exposure) < num_exposure:
        return dark

    num = ok.sum(axis=axis_exposure, where=True)
    for _ in range(num_iteration):
        reference = na.ndfilters.trimmed_mean_filter(
            array=signal,
            size={axis_slit: size},
            where=ok & ~dark,
            proportion=0.25,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            r = cast(na.ScalarArray, (signal / reference).explicit)
        r[~ok] = np.nan
        median = _nanpercentile(r, 50, axis=axis_exposure)
        deviation = np.abs(r - median)
        error = 1.25 * 1.4826 * _nanpercentile(deviation, 50, axis=axis_exposure)
        with np.errstate(divide="ignore", invalid="ignore"):
            error = error / np.sqrt(num)
        dark_new = (median < ratio) & (1 - median > significance * error)
        if np.all(dark_new == dark):
            break
        dark = dark_new

    return _dilate(dark, axis=(axis_slit,))
