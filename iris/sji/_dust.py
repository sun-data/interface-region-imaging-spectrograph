"""
The dust on the CCD of the slit-jaw imager.

The positions of the dust are taken from the maps of bad pixels which the
IRIS team measures from each flat field and publishes in the SolarSoft
database, and they are moved into the frames of a level 2 file the way
``IRIS_DUSTBUSTER`` in SolarSoft moves them.
"""

from typing import Any, NamedTuple
import functools
import re
import numpy as np
import scipy.io
import scipy.ndimage
import requests
import astropy.time
import sunpy.io.special
import iris

__all__ = []

_url_sswdb = "https://sohoftp.nascom.nasa.gov/sdb/iris/data/"
"""The directory of the SolarSoft database which holds the IRIS calibration files."""

_shape_ccd = (1096, 2072)
"""The number of rows and columns of the frame the addresses of the bad pixels count."""

_epoch_tai = astropy.time.Time("1958-01-01T00:00:00", scale="tai")
"""The epoch of the TAI seconds used by SolarSoft."""

_cdelt_level_2 = 0.16635
"""
The plate scale of an unsummed level 2 slit-jaw pixel in arcseconds,
``cdlt_1p5`` of the IRIS pointing database.
"""


class _Pointing(NamedTuple):
    """The geometry of one slit-jaw channel on the CCD."""

    slit_x: float
    """The column of the center of the slit, counted from one."""

    slit_y: float
    """The row of the center of the slit, counted from one."""

    cdelt: float
    """The plate scale of the CCD in arcseconds."""

    roll: float
    """The angle of the slit measured from the :math:`y` axis of the CCD, in degrees."""


_pointing = {
    "SJI_1330": _Pointing(537.30, 524.03, 0.16560, -0.200),
    "SJI_1400": _Pointing(529.32, 510.46, 0.16560, -0.224),
    "SJI_1600W": _Pointing(534.0, 516.7, 0.16560, -0.21),
    "SJI_2796": _Pointing(504.69, 503.402, 0.16790, 0.274),
    "SJI_2832": _Pointing(506.47, 502.22, 0.16790, 0.286),
    "SJI_5000W": _Pointing(505.2, 504.7, 0.16790, 0.28),
}
"""
The geometry of each slit-jaw channel on the CCD,
from version 11 of ``iris_mk_pointdb.pro`` in SolarSoft.
"""

_shift_max = 7
"""
The largest correction, in level 2 pixels, to the position of the dust
which is fit to the images, the same as ``IRIS_DUSTBUSTER``.
"""

_num_frames_shift = 32
"""The largest number of frames used to fit the correction to the position of the dust."""

_radius_disk = 880
"""
The largest helioprojective coordinate, in arcseconds,
of a pixel used to fit the correction to the position of the dust,
which keeps the dark sky off the limb out of the fit.
"""


@functools.cache
def _urls(num_retry: int = 5) -> tuple[str, str]:
    """
    The URLs of the latest index of flat fields in the SolarSoft database,
    and of the bad-pixel maps posted with it.

    Parameters
    ----------
    num_retry
        The number of times to try to connect to the server.
    """
    for _ in range(num_retry):
        try:
            response = requests.get(_url_sswdb, timeout=30)
            response.raise_for_status()
            break
        except requests.exceptions.RequestException:  # pragma: no cover
            pass
    else:  # pragma: no cover
        raise ConnectionError(f"Could not get {_url_sswdb}")

    stamps_flat = set(re.findall(r'"(\d{8}_\d{6})_flat\.genx"', response.text))
    stamps_badpix = set(re.findall(r'"(\d{8}_\d{6})_badpix\.geny"', response.text))

    stamp = max(stamps_flat & stamps_badpix)

    return (
        f"{_url_sswdb}{stamp}_flat.genx",
        f"{_url_sswdb}{stamp}_badpix.geny",
    )


def _ccd(
    window: str,
    time: astropy.time.Time,
    num_retry: int = 5,
) -> np.ndarray:
    """
    The bad pixels of the CCD measured from the flat field of a slit-jaw
    channel nearest in time, as ``iris_prep_get_badpix.pro`` finds them.

    Parameters
    ----------
    window
        The slit-jaw channel, such as ``"SJI_1400"``.
    time
        The time of the observation.
    num_retry
        The number of times to try to connect to the server.
    """
    url_flat, url_badpix = _urls(num_retry)

    (path_flat,) = iris.data.download([url_flat])
    (path_badpix,) = iris.data.download([url_badpix])

    index: list[dict[str, Any]] = sunpy.io.special.read_genx(str(path_flat))["SAVEGEN0"]
    index = [record for record in index if record["IMG_PATH"] == window]

    if not index:
        raise ValueError(f"There are no flat fields of {window}.")

    tai = (time - _epoch_tai).sec
    record = min(index, key=lambda r: abs(r["FILETAI"] - tai))

    maps: Any = scipy.io.readsav(str(path_badpix))
    address = maps["p0"][0][f"F{record['RECNUM']}"]

    result = np.zeros(_shape_ccd, dtype=bool)
    result.flat[address] = True

    return result


def _radius(window: str, binning: tuple[int, int]) -> int:
    """
    The number of CCD pixels by which ``IRIS_DUSTBUSTER`` grows each bad
    pixel, so that the mask covers the partly dark rim of each dust particle.

    Parameters
    ----------
    window
        The slit-jaw channel, such as ``"SJI_1400"``.
    binning
        The number of CCD pixels summed into each level 2 pixel along the
        :math:`x` and :math:`y` axes, ``SUMSPTRL`` and ``SUMSPAT``.
    """
    binning_x, binning_y = binning

    size = 4
    sizes_2832 = {1: 4, 2: 6, 4: 8, 8: 10}
    if window == "SJI_2832" and binning_y in sizes_2832:
        size = sizes_2832[binning_y]
        if binning_y >= 4 and binning_x >= 2:
            size = size // 4
        size = max(size, 4)

    # IDL's ``SMOOTH`` widens a window of even width by one
    width = size + 1 - size % 2

    return width // 2


def _offsets(
    window: str,
    time: astropy.time.Time,
    binning: tuple[int, int],
    num_retry: int = 5,
) -> np.ndarray:
    """
    The positions of the dusty pixels relative to the center of the slit,
    in level 2 pixels.

    The result has shape ``(N, 2)``, where the last axis is :math:`x, y`.

    Parameters
    ----------
    window
        The slit-jaw channel, such as ``"SJI_1400"``.
    time
        The time of the observation.
    binning
        The number of CCD pixels summed into each level 2 pixel along the
        :math:`x` and :math:`y` axes, ``SUMSPTRL`` and ``SUMSPAT``.
    num_retry
        The number of times to try to connect to the server.
    """
    if window not in _pointing:
        raise ValueError(f"Unrecognized slit-jaw channel {window}.")

    pointing = _pointing[window]

    ccd = _ccd(window, time, num_retry)

    radius = _radius(window, binning)
    structure = np.ones((2 * radius + 1, 2 * radius + 1), dtype=bool)
    ccd = scipy.ndimage.binary_dilation(ccd, structure=structure)

    binning_x, binning_y = binning

    row, column = np.argwhere(ccd).T
    position = np.stack([column // binning_x, row // binning_y], axis=~0)
    position = np.unique(position, axis=0).astype(float)

    # ``IRIS_DUSTBUSTER`` moves the mask down by half a pixel
    position[:, 1] -= 0.5

    # The center of the slit in summed pixels counted from zero
    summed = np.array(binning)
    slit = np.array([pointing.slit_x, pointing.slit_y])
    slit = (slit - 1) / summed + (summed - 1) / (2 * summed)

    x = position[:, 0] - slit[0]
    y = position[:, 1] - slit[1]

    magnification = pointing.cdelt / _cdelt_level_2
    roll = np.deg2rad(pointing.roll)
    cos, sin = np.cos(roll), np.sin(roll)

    return magnification * np.stack(
        [x * cos - y * sin, x * sin + y * cos],
        axis=~0,
    )


def _pixels(
    offsets: np.ndarray,
    slit: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    The frame, column, and row of every dusty pixel of every frame.

    Each offset falls between pixels, so like ``IRIS_DUSTBUSTER``,
    both the pixel below and to the left of it and the pixel above and to the
    right of it are taken.

    Parameters
    ----------
    offsets
        The output of :func:`_offsets`.
    slit
        The position of the center of the slit in each frame,
        in pixels counted from zero, with shape ``(num_frames, 2)``.
    """
    position = slit[:, np.newaxis, :] + offsets[np.newaxis, :, :]
    position = np.concatenate([np.floor(position), np.ceil(position)], axis=1)
    position = position.astype(int)

    t = np.broadcast_to(np.arange(slit.shape[0])[:, np.newaxis], position.shape[:2])

    return t.ravel(), position[..., 0].ravel(), position[..., 1].ravel()


def _shift(
    offsets: np.ndarray,
    slit: np.ndarray,
    image: np.ndarray,
    crval: np.ndarray,
    crpix: np.ndarray,
    cdelt: np.ndarray,
    pc: np.ndarray,
) -> tuple[int, int]:
    """
    The correction to the position of the dust which makes the dusty pixels
    darkest, as ``IRIS_DUSTBUSTER`` fits it,
    by searching every shift of up to :data:`_shift_max` pixels.

    Parameters
    ----------
    offsets
        The output of :func:`_offsets`.
    slit
        The position of the center of the slit in each frame,
        in pixels counted from zero, with shape ``(num_frames, 2)``.
    image
        The level 2 images, with shape ``(num_frames, num_y, num_x)``,
        and NaN outside the field of view.
    crval
        The helioprojective coordinates of the reference pixel of each frame,
        in arcseconds, with shape ``(num_frames, 2)``.
    crpix
        The reference pixel, counted from zero, with shape ``(2,)``.
    cdelt
        The plate scale along each axis in arcseconds, with shape ``(2,)``.
    pc
        The rotation matrix of each frame, with shape ``(num_frames, 2, 2)``.
    """
    num_t, num_y, num_x = image.shape

    frames = np.unique(np.linspace(0, num_t - 1, num=min(num_t, _num_frames_shift)))
    frames = frames.astype(int)

    t, x, y = _pixels(offsets, slit[frames])

    # The same pixel can be taken twice, from neighboring offsets
    _, unique = np.unique(np.stack([t, x, y]), axis=1, return_index=True)
    t, x, y = t[unique], x[unique], y[unique]

    # The helioprojective coordinates of each dusty pixel
    pixel = np.stack([x, y], axis=~0) - crpix
    position = crval[frames][t] + cdelt * np.einsum("nij,nj->ni", pc[frames][t], pixel)
    on_disk = np.all(np.abs(position) <= _radius_disk, axis=~0)
    t, x, y = t[on_disk], x[on_disk], y[on_disk]

    image = image[frames]

    shifts = [
        (shift_x, shift_y)
        for shift_x in range(-_shift_max, _shift_max + 1)
        for shift_y in range(-_shift_max, _shift_max + 1)
    ]
    # Ties go to the smallest shift
    shifts = sorted(shifts, key=lambda s: s[0] ** 2 + s[1] ** 2)

    result = (0, 0)
    darkest = np.inf
    for shift_x, shift_y in shifts:
        x_shifted = x + shift_x
        y_shifted = y + shift_y
        inside = (
            (x_shifted >= 0)
            & (x_shifted < num_x)
            & (y_shifted >= 0)
            & (y_shifted < num_y)
        )
        values = image[t[inside], y_shifted[inside], x_shifted[inside]]
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        mean = np.mean(values)
        if mean < darkest:
            darkest = mean
            result = (shift_x, shift_y)

    return result


def _mask(
    window: str,
    time: astropy.time.Time,
    binning: tuple[int, int],
    slit: np.ndarray,
    image: np.ndarray,
    crval: np.ndarray,
    crpix: np.ndarray,
    cdelt: np.ndarray,
    pc: np.ndarray,
    num_retry: int = 5,
) -> np.ndarray:
    """
    Whether each pixel of each level 2 frame is darkened by dust,
    with the same shape as `image`.

    The pixels outside the field of view, which are NaN in `image`,
    are never darkened by dust.

    Parameters
    ----------
    window
        The slit-jaw channel, such as ``"SJI_1400"``.
    time
        The time of the observation.
    binning
        The number of CCD pixels summed into each level 2 pixel along the
        :math:`x` and :math:`y` axes, ``SUMSPTRL`` and ``SUMSPAT``.
    slit
        The position of the center of the slit in each frame,
        in pixels counted from zero, with shape ``(num_frames, 2)``.
    image
        The level 2 images, with shape ``(num_frames, num_y, num_x)``,
        and NaN outside the field of view.
    crval
        The helioprojective coordinates of the reference pixel of each frame,
        in arcseconds, with shape ``(num_frames, 2)``.
    crpix
        The reference pixel, counted from zero, with shape ``(2,)``.
    cdelt
        The plate scale along each axis in arcseconds, with shape ``(2,)``.
    pc
        The rotation matrix of each frame, with shape ``(num_frames, 2, 2)``.
    num_retry
        The number of times to try to connect to the server.
    """
    offsets = _offsets(window, time, binning, num_retry)

    shift = _shift(
        offsets=offsets,
        slit=slit,
        image=image,
        crval=crval,
        crpix=crpix,
        cdelt=cdelt,
        pc=pc,
    )

    t, x, y = _pixels(offsets, slit + np.array(shift))

    _, num_y, num_x = image.shape
    inside = (x >= 0) & (x < num_x) & (y >= 0) & (y < num_y)

    result = np.zeros(image.shape, dtype=bool)
    result[t[inside], y[inside], x[inside]] = True

    return result & np.isfinite(image)
