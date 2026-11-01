import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import iris
from ._fit_line_test import observation

_shape = dict(time=1, detector_x=20, detector_y=64, wavelength=81)
"""The shape of the synthetic observations."""


def _line(velocity: na.AbstractScalar) -> na.AbstractScalar:
    """A Gaussian line on a faint continuum."""
    z = velocity / (20 * u.km / u.s)
    return 400 * u.DN * np.exp(-np.square(z) / 2) + 2 * u.DN


def _raster(seed: int = 0) -> iris.sg.SpectrographObservation:
    """A synthetic raster of :func:`_line`, without hits."""
    return observation(
        profile=_line,
        num_x=_shape["detector_x"],
        num_y=_shape["detector_y"],
        num_wavelength=_shape["wavelength"],
        seed=seed,
    )


def _hits(
    obs: iris.sg.SpectrographObservation,
    peak: u.Quantity,
    wavelength: tuple[int, ...],
    seed: int = 1,
) -> tuple[iris.sg.SpectrographObservation, list[dict[str, int]]]:
    """
    Add a hit to each exposure of `obs`, at a random position along the slit
    and at one of the pixels `wavelength`, with a negative lobe of 6% of its
    peak on the eight pixels around it, as level 2 data have.
    """
    rng = np.random.default_rng(seed)
    outputs = obs.outputs.copy()
    centers = []
    for x in range(_shape["detector_x"]):
        center = dict(
            time=0,
            detector_x=x,
            detector_y=int(rng.integers(3, _shape["detector_y"] - 3)),
            wavelength=int(rng.choice(wavelength)),
        )
        y, w = center["detector_y"], center["wavelength"]
        box = dict(
            time=0,
            detector_x=x,
            detector_y=slice(y - 1, y + 2),
            wavelength=slice(w - 1, w + 2),
        )
        outputs[box] = outputs[box] - 0.06 * peak
        outputs[center] = outputs[center] + 1.06 * peak
        centers.append(center)
    return dataclasses.replace(obs, outputs=outputs), centers


@pytest.mark.parametrize(
    argnames="wavelength,peak,fraction",
    argvalues=[
        ((10, 20, 60, 70), 300 * u.DN, 1),
        ((40,), 2000 * u.DN, 0.95),
    ],
    ids=["continuum", "line"],
)
def test_spikes(
    wavelength: tuple[int, ...],
    peak: u.Quantity,
    fraction: float,
) -> None:
    """
    Every hit is found, on the continuum and on the core of the bright line,
    with at least `fraction` of the pixels of its negative lobe, and few other
    pixels are.
    The Sun around a lobe on the line is underestimated where the line
    curves, so only the lobes of large hits are found there.
    """
    obs, centers = _hits(_raster(), peak=peak, wavelength=wavelength)
    obs.outputs[dict(detector_y=slice(0, 2))] = np.nan

    result = obs.spikes()

    assert result.shape == obs.outputs.shape
    num_lobe = 0
    for center in centers:
        y, w = center["detector_y"], center["wavelength"]
        box = center | dict(
            detector_y=slice(y - 1, y + 2),
            wavelength=slice(w - 1, w + 2),
        )
        assert result[center]
        num_lobe += int(np.sum(result[box].ndarray)) - 1
    assert num_lobe >= fraction * 8 * len(centers)
    assert not np.any(result[dict(detector_y=slice(0, 2))])
    assert np.mean(result) < 0.003


def test_spikes_dn() -> None:
    """The outputs must be in DN, which the limits of the tests are in."""
    obs = _raster()
    obs = dataclasses.replace(obs, outputs=obs.outputs.value)
    with pytest.raises(ValueError, match="DN"):
        obs.spikes()


def test_hot_pixels() -> None:
    """
    A pixel with a fixed pattern on faint Sun is hot, and so is a pixel which
    is a spike in most exposures, but a pixel with the same pattern on the
    bright line is not.
    """
    obs = _raster()
    faint = dict(detector_y=30, wavelength=10)
    bright = dict(detector_y=30, wavelength=40)
    for pixel in (faint, bright):
        obs.outputs[pixel] = obs.outputs[pixel] + 6 * u.DN

    spikes = na.ScalarArray.zeros(obs.outputs.shape, dtype=bool)
    often = dict(detector_y=50, wavelength=60)
    spikes[often | dict(detector_x=slice(0, 15))] = True

    result = obs.hot_pixels(spikes=spikes)

    assert result.shape == dict(detector_y=64, wavelength=81)
    assert result[faint]
    assert result[often]
    assert not result[bright]
    assert np.sum(result) == 2


def test_hot_pixels_default() -> None:
    """Without `spikes`, the hot pixels are found from :meth:`spikes`."""
    obs, _ = _hits(_raster(), peak=300 * u.DN, wavelength=(10,))
    spikes = obs.spikes()
    assert np.all(obs.hot_pixels() == obs.hot_pixels(spikes=spikes))


def test_dark_rows() -> None:
    """
    The rows behind a fiducial and at a dark end of the slit are dark, with
    the rows next to them.
    """
    obs = _raster()
    obs.outputs[dict(detector_y=30)] = 0.1 * obs.outputs[dict(detector_y=30)]
    obs.outputs[dict(detector_y=slice(60, None))] = (
        0.05 * obs.outputs[dict(detector_y=slice(60, None))]
    )
    obs.outputs[dict(detector_y=slice(0, 2))] = np.nan

    result = obs.dark_rows()

    assert result.shape == dict(detector_y=64)
    rows = np.flatnonzero(result.ndarray)
    assert rows.tolist() == [29, 30, 31, 59, 60, 61, 62, 63]


def test_dark_rows_noise() -> None:
    """The noise of a faint observation does not make any row dark."""

    def continuum(velocity: na.AbstractScalar) -> na.AbstractScalar:
        """A faint, flat continuum."""
        return 2 * u.DN + 0 * velocity.value

    obs = observation(continuum, num_x=30, num_y=200, seed=2)
    assert not np.any(obs.dark_rows())


def test_dark_rows_few() -> None:
    """
    No rows are found in fewer than 20 exposures, over which the structure
    of the Sun along the slit would not average out.
    """
    obs = _raster()[dict(detector_x=slice(0, 19))]
    obs.outputs[dict(detector_y=30)] = 0.1 * obs.outputs[dict(detector_y=30)]
    assert not np.any(obs.dark_rows())


def test_clean() -> None:
    """
    The outputs are NaN at the spikes, the hot pixels and the dark rows, and
    the same as before everywhere else, and the observation is not changed.
    """
    obs, _ = _hits(_raster(), peak=300 * u.DN, wavelength=(10, 70))
    obs.outputs[dict(detector_y=30)] = 0.1 * obs.outputs[dict(detector_y=30)]
    original = obs.outputs.copy()

    result = obs.clean()

    spikes = obs.spikes()
    mask = spikes | obs.hot_pixels(spikes=spikes) | obs.dark_rows()
    assert np.any(mask)
    assert np.all(np.isnan(result.outputs) == mask)
    assert np.all((result.outputs == obs.outputs) | mask)
    assert np.all(obs.outputs == original)


def test_clean_masks() -> None:
    """The masks which are given are used instead of the tests."""
    obs = _raster()
    spikes = na.ScalarArray.zeros(obs.outputs.shape, dtype=bool)
    spikes[dict(detector_x=3, detector_y=10, wavelength=20)] = True
    hot_pixels = na.ScalarArray.zeros(dict(detector_y=64, wavelength=81), dtype=bool)
    hot_pixels[dict(detector_y=20, wavelength=30)] = True
    dark_rows = na.ScalarArray.zeros(dict(detector_y=64), dtype=bool)
    dark_rows[dict(detector_y=40)] = True

    result = obs.clean(spikes=spikes, hot_pixels=hot_pixels, dark_rows=dark_rows)

    expected = spikes | hot_pixels | dark_rows
    assert np.all(np.isnan(result.outputs) == expected)
