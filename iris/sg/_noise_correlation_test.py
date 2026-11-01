import dataclasses
import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import iris
from ._fit_line import _velocity
from ._fit_line_test import observation


def _line(velocity: na.AbstractScalar) -> na.AbstractScalar:
    """A bright Gaussian line on a faint continuum."""
    z = velocity / (20 * u.km / u.s)
    return 400 * u.DN * np.exp(-np.square(z) / 2) + 2 * u.DN


def _correlated(
    axis: str,
    c: float,
    seed: int = 0,
) -> iris.sg.SpectrographObservation:
    """
    A synthetic raster of :func:`_line` whose noise along `axis` is that of
    white noise averaged with weights :math:`1` and :math:`c` over two
    neighboring pixels, so that the correlation of two neighboring pixels is
    :math:`c / (1 + c^2)`, and zero beyond.
    """
    obs = observation(_line, num_x=20, num_y=100, num_wavelength=81, seed=seed)
    noise_free = dataclasses.replace(
        obs,
        outputs=_line(_velocity(obs)).broadcast_to(obs.outputs.shape),
    )
    shape = dict(obs.outputs.shape)
    shape[axis] = shape[axis] + 1
    white = na.random.normal(0, 1, shape, seed=seed + 1)
    num = shape[axis]
    first = white[{axis: slice(None, num - 1)}]
    second = white[{axis: slice(1, None)}]
    noise = (first + c * second) / np.sqrt(1 + c**2)
    outputs = noise_free.outputs + noise * np.sqrt(noise_free.variance)
    return dataclasses.replace(obs, outputs=outputs)


@pytest.mark.parametrize("axis", ["wavelength", "detector_y"])
@pytest.mark.parametrize("c", [0, 0.25])
def test_noise_correlation(axis: str, c: float) -> None:
    """
    The correlation of the noise of neighboring pixels is found, despite the
    bright line, and that of pixels two apart is zero.
    """
    obs = _correlated(axis=axis, c=c)

    result = iris.sg.noise_correlation(obs, axis=axis, num_lag=3)

    assert isinstance(result, tuple)
    assert len(result) == 4
    assert result[0] == 1
    assert result[1] == pytest.approx(c / (1 + c**2), abs=0.03)
    assert result[2] == pytest.approx(0, abs=0.03)
    assert result[3] == pytest.approx(0, abs=0.03)


def test_noise_correlation_quantized() -> None:
    """
    Outputs in steps of 0.25 DN, as level 2 data are, give the same
    correlation.
    """
    obs = _correlated(axis="wavelength", c=0.25)
    step = 0.25 * u.DN
    quantized = dataclasses.replace(obs, outputs=np.round(obs.outputs / step) * step)

    result = iris.sg.noise_correlation(quantized, axis="wavelength", num_lag=3)
    expected = iris.sg.noise_correlation(obs, axis="wavelength", num_lag=3)

    assert result == pytest.approx(expected, abs=0.01)


def test_noise_correlation_spikes() -> None:
    """Spikes left in the outputs do not change the correlation."""
    obs = _correlated(axis="wavelength", c=0.25)
    expected = iris.sg.noise_correlation(obs, axis="wavelength")

    rng = np.random.default_rng(3)
    outputs = obs.outputs.copy()
    for x in range(20):
        index = dict(
            time=0,
            detector_x=x,
            detector_y=int(rng.integers(0, 100)),
            wavelength=int(rng.integers(0, 10)),
        )
        outputs[index] = outputs[index] + 500 * u.DN
    obs = dataclasses.replace(obs, outputs=outputs)

    result = iris.sg.noise_correlation(obs, axis="wavelength")

    assert result == pytest.approx(expected, abs=0.01)


def test_noise_correlation_where() -> None:
    """Only the pixels where `where` is true are used."""
    obs = _correlated(axis="wavelength", c=0.25)
    where = na.ScalarArray.zeros(obs.outputs.shape, dtype=bool)
    where[dict(detector_y=slice(0, 50))] = True
    outputs = obs.outputs.copy()
    outputs[dict(detector_y=slice(50, None))] = 0 * u.DN

    result = iris.sg.noise_correlation(
        dataclasses.replace(obs, outputs=outputs),
        axis="wavelength",
        where=where,
    )

    assert result[1] == pytest.approx(0.25 / (1 + 0.25**2), abs=0.04)


def test_noise_correlation_raster() -> None:
    """
    The noise of a level 2 raster is correlated by about 0.2 between
    neighboring pixels along wavelength and along the slit, and by a little
    less than zero between pixels two apart.
    """
    obs = iris.sg.open("2021-09-23T06:13")
    obs = obs[{obs.axis_detector_x: slice(0, 40)}]
    for axis in (obs.axis_wavelength, obs.axis_detector_y):
        result = iris.sg.noise_correlation(obs, axis=axis)
        assert result[0] == 1
        assert 0.1 < result[1] < 0.3
        assert -0.1 < result[2] < 0


def test_noise_correlation_errors() -> None:
    """
    The correlation is measured along an axis of each exposure, from at least
    two exposures, and from some pixels.
    """
    obs = _correlated(axis="wavelength", c=0)

    with pytest.raises(ValueError, match="along which the exposures vary"):
        iris.sg.noise_correlation(obs, axis="detector_x")

    with pytest.raises(ValueError, match="consecutive exposures"):
        iris.sg.noise_correlation(obs[dict(detector_x=0)], axis="wavelength")

    with pytest.raises(ValueError, match="no pixels"):
        iris.sg.noise_correlation(obs, axis="wavelength", where=False)
