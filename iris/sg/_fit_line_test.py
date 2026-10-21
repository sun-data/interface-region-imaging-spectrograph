from typing import Callable
import dataclasses
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import ndoptimize
import iris

_wavelength_rest = 1393.755 * u.AA
"""The rest wavelength of the synthetic observations, that of Si IV."""


def observation(
    profile: Callable[[na.AbstractScalar], na.AbstractScalar],
    num_x: int = 20,
    num_y: int = 20,
    num_wavelength: int = 81,
    wavelength_rest: u.Quantity = _wavelength_rest,
    dispersion: u.Quantity = 25.4 * u.mAA,
    seed: int = 0,
) -> iris.sg.SpectrographObservation:
    """
    A synthetic observation of a single raster, whose spectra are `profile`
    of the Doppler velocity of each wavelength pixel, plus the read and
    photon noise of :attr:`iris.sg.SpectrographObservation.variance`.

    Parameters
    ----------
    profile
        The noise-free spectra, in DN, as a function of the velocity relative
        to `wavelength_rest`.
    num_x
        The number of raster steps.
    num_y
        The number of pixels along the slit.
    num_wavelength
        The number of wavelength pixels, centered on `wavelength_rest`.
    wavelength_rest
        The rest wavelength of the observation.
    dispersion
        The width of a wavelength pixel.
    seed
        The seed of the noise.
    """
    obs = iris.sg.SpectrographObservation.empty(
        shape_base=dict(time=1),
        shape_wcs=dict(wavelength=num_wavelength, detector_x=num_x, detector_y=num_y),
    )
    inputs = obs.inputs
    inputs.time = na.ScalarArray(
        ndarray=astropy.time.Time(
            val=2459480.75 + np.arange(num_x + 1)[np.newaxis] / 86400 * 10,
            format="jd",
        ),
        axes=("time", obs.axis_detector_x),
    )
    inputs.wavelength_rest = wavelength_rest
    inputs.crval.wavelength = wavelength_rest
    inputs.crpix.components[obs.axis_wavelength] = (num_wavelength - 1) / 2
    inputs.cdelt.wavelength = dispersion
    inputs.cdelt.position.x = 0.35 * u.arcsec
    inputs.cdelt.position.y = 0.17 * u.arcsec
    inputs.pc.wavelength.components[obs.axis_wavelength] = 1
    inputs.pc.position.x.components[obs.axis_detector_x] = 1
    inputs.pc.position.y.components[obs.axis_detector_y] = 1

    velocity = inputs.velocity.cell_centers(obs.axis_wavelength)
    mean = profile(velocity).broadcast_to(obs.outputs.shape)
    noise_free = dataclasses.replace(obs, outputs=mean)
    obs.outputs = na.random.normal(mean, np.sqrt(noise_free.variance), seed=seed)
    return obs


def _gaussian_linear(velocity: na.AbstractScalar) -> na.AbstractScalar:
    """A Gaussian line which varies across the raster, on a sloped continuum."""
    x = na.linspace(0, 1, axis="detector_x", num=20)
    amplitude = (40 + 200 * x) * u.DN
    center = (-10 + 20 * x) * u.km / u.s
    width = 15 * u.km / u.s
    z = (velocity - center) / width
    continuum = 2 * u.DN + 0.005 * u.DN / (u.km / u.s) * velocity
    return amplitude * np.exp(-np.square(z) / 2) + continuum


def test_fit_line():
    """
    The fit of each spectrum of a synthetic raster recovers the line, and its
    errors are consistent with its uncertainties.
    """
    obs = observation(_gaussian_linear)
    result = iris.sg.fit_line(obs, velocity_max=100 * u.km / u.s)

    assert isinstance(result, na.ModelFitFunctionArray)
    assert np.all(result.converged)
    assert result.parameters.center.shape == {
        "time": 1,
        "detector_x": 20,
        "detector_y": 20,
    }

    x = na.linspace(0, 1, axis="detector_x", num=20)
    truth = dict(
        amplitude=(40 + 200 * x) * u.DN,
        center=(-10 + 20 * x) * u.km / u.s,
        width=15 * u.km / u.s,
        slope=0.005 * u.DN / (u.km / u.s),
        intercept=2 * u.DN,
    )
    for name, value in truth.items():
        error = result.parameters.components[name] - value
        sigma = np.sqrt(result.covariance.components[name].components[name])
        pull = (error / sigma).ndarray
        assert np.abs(np.mean(pull)) < 0.2, name
        assert 0.8 < np.std(pull) < 1.2, name

    # the reduced chi-squared of the right model and noise is about one
    chi2 = (result.chi2 / (result.num_points - 5)).ndarray
    assert 0.9 < np.mean(chi2) < 1.1


def test_fit_line_wavelength_rest():
    """
    The velocities are relative to the given rest wavelength, so a line which
    is not at the rest wavelength of the window is centered on its own.
    """
    shift = 40 * u.km / u.s
    wavelength_rest = _wavelength_rest * (1 + shift / (299792.458 * u.km / u.s))

    def profile(velocity: na.AbstractScalar) -> na.AbstractScalar:
        z = (velocity - shift) / (15 * u.km / u.s)
        return 100 * u.DN * np.exp(-np.square(z) / 2)

    obs = observation(profile, num_x=4, num_y=4)
    center = iris.sg.fit_line(obs, wavelength_rest=wavelength_rest).parameters.center
    assert np.all(np.abs(center) < 3 * u.km / u.s)
    center = iris.sg.fit_line(obs).parameters.center
    assert np.all(np.abs(center - shift) < 3 * u.km / u.s)


def test_fit_line_kappa():
    """A kappa line is fit by its own model, from the default guess."""
    truth = dict(
        amplitude=150 * u.DN,
        shift=5 * u.km / u.s,
        width=20 * u.km / u.s,
        kappa=1.5,
        bias=1 * u.DN,
        slope=0 * u.DN / (u.km / u.s),
    )

    def profile(velocity: na.AbstractScalar) -> na.AbstractScalar:
        return iris.sg.background.model_total(velocity, **truth)

    obs = observation(profile, num_x=10, num_y=10)
    result = iris.sg.fit_line(obs, model=iris.sg.models.kappa_linear)
    assert np.all(result.converged)
    p = result.parameters
    assert np.abs(np.median(p.center.ndarray) - truth["shift"]) < 1 * u.km / u.s
    assert np.abs(np.median(p.width.ndarray) - truth["width"]) < 1 * u.km / u.s
    assert np.abs(np.median(p.kappa.ndarray) - truth["kappa"]) < 0.3


def test_fit_line_guess():
    """
    A model whose parameters have no default guess needs one, and a guess
    which is given is used.
    """
    obs = observation(_gaussian_linear, num_y=2)
    with pytest.raises(ValueError, match="give one with `guess`"):
        iris.sg.fit_line(obs, model=ndoptimize.models.two_gaussians_linear)

    guess = na.CartesianNdVectorArray(
        dict(
            amplitude=100 * u.DN,
            center=0 * u.km / u.s,
            width=10 * u.km / u.s,
        )
    )
    result = iris.sg.fit_line(
        obs,
        model=ndoptimize.models.gaussian,
        guess=guess,
        max_iterations=0,
    )
    assert np.all(result.parameters.amplitude == guess.amplitude)


def test_variance():
    """
    The variance of each output is the read noise plus the photon noise, and
    needs outputs in DN.
    """
    obs = observation(_gaussian_linear, num_y=2)
    signal = np.maximum(obs.outputs, 0 * u.DN)
    expected = (3.1 * u.DN) ** 2 + signal * u.DN / 4
    assert np.allclose(obs.variance, expected)

    with pytest.raises(ValueError, match="outputs in DN"):
        dataclasses.replace(obs, outputs=obs.outputs.value * u.erg).variance


def test_read_noise():
    """The read noise of each channel of the spectrograph."""
    assert iris.sg.read_noise(1394 * u.AA) == 3.1 * u.DN
    assert iris.sg.read_noise(2796 * u.AA) == 1.2 * u.DN
