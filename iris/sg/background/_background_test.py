import pytest
import numpy as np
import astropy.units as u
import astropy.time
import named_arrays as na
import iris
from iris.sg._fit_line_test import observation, _wavelength_rest


@pytest.mark.parametrize("window", ["Si IV 1394", "Fe XII 1349"])
def test_estimate(
    window: str,
):
    obs = iris.sg.SpectrographObservation.from_time_range(
        time_start=astropy.time.Time("2021-09-23T06:00"),
        time_stop=astropy.time.Time("2021-09-23T07:00"),
        window=window,
    )

    result = iris.sg.background.estimate(
        obs=obs,
        axis_time=obs.axis_time,
        axis_wavelength=obs.axis_wavelength,
        axis_detector_x=obs.axis_detector_x,
        axis_detector_y=obs.axis_detector_y,
    )

    where = np.isfinite(result.outputs)
    assert np.all(result.outputs > -10 * u.DN, where=where)
    assert np.all(result.outputs < +10 * u.DN, where=where)


def test_fit():
    """
    The fit recovers the parameters of :func:`iris.sg.background.model_total`
    from noisy spectra, and names them as its arguments.
    """
    velocity = na.linspace(-150, 150, axis="wavelength", num=61) * u.km / u.s
    truth = dict(
        amplitude=na.ScalarArray(np.array([50.0, 100, 200]), "row") * u.DN,
        shift=na.ScalarArray(np.array([-5.0, 0, 8]), "row") * u.km / u.s,
        width=25 * u.km / u.s,
        kappa=1.5,
        bias=3 * u.DN,
        slope=0.004 * u.DN / (u.km / u.s),
    )
    data = na.random.normal(
        iris.sg.background.model_total(velocity, **truth),
        0.3 * u.DN,
        seed=0,
    )

    result = iris.sg.background.fit(
        data=data,
        velocity=velocity,
        axis_wavelength="wavelength",
    )

    # about five standard deviations of each parameter of the broadest line
    tolerance = dict(
        amplitude=1 * u.DN,
        shift=0.3 * u.km / u.s,
        width=1.5 * u.km / u.s,
        kappa=0.5,
        bias=0.3 * u.DN,
        slope=0.002 * u.DN / (u.km / u.s),
    )
    assert set(result.components) == set(truth)
    for name, value in truth.items():
        error = result.components[name] - value
        assert np.all(np.abs(error) < tolerance[name]), name


def test_fit_units():
    """Whether the fit converges does not depend on the units of the data."""
    velocity = na.linspace(-150, 150, axis="wavelength", num=61) * u.km / u.s
    spectra = iris.sg.background.model_total(
        velocity=velocity,
        amplitude=na.linspace(50, 200, axis="row", num=50) * u.DN,
        shift=0 * u.km / u.s,
        width=25 * u.km / u.s,
        kappa=1.5,
        bias=3 * u.DN,
        slope=0.004 * u.DN / (u.km / u.s),
    )
    data = na.random.normal(spectra, 0.3 * u.DN, seed=0)

    # more than the factor from DN to the radiance of IRIS spectra
    factor = 1e6 * u.erg / (u.cm**2 * u.sr * u.nm * u.s * u.DN)
    p = iris.sg.background.fit(data, velocity, "wavelength").components
    p_radiance = iris.sg.background.fit(data * factor, velocity, "wavelength")
    for name, value in p_radiance.components.items():
        scale = factor if name in ("amplitude", "bias", "slope") else 1
        assert np.allclose(value, p[name] * scale, rtol=1e-6), name


def test_fit_unconverged():
    """The parameters of a spectrum with nothing to fit are NaN."""
    velocity = na.linspace(-150, 150, axis="wavelength", num=61) * u.km / u.s
    spectrum = iris.sg.background.model_total(
        velocity=velocity,
        amplitude=100 * u.DN,
        shift=0 * u.km / u.s,
        width=25 * u.km / u.s,
        kappa=1.5,
        bias=3 * u.DN,
        slope=0 * u.DN / (u.km / u.s),
    )
    row = na.arange(0, 2, axis="row")
    data = np.where(row == 0, spectrum, np.nan * u.DN)

    result = iris.sg.background.fit(
        data=data,
        velocity=velocity,
        axis_wavelength="wavelength",
    )

    for name, value in result.components.items():
        assert np.all(np.isfinite(value[dict(row=0)])), name
        assert np.all(np.isnan(value[dict(row=1)])), name


def test_fit_continuum():
    """
    The line cannot take the place of a curve in the continuum, which made
    it large outside the pixels that are fit.
    """
    velocity = na.linspace(-440, 440, axis="wavelength", num=161) * u.km / u.s
    continuum = (
        3 * u.DN
        + 0.001 * u.DN / (u.km / u.s) * velocity
        + 3e-5 * u.DN / (u.km / u.s) ** 2 * np.square(velocity)
    )
    line = iris.sg.background.model_spectral_line(
        velocity=velocity,
        amplitude=na.linspace(0, 5, axis="row", num=40) * u.DN,
        shift=0 * u.km / u.s,
        width=20 * u.km / u.s,
        kappa=2,
    )
    data = na.random.normal(continuum + line, 0.3 * u.DN, seed=1)

    p = iris.sg.background.fit(
        data=data,
        velocity=velocity,
        axis_wavelength="wavelength",
    ).components

    line_fit = iris.sg.background.model_spectral_line(
        velocity=velocity,
        amplitude=p["amplitude"],
        shift=p["shift"],
        width=p["width"],
        kappa=p["kappa"],
    )
    outside = np.abs(velocity) > 150 * u.km / u.s
    assert np.all(np.abs(line_fit) < 3 * u.DN, where=outside)


def test_subtract_spectral_line():
    """
    The observation is left as it was, one line in an array is the line of
    the window, and lines closer than the pixels to which each is fit are an
    error.
    """

    def profile(velocity: na.AbstractScalar) -> na.AbstractScalar:
        return iris.sg.background.model_total(
            velocity=velocity,
            amplitude=60 * u.DN,
            shift=0 * u.km / u.s,
            width=20 * u.km / u.s,
            kappa=3,
            bias=3 * u.DN,
            slope=0 * u.DN / (u.km / u.s),
        )

    obs = observation(profile, num_x=4, num_y=6)
    avg = iris.sg.background.average(obs, axis=(obs.axis_time, obs.axis_detector_x))
    outputs = avg.outputs.copy()

    result = iris.sg.background.subtract_spectral_line(avg, avg.axis_wavelength)
    assert np.all(avg.outputs == outputs)

    for wavelength_rest in [
        u.Quantity([_wavelength_rest]),
        na.ScalarArray(u.Quantity([_wavelength_rest]), axes="line"),
    ]:
        result_array = iris.sg.background.subtract_spectral_line(
            obs=avg,
            axis_wavelength=avg.axis_wavelength,
            wavelength_rest=wavelength_rest,
        )
        assert np.allclose(result_array.outputs, result.outputs)

    shift = 100 * u.km / u.s
    wavelength_close = _wavelength_rest * (1 + shift / (299792.458 * u.km / u.s))
    with pytest.raises(ValueError, match="subtracted twice"):
        iris.sg.background.subtract_spectral_line(
            obs=avg,
            axis_wavelength=avg.axis_wavelength,
            wavelength_rest=u.Quantity([_wavelength_rest, wavelength_close]),
        )


def test_smooth():
    """
    The observation is left as it was, and a spectrum which is NaN, like one
    whose fit did not converge, is filled in from its neighbors.
    """

    def profile(velocity: na.AbstractScalar) -> na.AbstractScalar:
        return 3 * u.DN + 0.002 * u.DN / (u.km / u.s) * velocity

    obs = observation(profile, num_x=4, num_y=6)
    avg = iris.sg.background.average(obs, axis=(obs.axis_time, obs.axis_detector_x))
    avg.outputs[{avg.axis_detector_y: 2}] = np.nan
    outputs = avg.outputs.copy()

    result = iris.sg.background.smooth(
        obs=avg,
        axis_wavelength=avg.axis_wavelength,
        axis_detector_y=avg.axis_detector_y,
    )
    assert np.array_equal(avg.outputs, outputs, equal_nan=True)
    assert np.all(np.isfinite(result.outputs))


def test_estimate_lines():
    """
    The lines given by `wavelength_rest` are removed from the background,
    while a line which is not given is left in it.
    """
    shift = 250 * u.km / u.s
    wavelength_second = _wavelength_rest * (1 + shift / (299792.458 * u.km / u.s))

    def background(velocity: na.AbstractScalar) -> na.AbstractScalar:
        return 3 * u.DN + 0.002 * u.DN / (u.km / u.s) * velocity

    def profile(velocity: na.AbstractScalar) -> na.AbstractScalar:
        lines = sum(
            iris.sg.background.model_spectral_line(
                velocity=velocity - v0,
                amplitude=60 * u.DN,
                shift=0 * u.km / u.s,
                width=20 * u.km / u.s,
                kappa=3,
            )
            for v0 in (0 * u.km / u.s, shift)
        )
        return lines + background(velocity)

    obs = observation(profile, num_wavelength=161)

    def error(wavelength_rest: None | u.Quantity) -> na.AbstractScalar:
        result = iris.sg.background.estimate(
            obs=obs,
            axis_time=obs.axis_time,
            axis_wavelength=obs.axis_wavelength,
            axis_detector_x=obs.axis_detector_x,
            axis_detector_y=obs.axis_detector_y,
            wavelength_rest=wavelength_rest,
        )
        velocity = result.inputs.velocity.cell_centers()
        return np.abs(result.outputs - background(velocity))

    both = error(u.Quantity([_wavelength_rest, wavelength_second]))
    assert np.nanmax(both) < 1 * u.DN

    first = error(None)
    assert np.nanmax(first) > 5 * u.DN
