import pytest
import numpy as np
import astropy.units as u
import astropy.time
import iris


@pytest.mark.parametrize(
    argnames="obs",
    argvalues=[
        iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T07:00"),
        )
    ],
)
def test_estimate(
    obs: iris.sg.SpectrographObservation,
):
    wavelength_center = obs.wavelength_center.ndarray.mean()
    obs.inputs = obs.inputs.explicit
    obs.inputs.wavelength = obs.inputs.wavelength.to(
        unit=u.km / u.s,
        equivalencies=u.doppler_optical(wavelength_center),
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
