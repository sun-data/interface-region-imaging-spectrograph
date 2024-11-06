import pytest
import numpy as np
import astropy.units as u
import astropy.time
import iris


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T07:00"),
        )
    ],
)
class TestSpectrographObservation:

    def test_axis_time(self, array: iris.sg.SpectrographObservation):
        assert isinstance(array.axis_time, str)

    def test_axis_wavelength(self, array: iris.sg.SpectrographObservation):
        assert isinstance(array.axis_wavelength, str)

    def test_axis_detector_x(self, array: iris.sg.SpectrographObservation):
        assert isinstance(array.axis_detector_x, str)

    def test_axis_detector_y(self, array: iris.sg.SpectrographObservation):
        assert isinstance(array.axis_detector_y, str)

    def test_radiance(self, array: iris.sg.SpectrographObservation):
        result = array.radiance
        assert isinstance(result, iris.sg.SpectrographObservation)
        assert np.all(result.inputs == array.inputs)
        assert np.nansum(result.outputs) > 0 * u.erg / (u.cm**2 * u.sr * u.s * u.nm)
