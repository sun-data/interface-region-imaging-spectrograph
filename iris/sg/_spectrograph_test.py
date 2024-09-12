import pytest
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
