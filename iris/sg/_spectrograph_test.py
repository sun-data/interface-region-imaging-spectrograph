import pytest
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import iris


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T07:00"),
            nrt=True,
        ),
        iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2017-02-11T04:50"),
            time_stop=astropy.time.Time("2017-02-11T05:00"),
        ),
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

    def test_show(self, array: iris.sg.SpectrographObservation):
        result = array.show()
        assert isinstance(result, plt.Axes)

    def test_to_jshtml(self, array: iris.sg.SpectrographObservation):
        result = array.to_jshtml()
        assert isinstance(result, IPython.display.HTML)


def test_empty_custom_axes():

    axis_wavelength = "velocity"
    axis_detector_x = "x"
    axis_detector_y = "y"

    result = iris.sg.SpectrographObservation.empty(
        shape_base=dict(time=1),
        shape_wcs={
            axis_wavelength: 2,
            axis_detector_x: 3,
            axis_detector_y: 4,
        },
        axis_wavelength=axis_wavelength,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
    )

    axes = {axis_wavelength, axis_detector_x, axis_detector_y}
    assert set(result.inputs.crpix.components) == axes
    assert set(result.inputs.pc.wavelength.components) == axes
    assert set(result.inputs.pc.position.x.components) == axes
    assert set(result.inputs.pc.position.y.components) == axes
