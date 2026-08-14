import pytest
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.io.fits
import astropy.wcs
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


def test_inputs_against_astropy_wcs():
    """
    The coordinates must be the ones :mod:`astropy.wcs` makes of the same file.

    The keywords are put into the formula in the WCS paper, which is written
    out correctly, but that formula is given pixel coordinates counted from
    zero while `CRPIX` counts them from one. An implementation which knows
    nothing about this one is asked the same question, and is asked it on
    every axis: the wavelength axis is the one where a pixel is worth about
    3 km/s and would go unnoticed as an offset in Doppler shift.

    Built from a file rather than a time range, so that the coordinates and
    the keywords they are checked against come from the same file and not
    from two which happen to have been found by the same search.
    """
    window = "Si IV 1394"

    urls = iris.data.urls_hek(
        time_start=astropy.time.Time("2021-09-23T06:00"),
        time_stop=astropy.time.Time("2021-09-23T07:00"),
        spectrograph=True,
        sji=False,
    )
    path = iris.data.decompress(iris.data.download(urls[:1]))[0]

    result = iris.sg.SpectrographObservation.from_fits(path, window=window)

    hdul = astropy.io.fits.open(path)
    windows = [hdul[0].header.get(f"TDESC{h}") for h in range(len(hdul))]
    wcs = astropy.wcs.WCS(hdul[windows.index(window)])

    names = list(wcs.axis_type_names)

    # Without the projection, which :class:`named_arrays.AbstractWcsVector`
    # does not apply: it is the linear part of the transformation, and this
    # is a comparison of that against the same thing. Over a field this size
    # the projection is worth a few times ten to the minus five arcseconds
    # either way, so leaving it in would not hide the pixel this is looking
    # for, but taking it out makes the two answers the same answer.
    wcs.wcs.ctype = names
    columns = {
        "WAVE": result.axis_wavelength,
        "HPLN": result.axis_detector_x,
        "HPLT": result.axis_detector_y,
    }

    inputs = result.inputs
    shape = inputs.shape_wcs

    def value(quantity, unit):
        # Only the axes it has: the wavelength does not vary along the slit,
        # and `pc` says so, but the two spatial axes are mixed by the roll
        # and each depends on both.
        return quantity[{a: index[a] for a in quantity.shape}].ndarray.to_value(unit)

    for corner in ((0, 0, 0), (1, 2, 3), (0, -1, -1), (-1, -1, -1)):
        index = {
            columns[name]: corner[i] % shape[columns[name]]
            for i, name in enumerate(names)
        }

        # Astropy counts pixels from zero here, and the vertex of index `j`
        # lies half a pixel below the center of pixel `j`.
        pixel = [[index[columns[name]] - 0.5 for name in names]]
        expected = wcs.wcs_pix2world(pixel, 0)[0]
        expected = {name: expected[i] for i, name in enumerate(names)}

        assert np.isclose(value(inputs.wavelength, u.m), expected["WAVE"], rtol=1e-10)
        assert np.isclose(value(inputs.position.x, u.deg), expected["HPLN"], rtol=1e-10)
        assert np.isclose(value(inputs.position.y, u.deg), expected["HPLT"], rtol=1e-10)
