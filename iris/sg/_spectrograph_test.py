import pytest
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.time
import astropy.io.fits
import astropy.wcs
import named_arrays as na
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

    def test_mosaic(self, array: iris.sg.SpectrographObservation):
        result = array.mosaic()
        assert isinstance(result, iris.sg.SpectrographObservation)
        assert array.axis_time not in result.shape
        assert result.shape[array.axis_wavelength] == array.shape[array.axis_wavelength]
        assert np.nansum(result.outputs) > 0 * u.DN
        assert np.isfinite(result.timedelta).any()
        # A pixel no tile covers has no exposure time and no data, though a
        # pixel a tile covers can still be NaN at every wavelength.
        where_data = np.isfinite(result.outputs).any(array.axis_wavelength)
        assert not np.any(where_data & np.isnan(result.timedelta))
        time_result = result.inputs.time.ndarray.mean()
        time_array = array.inputs.time.ndarray.mean()
        assert abs(time_result - time_array) < 1 * u.s
        assert result.radiance.outputs.unit == array.radiance.outputs.unit


def _observation_synthetic() -> iris.sg.SpectrographObservation:
    """
    Two tiles on a one-arcsecond grid, both 6 by 4 pixels and 3 wavelengths.

    The second tile is 3 pixels to the right of and 2 pixels above the first,
    so that they overlap in a 3 by 2 pixel block, and is shifted by half a
    pixel in wavelength. The first tile is 1 DN everywhere but one NaN pixel,
    and the second is 3 DN everywhere.
    """
    axis_wavelength = "wavelength"
    axis_x = "detector_x"
    axis_y = "detector_y"

    num_x = 6
    num_y = 4
    num_wavelength = 3

    result = iris.sg.SpectrographObservation.empty(
        shape_base=dict(time=2),
        shape_wcs={
            axis_x: num_x,
            axis_y: num_y,
            axis_wavelength: num_wavelength,
        },
    )

    def pair(a: float, b: float, unit: u.Unit = u.dimensionless_unscaled):
        return na.ScalarArray(np.array([a, b]) * unit, axes="time")

    inputs = result.inputs
    inputs.wavelength_rest = 1394 * u.AA
    inputs.crval.wavelength = pair(1393, 1393, u.AA)
    inputs.crval.position.x = pair(0, 3, u.arcsec)
    inputs.crval.position.y = pair(0, 2, u.arcsec)
    inputs.cdelt.wavelength = pair(0.1, 0.1, u.AA)
    inputs.cdelt.position.x = pair(1, 1, u.arcsec)
    inputs.cdelt.position.y = pair(1, 1, u.arcsec)
    inputs.crpix.components[axis_wavelength] = pair(0, 0.5)
    inputs.pc.wavelength.components[axis_wavelength] = pair(1, 1)
    inputs.pc.position.x.components[axis_x] = pair(1, 1)
    inputs.pc.position.y.components[axis_y] = pair(1, 1)

    jd = np.stack([np.full(num_x + 1, 2461000.0), np.full(num_x + 1, 2461001.0)])
    inputs.time.ndarray = astropy.time.Time(jd, format="jd")

    result.outputs[dict(time=0)] = 1 * u.DN
    result.outputs[dict(time=1)] = 3 * u.DN
    result.outputs[{"time": 0, axis_x: 0, axis_y: 0, axis_wavelength: 0}] = np.nan

    result.timedelta = na.ScalarArray(
        ndarray=np.array([[2.0] * num_x, [4.0] * num_x]) * u.s,
        axes=("time", axis_x),
    )

    return result


def test_mosaic_synthetic():
    """
    Every pixel of the mosaic of two flat tiles is known in advance.
    """
    array = _observation_synthetic()

    result = array.mosaic()

    assert result.shape == dict(detector_x=9, detector_y=6, wavelength=3)

    # The first vertex is half a pixel outside the first pixel center.
    assert np.isclose(result.inputs.position.x.min().ndarray, -0.5 * u.arcsec)
    assert np.isclose(result.inputs.position.y.min().ndarray, -0.5 * u.arcsec)

    x = na.arange(0, 9, axis="detector_x")
    y = na.arange(0, 6, axis="detector_y")
    in_1 = (x < 6) & (y < 4)
    in_2 = (x >= 3) & (y >= 2)

    expected = na.ScalarArray(
        np.full((9, 6), np.nan), axes=("detector_x", "detector_y")
    )
    expected[in_1] = 1
    expected[in_2] = 3
    expected[in_1 & in_2] = 2

    # The NaN pixel of the first tile is the only thing covering its own
    # position, so the mosaic is NaN there and only there.
    expected_0 = expected.copy()
    expected_0[dict(detector_x=0, detector_y=0)] = np.nan

    # The second tile is shifted by half a pixel in wavelength, so its last
    # pixel covers only half of the last wavelength of the mosaic, and in the
    # overlap that wavelength is weighted 1 to 1/2 between the tiles.
    expected_2 = expected.copy()
    expected_2[in_1 & in_2] = (1 + 3 / 2) / (1 + 1 / 2)

    outputs = result.outputs.value
    for index_wavelength, expected_wavelength in enumerate(
        [expected_0, expected, expected_2]
    ):
        np.testing.assert_allclose(
            outputs[dict(wavelength=index_wavelength)].ndarray,
            expected_wavelength.ndarray,
            atol=1e-6,
        )

    expected_timedelta = expected.copy()
    expected_timedelta[in_1] = 2
    expected_timedelta[in_2] = 4
    expected_timedelta[in_1 & in_2] = 3
    np.testing.assert_allclose(
        result.timedelta.value.ndarray,
        expected_timedelta.ndarray,
        atol=1e-6,
    )

    # Vertices are covered if any pixel touching them is, and a vertex
    # touched only by one tile has that tile's time.
    time = result.inputs.time.ndarray
    assert time.shape == (10, 7)
    assert time.mask[8, 0]
    assert time.mask[0, 6]
    assert not time.mask[0, 0]
    assert not time.mask[9, 6]
    assert np.isclose(time[0, 0].jd, 2461000.0)
    assert np.isclose(time[9, 6].jd, 2461001.0)
    assert np.isclose(time[4, 3].jd, 2461000.5)

    # The mosaic can be done again, since it is itself a single tile.
    again = result.mosaic()
    np.testing.assert_allclose(
        again.outputs.value.ndarray,
        result.outputs.value.ndarray,
        atol=1e-6,
    )


def test_mosaic_cdelt():
    """
    A coarser mosaic of flat tiles is still flat, and has fewer pixels.
    """
    array = _observation_synthetic()

    result = array.mosaic(cdelt=2 * u.arcsec)

    assert result.shape == dict(detector_x=5, detector_y=3, wavelength=3)
    assert np.isclose(result.inputs.cdelt.position.x.ndarray, 2 * u.arcsec)
    assert np.isclose(result.inputs.cdelt.position.y.ndarray, 2 * u.arcsec)

    outputs = result.outputs.value[dict(wavelength=1)]
    assert np.isclose(outputs[dict(detector_x=0, detector_y=0)].ndarray, 1)
    assert np.isclose(outputs[dict(detector_x=4, detector_y=2)].ndarray, 3)
    assert np.isnan(outputs[dict(detector_x=4, detector_y=0)].ndarray)

    result = array.mosaic(cdelt=na.Cartesian2dVectorArray(1, 2) * u.arcsec)
    assert result.shape == dict(detector_x=9, detector_y=3, wavelength=3)


def test_mosaic_wavelength_rest_mismatch():
    array = _observation_synthetic()
    array.inputs.wavelength_rest = na.ScalarArray(
        ndarray=np.array([1394, 1403]) * u.AA,
        axes="time",
    )
    with pytest.raises(ValueError):
        array.mosaic()


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
