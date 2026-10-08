import pathlib
import pytest
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import astropy.wcs
import named_arrays as na
import iris
from iris.sji._slit_jaw import _interpolate_zeros

_time_start = astropy.time.Time("2021-09-23T06:13")
_time_stop = astropy.time.Time("2021-09-23T06:16")

_columns_pointing = [
    "XCENIX",
    "YCENIX",
    "PC1_1IX",
    "PC1_2IX",
    "PC2_1IX",
    "PC2_2IX",
]


def _path(window: str = "SJI_1400") -> pathlib.Path:
    """A level 2 slit-jaw file of the given channel."""
    urls = iris.data.urls_hek(
        time_start=_time_start,
        time_stop=_time_stop,
        spectrograph=False,
    )
    (url,) = [url for url in urls if f"_{window}_" in url]
    return iris.data.download([url])[0]


def _write(
    file: pathlib.Path,
    num_time: int,
    num_x: int,
    num_y: int,
    window: None | str = None,
    unknown: None | int | slice = None,
) -> pathlib.Path:
    """
    Write the first few frames of a real slit-jaw file, cut down to a corner
    of each image, as a new file.

    The corner keeps the pixel counted from zero, so `CRPIX` and the
    pointing describe the new file as well as the original.

    Parameters
    ----------
    file
        Where to write the new file.
    num_time
        The number of frames to keep.
    num_x
        The number of columns of each image to keep.
    num_y
        The number of rows of each image to keep.
    window
        If not :obj:`None`, the channel the new file says it is.
    unknown
        If not :obj:`None`, the index of the frames whose pointing is recorded
        as unknown.
    """
    with astropy.io.fits.open(_path()) as hdul:
        # Cut in memory, since cutting each row of a compressed file seeks
        # back through the compression for every row.
        data = hdul[0].section[:num_time][:, :num_y, :num_x]
        header = hdul[0].header.copy()
        aux = hdul[1].data[:num_time].copy()
        header_aux = hdul[1].header.copy()

    for key in ["BZERO", "BSCALE"]:
        header.remove(key, ignore_missing=True)

    if window is not None:
        header["TDESC1"] = window

    if unknown is not None:
        for key in _columns_pointing:
            aux[unknown, header_aux[key]] = 0

    hdul = astropy.io.fits.HDUList(
        hdus=[
            astropy.io.fits.PrimaryHDU(data, header),
            astropy.io.fits.ImageHDU(aux, header_aux),
        ]
    )
    hdul.writeto(file)

    return file


@pytest.mark.parametrize(
    argnames="array",
    argvalues=[
        iris.sji.SlitJawObservation.from_time_range(
            time_start=_time_start,
            time_stop=_time_stop,
        ),
        iris.sji.SlitJawObservation.from_time_range(
            time_start=_time_start,
            time_stop=_time_stop,
            window="SJI_2796",
            axis_time="t",
            axis_detector_x="x",
            axis_detector_y="y",
        ),
    ],
)
class TestSlitJawObservation:

    def test_axis_time(self, array: iris.sji.SlitJawObservation):
        assert isinstance(array.axis_time, str)

    def test_axis_detector_x(self, array: iris.sji.SlitJawObservation):
        assert isinstance(array.axis_detector_x, str)

    def test_axis_detector_y(self, array: iris.sji.SlitJawObservation):
        assert isinstance(array.axis_detector_y, str)

    def test_shape(self, array: iris.sji.SlitJawObservation):
        axes = {array.axis_time, array.axis_detector_x, array.axis_detector_y}
        assert set(array.outputs.shape) == axes
        assert set(array.inputs.crpix.components) == axes - {array.axis_time}
        assert set(array.inputs.pc.position.x.components) == axes - {array.axis_time}
        assert set(array.inputs.pc.position.y.components) == axes - {array.axis_time}

    def test_time(self, array: iris.sji.SlitJawObservation):
        time = array.inputs.time
        assert time.shape == {array.axis_time: array.shape[array.axis_time]}
        assert np.all(time.ndarray >= _time_start)
        assert np.all(time.ndarray < _time_stop)
        assert np.all(np.diff(time.ndarray.jd) > 0)

    def test_timedelta(self, array: iris.sji.SlitJawObservation):
        timedelta = array.timedelta
        assert timedelta.shape == {array.axis_time: array.shape[array.axis_time]}
        assert np.all(timedelta > 0 * u.s)

    def test_wavelength(self, array: iris.sji.SlitJawObservation):
        assert array.inputs.wavelength.ndarray in [1400, 2796] * u.AA

    def test_outputs(self, array: iris.sji.SlitJawObservation):
        outputs = array.outputs
        assert outputs.unit == u.DN
        assert np.all(np.isnan(outputs) | (outputs > -200 * u.DN))
        assert np.nansum(outputs) > 0 * u.DN

    def test_getitem(self, array: iris.sji.SlitJawObservation):
        index = {array.axis_time: 1}
        result = array[index]
        assert isinstance(result, iris.sji.SlitJawObservation)
        assert result.timedelta.shape == {}
        assert result.timedelta == array.timedelta[index]


def test_inputs_against_astropy_wcs():
    """
    The coordinates of each frame must be the ones :mod:`astropy.wcs` makes
    of the pointing recorded for that frame.

    The primary header of a slit-jaw file holds a single pointing for the
    whole observation, so the WCS of each frame is assembled from the
    auxiliary data the way :func:`irispy.io.read_files` assembles it.
    """
    path = _path()

    result = iris.sji.SlitJawObservation.from_fits(path)

    with astropy.io.fits.open(path) as hdul:
        header = hdul[0].header.copy()
        aux = hdul[1].data.copy()
        header_aux = hdul[1].header.copy()

    inputs = result.inputs
    shape = inputs.shape_wcs
    axis_x = result.axis_detector_x
    axis_y = result.axis_detector_y

    for t in (0, 1, -1):
        row = aux[t]

        # The coordinates of one frame, rather than of every frame.
        position = inputs[
            {result.axis_time: t % result.shape[result.axis_time]}
        ].position

        wcs = astropy.wcs.WCS(naxis=2)

        # Without the projection, which :class:`named_arrays.AbstractWcsVector`
        # does not apply, see the test of the same name for the spectrograph.
        wcs.wcs.ctype = ["HPLN", "HPLT"]
        wcs.wcs.cunit = ["arcsec", "arcsec"]
        wcs.wcs.crpix = [header["CRPIX1"], header["CRPIX2"]]
        wcs.wcs.cdelt = [header["CDELT1"], header["CDELT2"]]
        wcs.wcs.crval = [row[header_aux["XCENIX"]], row[header_aux["YCENIX"]]]
        wcs.wcs.pc = [
            [row[header_aux["PC1_1IX"]], row[header_aux["PC1_2IX"]]],
            [row[header_aux["PC2_1IX"]], row[header_aux["PC2_2IX"]]],
        ]

        for corner in ((0, 0), (1, 2), (0, -1), (-1, -1)):
            index = {
                axis_x: corner[0] % shape[axis_x],
                axis_y: corner[1] % shape[axis_y],
            }

            # Astropy counts pixels from zero here, and the vertex of index `j`
            # lies half a pixel below the center of pixel `j`.
            pixel = [[index[axis_x] - 0.5, index[axis_y] - 0.5]]
            expected = wcs.wcs_pix2world(pixel, 0)[0] * u.deg

            assert np.isclose(position.x[index].ndarray, expected[0], rtol=1e-10)
            assert np.isclose(position.y[index].ndarray, expected[1], rtol=1e-10)


def test_from_fits_time_range():
    path = _path()

    every = iris.sji.SlitJawObservation.from_fits(path)
    time = every.inputs.time.ndarray

    result = iris.sji.SlitJawObservation.from_fits(
        path=path,
        time_start=time[2],
        time_stop=time[5],
    )

    index = {every.axis_time: slice(2, 5)}
    assert np.all(result.inputs.time.ndarray == time[2:5])
    assert np.all(result.timedelta == every.timedelta[index])
    assert np.all(
        (result.outputs == every.outputs[index])
        | (np.isnan(result.outputs) & np.isnan(every.outputs[index]))
    )


def test_from_fits_concatenate(tmp_path: pathlib.Path):
    a = _write(tmp_path / "a.fits", num_time=3, num_x=40, num_y=30)
    b = _write(tmp_path / "b.fits", num_time=2, num_x=20, num_y=50)

    result = iris.sji.SlitJawObservation.from_fits([a, b])

    axis_t = result.axis_time
    axis_x = result.axis_detector_x
    axis_y = result.axis_detector_y

    assert result.shape == {axis_t: 5, axis_y: 50, axis_x: 40}
    assert result.inputs.shape_wcs == {axis_x: 41, axis_y: 51}

    pad_a = {axis_t: slice(None, 3), axis_y: slice(30, None)}
    pad_b = {axis_t: slice(3, None), axis_x: slice(20, None)}
    assert np.all(np.isnan(result.outputs[pad_a]))
    assert np.all(np.isnan(result.outputs[pad_b]))

    image_a = iris.sji.SlitJawObservation.from_fits(a).outputs
    image_b = iris.sji.SlitJawObservation.from_fits(b).outputs
    for image, index in [
        (image_a, {axis_t: slice(None, 3), axis_y: slice(None, 30)}),
        (image_b, {axis_t: slice(3, None), axis_x: slice(None, 20)}),
    ]:
        image_result = result.outputs[index]
        assert np.all(
            (image_result == image) | (np.isnan(image_result) & np.isnan(image))
        )


def test_from_fits_different_channels(tmp_path: pathlib.Path):
    a = _write(tmp_path / "a.fits", num_time=1, num_x=10, num_y=10)
    b = _write(tmp_path / "b.fits", num_time=1, num_x=10, num_y=10, window="SJI_2796")

    with pytest.raises(ValueError, match="different channels"):
        iris.sji.SlitJawObservation.from_fits([a, b])


def test_from_fits_no_frames():
    with pytest.raises(ValueError, match="No frames"):
        iris.sji.SlitJawObservation.from_fits(
            path=_path(),
            time_start="2021-09-23T05:00",
            time_stop="2021-09-23T06:00",
        )


def test_from_fits_unknown_pointing(tmp_path: pathlib.Path):
    path = _write(tmp_path / "a.fits", num_time=3, num_x=10, num_y=10, unknown=1)

    result = iris.sji.SlitJawObservation.from_fits(path)

    with astropy.io.fits.open(path) as hdul:
        aux = hdul[1].data
        header_aux = hdul[1].header

    inputs = result.inputs
    actual = {
        "XCENIX": inputs.crval.position.x.ndarray.to_value(u.arcsec),
        "YCENIX": inputs.crval.position.y.ndarray.to_value(u.arcsec),
        "PC1_1IX": inputs.pc.position.x.components[result.axis_detector_x].ndarray,
        "PC1_2IX": inputs.pc.position.x.components[result.axis_detector_y].ndarray,
        "PC2_1IX": inputs.pc.position.y.components[result.axis_detector_x].ndarray,
        "PC2_2IX": inputs.pc.position.y.components[result.axis_detector_y].ndarray,
    }

    for key in _columns_pointing:
        column = aux[:, header_aux[key]]
        assert column[1] == 0
        assert actual[key][0] == column[0]
        assert np.isclose(actual[key][1], (column[0] + column[2]) / 2, rtol=1e-12)
        assert actual[key][2] == column[2]


def test_from_fits_unknown_pointing_every_frame(tmp_path: pathlib.Path):
    path = _write(
        tmp_path / "a.fits", num_time=3, num_x=10, num_y=10, unknown=slice(None)
    )

    with pytest.raises(ValueError, match=r"XCENIX of .*a\.fits is zero"):
        iris.sji.SlitJawObservation.from_fits(path)


def test_timedelta_default():
    """Each instance gets its own default exposure time."""
    a = iris.sji.SlitJawObservation(inputs=na.ScalarArray(0), outputs=na.ScalarArray(0))
    b = iris.sji.SlitJawObservation(inputs=na.ScalarArray(0), outputs=na.ScalarArray(0))
    assert a.timedelta == 0 * u.s
    assert a.timedelta is not b.timedelta


def test_from_time_range_no_window():
    with pytest.raises(ValueError, match="No SJI_1330 observations"):
        iris.sji.SlitJawObservation.from_time_range(
            time_start="2019-09-30T18:00",
            time_stop="2019-09-30T18:01",
            window="SJI_1330",
        )


@pytest.mark.parametrize(
    argnames="a,expected",
    argvalues=[
        (np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([1.0, 0.0, 3.0]), np.array([1.0, 2.0, 3.0])),
        (np.array([0.0, 0.0, 3.0, 5.0]), np.array([3.0, 3.0, 3.0, 5.0])),
    ],
)
def test_interpolate_zeros(a: np.ndarray, expected: np.ndarray):
    assert np.all(_interpolate_zeros(a) == expected)


def test_interpolate_zeros_every_value():
    with pytest.raises(ValueError, match="Every value"):
        _interpolate_zeros(np.zeros(3))


def test_position_units():
    result = iris.sji.SlitJawObservation.from_fits(_path())
    position = result.inputs.position
    assert isinstance(position, na.Cartesian2dVectorArray)
    assert position.x.unit.is_equivalent(u.arcsec)
