from typing import Any, Callable, Mapping, TypeVar, cast
from typing_extensions import Self
import dataclasses
import pathlib
import IPython.display
import numpy as np
import matplotlib.animation
import matplotlib.colors
import matplotlib.axes
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import astropy.units as u
import astropy.constants
import astropy.time
import astropy.wcs
import astropy.io.fits
import astropy.visualization
import named_arrays as na
import iris

__all__ = [
    "SpectrographObservation",
]

_ArrayT = TypeVar("_ArrayT", bound=na.AbstractArray)


def _index(a: _ArrayT, item: Mapping[str, int | slice]) -> _ArrayT:
    """
    Index an array without losing its type.

    :meth:`named_arrays.AbstractArray.__getitem__` is declared to return the
    abstract base class, which hides the components from the type checker.
    """
    return cast(_ArrayT, a[item])


def _scalar(a: Any) -> na.ScalarArray:
    """
    Normalize a quantity or a scalar array to an explicit scalar array.
    """
    if isinstance(a, na.AbstractScalarArray):
        return cast(na.ScalarArray, a.explicit)
    return na.ScalarArray(a)


def _vector(a: Any) -> na.Cartesian2dVectorArray[na.ScalarArray, na.ScalarArray]:
    """
    Normalize a 2D vector to one with explicit scalar array components.
    """
    return na.Cartesian2dVectorArray(x=_scalar(a.x), y=_scalar(a.y))


def _quantity(a: na.ScalarArray) -> u.Quantity:
    """
    The value of a scalar array with no axes, as a quantity.
    """
    return u.Quantity(a.ndarray_aligned(tuple(a.shape)))


def _extent(a: na.ScalarArray) -> tuple[u.Quantity, u.Quantity]:
    """
    The smallest and largest values of a scalar array, as quantities.
    """
    ndarray = a.ndarray_aligned(tuple(a.shape))
    return u.Quantity(ndarray.min()), u.Quantity(ndarray.max())


def _value(a: np.ndarray, unit: u.UnitBase) -> np.ndarray:
    """
    Strip a known unit from an array.

    The unit comes from :func:`named_arrays.unit_normalized`, which answers
    with a dimensionless unit rather than :obj:`None`, so there is no
    unitless case to handle separately.
    """
    return np.asarray(u.Quantity(a).to_value(unit))


def _ratio(a: Any, b: Any) -> float:
    """
    The dimensionless ratio of two quantities, whatever their units.

    Rounding a ratio like ``arcsec / deg`` before converting it would round
    the wrong number, since :mod:`astropy` does not simplify the unit.
    """
    ratio = u.Quantity(a / b).to(u.dimensionless_unscaled)
    return float(np.asarray(ratio))


def _regrid(
    weights: tuple[na.AbstractScalar, dict[str, int], dict[str, int]],
    values: na.AbstractScalarArray,
) -> na.ScalarArray:
    """
    Resample an array using weights from :func:`named_arrays.regridding.weights`.
    """
    result = na.regridding.regrid_from_weights(*weights, values_input=values)
    return cast(na.ScalarArray, result)


@dataclasses.dataclass(eq=False, repr=False)
class SpectrographObservation(
    na.FunctionArray[
        na.ExplicitTemporalWcsDopplerPositionalVectorArray,
        na.AbstractScalarArray,
    ]
):
    """
    A sequence of observations captured by the IRIS spectrograph.

    Examples
    --------

    Load a IRIS raster and plot as an RGB image using the
    :mod:`named_arrays.colorsynth` module.

    .. jupyter-execute::

        import iris

        # Load a 320-step raster
        obs = iris.sg.open("2021-09-23T06:13")

        # Display the first raster as a false-color image
        obs.show();
    """

    timedelta: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The exposure time for each frame in the observation.
    """

    axis_time: str = "time"
    """The logical axis corresponding to changes in time."""

    axis_wavelength: str = "wavelength"
    """The logical axis corresponding to changes in wavelength."""

    axis_detector_x: str = "detector_x"
    """The logical axis corresponding to changes in detector :math:`x`-coordinate."""

    axis_detector_y: str = "detector_y"
    """The logical axis corresponding to changes in detector :math:`y`-coordinate."""

    @classmethod
    def from_time_range(
        cls,
        time_start: None | astropy.time.Time = None,
        time_stop: None | astropy.time.Time = None,
        description: str = "",
        obs_id: None | int = None,
        window: str = "Si IV 1394",
        axis_time: str = "time",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        limit: int = 200,
        nrt: bool = False,
        num_retry: int = 5,
    ) -> "SpectrographObservation":
        """
        Given an OBSID and a time range, automatically download the data and
        construct an instance of :class:`SpectrographObservation`.

        Parameters
        ----------
        time_start
            The start time of the search period. If :obj:`None`, the start of operations,
            2013-07-20 will be used.
        time_stop
            The end time of the search period. If :obj:`None`, the current time will be used.
        description
            The description of the observation. If an empty string, observations with
            any description will be returned.
        obs_id
            The OBSID of the observation, a number which describes the size, cadence,
            etc. of the observation. If :obj:`None`, all OBSIDs will be used.
        window
            The spectral window to load.
        axis_time
            The logical axis corresponding to changes in time.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        limit
            The maximum number of observations returned by the query.
            Note that this is not the same as the number of files since there
            are several files per observation.
        nrt
            Whether to return results with near-real-time (NRT) data.
        num_retry
            The number of times to try to connect to the server.

        """
        kwargs = dict(
            time_start=time_start,
            time_stop=time_stop,
            description=description,
            obs_id=obs_id,
            limit=limit,
            spectrograph=True,
            sji=False,
            deconvolved=False,
            num_retry=num_retry,
        )

        urls = iris.data.urls_hek(
            nrt=False,
            **kwargs,
        )

        if nrt:
            urls += iris.data.urls_hek(
                nrt=True,
                **kwargs,
            )

        archives = iris.data.download(urls)
        fits = iris.data.decompress(archives)
        fits = na.ScalarArray(np.array(fits), axes="time")

        return cls.from_fits(
            path=fits,
            window=window,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    @classmethod
    def from_fits(
        cls,
        path: pathlib.Path | na.ScalarArray[pathlib.Path],
        window: str = "Si IV 1394",
        axis_time: str = "time",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> "SpectrographObservation":
        """
        Given a single FITS file or an array of FITS files with the same OBSID,
        construct a SpectrographObservation object.

        Parameters
        ----------
        path
            A single FITS file or an array of FITS files to load.
        window
            The spectral window to load.
        axis_time
            The logical axis corresponding to changes in time.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        """

        path = na.asarray(path)
        shape_base = path.shape

        hdul_prototype = astropy.io.fits.open(path.ndarray.item(0))

        header_primary = hdul_prototype[0].header
        windows = [
            header_primary[f"TDESC{h}"] if f"TDESC{h}" in header_primary else None
            for h in range(len(hdul_prototype))
        ]

        if window in windows:
            index_window = windows.index(window)
        else:  # pragma: nocover
            raise ValueError(f"{window=} not in {windows=}")

        hdu_prototype = hdul_prototype[index_window]

        wcs_prototype = astropy.wcs.WCS(hdu_prototype)

        axes_wcs = list(reversed(wcs_prototype.axis_type_names))

        iw = axes_wcs.index("WAVE")
        ix = axes_wcs.index("HPLN")
        iy = axes_wcs.index("HPLT")

        axes_wcs[iw] = axis_wavelength
        axes_wcs[ix] = axis_detector_x
        axes_wcs[iy] = axis_detector_y

        shape_wcs = wcs_prototype.array_shape
        shape_wcs = {ax: sz for ax, sz in zip(axes_wcs, shape_wcs)}

        index_max = {
            axis_wavelength: slice(None, shape_wcs[axis_wavelength]),
            axis_detector_x: slice(None, shape_wcs[axis_detector_x]),
            axis_detector_y: slice(None, shape_wcs[axis_detector_y]),
        }

        self = cls.empty(
            shape_base=shape_base,
            shape_wcs=shape_wcs,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

        for index in path.ndindex():
            file = path[index].ndarray

            hdul = astropy.io.fits.open(file)
            hdu = hdul[index_window]
            hdu_aux = hdul[~1]

            detector_type = hdul[0].header[f"TDET{index_window}"]

            key_timedelta = "EXPTIMEF" if "FUV" in detector_type else "EXPTIMEN"
            timedelta = hdu_aux.data[..., hdu_aux.header[key_timedelta]] << u.s
            self.timedelta[index] = na.ScalarArray(timedelta, axis_detector_x)

            outputs_index = na.ScalarArray(
                ndarray=hdu.data << u.DN,
                axes=tuple(shape_wcs),
            )[index_max]

            shape_index = outputs_index.shape

            index_min = {
                axis_wavelength: slice(None, shape_index[axis_wavelength]),
                axis_detector_x: slice(None, shape_index[axis_detector_x]),
                axis_detector_y: slice(None, shape_index[axis_detector_y]),
            }

            self.outputs[index | index_min] = outputs_index

            timedelta_frame = hdu_aux.data[..., hdu_aux.header["Time"]] * u.s
            timedelta_avg = np.diff(timedelta_frame).mean()
            timedelta_last = timedelta_frame[~0] + timedelta_avg
            timedelta_frame = np.append(timedelta_frame, timedelta_last)

            time_start = astropy.time.Time(hdul[0].header["STARTOBS"])

            time = time_start + timedelta_frame
            time = na.ScalarArray(time.jd, axis_detector_x)
            self.inputs.time[index] = time

            wcs = astropy.wcs.WCS(hdu).wcs

            crval = self.inputs.crval
            crval.wavelength[index] = wcs.crval[~iw] << u.m
            crval.position.x[index] = wcs.crval[~ix] << u.deg
            crval.position.y[index] = wcs.crval[~iy] << u.deg

            # One less than the FITS keyword, which counts pixels from one
            # where :class:`named_arrays.AbstractWcsVector` counts them from
            # zero. The formula they are put into is the one in the WCS
            # paper, but the pixel coordinates it is given come from
            # :func:`named_arrays.indices` and so start at zero, and a
            # reference point measured from the other end of the first pixel
            # moves everything by one: a third of an arcsecond across the
            # raster, a sixth along the slit, and, on the wavelength axis,
            # about 3 km/s of Doppler shift.
            crpix = self.inputs.crpix
            crpix.components[axis_wavelength][index] = wcs.crpix[~iw] - 1
            crpix.components[axis_detector_x][index] = wcs.crpix[~ix] - 1
            crpix.components[axis_detector_y][index] = wcs.crpix[~iy] - 1

            cdelt = self.inputs.cdelt
            cdelt.wavelength[index] = wcs.cdelt[~iw] << u.m
            cdelt.position.x[index] = wcs.cdelt[~ix] << u.deg
            cdelt.position.y[index] = wcs.cdelt[~iy] << u.deg

            pc = self.inputs.pc
            pc.wavelength.components[axis_wavelength][index] = wcs.pc[~iw, ~iw]
            pc.wavelength.components[axis_detector_x][index] = wcs.pc[~iw, ~ix]
            pc.wavelength.components[axis_detector_y][index] = wcs.pc[~iw, ~iy]
            pc.position.x.components[axis_wavelength][index] = wcs.pc[~ix, ~iw]
            pc.position.x.components[axis_detector_x][index] = wcs.pc[~ix, ~ix]
            pc.position.x.components[axis_detector_y][index] = wcs.pc[~ix, ~iy]
            pc.position.y.components[axis_wavelength][index] = wcs.pc[~iy, ~iw]
            pc.position.y.components[axis_detector_x][index] = wcs.pc[~iy, ~ix]
            pc.position.y.components[axis_detector_y][index] = wcs.pc[~iy, ~iy]

            key_center = f"TWAVE{index_window}"
            self.inputs.wavelength_rest[index] = hdul[0].header[key_center] * u.AA

        t = astropy.time.Time(
            val=self.inputs.time.ndarray,
            format="jd",
        )
        t.format = "isot"
        self.inputs.time.ndarray = t

        where_invalid = self.outputs == -200 * u.DN
        self.outputs[where_invalid] = np.nan

        w0 = self.inputs.wavelength_rest
        if np.all(w0[{self.axis_time: 0}] == w0):
            w0 = w0[{self.axis_time: 0}]

        if not w0.shape:
            w0 = w0.ndarray

        self.inputs.wavelength_rest = w0

        return self

    @classmethod
    def empty(
        cls,
        shape_base: dict[str, int],
        shape_wcs: dict[str, int],
        axis_time: str = "time",
        axis_wavelength: str = "wavelength",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
    ) -> "SpectrographObservation":
        """
        Create an empty SpectrographObservation object.

        Parameters
        ----------
        shape_base
            The shape of the result excluding the axes handled by WCS.
        shape_wcs
            The shape of the axes handled by WCS.
        axis_time
            The logical axis corresponding to changes in time.
        axis_wavelength
            The logical axis corresponding to changes in wavelength.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        """

        vshape_wcs = {a: shape_wcs[a] + 1 for a in shape_wcs}

        shape_time = shape_base | {axis_detector_x: shape_wcs[axis_detector_x]}
        vshape_time = shape_base | {axis_detector_x: vshape_wcs[axis_detector_x]}

        inputs = na.ExplicitTemporalWcsDopplerPositionalVectorArray(
            time=na.ScalarArray.zeros(vshape_time),
            wavelength_rest=na.ScalarArray.zeros(shape_base) << u.AA,
            crval=na.SpectralPositionalVectorArray(
                wavelength=na.ScalarArray.zeros(shape_base) << u.AA,
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.zeros(shape_base) << u.arcsec,
                    y=na.ScalarArray.zeros(shape_base) << u.arcsec,
                ),
            ),
            crpix=na.CartesianNdVectorArray(
                components={
                    axis_wavelength: na.ScalarArray.zeros(shape_base),
                    axis_detector_x: na.ScalarArray.zeros(shape_base),
                    axis_detector_y: na.ScalarArray.zeros(shape_base),
                }
            ),
            cdelt=na.SpectralPositionalVectorArray(
                wavelength=na.ScalarArray.zeros(shape_base) << u.AA,
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.zeros(shape_base) << u.arcsec,
                    y=na.ScalarArray.zeros(shape_base) << u.arcsec,
                ),
            ),
            pc=na.SpectralPositionalMatrixArray(
                wavelength=na.CartesianNdVectorArray(
                    components={
                        axis_wavelength: na.ScalarArray.zeros(shape_base),
                        axis_detector_x: na.ScalarArray.zeros(shape_base),
                        axis_detector_y: na.ScalarArray.zeros(shape_base),
                    },
                ),
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components={
                            axis_wavelength: na.ScalarArray.zeros(shape_base),
                            axis_detector_x: na.ScalarArray.zeros(shape_base),
                            axis_detector_y: na.ScalarArray.zeros(shape_base),
                        },
                    ),
                    y=na.CartesianNdVectorArray(
                        components={
                            axis_wavelength: na.ScalarArray.zeros(shape_base),
                            axis_detector_x: na.ScalarArray.zeros(shape_base),
                            axis_detector_y: na.ScalarArray.zeros(shape_base),
                        },
                    ),
                ),
            ),
            shape_wcs=vshape_wcs,
        )

        shape = na.broadcast_shapes(shape_base, shape_wcs)
        outputs = na.ScalarArray.zeros(shape) << u.DN

        timedelta = na.ScalarArray.zeros(shape_time) * u.s

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=timedelta,
            axis_time=axis_time,
            axis_wavelength=axis_wavelength,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    @property
    def radiance(self) -> Self:
        """
        Convert to radiometric units using :func:`iris.sg.effective_area`.
        """

        time = self.inputs.time.ndarray.mean()
        wavelength = self.inputs.wavelength

        lower = {self.axis_wavelength: slice(None, ~0)}
        upper = {self.axis_wavelength: slice(+1, None)}
        wavelength = (wavelength[lower] + wavelength[upper]) / 2

        energy = astropy.constants.h * astropy.constants.c / wavelength / u.ph

        gain = iris.sg.dn_to_photons(wavelength)

        area_eff = iris.sg.effective_area(time, wavelength)

        pix_xy = np.diff(self.inputs.position, axis=self.axis_detector_y).length
        pix_xy = pix_xy.mean(self.axis_detector_x)

        pix_lambda = np.diff(self.inputs.wavelength, axis=self.axis_wavelength)

        t_exp = self.timedelta

        w_slit = iris.sg.width_slit

        factor = energy * gain / (area_eff * pix_xy * pix_lambda * t_exp * w_slit)

        outputs = self.outputs * factor

        outputs = outputs.to(u.erg / (u.cm**2 * u.sr * u.nm * u.s))

        return dataclasses.replace(
            self,
            outputs=outputs,
        )

    def mosaic(
        self,
        cdelt: None | u.Quantity | na.AbstractCartesian2dVectorArray = None,
    ) -> Self:
        """
        Assemble the rasters along the time axis into a single mosaic.

        Each raster (a tile of the mosaic) is resampled onto a common grid
        using :func:`named_arrays.regridding.regrid` with the ``conservative``
        method, first along the wavelength axis, onto the wavelength grid
        of the first tile, and then along the two spatial axes, onto an
        axis-aligned helioprojective grid which covers every tile.
        Each pixel of the result is the coverage-weighted mean of the tile
        pixels overlapping it, so tiles which overlap are averaged,
        and pixels which no tile covers are NaN.

        The tiles are placed at the helioprojective coordinates recorded in
        their headers, no correction for solar rotation is applied.
        The time of each vertex of the result is the mean time of the tile
        pixels within half a pixel of it, and is masked where there are none.

        Parameters
        ----------
        cdelt
            The plate scale of the mosaic.
            If :obj:`None`, the plate scale of the first tile is used.
            If a scalar, the same plate scale is used for both spatial axes.

        Examples
        --------

        Assemble a sequence of five rasters, whose pointing drifts by about
        an arcsecond between the first and the last, into a single image.

        .. jupyter-execute::

            import iris

            # Load a sequence of rasters
            tiles = iris.sg.open(
                time="2017-02-11T04:50",
                time_stop="2017-02-11T05:00",
            )

            # Assemble the rasters into a single image
            mosaic = tiles.mosaic()

            # Display the mosaic as a false-color image
            mosaic.show();

        A full-Sun mosaic is assembled the same way, by asking for the
        OBSID which took it. That is tens of gigabytes of raster, so it is
        not run here:

        .. code-block:: python

            tiles = iris.sg.open(
                time="2026-09-02T04:45",
                time_stop="2026-09-03T00:00",
                obs_id=3600108078,
            )
            mosaic = tiles.mosaic()
        """

        axis_time = self.axis_time
        axis_wavelength = self.axis_wavelength
        axis_x = self.axis_detector_x
        axis_y = self.axis_detector_y
        axis_xy = (axis_x, axis_y)
        axes = (axis_x, axis_y, axis_wavelength)

        inputs = self.inputs

        def select(a: _ArrayT, index: int) -> _ArrayT:
            """Pick one tile out of an array, if it varies from tile to tile."""
            if axis_time in a.shape:
                a = _index(a, {axis_time: index})
            return a

        wavelength_rest = _scalar(inputs.wavelength_rest)
        if not (wavelength_rest == select(wavelength_rest, 0)).all():
            raise ValueError("every tile must have the same rest wavelength")
        wavelength_rest = select(wavelength_rest, 0)

        position = _vector(inputs.position)
        wavelength = _scalar(inputs.wavelength)
        outputs = _scalar(self.outputs)
        timedelta = _scalar(self.timedelta)

        # Vertices of a tile with a masked time, which a mosaic has where no
        # tile covered it, are NaN here, and so contribute nothing below.
        time = astropy.time.Time(_scalar(inputs.time).ndarray)
        jd = np.array(time.jd, dtype=float)
        jd[np.asarray(time.mask, dtype=bool)] = np.nan
        jd = na.ScalarArray(jd, axes=inputs.time.axes)

        if cdelt is None:
            cdelt_first = select(inputs.cdelt.position, 0)
            dx = _quantity(_scalar(cdelt_first.x))
            dy = _quantity(_scalar(cdelt_first.y))
        elif isinstance(cdelt, na.AbstractCartesian2dVectorArray):
            dx = _quantity(_scalar(cdelt.x))
            dy = _quantity(_scalar(cdelt.y))
        else:
            dx = dy = u.Quantity(cdelt)

        # The grid of the mosaic is the smallest one, with the requested
        # plate scale, which holds every vertex of every tile.
        x_min, x_max = _extent(position.x)
        y_min, y_max = _extent(position.y)
        # Rounded first, so that a grid which already fits, like that
        # of a mosaic, is not enlarged by a pixel by rounding error.
        num_x = int(np.ceil(round(_ratio(x_max - x_min, dx), 6)))
        num_y = int(np.ceil(round(_ratio(y_max - y_min, dy), 6)))
        x_start = (x_min + x_max) / 2 - dx * num_x / 2
        y_start = (y_min + y_max) / 2 - dy * num_y / 2

        num_wavelength = self.shape[axis_wavelength]

        shape_xy = {axis_x: num_x, axis_y: num_y}
        shape_wcs = shape_xy | {axis_wavelength: num_wavelength}
        vshape_wcs = {a: shape_wcs[a] + 1 for a in shape_wcs}
        vshape_xy = {a: shape_xy[a] + 1 for a in shape_xy}

        # The reference pixel is the first one, and `crval` is its center,
        # which lies half a pixel inside the first vertex.
        inputs_result = na.ExplicitTemporalWcsDopplerPositionalVectorArray(
            time=na.ScalarArray.zeros(vshape_xy),
            wavelength_rest=wavelength_rest,
            crval=na.SpectralPositionalVectorArray(
                wavelength=select(inputs.crval.wavelength, 0),
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray(x_start + dx / 2),
                    y=na.ScalarArray(y_start + dy / 2),
                ),
            ),
            crpix=na.CartesianNdVectorArray(
                components={
                    axis_wavelength: select(
                        inputs.crpix.components[axis_wavelength], 0
                    ),
                    axis_x: na.ScalarArray(0),
                    axis_y: na.ScalarArray(0),
                }
            ),
            cdelt=na.SpectralPositionalVectorArray(
                wavelength=select(inputs.cdelt.wavelength, 0),
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray(dx),
                    y=na.ScalarArray(dy),
                ),
            ),
            pc=na.SpectralPositionalMatrixArray(
                wavelength=na.CartesianNdVectorArray(
                    components={
                        axis_wavelength: na.ScalarArray(1),
                        axis_x: na.ScalarArray(0),
                        axis_y: na.ScalarArray(0),
                    },
                ),
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components={
                            axis_wavelength: na.ScalarArray(0),
                            axis_x: na.ScalarArray(1),
                            axis_y: na.ScalarArray(0),
                        },
                    ),
                    y=na.CartesianNdVectorArray(
                        components={
                            axis_wavelength: na.ScalarArray(0),
                            axis_x: na.ScalarArray(0),
                            axis_y: na.ScalarArray(1),
                        },
                    ),
                ),
            ),
            shape_wcs=vshape_wcs,
        )

        wavelength_result = _scalar(inputs_result.wavelength)
        position_result = _vector(inputs_result.position)

        # A second spatial grid whose cells are centered on the vertices of
        # the mosaic, onto which the times of the tiles are resampled.
        position_vertex = na.Cartesian2dVectorArray(
            x=na.ScalarArray(
                x_start + dx * (np.arange(num_x + 2) - 0.5), axes=(axis_x,)
            ),
            y=na.ScalarArray(
                y_start + dy * (np.arange(num_y + 2) - 0.5), axes=(axis_y,)
            ),
        )

        # The sums of the resampled values and of the resampled coverage,
        # whose ratio is the coverage-weighted mean.
        unit = na.unit_normalized(outputs)
        unit_timedelta = na.unit_normalized(timedelta)

        shape = tuple(shape_wcs[a] for a in axes)
        num_outputs = np.zeros(shape)
        den_outputs = np.zeros(shape)

        num_timedelta = np.zeros((num_x, num_y))
        den_timedelta = np.zeros((num_x, num_y))

        num_time = np.zeros((num_x + 1, num_y + 1))
        den_time = np.zeros((num_x + 1, num_y + 1))

        num_tiles = self.shape.get(axis_time, 1)

        for i in range(num_tiles):

            # NaN pixels of the tile contribute neither to the sum of the
            # values nor to the coverage, so that they are averaged out
            # rather than spread by the resampling.
            values = _value(select(outputs, i).ndarray_aligned(axes), unit)
            where = np.isfinite(values)
            values = np.where(where, values, 0)
            coverage = where.astype(float)

            weights_wavelength = na.regridding.weights(
                coordinates_input=select(wavelength, i),
                coordinates_output=wavelength_result,
                axis_input=axis_wavelength,
                axis_output=axis_wavelength,
                method="conservative",
            )
            values_tile = _regrid(
                weights=weights_wavelength,
                values=na.ScalarArray(values, axes=axes),
            )
            coverage_tile = _regrid(
                weights=weights_wavelength,
                values=na.ScalarArray(coverage, axes=axes),
            )

            # Only the part of the mosaic which the tile can touch is
            # resampled onto, with a margin of one pixel on every side.
            position_tile = select(position, i)
            x_min_tile, x_max_tile = _extent(position_tile.x)
            y_min_tile, y_max_tile = _extent(position_tile.y)
            ix0 = int(np.floor(_ratio(x_min_tile - x_start, dx)))
            ix1 = int(np.ceil(_ratio(x_max_tile - x_start, dx)))
            iy0 = int(np.floor(_ratio(y_min_tile - y_start, dy)))
            iy1 = int(np.ceil(_ratio(y_max_tile - y_start, dy)))
            ix0 = max(ix0 - 1, 0)
            iy0 = max(iy0 - 1, 0)
            ix1 = min(ix1 + 1, num_x)
            iy1 = min(iy1 + 1, num_y)

            slice_cell = (slice(ix0, ix1), slice(iy0, iy1))
            slice_vertex = (slice(ix0, ix1 + 1), slice(iy0, iy1 + 1))
            index_vertex = {
                axis_x: slice(ix0, ix1 + 1),
                axis_y: slice(iy0, iy1 + 1),
            }
            index_vertex_cell = {
                axis_x: slice(ix0, ix1 + 2),
                axis_y: slice(iy0, iy1 + 2),
            }

            weights_xy = na.regridding.weights(
                coordinates_input=position_tile,
                coordinates_output=_index(position_result, index_vertex),
                axis_input=axis_xy,
                axis_output=axis_xy,
                method="conservative",
            )
            num_outputs[slice_cell] += _regrid(
                weights=weights_xy,
                values=values_tile,
            ).ndarray_aligned(axes)
            den_outputs[slice_cell] += _regrid(
                weights=weights_xy,
                values=coverage_tile,
            ).ndarray_aligned(axes)

            shape_tile = tuple(position_tile.shape[a] - 1 for a in axis_xy)
            ones_tile = na.ScalarArray.ones(dict(zip(axis_xy, shape_tile)))
            coverage_xy = _regrid(
                weights=weights_xy,
                values=ones_tile,
            ).ndarray_aligned(axis_xy)

            timedelta_tile = select(timedelta, i).ndarray_aligned(axis_xy)
            timedelta_tile = _value(timedelta_tile, unit_timedelta)
            timedelta_tile = np.broadcast_to(timedelta_tile, shape_tile)
            num_timedelta[slice_cell] += _regrid(
                weights=weights_xy,
                values=na.ScalarArray(timedelta_tile, axes=axis_xy),
            ).ndarray_aligned(axis_xy)
            den_timedelta[slice_cell] += coverage_xy

            # The time of a pixel of the tile is the mean of the times of
            # the vertices around it, along whichever axes it varies.
            jd_tile = select(jd, i).ndarray_aligned(axis_xy)
            if jd_tile.shape[0] > 1:
                jd_tile = (jd_tile[:-1] + jd_tile[1:]) / 2
            if jd_tile.shape[1] > 1:
                jd_tile = (jd_tile[:, :-1] + jd_tile[:, 1:]) / 2
            jd_tile = np.broadcast_to(jd_tile, shape_tile)
            where_jd = np.isfinite(jd_tile)
            jd_tile = np.where(where_jd, jd_tile, 0)

            weights_vertex = na.regridding.weights(
                coordinates_input=position_tile,
                coordinates_output=_index(position_vertex, index_vertex_cell),
                axis_input=axis_xy,
                axis_output=axis_xy,
                method="conservative",
            )
            num_time[slice_vertex] += _regrid(
                weights=weights_vertex,
                values=na.ScalarArray(jd_tile, axes=axis_xy),
            ).ndarray_aligned(axis_xy)
            den_time[slice_vertex] += _regrid(
                weights=weights_vertex,
                values=na.ScalarArray(where_jd.astype(float), axes=axis_xy),
            ).ndarray_aligned(axis_xy)

        with np.errstate(divide="ignore", invalid="ignore"):
            outputs_result = num_outputs / den_outputs
            timedelta_result = num_timedelta / den_timedelta
            jd_result = num_time / den_time

        outputs_result[den_outputs == 0] = np.nan
        timedelta_result[den_timedelta == 0] = np.nan

        outputs_result = u.Quantity(outputs_result, unit)
        timedelta_result = u.Quantity(timedelta_result, unit_timedelta)

        # Vertices which no tile covers have no time, and are masked.
        # The value underneath the mask is the mean time of the rest,
        # since :class:`astropy.time.Time` insists on a finite one.
        where_time = den_time != 0
        jd_fill = jd_result[where_time].mean() if where_time.any() else 0.0
        jd_result = np.where(where_time, jd_result, jd_fill)
        time_result = astropy.time.Time(
            val=np.ma.array(jd_result, mask=~where_time),
            format="jd",
        )
        time_result.format = "isot"
        inputs_result.time = na.ScalarArray(
            ndarray=cast(np.ndarray, time_result),
            axes=axis_xy,
        )

        return dataclasses.replace(
            self,
            inputs=inputs_result,
            outputs=na.ScalarArray(outputs_result, axes=axes),
            timedelta=na.ScalarArray(timedelta_result, axes=axis_xy),
        )

    def show(
        self,
        index_time: int = 0,
        ax: plt.Axes = None,
        cax: plt.Axes = None,
        norm: None | Callable = None,
        vmin: None | float | u.Quantity | na.AbstractScalar = None,
        vmax: None | float | u.Quantity | na.AbstractScalar = None,
        velocity_min: u.Quantity = -100 * u.km / u.s,
        velocity_max: u.Quantity = +100 * u.km / u.s,
        cbar_fraction: float = 0.1,
    ) -> plt.Axes:
        """
        Display a single raster of this dataset as a false-color image.

        Parameters
        ----------
        index_time
            The index along the time axis to show.
        ax
            The :mod:`matplotlib` axes on which to plot the image.
            If :obj:`None`, a new figure is created.
        cax
            The axes on which to plot the colorbar.
            If :obj:`None`, space is stolen from `ax` to create a new set of axes.
        norm
            The normalization method used to scale data into the range [0, 1] before
            mapping to colors.
        vmin
            The minimum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        vmax
            The maximum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        velocity_min
            The minimum Doppler velocity of the data range.
        velocity_max
            The maximum Doppler velocity of the data range.
        cbar_fraction
            The fraction of the space to use for the colorbar axes if `cax`
            is :obj:`None`.
        """
        a = self

        if self.axis_time in self.shape:
            a = a[{self.axis_time: index_time}]

        wavelength_center = na.as_named_array(a.inputs.wavelength_rest).ndarray

        axis_wavelength = self.axis_wavelength
        axis_x = self.axis_detector_x
        axis_y = self.axis_detector_y

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(8, 8),
                constrained_layout=True,
            )

        if cax is None:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes(
                position="right",
                size=f"{cbar_fraction * 100}%",
                pad=1,
            )

        if vmin is None:
            vmin = 0

        if vmax is None:
            vmax = np.nanpercentile(
                a=a.outputs,
                q=99.5,
                axis=(axis_x, axis_y),
            )

        with astropy.visualization.quantity_support():
            cax_twin = cax.twinx()
            colorbar = na.plt.rgbmesh(
                a.inputs.velocity,
                a.inputs.position.x,
                a.inputs.position.y,
                C=a.outputs,
                axis_wavelength=axis_wavelength,
                ax=ax,
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                wavelength_min=velocity_min,
                wavelength_max=velocity_max,
            )
            na.plt.pcolormesh(
                colorbar.inputs.x,
                colorbar.inputs.y.to(
                    u.AA,
                    equivalencies=u.doppler_radio(wavelength_center),
                ),
                C=colorbar.outputs,
                axis_rgb=axis_wavelength,
                ax=cax,
            )
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb=axis_wavelength,
                ax=cax_twin,
            )

            ax.set_title(a.inputs.time.ndarray.mean())
            ax.set_aspect("equal")
            ax.set_xlabel(f"helioprojective $x$ ({ax.get_xlabel()})")
            ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")
            cax.set_ylim(
                velocity_min.to(u.AA, equivalencies=u.doppler_radio(wavelength_center)),
                velocity_max.to(u.AA, equivalencies=u.doppler_radio(wavelength_center)),
            )
            cax_twin.set_ylim(velocity_min, velocity_max)

        return ax

    def _animate(
        self,
        norm: None | Callable = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        velocity_min: u.Quantity = -100 * u.km / u.s,
        velocity_max: u.Quantity = +100 * u.km / u.s,
        cbar_fraction: float = 0.1,
    ) -> matplotlib.animation.FuncAnimation:
        """
        Create an animation using the frames in this dataset.

        Parameters
        ----------
        norm
            The normalization method used to scale data into the range [0, 1] before
            mapping to colors.
        vmin
            The minimum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        vmax
            The maximum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        velocity_min
            The minimum Doppler velocity of the data range.
        velocity_max
            The maximum Doppler velocity of the data range.
        cbar_fraction
            The fraction of the space to use for the colorbar axes.
        """
        wavelength_center = self.inputs.wavelength_rest

        axis_time = self.axis_time
        axis_wavelength = self.axis_wavelength
        axis_x = self.axis_detector_x
        axis_y = self.axis_detector_y

        if vmin is None:
            vmin = 0

        if vmax is None:
            vmax = np.nanpercentile(
                a=self.outputs,
                q=99.5,
                axis=(axis_time, axis_x, axis_y),
            )

        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                figsize=(6, 6),
                gridspec_kw=dict(width_ratios=[1 - cbar_fraction, cbar_fraction]),
                constrained_layout=True,
                dpi=200,
            )
            ax[1].xaxis.set_ticks_position("top")
            ax[1].xaxis.set_label_position("top")
            ax[1].ticklabel_format(useOffset=False)
            ax2 = ax[1].twinx()
            x = self.inputs.position.x
            y = self.inputs.position.y
            ani, colorbar = na.plt.rgbmovie(
                self.inputs.time.mean(axis_x),
                self.inputs.velocity,
                x,
                y,
                C=self.outputs,
                axis_time=axis_time,
                axis_wavelength=axis_wavelength,
                ax=ax[0],
                vmin=vmin,
                vmax=vmax,
                norm=norm,
                wavelength_min=velocity_min,
                wavelength_max=velocity_max,
            )
            if axis_time in colorbar.shape:
                colorbar = colorbar[{axis_time: 0}]
            na.plt.pcolormesh(
                colorbar.inputs.x,
                colorbar.inputs.y.to(
                    u.AA,
                    equivalencies=u.doppler_radio(wavelength_center),
                ),
                C=colorbar.outputs,
                axis_rgb=axis_wavelength,
                ax=ax[1],
            )
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb=axis_wavelength,
                ax=ax2,
            )

            ax[0].set_aspect("equal")
            ax[0].set_xlabel(f"helioprojective $x$ ({x.unit:latex_inline})")
            ax[0].set_ylabel(f"helioprojective $y$ ({y.unit:latex_inline})")
            ax[1].set_ylim(
                velocity_min.to(u.AA, equivalencies=u.doppler_radio(wavelength_center)),
                velocity_max.to(u.AA, equivalencies=u.doppler_radio(wavelength_center)),
            )
            ax2.set_ylim(velocity_min, velocity_max)

            return ani

    def to_jshtml(
        self,
        norm: None | str | matplotlib.colors.Normalize = None,
        vmin: None | na.ArrayLike = None,
        vmax: None | na.ArrayLike = None,
        velocity_min: u.Quantity = -100 * u.km / u.s,
        velocity_max: u.Quantity = +100 * u.km / u.s,
        cbar_fraction: float = 0.1,
        fps: None | float = None,
    ) -> IPython.display.HTML:
        """
        Create a Javascript animation of this observation.

        Parameters
        ----------
        norm
            The normalization method used to scale data into the range [0, 1] before
            mapping to colors.
        vmin
            The minimum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        vmax
            The maximum value of the data range.
            If `norm` is :obj:`None`, this parameter will be ignored.
        velocity_min
            The minimum Doppler velocity of the data range.
        velocity_max
            The maximum Doppler velocity of the data range.
        cbar_fraction
            The fraction of the space to use for the colorbar axes.
        fps
            The frames per second of the animation.
        """
        ani = self._animate(
            norm=norm,
            vmin=vmin,
            vmax=vmax,
            velocity_min=velocity_min,
            velocity_max=velocity_max,
            cbar_fraction=cbar_fraction,
        )

        result = ani.to_jshtml(fps=fps)
        result = IPython.display.HTML(result)

        plt.close(ani._fig)

        return result
