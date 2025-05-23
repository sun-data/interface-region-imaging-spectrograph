from typing_extensions import Self
import dataclasses
import pathlib
import numpy as np
import astropy.units as u
import astropy.constants
import astropy.time
import astropy.wcs
import astropy.io.fits
import named_arrays as na
import iris

__all__ = [
    "SpectrographObservation",
]


@dataclasses.dataclass(eq=False, repr=False)
class SpectrographObservation(
    na.FunctionArray[
        na.AbstractTemporalSpectralPositionalVectorArray,
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

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.time
        import astropy.visualization
        import named_arrays as na
        import iris

        # Load a 320-step raster
        obs = iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T07:00"),
        )

        # Calculate the mean rest wavelength of the
        # brightest spectral line
        wavelength_center = obs.wavelength_center.mean().ndarray

        # Define the wavelength range that will be colorized
        wavelength_min = wavelength_center - 0.5 * u.AA
        wavelength_max = wavelength_center + 0.5 * u.AA

        # Define the spectral normalization curve
        vmax = np.nanpercentile(
            a=obs.outputs,
            q=99,
            axis=(obs.axis_time, obs.axis_detector_x, obs.axis_detector_y),
        )

        # Isolate the first raster of the observation
        index = {obs.axis_time: 0}

        # Plot the result as an RGB image
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                figsize=(6, 6),
                gridspec_kw=dict(width_ratios=[.9,.1]),
                constrained_layout=True,
            )
            colorbar = na.plt.rgbmesh(
                obs.inputs[index],
                C=obs.outputs[index],
                axis_wavelength=obs.axis_wavelength,
                ax=ax[0],
                vmin=0 * u.DN,
                vmax=vmax,
                wavelength_min=wavelength_min,
                wavelength_max=wavelength_max,
            )
            na.plt.pcolormesh(
                C=colorbar,
                axis_rgb=obs.axis_wavelength,
                ax=ax[1],
            )
            ax[0].set_aspect("equal")
            ax[0].set_xlabel(f"helioprojective $x$ ({ax[0].get_xlabel()})")
            ax[0].set_ylabel(f"helioprojective $y$ ({ax[0].get_ylabel()})")
            ax[0].set_title(obs.inputs.time[index].ndarray)
            ax[1].yaxis.tick_right()
            ax[1].yaxis.set_label_position("right")
            ax[1].set_ylim(wavelength_min, wavelength_max)
    """

    timedelta: u.Quantity | na.AbstractScalar = 0 * u.s
    """
    The exposure time for each frame in the observation.
    """

    wavelength_center: None | u.Quantity | na.AbstractScalar = None
    """
    TThe rest wavelength of the dominant spectral line in the observation.
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
        urls = iris.data.urls_hek(
            time_start=time_start,
            time_stop=time_stop,
            description=description,
            obs_id=obs_id,
            limit=limit,
            nrt=nrt,
            spectrograph=True,
            sji=False,
            deconvolved=False,
            num_retry=num_retry,
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

            self.outputs[index] = na.ScalarArray(
                ndarray=hdu.data << u.DN,
                axes=tuple(shape_wcs),
            )[index_max]

            time = astropy.time.Time(hdul[0].header["DATE_OBS"]).jd
            self.inputs.time[index] = time

            wcs = astropy.wcs.WCS(hdu).wcs

            crval = self.inputs.crval
            crval.wavelength[index] = wcs.crval[~iw] << u.m
            crval.position.x[index] = wcs.crval[~ix] << u.deg
            crval.position.y[index] = wcs.crval[~iy] << u.deg

            crpix = self.inputs.crpix
            crpix.components[axis_wavelength][index] = wcs.crpix[~iw]
            crpix.components[axis_detector_x][index] = wcs.crpix[~ix]
            crpix.components[axis_detector_y][index] = wcs.crpix[~iy]

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
            self.wavelength_center[index] = hdul[0].header[key_center] * u.AA

        t = astropy.time.Time(
            val=self.inputs.time.ndarray,
            format="jd",
        )
        t.format = "isot"
        self.inputs.time.ndarray = t

        where_invalid = self.outputs == -200 * u.DN
        self.outputs[where_invalid] = np.nan

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

        inputs = na.ExplicitTemporalWcsSpectralPositionalVectorArray(
            time=na.ScalarArray.zeros(shape_base),
            crval=na.SpectralPositionalVectorArray(
                wavelength=na.ScalarArray.empty(shape_base) << u.AA,
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.empty(shape_base) << u.arcsec,
                    y=na.ScalarArray.empty(shape_base) << u.arcsec,
                ),
            ),
            crpix=na.CartesianNdVectorArray(
                components=dict(
                    wavelength=na.ScalarArray.empty(shape_base),
                    detector_x=na.ScalarArray.empty(shape_base),
                    detector_y=na.ScalarArray.empty(shape_base),
                )
            ),
            cdelt=na.SpectralPositionalVectorArray(
                wavelength=na.ScalarArray.empty(shape_base) << u.AA,
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.empty(shape_base) << u.arcsec,
                    y=na.ScalarArray.empty(shape_base) << u.arcsec,
                ),
            ),
            pc=na.SpectralPositionalMatrixArray(
                wavelength=na.CartesianNdVectorArray(
                    components=dict(
                        wavelength=na.ScalarArray.empty(shape_base),
                        detector_x=na.ScalarArray.empty(shape_base),
                        detector_y=na.ScalarArray.empty(shape_base),
                    ),
                ),
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components=dict(
                            wavelength=na.ScalarArray.empty(shape_base),
                            detector_x=na.ScalarArray.empty(shape_base),
                            detector_y=na.ScalarArray.empty(shape_base),
                        ),
                    ),
                    y=na.CartesianNdVectorArray(
                        components=dict(
                            wavelength=na.ScalarArray.empty(shape_base),
                            detector_x=na.ScalarArray.empty(shape_base),
                            detector_y=na.ScalarArray.empty(shape_base),
                        ),
                    ),
                ),
            ),
            shape_wcs={a: shape_wcs[a] + 1 for a in shape_wcs},
        )

        shape = na.broadcast_shapes(shape_base, shape_wcs)
        outputs = na.ScalarArray.empty(shape) << u.DN

        shape_timedelta = shape_base | {axis_detector_x: shape_wcs[axis_detector_x]}
        timedelta = na.ScalarArray.empty(shape_timedelta) * u.s

        wavelength_center = na.ScalarArray.empty(shape_base) << u.AA

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=timedelta,
            wavelength_center=wavelength_center,
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

        wavelength = self.inputs.wavelength

        lower = {self.axis_wavelength: slice(None, ~0)}
        upper = {self.axis_wavelength: slice(+1, None)}
        wavelength = (wavelength[lower] + wavelength[upper]) / 2

        energy = astropy.constants.h * astropy.constants.c / wavelength / u.ph

        gain = iris.sg.gain

        area_eff = iris.sg.effective_area(wavelength)

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
