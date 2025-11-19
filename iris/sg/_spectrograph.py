from typing import Callable
from typing_extensions import Self
import dataclasses
import pathlib
import IPython.display
import numpy as np
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

    @property
    def velocity_doppler(self):
        """
        The Doppler velocity of each wavelength bin in the observation.
        """
        return self.inputs.wavelength.to(
            unit=u.km / u.s,
            equivalencies=u.doppler_radio(self.wavelength_center),
        )

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

        time_start = astropy.time.Time(header_primary["STARTOBS"])

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

            time = time_start + timedelta_frame
            time = na.ScalarArray(time.jd, axis_detector_x)
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

        w0 = self.wavelength_center
        if np.all(w0[{self.axis_time: 0}] == w0):
            w0 = w0[{self.axis_time: 0}]

        if not w0.shape:
            w0 = w0.ndarray

        self.wavelength_center = w0

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

        inputs = na.ExplicitTemporalWcsSpectralPositionalVectorArray(
            time=na.ScalarArray.zeros(vshape_time),
            crval=na.SpectralPositionalVectorArray(
                wavelength=na.ScalarArray.zeros(shape_base) << u.AA,
                position=na.Cartesian2dVectorArray(
                    x=na.ScalarArray.zeros(shape_base) << u.arcsec,
                    y=na.ScalarArray.zeros(shape_base) << u.arcsec,
                ),
            ),
            crpix=na.CartesianNdVectorArray(
                components=dict(
                    wavelength=na.ScalarArray.zeros(shape_base),
                    detector_x=na.ScalarArray.zeros(shape_base),
                    detector_y=na.ScalarArray.zeros(shape_base),
                )
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
                    components=dict(
                        wavelength=na.ScalarArray.zeros(shape_base),
                        detector_x=na.ScalarArray.zeros(shape_base),
                        detector_y=na.ScalarArray.zeros(shape_base),
                    ),
                ),
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components=dict(
                            wavelength=na.ScalarArray.zeros(shape_base),
                            detector_x=na.ScalarArray.zeros(shape_base),
                            detector_y=na.ScalarArray.zeros(shape_base),
                        ),
                    ),
                    y=na.CartesianNdVectorArray(
                        components=dict(
                            wavelength=na.ScalarArray.zeros(shape_base),
                            detector_x=na.ScalarArray.zeros(shape_base),
                            detector_y=na.ScalarArray.zeros(shape_base),
                        ),
                    ),
                ),
            ),
            shape_wcs=vshape_wcs,
        )

        shape = na.broadcast_shapes(shape_base, shape_wcs)
        outputs = na.ScalarArray.zeros(shape) << u.DN

        timedelta = na.ScalarArray.zeros(shape_time) * u.s

        wavelength_center = na.ScalarArray.zeros(shape_base) << u.AA

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

        wavelength_center = a.wavelength_center

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
                a.velocity_doppler,
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
        wavelength_center = self.wavelength_center

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
                self.velocity_doppler,
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
