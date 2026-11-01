from typing import Callable, cast
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
from . import _cleaning

__all__ = [
    "SpectrographObservation",
]


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

        urls = iris.data._prefer_final(urls)

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

    @property
    def variance(self) -> na.AbstractScalar:
        """
        The variance of the noise of each output, in :math:`\\text{DN}^2`.

        It is the read noise of the CCD, :func:`iris.sg.read_noise`, plus the
        photon noise of the signal, which is the variance of the number of
        photons, the output times :func:`iris.sg.dn_to_photons`, converted
        back to DN.
        A negative output, which is noise around a dark current that has
        been subtracted, has only the read noise.

        The outputs must be in DN, as they are when the observation is
        loaded, and each must be a single exposure: an average of several
        has less noise than this.

        Raises
        ------
        ValueError
            If the outputs are not in DN.
        """
        outputs = self.outputs
        if na.unit(outputs) != u.DN:
            raise ValueError(
                f"The variance is computed from outputs in DN, got outputs "
                f"in {na.unit(outputs)}."
            )
        wavelength = self.inputs.wavelength_rest
        signal = np.maximum(outputs, 0 * u.DN)
        return np.square(iris.sg.read_noise(wavelength)) + signal * u.ph / (
            iris.sg.dn_to_photons(wavelength)
        )

    def _data(self) -> na.ScalarArray:
        """
        The values of the outputs in DN, without units, for the tests of
        :meth:`spikes`, :meth:`hot_pixels`, and :meth:`dark_rows`.

        Raises
        ------
        ValueError
            If the outputs are not in DN.
        """
        outputs = self.outputs
        if na.unit(outputs) != u.DN:
            raise ValueError(
                f"The tests for bad pixels need outputs in DN, got outputs "
                f"in {na.unit(outputs)}."
            )
        return cast(na.ScalarArray, outputs.value.explicit)

    @property
    def _axis_exposure(self) -> tuple[str, ...]:
        """
        The axes along which the exposures vary, :attr:`axis_time` and
        :attr:`axis_detector_x`, those of them which the outputs have.
        """
        axis = (self.axis_time, self.axis_detector_x)
        return tuple(a for a in axis if a in self.outputs.shape)

    def spikes(
        self,
        sigclip: float = 4.5,
        objlim: float = 5,
        satlevel: u.Quantity = 16000 * u.DN,
        limit: u.Quantity = 80 * u.DN,
        sigma: float = 4.5,
    ) -> na.ScalarArray:
        r"""
        Find the pixels hit by energetic particles, such as cosmic rays and
        the protons trapped in the South Atlantic Anomaly.

        Two tests find the hits, since each finds hits which the other
        misses:

        - L.A.Cosmic :cite:p:`vanDokkum2001`, on each exposure on its own,
          through :func:`named_arrays.despike_mask`, with the gain,
          :func:`iris.sg.dn_to_photons`, and the read noise,
          :func:`iris.sg.read_noise`, of IRIS.
          It finds a hit by its sharp edges, so it finds faint hits on faint
          Sun.
          On bright Sun it also flags the edges of bright features and the
          thin, bright strands of lines such as :math:`\text{Si\,IV}`, which
          are as sharp as a hit, so its flags are kept only where the Sun is
          fainter than `limit`.
          Without that limit, on the :math:`\text{Si\,IV}` line of a dense
          raster, it flagged 1.7% of the pixels of 320 to 1280 DN, and 4.2%
          of those above, most of them the Sun.
        - The test of ``iris_prep_despike.pro`` by C. Kankelborg in
          SolarSoft, which flags the pixels more than `sigma` standard
          deviations above the mean of the good pixels in the box of
          :math:`5 \times 5 \times 5` pixels around them, along the slit,
          wavelength, and :attr:`axis_detector_x`.
          The pixels it flags are no longer good, and it searches again
          until it finds no more, up to 10 times.
          Since the box spans the neighboring exposures, it finds hits on
          bright Sun, but fewer of them than L.A.Cosmic does on faint Sun.

        Level 2 data are resampled from the pixels of the detector, which
        leaves a negative lobe around each hit, a median of 6% of its peak.
        So the pixels next to a hit which are more than two standard
        deviations of their noise below the Sun around them are hits too.
        The Sun around a pixel is the mean of the middle 5 of the 25 pixels in
        the :math:`5 \times 5` box around it, along the slit and wavelength,
        which a hit and its lobe do not change.
        On a bright, narrow line, that mean is less than the Sun at the
        pixels next to the core, where the line curves, so only the deep
        lobes of large hits are found there.

        Parameters
        ----------
        sigclip
            The detection limit of L.A.Cosmic, in standard deviations of the
            noise.
        objlim
            The contrast limit of L.A.Cosmic between a hit and the fine
            structure of the Sun around it.
        satlevel
            The level at and above which the detector is saturated.
            L.A.Cosmic leaves saturated pixels on bright Sun, and the pixels
            around them, alone.
        limit
            The brightest Sun around a pixel on which the flags of L.A.Cosmic
            are kept.
            An infinite limit keeps them all.
        sigma
            The detection limit of the test of ``iris_prep_despike.pro``, in
            standard deviations.

        Returns
        -------
        :
            Whether each pixel was hit, along the same axes as the outputs.
            Pixels without data, which are NaN, are not.

        Raises
        ------
        ValueError
            If the outputs are not in DN.

        See Also
        --------
        clean : A copy of the observation without the hits.
        """
        wavelength = self.inputs.wavelength_rest
        gain = iris.sg.dn_to_photons(wavelength)
        read_noise = iris.sg.read_noise(wavelength)
        result = _cleaning.spikes(
            data=self._data(),
            axis_exposure=self.axis_detector_x,
            axis_image=(self.axis_detector_y, self.axis_wavelength),
            gain=gain.to(u.ph / u.DN).value,
            read_noise=read_noise.to(u.DN).value,
            sigclip=sigclip,
            objlim=objlim,
            satlevel=satlevel.to(u.DN).value,
            limit=limit.to(u.DN).value,
            sigma=sigma,
        )
        return cast(na.ScalarArray, result)

    def hot_pixels(
        self,
        spikes: None | na.AbstractScalar = None,
        fraction: float = 0.5,
        threshold: u.Quantity = 2 * u.DN,
        limit: u.Quantity = 20 * u.DN,
    ) -> na.ScalarArray:
        """
        Find the hot pixels of the detector, which are too bright in every
        exposure.

        A pixel is hot if it is a spike, :meth:`spikes`, in more than
        `fraction` of the exposures in which it has data, or if its fixed
        pattern is larger than `threshold` either way, where the Sun is
        fainter than `limit`.
        The fixed pattern is found along the slit: it is the interquartile
        mean of each pixel over the exposures, without the spikes, minus
        the 25% trimmed mean of that over the 15 pixels around it along the
        slit.
        It must also be more than five times its uncertainty, from the noise
        of the outputs, :attr:`variance`, so that a short observation does
        not make noisy pixels hot.
        The hot pixels are masked rather than subtracted, since a fixed
        pattern found from the observation carries some of the Sun with it.

        Parameters
        ----------
        spikes
            The pixels hit by energetic particles.
            If :obj:`None` (the default), :meth:`spikes` with its default
            arguments.
        fraction
            The largest fraction of the exposures in which a pixel which is
            not hot is a spike.
        threshold
            The largest fixed pattern of a pixel which is not hot.
        limit
            The brightest Sun, smoothed along the slit as the fixed pattern
            is, on which the fixed pattern is tested.
            The structure of a bright line along the slit is larger than the
            fixed pattern, so it would make many of its pixels hot.

        Returns
        -------
        :
            Whether each pixel is hot, along the axes of the outputs other
            than :attr:`axis_time` and :attr:`axis_detector_x`.

        Raises
        ------
        ValueError
            If the outputs are not in DN.

        See Also
        --------
        clean : A copy of the observation without the hot pixels.
        """
        if spikes is None:
            spikes = self.spikes()
        result = _cleaning.hot_pixels(
            data=self._data(),
            variance=cast(na.ScalarArray, self.variance.to(u.DN**2).value),
            spikes=spikes,
            axis_exposure=self._axis_exposure,
            axis_slit=self.axis_detector_y,
            fraction=fraction,
            threshold=threshold.to(u.DN).value,
            limit=limit.to(u.DN).value,
        )
        return cast(na.ScalarArray, result)

    def dark_rows(
        self,
        ratio: float = 0.5,
    ) -> na.ScalarArray:
        """
        Find the rows along the slit which are darker than the rows around
        them in every exposure, such as those behind the fiducial marks on
        the slit, and the dark ends of the slit.

        The signal of a row in an exposure is its mean over the brightest
        quarter of the wavelengths, where the stray light of the
        spectrograph, which the fiducial marks do not darken, hides them the
        least.
        A row is dark if the median over the exposures of the ratio of its
        signal to that of the 21 rows around it, the 25% trimmed mean of
        their signals, is less than `ratio`, by more than five times the
        uncertainty of the median, from the scatter of the ratio over the
        exposures.
        The dark rows are left out of the rows around each row, and the
        search is repeated until it finds no more, so that a dark end of the
        slit is found whole.
        The rows next to each dark row are dark too, since they are partly
        dark.

        The structure of the Sun along the slit averages out over the
        exposures of a raster of many steps, but not over a few steps, or a
        sit-and-stare observation, where it can be taken for dark rows.
        With fewer than 20 exposures, no rows are found.
        The fiducial marks of observations summed along the slit are less
        dark, and can need a larger `ratio`.

        Parameters
        ----------
        ratio
            The largest ratio of the signal of a dark row to that of the rows
            around it.

        Returns
        -------
        :
            Whether each row is dark, along :attr:`axis_detector_y`.

        Raises
        ------
        ValueError
            If the outputs are not in DN.

        See Also
        --------
        clean : A copy of the observation without the dark rows.
        """
        result = _cleaning.dark_rows(
            data=self._data(),
            axis_exposure=self._axis_exposure,
            axis_slit=self.axis_detector_y,
            axis_wavelength=self.axis_wavelength,
            ratio=ratio,
        )
        return cast(na.ScalarArray, result)

    def clean(
        self,
        spikes: None | na.AbstractScalar = None,
        hot_pixels: None | na.AbstractScalar = None,
        dark_rows: None | na.AbstractScalar = None,
    ) -> Self:
        """
        A copy of this observation whose outputs are NaN where the detector
        does not see the Sun as it is: at the pixels hit by energetic
        particles, the hot pixels, and the dark rows.

        Other functions ignore NaN outputs, such as
        :func:`iris.sg.fit_line`, which leaves them out of each fit.

        Parameters
        ----------
        spikes
            The pixels hit by energetic particles.
            If :obj:`None` (the default), :meth:`spikes` with its default
            arguments.
        hot_pixels
            The hot pixels.
            If :obj:`None` (the default), :meth:`hot_pixels` of `spikes`.
        dark_rows
            The dark rows.
            If :obj:`None` (the default), :meth:`dark_rows` with its default
            arguments.

        Raises
        ------
        ValueError
            If the outputs are not in DN and one of the masks is not given.

        Examples
        --------

        Clean the first 40 exposures of a raster, and compare one exposure
        before and after.

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            import astropy.visualization
            import named_arrays as na
            import iris

            # Load the first 40 exposures of a 320-step raster
            obs = iris.sg.open("2021-09-23T06:13")
            obs = obs[{obs.axis_detector_x: slice(0, 40)}]

            # Mask the spikes, the hot pixels, and the dark rows
            cleaned = obs.clean()

            # Plot one exposure before and after, with the masked pixels
            # in red
            index = {obs.axis_time: 0, obs.axis_detector_x: 20}
            with astropy.visualization.quantity_support():
                fig, ax = plt.subplots(
                    ncols=2,
                    figsize=(8, 6),
                    sharex=True,
                    sharey=True,
                    constrained_layout=True,
                )
                for axes, o, title in zip(ax, (obs, cleaned), ("level 2", "cleaned")):
                    axes.set_facecolor("red")
                    na.plt.imshow(
                        o.outputs[index].value,
                        axis_x=obs.axis_wavelength,
                        axis_y=obs.axis_detector_y,
                        ax=axes,
                        cmap="gray",
                        vmin=0,
                        vmax=20,
                        interpolation="none",
                    )
                    axes.set_title(title)
                    axes.set_xlabel("wavelength (pixels)")
                ax[0].set_ylabel("position along the slit (pixels)")
        """
        if spikes is None:
            spikes = self.spikes()
        if hot_pixels is None:
            hot_pixels = self.hot_pixels(spikes=spikes)
        if dark_rows is None:
            dark_rows = self.dark_rows()
        outputs = cast(na.ScalarArray, self.outputs.explicit.copy())
        outputs[spikes | hot_pixels | dark_rows] = np.nan
        return dataclasses.replace(self, outputs=outputs)

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
