from typing import Sequence
import typing
import os
import pathlib
import warnings
import dataclasses
import numpy as np
import astropy.units as u
import astropy.time
import astropy.io.fits
import regridding
import named_arrays as na
import iris
from . import _dust

__all__ = [
    "SlitJawObservation",
]

_columns_pointing = [
    "XCENIX",
    "YCENIX",
    "PC1_1IX",
    "PC1_2IX",
    "PC2_1IX",
    "PC2_2IX",
]
"""The columns of the auxiliary data which hold the pointing of each frame."""


@dataclasses.dataclass(eq=False, repr=False)
class SlitJawObservation(
    na.FunctionArray[
        na.ExplicitTemporalSpectralWcsPositionalVectorArray,
        na.ScalarArray,
    ]
):
    """
    A sequence of images captured by the IRIS slit-jaw imager (SJI).

    Each image is one frame along :attr:`axis_time`.
    IRIS follows the rotation of the Sun, so the pointing changes from one
    frame to the next,
    and the coordinates of each frame are found from the pointing recorded
    for its exposure in the auxiliary data of the level 2 file
    rather than from the single pointing in its primary header.

    The time of each frame, ``inputs.time``, is the start of its exposure,
    as in the other sun-data packages.
    irispy uses the midpoint of each exposure instead,
    which is ``inputs.time + timedelta / 2``.

    Examples
    --------

    Load the Si IV slit-jaw images captured while the EUV Snapshot Imaging
    Spectrograph (ESIS) was observing the Sun on 2019 September 30,
    and display the first one.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import iris

        # Load the slit-jaw images captured during the ESIS flight
        obs = iris.sji.open(
            time="2019-09-30T18:06:11",
            time_stop="2019-09-30T18:11:01",
        )

        # Select the first image
        image = obs[{obs.axis_time: 0}]

        # Display the first image
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(constrained_layout=True)
            na.plt.pcolormesh(
                image.inputs.position.x,
                image.inputs.position.y,
                C=image.outputs,
                ax=ax,
                cmap="gray",
                vmin=0 * u.DN,
                vmax=50 * u.DN,
            )
            ax.set_aspect("equal")
            ax.set_title(image.inputs.time.ndarray)
            ax.set_xlabel(f"helioprojective $x$ ({ax.get_xlabel()})")
            ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")
    """

    timedelta: u.Quantity | na.AbstractScalar = dataclasses.field(
        default_factory=lambda: 0 * u.s,
    )
    """
    The exposure time of each frame.
    """

    dust: None | na.ScalarArray = None
    """
    Whether each pixel is darkened by dust on the CCD of the slit-jaw imager,
    or :obj:`None` if the dust was not loaded.

    The dust is found from the map of bad pixels which the IRIS team measures
    from the flat field of each channel and publishes in the SolarSoft
    database, using the flat field nearest in time to the observation.
    Each bad pixel is grown by a few pixels, to cover the partly dark rim of
    each dust particle, and moved into each frame the way ``IRIS_DUSTBUSTER``
    in SolarSoft moves it:
    rotated and scaled about the center of the slit on the CCD,
    placed relative to the center of the slit in that frame,
    and moved by the shift of up to 7 pixels which makes the dusty pixels
    of the observation darkest.

    The dust stays put on the CCD while each level 2 frame is shifted so that
    the Sun stays put,
    so the dust moves from frame to frame along with the slit.
    """

    axis_time: str = "time"
    """The logical axis corresponding to changes in time."""

    axis_detector_x: str = "detector_x"
    """The logical axis corresponding to changes in detector :math:`x`-coordinate."""

    axis_detector_y: str = "detector_y"
    """The logical axis corresponding to changes in detector :math:`y`-coordinate."""

    @classmethod
    def from_time_range(
        cls,
        time_start: None | str | astropy.time.Time = None,
        time_stop: None | str | astropy.time.Time = None,
        description: str = "",
        obs_id: None | int = None,
        window: str = "SJI_1400",
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        dust: bool = True,
        limit: int = 200,
        nrt: bool = False,
        num_retry: int = 5,
    ) -> "SlitJawObservation":
        """
        Download the slit-jaw images which began during a given time range
        and construct an instance of :class:`SlitJawObservation`.

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
            The slit-jaw channel to load: ``"SJI_1330"``, ``"SJI_1400"``,
            ``"SJI_2796"``, ``"SJI_2832"``, or ``"SJI_5000W"``.
        axis_time
            The logical axis corresponding to changes in time.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        dust
            Whether to find the pixels darkened by dust, :attr:`dust`,
            which downloads the maps of bad pixels from the SolarSoft
            database the first time.
        limit
            The maximum number of observations returned by the query.
        nrt
            Whether to return results with near-real-time (NRT) data.
        num_retry
            The number of times to try to connect to the server.
        """
        if time_start is not None:
            time_start = astropy.time.Time(time_start)

        if time_stop is not None:
            time_stop = astropy.time.Time(time_stop)

        files = _download(
            time_start=time_start,
            time_stop=time_stop,
            description=description,
            obs_id=obs_id,
            window=window,
            limit=limit,
            nrt=nrt,
            num_retry=num_retry,
        )

        return cls.from_fits(
            path=files,
            time_start=time_start,
            time_stop=time_stop,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
            dust=dust,
            num_retry=num_retry,
        )

    @classmethod
    def from_fits(
        cls,
        path: str | os.PathLike | Sequence[str | os.PathLike],
        time_start: None | str | astropy.time.Time = None,
        time_stop: None | str | astropy.time.Time = None,
        axis_time: str = "time",
        axis_detector_x: str = "detector_x",
        axis_detector_y: str = "detector_y",
        dust: bool = True,
        num_retry: int = 5,
    ) -> "SlitJawObservation":
        """
        Load the frames of one or more level 2 slit-jaw files which began
        during a given time range.

        The frames of every file are placed one after another along
        `axis_time`, in the order the files are given,
        and images smaller than the largest one are padded with NaN.

        Only the frames which are kept are read from each file, so a short
        time range can be loaded from a long observation without holding
        the whole observation in memory.

        Parameters
        ----------
        path
            A level 2 slit-jaw file, or a sequence of files of the same channel.
            The files may be compressed with gzip.
        time_start
            The earliest start time of the frames to keep.
            If :obj:`None`, frames are kept from the start of each file.
        time_stop
            The time before which a frame must begin to be kept.
            If :obj:`None`, frames are kept until the end of each file.
        axis_time
            The logical axis corresponding to changes in time.
        axis_detector_x
            The logical axis corresponding to changes in detector :math:`x`-coordinate.
        axis_detector_y
            The logical axis corresponding to changes in detector :math:`y`-coordinate.
        dust
            Whether to find the pixels darkened by dust, :attr:`dust`,
            which downloads the maps of bad pixels from the SolarSoft
            database the first time.
        num_retry
            The number of times to try to connect to the SolarSoft database.

        Notes
        -----
        The level 2 pipeline marks missing pixels, and the pixels outside
        the field of view, with the same value, and both are set to NaN.

        The level 1 pipeline sets the pixels which are known to be bad,
        including the ones covered by dust, to zero,
        and they are left as they are.
        :meth:`remove_dust` replaces the pixels darkened by dust.

        The level 2 pipeline records the pointing of an exposure as zero
        when it is unknown.
        As :func:`irispy.io.read_files` does, each zero in the pointing
        columns is replaced by the value interpolated from the other frames
        of the same file.
        """
        if isinstance(path, (str, os.PathLike)):
            path = [path]

        if time_start is not None:
            time_start = astropy.time.Time(time_start)

        if time_stop is not None:
            time_stop = astropy.time.Time(time_stop)

        window = None
        wavelength = None
        frames = {
            key: []
            for key in [
                "TIME",
                "EXPTIMES",
                *_columns_pointing,
                "CRPIX1",
                "CRPIX2",
                "CDELT1",
                "CDELT2",
            ]
        }
        selected = []
        shape_y = 0
        shape_x = 0

        # Find the frames of each file which began during the time range
        # from the auxiliary data alone, before reading any images.
        for file in path:
            with astropy.io.fits.open(file) as hdul:
                hdu = hdul[0]
                hdu_aux = hdul[1]
                header = hdu.header
                aux = hdu_aux.data
                header_aux = hdu_aux.header

                time = astropy.time.Time(header["STARTOBS"])
                time = time + aux[:, header_aux["TIME"]] * u.s

                where = np.ones(time.shape, dtype=bool)
                if time_start is not None:
                    where &= time >= time_start
                if time_stop is not None:
                    where &= time < time_stop

                (index,) = np.nonzero(where)
                if index.size == 0:
                    continue

                if window is None:
                    window = header["TDESC1"]
                    wavelength = header["TWAVE1"]
                elif header["TDESC1"] != window:
                    raise ValueError(
                        f"The files are of different channels, "
                        f"{window} and {header['TDESC1']} in {file}."
                    )

                # The frames of a file are in the order they were captured,
                # so the frames in the time range are contiguous, and can be
                # read in one pass through a compressed file.
                index = slice(int(index[0]), int(index[~0]) + 1)
                num = index.stop - index.start

                frames["TIME"].append(time[index])
                frames["EXPTIMES"].append(aux[index, header_aux["EXPTIMES"]])

                pointing = dict()
                for key in _columns_pointing:
                    try:
                        column = _interpolate_zeros(aux[:, header_aux[key]])
                    except ValueError as e:
                        raise ValueError(
                            f"The pointing column {key} of {file} is zero, "
                            f"which means unknown, in every frame."
                        ) from e
                    pointing[key] = column[index]
                    frames[key].append(pointing[key])

                for key in ["CRPIX1", "CRPIX2", "CDELT1", "CDELT2"]:
                    frames[key].append(np.full(num, header[key]))

                shape_y = max(shape_y, int(header["NAXIS2"]))
                shape_x = max(shape_x, int(header["NAXIS1"]))

                geometry = None
                if dust:
                    # One less than the auxiliary data,
                    # which counts pixels from one.
                    slit = [
                        _interpolate_zeros(aux[:, header_aux[key]])[index] - 1
                        for key in ["SLTPX1IX", "SLTPX2IX"]
                    ]
                    pc = [
                        pointing[key]
                        for key in ["PC1_1IX", "PC1_2IX", "PC2_1IX", "PC2_2IX"]
                    ]
                    geometry = dict(
                        window=header["TDESC1"],
                        time=astropy.time.Time(header["DATE_OBS"]),
                        binning=(header["SUMSPTRL"], header["SUMSPAT"]),
                        slit=np.stack(slit, axis=~0),
                        crval=np.stack(
                            [pointing["XCENIX"], pointing["YCENIX"]],
                            axis=~0,
                        ),
                        crpix=np.array([header["CRPIX1"], header["CRPIX2"]]) - 1,
                        cdelt=np.array([header["CDELT1"], header["CDELT2"]]),
                        pc=np.stack(pc, axis=~0).reshape(num, 2, 2),
                    )

                selected.append((file, index, geometry))

        if not selected:
            raise ValueError(
                f"No frames in {path} began between {time_start} and {time_stop}."
            )

        # Read the frames into one array, padded with NaN,
        # so that the images are held in memory only once.
        num_t = sum(index.stop - index.start for file, index, geometry in selected)
        image = np.full((num_t, shape_y, shape_x), np.nan, dtype=np.float32)
        mask = np.zeros(image.shape, dtype=bool)
        t = 0
        for file, index, geometry in selected:
            with astropy.io.fits.open(file) as hdul:
                data = hdul[0].section[index]
            data[data == -200] = np.nan
            num, num_y, num_x = data.shape
            image[t : t + num, :num_y, :num_x] = data
            if geometry is not None:
                mask[t : t + num, :num_y, :num_x] = _dust._mask(
                    image=data,
                    num_retry=num_retry,
                    **geometry,
                )
            t += num

        axes = (axis_time, axis_detector_y, axis_detector_x)

        dusty = None
        if dust:
            dusty = na.ScalarArray(mask, axes=axes)

        def scalar(key: str, unit: None | u.UnitBase = None) -> na.ScalarArray:
            a = np.concatenate(frames[key])
            if unit is not None:
                a = a << unit
            return na.ScalarArray(a, axis_time)

        time = astropy.time.Time(np.concatenate(frames["TIME"]), format="isot")

        inputs = na.ExplicitTemporalSpectralWcsPositionalVectorArray(
            time=na.ScalarArray(time, axis_time),
            wavelength=na.ScalarArray(wavelength * u.AA),
            crval=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=scalar("XCENIX", u.arcsec),
                    y=scalar("YCENIX", u.arcsec),
                ),
            ),
            # One less than the FITS keyword, which counts pixels from one
            # where :class:`named_arrays.AbstractWcsVector` counts them from
            # zero.
            crpix=na.CartesianNdVectorArray(
                components={
                    axis_detector_x: scalar("CRPIX1") - 1,
                    axis_detector_y: scalar("CRPIX2") - 1,
                },
            ),
            cdelt=na.PositionalVectorArray(
                position=na.Cartesian2dVectorArray(
                    x=scalar("CDELT1", u.arcsec),
                    y=scalar("CDELT2", u.arcsec),
                ),
            ),
            pc=na.PositionalMatrixArray(
                position=na.Cartesian2dMatrixArray(
                    x=na.CartesianNdVectorArray(
                        components={
                            axis_detector_x: scalar("PC1_1IX"),
                            axis_detector_y: scalar("PC1_2IX"),
                        },
                    ),
                    y=na.CartesianNdVectorArray(
                        components={
                            axis_detector_x: scalar("PC2_1IX"),
                            axis_detector_y: scalar("PC2_2IX"),
                        },
                    ),
                ),
            ),
            shape_wcs={
                axis_detector_x: shape_x + 1,
                axis_detector_y: shape_y + 1,
            },
        )

        outputs = na.ScalarArray(
            ndarray=image << u.DN,
            axes=axes,
        )

        return cls(
            inputs=inputs,
            outputs=outputs,
            timedelta=scalar("EXPTIMES", u.s),
            dust=dusty,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

    def remove_dust(
        self,
        num_iterations: int = 100,
    ) -> "SlitJawObservation":
        """
        Replace the pixels darkened by dust, :attr:`dust`,
        with an estimate of the Sun behind the dust.

        As ``IRIS_DUSTBUSTER`` in SolarSoft does,
        each dusty pixel is first replaced by the median of the same pixel in
        the two frames before it and the two frames after it in which that
        pixel is not dusty, each scaled by the ratio of the exposure times.
        The dust moves with the slit while the Sun stays put,
        so when IRIS is rastering, the neighboring frames often see the Sun
        behind the dust.

        The dusty pixels which none of those frames see,
        such as the middle of a large dust particle,
        or every dusty pixel of a sit-and-stare observation,
        are then filled by :func:`regridding.fill`,
        which relaxes each one toward the mean of its neighbors in the same
        frame.

        The result is a cosmetic estimate rather than a measurement,
        and :attr:`dust` still marks the pixels which were replaced.

        Parameters
        ----------
        num_iterations
            The number of Gauss-Seidel iterations used by :func:`regridding.fill`.

        Examples
        --------

        Remove the dust from the Si IV slit-jaw images captured while ESIS was
        observing the Sun, and compare a dusty region of one frame before and
        after.

        .. jupyter-execute::

            import matplotlib.pyplot as plt
            import astropy.units as u
            import astropy.visualization
            import named_arrays as na
            import iris

            # Load the slit-jaw images captured during the ESIS flight
            obs = iris.sji.open(
                time="2019-09-30T18:06:11",
                time_stop="2019-09-30T18:11:01",
            )

            # Replace the pixels darkened by dust
            clean = obs.remove_dust()

            # A dusty region of one frame
            index = {
                obs.axis_time: 10,
                obs.axis_detector_x: slice(90, 170),
                obs.axis_detector_y: slice(35, 115),
            }

            # Display the region before and after
            unit = obs.inputs.position.x.unit
            with astropy.visualization.quantity_support():
                fig, axs = plt.subplots(
                    ncols=2,
                    figsize=(8, 4.5),
                    sharex=True,
                    sharey=True,
                    constrained_layout=True,
                )
                for ax, o, title in zip(axs, [obs, clean], ["level 2", "dust removed"]):
                    region = o[index]
                    na.plt.pcolormesh(
                        region.inputs.position.x,
                        region.inputs.position.y,
                        C=region.outputs,
                        ax=ax,
                        cmap="gray",
                        vmin=0 * u.DN,
                        vmax=50 * u.DN,
                    )
                    ax.set_aspect("equal")
                    ax.set_title(title)
                    ax.set_xlabel(f"helioprojective $x$ ({unit:latex_inline})")
                axs[0].set_ylabel(f"helioprojective $y$ ({unit:latex_inline})")
        """
        dust = self.dust

        if dust is None:
            raise ValueError(
                "The dust was not loaded, so it cannot be removed. "
                "Load the images with `dust=True`."
            )

        axis_t = self.axis_time
        axis_x = self.axis_detector_x
        axis_y = self.axis_detector_y

        timedelta = na.as_named_array(self.timedelta)

        shape = na.shape_broadcasted(self.outputs, dust, timedelta)
        axes = tuple(shape)

        # The images, the dust, and the exposure times as arrays of numbers,
        # broadcast against each other with the axes in the order of `axes`
        unit = na.unit(self.outputs)
        images = np.array(na.broadcast_to(self.outputs, shape).value.ndarray, float)
        dust = np.array(na.broadcast_to(dust, shape).ndarray, bool)
        timedelta = typing.cast(na.ScalarArray, na.broadcast_to(timedelta, shape))
        timedelta = np.array(timedelta.value.ndarray, float)
        remaining = dust.copy()

        if axis_t in shape:
            ax_t = axes.index(axis_t)
            num_t = shape[axis_t]

            # The index of every dusty pixel,
            # and of the same pixel in the two frames before it and the two
            # frames after it
            index = np.nonzero(dust)
            frame = index[ax_t][:, np.newaxis] + np.array([-2, -1, 1, 2])
            inside = (frame >= 0) & (frame < num_t)
            index_neighbor = tuple(
                frame % num_t if ax == ax_t else i[:, np.newaxis]
                for ax, i in enumerate(index)
            )

            # The rate of each neighbor which is not dusty itself
            neighbors = images[index_neighbor] / timedelta[index_neighbor]
            valid = inside & ~dust[index_neighbor] & np.isfinite(neighbors)
            neighbors[~valid] = np.nan

            with warnings.catch_warnings():
                # The pixels which no neighboring frame sees are expected,
                # and are filled below.
                warnings.simplefilter("ignore", category=RuntimeWarning)
                estimate = np.nanmedian(neighbors, axis=~0) * timedelta[index]

            seen = np.isfinite(estimate)
            index_seen = tuple(i[seen] for i in index)

            images[index_seen] = estimate[seen]
            remaining[index_seen] = False

        if np.any(remaining):
            images = regridding.fill(
                a=images,
                where=remaining,
                axis=(axes.index(axis_y), axes.index(axis_x)),
                num_iterations=num_iterations,
            )

        outputs = na.ScalarArray(images, axes=axes)
        if unit is not None:
            outputs = outputs << unit

        return self.replace(outputs=outputs)


def _download(
    time_start: None | astropy.time.Time,
    time_stop: None | astropy.time.Time,
    description: str,
    obs_id: None | int,
    window: str,
    limit: int,
    nrt: bool,
    num_retry: int,
) -> list[pathlib.Path]:
    """
    Download the level 2 files of one slit-jaw channel of every observation
    which the Heliophysics Event Knowledge Base (HEK) finds in a time range.

    Parameters
    ----------
    time_start
        The start time of the search period.
    time_stop
        The end time of the search period.
    description
        The description of the observation.
    obs_id
        The OBSID of the observation.
    window
        The slit-jaw channel to download.
    limit
        The maximum number of observations returned by the query.
    nrt
        Whether to include near-real-time (NRT) data,
        which is used only for the observations whose final data is not
        yet published.
    num_retry
        The number of times to try to connect to the server.
    """
    urls = []
    for nrt_query in [False, True] if nrt else [False]:
        urls += iris.data.urls_hek(
            time_start=time_start,
            time_stop=time_stop,
            description=description,
            obs_id=obs_id,
            limit=limit,
            nrt=nrt_query,
            spectrograph=False,
            sji=True,
            deconvolved=False,
            num_retry=num_retry,
        )

    urls = iris.data._prefer_final(urls)

    # Level 2 slit-jaw files are named after their channel, as in
    # ``iris_l2_20190930_171911_3604109624_SJI_1400_t000.fits.gz``,
    # so only the requested channel needs to be downloaded.
    urls = [url for url in urls if f"_{window}_" in url]

    if not urls:
        raise ValueError(
            f"No {window} observations between {time_start} and {time_stop}."
        )

    return iris.data.download(urls)


def _interpolate_zeros(a: np.ndarray) -> np.ndarray:
    """
    Replace each zero in the given column of auxiliary data by the value
    interpolated from the nonzero values around it.

    Parameters
    ----------
    a
        A column of the auxiliary data of a level 2 slit-jaw file,
        with one value per exposure.
    """
    where = a == 0

    if not np.any(where):
        return a

    if np.all(where):
        raise ValueError("Every value in the column is zero.")

    index = np.arange(a.size)

    result = a.copy()
    result[where] = np.interp(
        x=index[where],
        xp=index[~where],
        fp=a[~where],
    )

    return result
