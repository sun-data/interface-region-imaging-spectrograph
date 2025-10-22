import astropy.units as u
import astropy.time
from ._spectrograph import SpectrographObservation

__all__ = [
    "open",
]


def open(
    time: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    description: str = "",
    obs_id: None | int = None,
    window: str = "Si IV 1394",
    axis_time: str = "time",
    axis_wavelength: str = "wavelength",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    limit: int = 200,
    nrt: bool = True,
    num_retry: int = 5,
) -> SpectrographObservation:
    """
    Download an IRIS observation and load it into memory as an instance of
    :class:`~iris.sg.SpectrographObservation`.

    Parameters
    ----------
    time
        The start time of the search period.
    time_stop
        The end time of the search period.
        If :obj:`None`, 1 second will be added to `time` which usually has the
        effect of selecting one observation.
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

    time = astropy.time.Time(time)

    if time_stop is None:
        time_stop = time + 1 * u.min

    time_stop = astropy.time.Time(time_stop)

    return SpectrographObservation.from_time_range(
        time_start=time,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        window=window,
        axis_time=axis_time,
        axis_wavelength=axis_wavelength,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        limit=limit,
        nrt=nrt,
        num_retry=num_retry,
    )
