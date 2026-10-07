import astropy.units as u
import astropy.time
from ._slit_jaw import SlitJawObservation

__all__ = [
    "open",
]


def open(
    time: str | astropy.time.Time,
    time_stop: None | str | astropy.time.Time = None,
    description: str = "",
    obs_id: None | int = None,
    window: str = "SJI_1400",
    axis_time: str = "time",
    axis_detector_x: str = "detector_x",
    axis_detector_y: str = "detector_y",
    limit: int = 200,
    nrt: bool = True,
    num_retry: int = 5,
) -> SlitJawObservation:
    """
    Download the IRIS slit-jaw images which began during a given time range
    and load them into memory as an instance of
    :class:`~iris.sji.SlitJawObservation`.

    Parameters
    ----------
    time
        The start time of the search period.
    time_stop
        The end time of the search period.
        If :obj:`None`, 1 minute will be added to `time`.
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
    limit
        The maximum number of observations returned by the query.
    nrt
        Whether to return results with near-real-time (NRT) data.
    num_retry
        The number of times to try to connect to the server.
    """

    time = astropy.time.Time(time)

    if time_stop is None:
        time_stop = time + 1 * u.min

    time_stop = astropy.time.Time(time_stop)

    return SlitJawObservation.from_time_range(
        time_start=time,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        window=window,
        axis_time=axis_time,
        axis_detector_x=axis_detector_x,
        axis_detector_y=axis_detector_y,
        limit=limit,
        nrt=nrt,
        num_retry=num_retry,
    )
