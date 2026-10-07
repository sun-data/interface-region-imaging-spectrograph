import astropy.units as u
import astropy.time
from ._slit_jaw import SlitJawObservation, _download

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
    dust: bool = True,
    limit: int = 200,
    nrt: bool = True,
    num_retry: int = 5,
) -> SlitJawObservation:
    """
    Download IRIS slit-jaw images and load them into memory as an instance of
    :class:`~iris.sji.SlitJawObservation`.

    If `time_stop` is :obj:`None`, every frame of the observations running
    during the minute after `time` is loaded, as :func:`iris.sg.open` does.
    Otherwise, only the frames which began between `time` and `time_stop`
    are loaded.

    Parameters
    ----------
    time
        The start time of the search period.
    time_stop
        The end time of the search period.
        If :obj:`None`, the search period is the minute after `time`,
        and every frame of the observations found is loaded.
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
        Whether to find the pixels darkened by dust,
        :attr:`~iris.sji.SlitJawObservation.dust`,
        which downloads the maps of bad pixels from the SolarSoft database
        the first time.
    limit
        The maximum number of observations returned by the query.
    nrt
        Whether to return results with near-real-time (NRT) data.
    num_retry
        The number of times to try to connect to the server.
    """

    time = astropy.time.Time(time)

    if time_stop is None:
        files = _download(
            time_start=time,
            time_stop=time + 1 * u.min,
            description=description,
            obs_id=obs_id,
            window=window,
            limit=limit,
            nrt=nrt,
            num_retry=num_retry,
        )
        return SlitJawObservation.from_fits(
            path=files,
            axis_time=axis_time,
            axis_detector_x=axis_detector_x,
            axis_detector_y=axis_detector_y,
        )

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
        dust=dust,
        limit=limit,
        nrt=nrt,
        num_retry=num_retry,
    )
