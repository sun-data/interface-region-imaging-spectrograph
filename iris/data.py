"""
Utilities for finding and downloading IRIS data
"""

from __future__ import annotations
import requests
import urlpath
import astropy.time

__all__ = [
    "query_hek",
    "urls_hek",
]


def query_hek(
    time_start: None | astropy.time.Time = None,
    time_stop: None | astropy.time.Time = None,
    description: str = "",
    obs_id: None | int = None,
    limit: int = 200,
) -> str:
    """
    Constructs a query that can be sent to the Heliophysics Event Knowledge
    Base (HEK) to receive a list of URLs.

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
        the OBSID of the observation, a number which describes the size, cadence,
        etc. of the observation. If :obj:`None`, all OBSIDs will be used.
    limit
        the maximum number of files returned by the query

    Examples
    --------

    Construct a query for the first 100 A1: QS monitoring observations in 2023

    .. jupyter-execute::

        import astropy.time
        import iris

        iris.data.query_hek(
            time_start=astropy.time.Time("2023-01-01T00:00"),
            time_stop=astropy.time.Time("2024-01-01T00:00"),
            description="A1: QS monitoring",
            limit=100,
        )
    """

    format_spec = "%Y-%m-%dT%H:%M"

    if time_start is None:
        time_start = astropy.time.Time("2013-07-20T00:00")

    if time_stop is None:
        time_stop = astropy.time.Time.now()

    query_hek = (
        "https://www.lmsal.com/hek/hcr?cmd=search-events3"
        "&outputformat=json"
        f"&startTime={time_start.strftime(format_spec)}"
        f"&stopTime={time_stop.strftime(format_spec)}"
        "&hasData=true"
        "&hideMostLimbScans=true"
        f"&obsDesc={description}"
        f"&limit={limit}"
    )
    if obs_id is not None:
        query_hek += f"&obsId={obs_id}"

    return query_hek


def urls_hek(
    time_start: None | astropy.time.Time = None,
    time_stop: None | astropy.time.Time = None,
    description: str = "",
    obs_id: None | int = None,
    limit: int = 200,
    spectrograph: bool = True,
    sji: bool = True,
) -> list[urlpath.URL]:
    """
    Find a list of URLs to download matching the given parameters.

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
        the OBSID of the observation, a number which describes the size, cadence,
        etc. of the observation. If :obj:`None`, all OBSIDs will be used.
    limit
        the maximum number of files returned by the query
    spectrograph
        Boolean flag controlling whether to include spectrograph data.
    sji
        Boolean flag controlling whether to include SJI data.

    Examples
    --------
    Find the URLs of the first 5 "A1: QS monitoring" spectrograph observations
    in 2023.

    .. jupyter-execute::

        import astropy.time
        import iris

        iris.data.urls_hek(
            time_start=astropy.time.Time("2023-01-01T00:00"),
            time_stop=astropy.time.Time("2024-01-01T00:00"),
            description="A1: QS monitoring",
            limit=5,
            sji=False,
        )
    """
    query = query_hek(
        time_start=time_start,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        limit=limit,
    )

    response = None
    while response is None:
        try:
            response = requests.get(query, timeout=5)
        except requests.exceptions.RequestException:
            pass

    response = response.json()

    result = []
    for event in response["Events"]:
        for group in event["groups"]:

            url = group["comp_data_url"]

            if spectrograph:
                if "raster" in url:
                    result.append(urlpath.URL(url))
            if sji:
                if "SJI" in url:
                    result.append(urlpath.URL(url))

    return result
