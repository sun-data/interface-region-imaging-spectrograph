"""
Utilities for finding and downloading IRIS data
"""

from __future__ import annotations
import pathlib
import requests
import urlpath
import astropy.time

__all__ = [
    "query_hek",
    "urls_hek",
    "download",
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
    deconvolved: bool = False,
    num_retry: int = 5,
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
        The maximum number of observations returned by the query.
        Note that this is not the same as the number of files since there
        are several files per observation.
    spectrograph
        Boolean flag controlling whether to include spectrograph data.
    sji
        Boolean flag controlling whether to include SJI data.
    deconvolved
        Boolean flag controlling whether to include the deconvolved slitjaw
        imagery. Has no effect if ``sji`` is :obj:`False`.
    num_retry
        The number of times to try to connect to the server.

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

    for i in range(num_retry):
        try:
            response = requests.get(query, timeout=5).json()
            break
        except requests.exceptions.RequestException:  # pragma: no cover
            pass
    else:  # pragma: no cover
        raise ConnectionError(f"Could not get query {query}")

    result = []
    for event in response["Events"]:
        for group in event["groups"]:

            url = group["comp_data_url"]

            if spectrograph:
                if "raster" in url:
                    result.append(urlpath.URL(url))
            if sji:
                if "SJI" in url:
                    if "deconvolved" in url:
                        if deconvolved:
                            result.append(urlpath.URL(url))
                    else:
                        result.append(urlpath.URL(url))

    return result


def download(
    urls: list[urlpath.URL],
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
) -> list[pathlib.Path]:
    """
    Download the given URLs to a specified directory.
    If `overwrite` is :obj:`False`, the file will not be downloaded if it exists.

    Parameters
    ----------
    urls
        The URLs to download.
    directory
        The directory to place the downloaded files.
    overwrite
        Boolean flag controlling whether to overwrite existing files.


    Examples
    --------
    Download the most recent "A1: QS monitoring" SJI files

    .. jupyter-execute::

        import iris

        urls = iris.data.urls_hek(
            description="A1: QS monitoring",
            limit=1,
            spectrograph=False,
        )

        iris.data.download(urls)
    """
    if directory is None:
        directory = pathlib.Path.home() / ".iris/cache"

    directory.mkdir(parents=True, exist_ok=True)

    result = []
    for url in urls:

        file = directory / url.name

        if overwrite or not file.exists():
            r = requests.get(url)
            with open(file, "wb") as f:
                f.write(r.content)

        result.append(file)

    return result
