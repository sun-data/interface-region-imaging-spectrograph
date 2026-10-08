"""
Utilities for finding and downloading IRIS data
"""

from __future__ import annotations
from typing import Sequence
import pathlib
import shutil
import requests
import astropy.units as u
import astropy.time
import iris

__all__ = [
    "query_hek",
    "urls_hek",
    "download",
    "decompress",
]


def _ceil_minute(time: astropy.time.Time) -> astropy.time.Time:
    """
    Round a time up to the next whole minute.

    HEK takes times to the minute, and cutting the stop time of a search to
    the minute would leave out the observations which begin after the cut.

    Parameters
    ----------
    time
        The time to round.
    """
    result = astropy.time.Time(time.strftime("%Y-%m-%dT%H:%M"), scale=time.scale)
    if result < time:
        result = result + 1 * u.min
    return result


def query_hek(
    time_start: None | astropy.time.Time = None,
    time_stop: None | astropy.time.Time = None,
    description: str = "",
    obs_id: None | int = None,
    limit: int = 200,
    nrt: bool = False,
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
        HEK takes times to the minute,
        so this is rounded up to the next whole minute,
        which keeps the observations that begin during its last partial minute.
    description
        The description of the observation. If an empty string, observations with
        any description will be returned.
    obs_id
        the OBSID of the observation, a number which describes the size, cadence,
        etc. of the observation. If :obj:`None`, all OBSIDs will be used.
    limit
        the maximum number of files returned by the query
    nrt
        Whether to return results with near-real-time (NRT) data.

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

    stop = _ceil_minute(time_stop)

    if nrt:
        hasData = "false"
    else:
        hasData = "true"

    query_hek = (
        "https://www.lmsal.com/hek/hcr?cmd=search-events3"
        "&outputformat=json"
        f"&startTime={time_start.strftime(format_spec)}"
        f"&stopTime={stop.strftime(format_spec)}"
        f"&hasData={hasData}"
        "&hideMostLimbScans=true"
        f"&obsDesc={description}"
        f"&limit={limit}"
    )
    if obs_id is not None:
        query_hek += f"&obsId={obs_id}"

    return query_hek


@iris.memory.cache
def urls_hek(
    time_start: None | astropy.time.Time = None,
    time_stop: None | astropy.time.Time = None,
    description: str = "",
    obs_id: None | int = None,
    limit: int = 200,
    nrt: bool = False,
    spectrograph: bool = True,
    sji: bool = True,
    deconvolved: bool = False,
    num_retry: int = 5,
) -> list[str]:
    """
    Find a list of URLs to download matching the given parameters.

    Parameters
    ----------
    time_start
        The start time of the search period. If :obj:`None`, the start of operations,
        2013-07-20 will be used.
    time_stop
        The end time of the search period. If :obj:`None`, the current time will be used.
        It is rounded up to the next whole minute, as in :func:`query_hek`.
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
    nrt
        Whether to return results with near-real-time (NRT) data.

    Examples
    --------
    Find the URLs of the last 5 "A1: QS monitoring" spectrograph observations
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
        nrt=nrt,
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
            url = url.replace("data_lmsal", "data")

            url_str = str(url)

            if spectrograph:
                if "raster" in url_str:
                    result.append(url)
            if sji:
                if "SJI" in url_str:
                    if "deconvolved" in url_str:
                        if deconvolved:
                            result.append(url)
                    else:
                        result.append(url)

    return result


def download(
    urls: list[str],
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
) -> list[pathlib.Path]:
    """
    Download the given URLs to a specified directory.
    If `overwrite` is :obj:`False`, the file will not be downloaded if it exists.

    A near-real-time (NRT) file has the same name as the final file of the
    same observation, so the NRT files are placed in a subdirectory, ``nrt``,
    where they cannot be mistaken for the final files.

    Parameters
    ----------
    urls
        The URLs to download.
    directory
        The directory to place the downloaded files.
    overwrite
        Boolean flag controlling whether to overwrite existing files.

    Returns
    -------
    The paths of the downloaded files, sorted by file name.


    Examples
    --------
    Download the last "A1: QS monitoring" spectrograph file in 2023.

    .. jupyter-execute::

        import astropy.time
        import iris

        urls = iris.data.urls_hek(
            time_start=astropy.time.Time("2023-01-01T00:00"),
            time_stop=astropy.time.Time("2024-01-01T00:00"),
            description="A1: QS monitoring",
            limit=1,
            sji=False,
        )

        iris.data.download(urls)
    """
    if directory is None:
        directory = pathlib.Path.home() / ".iris/cache"

    directory.mkdir(parents=True, exist_ok=True)

    result = []
    for url in urls:

        file = directory / url.split("/")[~0]
        if _is_nrt(url):
            file = directory / "nrt" / file.name

        if overwrite or not file.exists():
            file.parent.mkdir(parents=True, exist_ok=True)
            r = requests.get(url, stream=True)
            with open(file, "wb") as f:
                f.write(r.content)

        result.append(file)

    # The file names start with the time of the observation
    return sorted(result, key=lambda file: file.name)


def _is_nrt(url: str) -> bool:
    """
    Whether a URL is of near-real-time (NRT) data,
    which LMSAL keeps in directories such as ``level2_nrt_compressed``.

    Parameters
    ----------
    url
        The URL of an IRIS file.
    """
    return any("_nrt" in part for part in url.split("/")[:~0])


def _prefer_final(urls: list[str]) -> list[str]:
    """
    Remove the repeated URLs and the near-real-time (NRT) URLs of the files
    whose final version is also in the list.

    The NRT and the final file of an observation have the same name,
    so the final file replaces the NRT file once LMSAL publishes it.

    Parameters
    ----------
    urls
        URLs of IRIS files, final and NRT.
    """
    urls = list(dict.fromkeys(urls))
    final = {url.split("/")[~0] for url in urls if not _is_nrt(url)}
    return [url for url in urls if not (_is_nrt(url) and url.split("/")[~0] in final)]


def decompress(
    archives: Sequence[pathlib.Path],
    directory: None | pathlib.Path = None,
    overwrite: bool = False,
) -> list[pathlib.Path]:
    """
    Decompress a list of ``.tar.gz`` files.

    Each ``.tar.gz`` file is decompressed and the ``.fits`` files within the
    archive are appended to the returned list.

    Parameters
    ----------
    archives
        A list of ``.tar.gz`` files to decompress.
    directory
        A filesystem directory to place the decompressed results.
        If :obj:`None`, the directory of the ``.tar.gz`` archive will be used.
    overwrite
        If the file already exists, it will be overwritten.

    Examples
    --------
    Download the most last "A1: QS monitoring" spectrograph file in 2023 and
    decompress it into a list of ``.fits`` files.

    .. jupyter-execute::

        import astropy.time
        import iris

        # Find the URL of the .tar.gz archive
        urls = iris.data.urls_hek(
            time_start=astropy.time.Time("2023-01-01T00:00"),
            time_stop=astropy.time.Time("2024-01-01T00:00"),
            description="A1: QS monitoring",
            limit=1,
            sji=False,
        )

        # Download the .tar.gz archive
        archives = iris.data.download(urls)

        # Decompress the .tar.gz archive into a list of fits files
        iris.data.decompress(archives)
    """

    result = []

    for archive in archives:

        parent = archive.parent if directory is None else directory

        destination = parent / pathlib.Path(archive.stem).stem

        if overwrite or not destination.exists():
            shutil.unpack_archive(archive, extract_dir=destination, filter="data")

        files = sorted(destination.rglob("*.fits"))
        result = result + files

    return result
