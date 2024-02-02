"""
Utilities for finding and downloading IRIS data
"""

import astropy.time

__all__ = [
    "query_hek",
]


def query_hek(
    time_start: astropy.time.Time | None = None,
    time_stop: astropy.time.Time | None = None,
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
        The start time of the search period
    time_stop
        the end time of the search period
    description
        the description of the observation
    obs_id
        the OBSID of the observation, a number which describes the size, cadence,
        etc. of the observation
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
