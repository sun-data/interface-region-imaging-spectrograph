from __future__ import annotations
import pytest
import astropy.time
import iris


@pytest.mark.parametrize("time_start", [None])
@pytest.mark.parametrize("time_stop", [None])
@pytest.mark.parametrize("description", [""])
@pytest.mark.parametrize("obs_id", [None, 3882010194])
@pytest.mark.parametrize("limit", [5])
def test_query_hek(
    time_start: None | astropy.time.Time,
    time_stop: None | astropy.time.Time,
    description: str,
    obs_id: None | str,
    limit: int,
):
    result = iris.data.query_hek(
        time_start=time_start,
        time_stop=time_stop,
        description=description,
        obs_id=obs_id,
        limit=limit,
    )
    assert isinstance(result, str)
