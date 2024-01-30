from __future__ import annotations
import pytest
import astropy.units as u
import astropy.time
import iris


@pytest.mark.parametrize(
    argnames="time_start",
    argvalues=[
        "2024-01-29T00:00:00",
    ],
)
@pytest.mark.parametrize(
    argnames="time_stop",
    argvalues=[
        "2024-01-29T04:00:00",
    ],
)
@pytest.mark.parametrize(
    argnames="timedelta_raster",
    argvalues=[
        1 * u.hr,
    ],
)
@pytest.mark.parametrize(
    argnames="timedelta_slew",
    argvalues=[
        20 * u.min,
    ],
)
def test_num_repeats(
    time_start: str | astropy.time.Time,
    time_stop: str | astropy.time.Time,
    timedelta_raster: u.Quantity,
    timedelta_slew: u.Quantity,
):
    result = iris.planning.num_repeats(
        time_start=time_start,
        time_stop=time_stop,
        timedelta_raster=timedelta_raster,
        timedelta_slew=timedelta_slew,
    )
    assert isinstance(result, float)
    assert result > 0
