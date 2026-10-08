import pytest
import numpy as np
import astropy.time
import iris


@pytest.mark.parametrize(
    argnames="time",
    argvalues=[
        "2021-09-23T06:20",
    ],
)
def test_open(
    time: str | astropy.time.Time,
):
    """Without a stop time, every frame of the observation is loaded."""
    result = iris.sji.open(time)

    assert isinstance(result, iris.sji.SlitJawObservation)

    t = result.inputs.time.ndarray
    assert result.shape[result.axis_time] == 80
    assert np.min(t) < astropy.time.Time(time)
    assert np.max(t) > astropy.time.Time(time)


@pytest.mark.parametrize(
    argnames="time,time_stop",
    argvalues=[
        ("2021-09-23T06:20", "2021-09-23T06:22"),
    ],
)
def test_open_time_stop(
    time: str,
    time_stop: str,
):
    """With a stop time, only the frames which began in the range are loaded."""
    result = iris.sji.open(time, time_stop)

    t = result.inputs.time.ndarray
    assert result.shape[result.axis_time] > 0
    assert np.all(t >= astropy.time.Time(time))
    assert np.all(t < astropy.time.Time(time_stop))
