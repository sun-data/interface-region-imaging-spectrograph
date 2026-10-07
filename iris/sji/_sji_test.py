import pytest
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
    result = iris.sji.open(time)

    assert isinstance(result, iris.sji.SlitJawObservation)
