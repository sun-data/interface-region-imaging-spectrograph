import pytest
import astropy.time
import iris


@pytest.mark.parametrize(
    argnames="time",
    argvalues=[
        "2021-09-23T06:13",
    ],
)
def test_open(
    time: str | astropy.time.Time,
):
    result = iris.sg.open(time)

    assert isinstance(result, iris.sg.SpectrographObservation)
