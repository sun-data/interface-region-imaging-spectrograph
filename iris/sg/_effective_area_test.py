import pytest
import astropy.units as u
import astropy.time
import named_arrays as na
import iris


@pytest.mark.parametrize(
    argnames="time",
    argvalues=[
        astropy.time.Time("2014-01-01"),
    ],
)
@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        1335 * u.AA,
        na.linspace(1300, 3000, axis="w", num=11) * u.AA,
    ],
)
def test_effective_area(
    time: astropy.time.Time | na.AbstractScalarArray,
    wavelength: u.Quantity | na.AbstractScalarArray,
):
    result = iris.sg.effective_area(
        time=time,
        wavelength=wavelength,
    )

    assert result.sum() > 0 * u.cm**2
