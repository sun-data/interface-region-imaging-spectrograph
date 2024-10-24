import pytest
import astropy.units as u
import named_arrays as na
import iris


@pytest.mark.parametrize(
    argnames="wavelength",
    argvalues=[
        1330 * u.AA,
        na.linspace(1300, 1400, axis="w", num=11) * u.AA,
    ],
)
def test_effective_area(
    wavelength: u.Quantity | na.AbstractScalar,
):
    result = iris.sg.effective_area(wavelength)

    assert result.sum() > 0 * u.cm**2
