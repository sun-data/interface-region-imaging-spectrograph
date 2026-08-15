import pytest
import numpy as np
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

    # NaN where the response file has nothing to say, which a grid reaching
    # far outside the FUV and NUV bands is expected to include, so the sum
    # has to be taken over the wavelengths with a calibration.
    assert np.nansum(result) > 0 * u.cm**2


def test_effective_area_outside_band():
    """
    No calibration outside the nominal bands, and an answer inside them.

    `irispy-lmsal` 0.8.1 returns NaN for wavelengths its response file does
    not cover, where it used to return a number interpolated from nothing.
    This is the behavior `radiance` relies on: dividing by that number
    amplified whatever was in the out-of-band pixels, while dividing by NaN
    marks them as uncalibrated.
    """
    time = astropy.time.Time("2014-01-01")

    wavelength = na.ScalarArray(
        ndarray=[1000, 1394, 2796, 5000] * u.AA,
        axes="w",
    )

    result = iris.sg.effective_area(time, wavelength)

    finite = np.isfinite(result).ndarray
    assert not finite[0]
    assert finite[1]
    assert finite[2]
    assert not finite[3]
