import numpy as np
import astropy.units as u
import astropy.time
import irispy.utils.response
import named_arrays as na

__all__ = [
    "dn_to_photons",
    "width_slit",
    "effective_area",
]


width_slit = 1 / 3 * u.arcsec
"""
The angular subtent of the spectrographic slit.
"""


def dn_to_photons(
    wavelength: u.Quantity | na.AbstractScalar,
) -> u.Quantity:
    """
    Return the conversion factor between data numbers (DN) and photons
    given by :cite:t:`Wulser2018`.

    Parameters
    ----------
    wavelength
        The wavelength at which to compute the conversion factor.
    """
    return np.where(
        wavelength > 2000 * u.AA,
        18 * u.ph / u.DN,
        5 * u.ph / u.DN,
    )


def effective_area(
    time: astropy.time.Time | na.AbstractScalarArray,
    wavelength: u.Quantity | na.AbstractScalarArray,
) -> na.AbstractScalar:
    """
    Load the effective area of the spectrograph.

    This function uses :func:`irispy.utils.response.get_interpolated_effective_area`
    to find the effective area for a given time and wavelength.

    Parameters
    ----------
    time
        The time at which to calculate the effective area.
        Must be only a single time, an array of times is not supported.
    wavelength
        The wavelength of the incident light at which to evaluate the effective
        area.

    Examples
    --------

    Plot the effective area of the spectrograph as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.time
        import astropy.visualization
        import named_arrays as na
        import iris

        # Define a time at which to evalutate the effective area
        time = astropy.time.Time("2014-01-01")

        # Define a wavelength grid
        wavelength = na.linspace(
            start=1250 * u.AA,
            stop=3000 * u.AA,
            axis="wavelength",
            num=1001,
        )

        # Compute the effective area
        area = iris.sg.effective_area(time, wavelength)

        # Plot the effective area as a function of wavelength
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                wavelength,
                area,
            )
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel(f"effective area ({ax.get_ylabel()})")

    """
    shape = na.shape_broadcasted(time, wavelength)

    time = na.as_named_array(time)
    wavelength = na.as_named_array(wavelength)

    time = time.ndarray_aligned(shape)
    wavelength = wavelength.ndarray_aligned(shape)

    response = irispy.utils.response.get_latest_response(time)

    if np.min(wavelength) > 2000 * u.AA:
        detector_type = "NUV"
    else:
        detector_type = "FUV"

    area = irispy.utils.response.get_interpolated_effective_area(
        iris_response=response,
        detector_type=detector_type,
        obs_wavelength=wavelength,
    )

    area = na.ScalarArray(
        ndarray=area,
        axes=tuple(shape),
    )

    return area
