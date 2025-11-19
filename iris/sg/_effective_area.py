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

    Reproduce Figure 22 of :cite:t:`Wulser2018`,
    the effective area of the FUV spectrograph channel on March 1st, 2015.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.time
        import astropy.visualization
        import named_arrays as na
        import iris

        # Define the time at which to evaluate the effective area
        time = astropy.time.Time("2015-03-01")

        # Define the wavelength grid
        wavelength = na.linspace(
            start=1320 * u.AA,
            stop=1420 * u.AA,
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

    |

    Reproduce Figure 23 of :cite:t:`Wulser2018`,
    the effective area of the NUV spectrograph channel on October 20th, 2014.

    .. jupyter-execute::

        # Define the time at which to evaluate the effective area
        time = astropy.time.Time("2014-10-20")

        # Define a wavelength grid
        wavelength = na.linspace(
            start=2780 * u.AA,
            stop=2840 * u.AA,
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
    if time.size != 1:  # pragma: nocover
        raise ValueError(f"arrays of times are not supported, got {time.shape=}")

    wavelength = na.as_named_array(wavelength)

    shape = wavelength.shape

    wavelength = wavelength.ndarray

    response = irispy.utils.response.get_latest_response(time)

    area_fuv = irispy.utils.response.get_interpolated_effective_area(
        iris_response=response,
        detector_type="FUV",
        obs_wavelength=wavelength,
    )

    area_nuv = irispy.utils.response.get_interpolated_effective_area(
        iris_response=response,
        detector_type="NUV",
        obs_wavelength=wavelength,
    )

    area = np.where(
        wavelength > 2000 * u.AA,
        area_nuv,
        area_fuv,
    )

    area = na.ScalarArray(
        ndarray=area,
        axes=tuple(shape),
    )

    return area
