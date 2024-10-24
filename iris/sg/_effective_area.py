import scipy.io
import astropy.units as u
import named_arrays as na
import iris

__all__ = [
    "effective_area",
]


def effective_area(wavelength: u.Quantity | na.AbstractScalar) -> na.AbstractScalar:
    """
    Load the effective area of the spectrograph.

    Currently only Version 1 is implemented.

    Parameters
    ----------
    wavelength
        The wavelength of the incident light at which to evaluate the effective
        area.

    Examples
    --------

    Plot the effective area of the spectrograph as a function of wavelength.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import iris

        # Define a wavelength grid
        wavelength = na.linspace(
            start=1250 * u.AA,
            stop=3000 * u.AA,
            axis="wavelength",
            num=1001,
        )

        # Compute the effective area
        area = iris.sg.effective_area(wavelength)

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

    files = iris.response.files()

    file_v1 = files[0]

    struct_v1 = scipy.io.readsav(file_v1)["p0"]

    wavelength_v1 = struct_v1["LAMBDA"][0] * u.nm
    area_v1 = struct_v1["AREA_SG"][0] * u.cm**2

    axis = "_dummy"
    axes = ("channel", axis)

    axis = "_dummy"

    wavelength_v1 = na.ScalarArray(wavelength_v1, axes=axis)
    area_v1 = na.ScalarArray(area_v1, axes=axes).sum("channel")

    return na.interp(
        x=wavelength,
        xp=wavelength_v1,
        fp=area_v1,
        axis=axis,
    )
