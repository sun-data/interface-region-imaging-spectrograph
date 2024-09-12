"""
Utilities for estimating the stray light background of FUV spectrograph images.
"""

import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "model_background",
    "model_spectral_line",
    "model_total",
    "fit",
    "average",
    "subtract_spectral_line",
]


def model_background(
    velocity: na.AbstractScalar,
    bias: u.Quantity | na.AbstractScalar,
    slope: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    A simple linear model of the background in the proximity of a spectral
    line, given by

    .. math::

        y(v) = m v + b

    where :math:`m` is the slope,
    :math:`v` is the velocity,
    and :math:`b` is the bias.

    Parameters
    ----------
    velocity
        The test points at which to evaluate the model background.
    bias
        The :math:`y`-intercept of the model.
    slope
        The linear slope of the model.
    """
    return slope * velocity + bias


def model_spectral_line(
    velocity: na.AbstractScalar,
    amplitude: na.AbstractScalar,
    shift: na.AbstractScalar,
    width: na.AbstractScalar,
    kappa: na.AbstractScalar,
) -> na.AbstractScalar:
    r"""
    A simple model of a spectral line profile.

    This function uses a kappa distribution,

    .. math::

        y(v) = A \left( 1 + \frac{(v - v_0)^2}{\kappa \sigma^2} \right)^{-\kappa - 1},

    to model the spectral line profile,
    where :math:`A` is the amplitude,
    :math:`v` is the velocity,
    :math:`v` is the shift,
    :math:`\sigma` is the width,
    and :math:`\kappa` is a parameter which controls the thickness of the tails.

    Parameters
    ----------
    velocity
        The test points at which to evaluate the model background.
    amplitude
        The height of the spectral line.
    shift
        The center of the spectral line.
    width
        The width of the spectral line.
    kappa
        The fatness of the tails of the spectral line.
    """
    x_squared = np.square(velocity - shift)
    width_squared = np.square(width)
    y = amplitude * (1 + x_squared / (kappa * width_squared)) ** (-kappa - 1)
    return y


def model_total(
    velocity: na.AbstractScalar,
    amplitude: na.AbstractScalar,
    shift: na.AbstractScalar,
    width: na.AbstractScalar,
    kappa: na.AbstractScalar,
    bias: u.Quantity | na.AbstractScalar,
    slope: u.Quantity | na.AbstractScalar,
) -> na.AbstractScalar:
    """
    The sum of :func:`model_background` and :func:`model_spectral_line`.

    Parameters
    ----------
    velocity
        The test points at which to evaluate the model background.
    amplitude
        The height of the spectral line.
    shift
        The center of the spectral line.
    width
        The width of the spectral line.
    kappa
        The fatness of the tails of the spectral line.
    bias
        The :math:`y`-intercept of the model.
    slope
        The linear slope of the model.
    """
    line = model_spectral_line(
        velocity=velocity,
        amplitude=amplitude,
        shift=shift,
        width=width,
        kappa=kappa,
    )
    bg = model_background(
        velocity=velocity,
        bias=bias,
        slope=slope,
    )
    return line + bg


def _objective(
    data: na.AbstractScalar,
    velocity: na.AbstractScalar,
    axis_wavelength: str,
    amplitude: na.AbstractScalar,
    shift: na.AbstractScalar,
    width: na.AbstractScalar,
    kappa: na.AbstractScalar,
    bias: u.Quantity | na.AbstractScalar,
    slope: u.Quantity | na.AbstractScalar,
    where: bool | na.AbstractScalar = True,
) -> na.AbstractScalar:
    """
    The function which is minimized during background fitting

    Parameters
    ----------
    data
        The observation to be fitted.
    velocity
        The Doppler velocity for each point in the observation.
    axis_wavelength
        The logical axis corresponding to increasing wavelength (velocity).
    amplitude
        The height of the spectral line.
    shift
        The center of the spectral line.
    width
        The width of the spectral line.
    kappa
        The fatness of the tails of the spectral line.
    bias
        The :math:`y`-intercept of the model.
    slope
        The linear slope of the model.
    where
        The points in the observation to consider when fitting.
    """

    model = model_total(
        velocity=velocity,
        amplitude=amplitude,
        shift=shift,
        width=width,
        kappa=kappa,
        bias=bias,
        slope=slope,
    )

    diff = data - model

    result = np.sqrt(np.sum(np.square(diff), axis=axis_wavelength, where=where))

    return result


def fit(
    data: na.AbstractScalar,
    velocity: na.AbstractScalar,
    axis_wavelength: str,
    where: bool | na.AbstractScalar = True,
) -> na.CartesianNdVectorArray:
    """
    Compute the parameters of :func:`model_total` which best fit `data` using
    :func:`named_arrays.optimize.minimize_gradient_descent`.

    Parameters
    ----------
    data
        The observation to be fitted.
    velocity
        The Doppler velocity for each point in the observation
    axis_wavelength
        The logical axis corresponding to increasing wavelength (velocity).
    where
        The points in the observation to consider when fitting.
    """

    def function(x: na.CartesianNdVectorArray):
        x = x.components

        return _objective(
            data=data,
            velocity=velocity,
            axis_wavelength=axis_wavelength,
            amplitude=x["amplitude"],
            shift=x["shift"],
            width=x["width"],
            kappa=x["kappa"],
            bias=x["bias"],
            slope=x["slope"],
            where=where,
        )

    data_nan = np.where(where, data, np.nan)

    guess = dict(
        # amplitude=np.percentile(img, 99, axis=axis_wavelength),
        amplitude=np.nanpercentile(
            a=data_nan,
            q=99.9,
            axis=axis_wavelength,
        ),
        shift=0 * u.km / u.s,
        width=20 * u.km / u.s,
        kappa=1.1,
        bias=0 * u.DN,
        slope=0 * u.DN / (u.km / u.s),
    )
    guess = na.CartesianNdVectorArray.from_components(guess)

    def callback(i, x, f, c):
        if i % 10 == 0:
            print(f"{i=}\n{x.mean()=}\n{f.mean()=}\n{(~c).sum()=}")

    result = na.optimize.minimum_gradient_descent(
        function=function,
        guess=guess,
        step_size=na.CartesianNdVectorArray.from_components(
            components=dict(
                amplitude=0.01 * u.DN,
                shift=0.01 * (u.km / u.s) ** 2 / u.DN,
                width=0.01 * (u.km / u.s) ** 2 / u.DN,
                bias=0.001 * u.DN,
                kappa=0.0001 / u.DN,
                slope=1e-6 * u.DN / (u.km / u.s) ** 2,
            ),
        ),
        momentum=0.9,
        min_gradient=na.CartesianNdVectorArray.from_components(
            components=dict(
                amplitude=0.5,
                shift=0.5 * u.DN / (u.km / u.s),
                width=0.5 * u.DN / (u.km / u.s),
                kappa=0.5 * u.DN,
                bias=1,
                slope=110 * (u.km / u.s),
            )
        ),
        max_iterations=10000,
        callback=callback,
    )
    return result


def average(
    obs: na.FunctionArray,
    axis: str | tuple[str, ...],
) -> na.FunctionArray:
    """
    Compute the average along the given axis using :func:`numpy.nanmedian`

    Parameters
    ----------
    obs
        The observation to be averaged.
    axis
        The logical axis along which to take the average.

    Examples
    --------

    Download an IRIS spectrograph observation and compute the average along
    the raster average.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.visualization
        import named_arrays as na
        import iris

        # Load a spectrograph observation
        obs = iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T07:00"),
        )

        # Save the time and raster axes
        axis = (obs.axis_time, obs.axis_detector_x)

        # Compute the average along the time and raster axes
        avg = iris.sg.background.average(
            data=obs.outputs,
            axis=axis,
        )

        # Plot the result
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            img = na.plt.pcolormesh(
                avg.inputs.wavelength,
                avg.inputs.position.y,
                C=avg.outputs.value,
            )
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")
            fig.colorbar(
                mappable=img.ndarray.item(),
                ax=ax,
                label=f"average spectral radiance ({obs.outputs.unit})",
            )
    """
    obs = obs.copy_shallow()
    obs.inputs = na.TemporalSpectralPositionalVectorArray(
        time=obs.inputs.time.ndarray.mean(),
        wavelength=obs.inputs.wavelength.mean(axis),
        position=obs.inputs.position.mean(axis),
    )
    obs.outputs = np.nanmedian(obs.outputs, axis=axis)
    return obs


def subtract_spectral_line(
    obs: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    axis_wavelength: str,
) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
    """
    Fit `obs` using :func:`fit` and subtract the component from
    :func:`model_spectral_line`.

    Parameters
    ----------
    obs
        The observation to remove the spectral line from.
    axis_wavelength
        The logical axis corresponding to increasing wavelength.

    Examples
    --------

    Subtract the spectral line from an averaged spectrograph observation.

    .. jupyter-execute::

        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import iris

        # Load a spectrograph observation
        obs = iris.sg.SpectrographObservation.from_time_range(
            time_start=astropy.time.Time("2021-09-23T06:00"),
            time_stop=astropy.time.Time("2021-09-23T07:00"),
        )

        # Calculate the mean rest wavelength of the
        # brightest spectral line
        wavelength_center = obs.wavelength_center.ndarray.mean()

        # Convert wavelength to velocity units
        obs.inputs = obs.inputs.explicit
        obs.inputs.wavelength = obs.inputs.wavelength.to(
            unit=u.km / u.s,
            equivalencies=u.doppler_optical(wavelength_center),
        )

        # Save the time and raster axes
        axis = (obs.axis_time, obs.axis_detector_x)

        # Compute the average along the time and raster axes
        avg = iris.sg.background.average(
            obs=obs,
            axis=axis,
        )

        # Subtract the spectral line from the average
        bg = iris.sg.background.subtract_spectral_line(
            obs=avg,
            axis_wavelength=obs.axis_wavelength,
        )

        # Plot the result
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            img = na.plt.pcolormesh(
                bg.inputs.wavelength,
                bg.inputs.position.y,
                C=bg.outputs.value,
            )
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")
            fig.colorbar(
                mappable=img.ndarray.item(),
                ax=ax,
                label=f"average spectral radiance ({obs.outputs.unit})",
            )
    """

    where_crop = np.isfinite(obs.outputs).mean(obs.axis_wavelength) > .7

    velocity = obs.inputs.wavelength[where_crop]

    where = np.abs(velocity) < 150 * u.km / u.s

    data = obs.outputs[where_crop]

    parameters = fit(
        data=data,
        velocity=velocity,
        axis_wavelength=axis_wavelength,
        where=where
    )

    model_fit_line = model_spectral_line(
        velocity=velocity,
        amplitude=parameters.components["amplitude"],
        shift=parameters.components["shift"],
        width=parameters.components["width"],
        kappa=parameters.components["kappa"]
    )

    obs = obs.copy_shallow()
    obs.outputs[where_crop] = data - model_fit_line

    return obs
