import numpy as np
import astropy.units as u
import named_arrays as na

__all__ = [
    "average",
    "model_background",
    "model_spectral_line",
    "model_total",
    "fit",
    "subtract_spectral_line",
    "smooth",
    "estimate",
]


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
            obs=obs,
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
        time=obs.inputs.time.ndarray.jd.mean(),
        wavelength=obs.inputs.wavelength.mean(axis),
        position=obs.inputs.position.mean(axis),
    )
    obs.outputs = np.nanmedian(obs.outputs, axis=axis)
    return obs


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
    :math:`v_0` is the shift,
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
    the gradient descent method.

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

    Examples
    --------

    Fit a spectrograph image and display the average actual line profile
    compared to the average fitted line profile.

    .. jupyter-execute::

        import numpy as np
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

        # Save the time and raster axes
        axis = (obs.axis_time, obs.axis_detector_x)

        # Compute the average along the time and raster axes
        avg = iris.sg.background.average(
            obs=obs,
            axis=axis,
        )

        # Ignore line profiles that are mostly NaN
        where_crop = np.isfinite(avg.outputs).mean(avg.axis_wavelength) > 0.7
        data = avg.outputs[where_crop]

        # Calculate the mean rest wavelength of the
        # brightest spectral line
        wavelength_center = avg.wavelength_center.ndarray.mean()

        # Convert wavelength to velocity units
        velocity = avg.inputs.wavelength.to(
            unit=u.km / u.s,
            equivalencies=u.doppler_optical(wavelength_center),
        )
        velocity = velocity[where_crop]

        # Fit the data within +/- 150 km/s of line center
        where = np.abs(velocity) < 150 * u.km / u.s
        parameters = iris.sg.background.fit(
            data=data,
            velocity=velocity,
            axis_wavelength=obs.axis_wavelength,
            where=where,
        )

        # Evaluate the model with the best-fit parameters
        data_fit = iris.sg.background.model_total(
            velocity=velocity,
            amplitude=parameters.components["amplitude"],
            shift=parameters.components["shift"],
            width=parameters.components["width"],
            kappa=parameters.components["kappa"],
            bias=parameters.components["bias"],
            slope=parameters.components["slope"],
        )

        # Plot the average data and model
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                velocity.mean(obs.axis_detector_y),
                data.mean(obs.axis_detector_y),
                label="data",
            )
            na.plt.plot(
                velocity.mean(obs.axis_detector_y),
                data_fit.mean(obs.axis_detector_y),
                label="fit",
            )
            na.plt.plot(
                velocity.mean(obs.axis_detector_y),
                (data - data_fit).mean(obs.axis_detector_y),
                label="difference"
            )
            ax.set_xlim(-200, 200)
            ax.set_xlabel(f"Doppler velocity ({ax.get_xlabel()})")
            ax.set_ylabel(f"mean spectral radiance ({ax.get_ylabel()})")
            ax.legend();
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
    )
    return result


def subtract_spectral_line(
    obs: na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray],
    axis_wavelength: str,
) -> na.FunctionArray[na.SpectralPositionalVectorArray, na.ScalarArray]:
    """
    Fit `obs` using :func:`fit` and subtract the :func:`model_spectral_line`
    component.

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
                vmin=-5,
                vmax=+5,
            )
            ax.set_xlabel(f"wavelength ({ax.get_xlabel()})")
            ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")
            fig.colorbar(
                mappable=img.ndarray.item(),
                ax=ax,
                label=f"average spectral radiance ({obs.outputs.unit})",
            )
    """

    where_crop = np.isfinite(obs.outputs).mean(obs.axis_wavelength) > 0.7

    velocity = obs.inputs.wavelength[where_crop]

    where = np.abs(velocity) < 150 * u.km / u.s

    data = obs.outputs[where_crop]

    parameters = fit(
        data=data,
        velocity=velocity,
        axis_wavelength=axis_wavelength,
        where=where,
    )

    model_fit_line = model_spectral_line(
        velocity=velocity,
        amplitude=parameters.components["amplitude"],
        shift=parameters.components["shift"],
        width=parameters.components["width"],
        kappa=parameters.components["kappa"],
    )

    obs = obs.copy_shallow()
    obs.outputs[where_crop] = data - model_fit_line

    return obs


def smooth(
    obs: na.FunctionArray,
    axis_wavelength: str,
    axis_detector_y: str,
) -> na.FunctionArray:
    """
    Smooth the given observation using
    :func:`named_arrays.ndfilters.trimmed_mean_filter`.

    Parameters
    ----------
    obs
        The observation to be smoothed.
    axis_wavelength
        The logical axis corresponding to increasing wavelength.
    axis_detector_y
        The logical axis corresponding to increasing position along the slit.

    Examples
    --------

    Compute a smoothed version of an average spectrograph observation.

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
        bg = iris.sg.background.smooth(
            obs=avg,
            axis_wavelength=obs.axis_wavelength,
            axis_detector_y=obs.axis_detector_y,
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
    obs.outputs = na.ndfilters.trimmed_mean_filter(
        array=obs.outputs,
        size={axis_wavelength: 21},
        where=np.isfinite(obs.outputs),
        mode="truncate",
        proportion=0.2,
    )
    obs.outputs = na.ndfilters.trimmed_mean_filter(
        array=obs.outputs,
        size={axis_detector_y: 21},
        where=np.isfinite(obs.outputs),
        mode="truncate",
        proportion=0.2,
    )
    return obs


def estimate(
    obs: na.FunctionArray,
    axis_time: str,
    axis_wavelength: str,
    axis_detector_x: str,
    axis_detector_y: str,
) -> na.FunctionArray:
    """
    Estimate the background from a given spectrograph observation.

    This function applies :func:`average`, :func:`subtract_spectral_line`,
    and :func:`smooth` in succession to estimate the background.

    Parameters
    ----------
    obs
        The observation to estimate the background from.
    axis_time
        The logical axis corresponding to changing raster number.
    axis_wavelength
        The logical axis corresponding to changing wavelength.
    axis_detector_x
        The logical axis corresponding the changing position perpendicular to
        the slit.
    axis_detector_y
        The logical axis corresponding to changing position along the slit.

    Examples
    --------

    Estimate the background from a spectrograph observation

    .. jupyter-execute::

        import numpy as np
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

        # Subtract the spectral line from the average
        bg = iris.sg.background.estimate(
            obs=obs,
            axis_time=obs.axis_time,
            axis_wavelength=obs.axis_wavelength,
            axis_detector_x=obs.axis_detector_x,
            axis_detector_y=obs.axis_detector_y,
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

    |

    Subtract the background from the observation and compare to the original.

    .. jupyter-execute::

        # Remove background from spectrograph observation
        obs_nobg = obs.copy_shallow()
        obs_nobg.outputs = obs.outputs - bg.outputs

        # Select the first raster to plot
        index = {obs.axis_time: 0}

        # Plot the original compared to the background-subtracted
        # observation.
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=2,
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            mappable = plt.cm.ScalarMappable(
                norm=plt.Normalize(vmin=0, vmax=5),
            )
            ax[0].set_title("original")
            na.plt.pcolormesh(
                obs.inputs.position.x[index].mean(obs.axis_wavelength),
                obs.inputs.position.y[index].mean(obs.axis_wavelength),
                C=np.nanmean(obs.outputs.value[index], axis=obs.axis_wavelength),
                ax=ax[0],
                norm=mappable.norm,
                cmap=mappable.cmap,
            )
            ax[1].set_title("corrected")
            na.plt.pcolormesh(
                obs_nobg.inputs.position.x[index].mean(obs.axis_wavelength),
                obs_nobg.inputs.position.y[index].mean(obs.axis_wavelength),
                C=np.nanmean(obs_nobg.outputs.value[index], axis=obs.axis_wavelength),
                ax=ax[1],
                norm=mappable.norm,
                cmap=mappable.cmap,
            )
            ax[0].set_xlabel(f"helioprojective $x$ ({ax[0].get_xlabel()})")
            ax[0].set_ylabel(f"helioprojective $y$ ({ax[0].get_ylabel()})")
            fig.colorbar(
                mappable=mappable,
                ax=ax,
                label=f"mean spectral radiance ({obs.outputs.unit:latex_inline})",
            )

    |

    Plot the original and corrected median spectral line profiles

    .. jupyter-execute::

        # Define the axes to average over
        axis_txy = (
            obs.axis_time,
            obs.axis_detector_x,
            obs.axis_detector_y,
        )

        # Plot the result
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots()
            na.plt.plot(
                obs.inputs.wavelength.mean(axis=axis_txy),
                np.nanmedian(obs.outputs, axis=axis_txy),
                label="original",
            )
            na.plt.plot(
                obs_nobg.inputs.wavelength.mean(axis=axis_txy),
                np.nanmedian(obs_nobg.outputs, axis=axis_txy),
                label="corrected",
            )
            ax.set_xlabel(f"Doppler velocity ({ax.get_xlabel()})")
            ax.set_ylabel(f"median spectral radiance ({ax.get_ylabel()})")
            ax.set_ylim(top=10)
            ax.legend()
    """
    avg = average(
        obs=obs,
        axis=(axis_time, axis_detector_x),
    )
    bg_0 = subtract_spectral_line(
        obs=avg,
        axis_wavelength=axis_wavelength,
    )
    bg_1 = smooth(
        obs=bg_0,
        axis_wavelength=axis_wavelength,
        axis_detector_y=axis_detector_y,
    )
    return bg_1
