from typing import cast
import dataclasses
import numpy as np
import astropy.units as u
import named_arrays as na
from .._spectrograph import SpectrographObservation
from ..models import kappa_linear

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
    obs: SpectrographObservation,
    axis: str | tuple[str, ...],
) -> SpectrographObservation:
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
    shape = obs.inputs.shape
    obs.inputs = obs.inputs.explicit.replace(
        time=obs.inputs.time.ndarray.jd.mean(),
        wavelength=obs.inputs.wavelength.broadcast_to(shape).mean(axis),
        position=obs.inputs.position.broadcast_to(shape).mean(axis),
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


def fit(
    data: na.AbstractScalar,
    velocity: na.AbstractScalar,
    axis_wavelength: str,
    where: bool | na.AbstractScalar = True,
) -> na.CartesianNdVectorArray:
    """
    Compute the parameters of :func:`model_total` which best fit `data`,
    using the Levenberg-Marquardt method of :func:`ndoptimize.least_squares`.

    The compiled form of :func:`model_total`,
    :func:`iris.sg.models.kappa_linear`, is fit to each spectrum along
    `axis_wavelength` with :class:`named_arrays.ModelFitFunctionArray`,
    giving every point the same weight.

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

    Returns
    -------
    :
        The best-fit parameters, a vector whose components are the arguments
        of :func:`model_total` named ``amplitude``, ``shift``, ``width``,
        ``kappa``, ``bias``, and ``slope``.

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

        # Convert wavelength to velocity units
        velocity = avg.inputs.velocity.cell_centers()
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

    # the model sees each parameter in the unit of its guess, so the guess is
    # in the units of the velocity and of the data
    unit_velocity = na.unit_normalized(velocity)
    unit_data = na.unit_normalized(data)

    guess = na.CartesianNdVectorArray(
        dict(
            amplitude=np.nanpercentile(
                a=np.where(where, data, np.nan),
                q=99.9,
                axis=axis_wavelength,
            ),
            center=0 * unit_velocity,
            width=(20 * u.km / u.s).to(unit_velocity),
            kappa=1.1,
            slope=0 * unit_data / unit_velocity,
            intercept=0 * unit_data,
        )
    )

    result = na.ModelFitFunctionArray(
        inputs=velocity,
        outputs=data,
        model=kappa_linear,
        guess=guess,
        variance=1 * unit_data**2,
        axis_model=axis_wavelength,
        where_model=where,
        lower=na.CartesianNdVectorArray(
            dict(width=1 * u.km / u.s, kappa=0.1),
        ),
        upper=na.CartesianNdVectorArray(
            dict(kappa=1000),
        ),
    )

    p = result.parameters.components
    return na.CartesianNdVectorArray(
        dict(
            amplitude=p["amplitude"],
            shift=p["center"],
            width=p["width"],
            kappa=p["kappa"],
            bias=p["intercept"],
            slope=p["slope"],
        )
    )


def subtract_spectral_line(
    obs: SpectrographObservation,
    axis_wavelength: str,
    wavelength_rest: None | u.Quantity | na.AbstractScalar = None,
) -> SpectrographObservation:
    """
    Fit `obs` using :func:`fit` and subtract the :func:`model_spectral_line`
    component.

    Each line is fit to the pixels within :math:`150 \\, \\text{km/s}` of its
    rest wavelength.

    Parameters
    ----------
    obs
        The observation to remove the spectral line from.
    axis_wavelength
        The logical axis corresponding to increasing wavelength.
    wavelength_rest
        The rest wavelengths of the spectral lines to remove.
        Several lines are given along an axis which `obs` does not have, or
        as a one-dimensional :class:`~astropy.units.Quantity`.
        If :obj:`None` (the default), the rest wavelength of `obs`, that of
        the line which names its spectral window, so that any other line in
        the window is left in the result.

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

    inputs = obs.inputs
    if wavelength_rest is not None:
        if not isinstance(wavelength_rest, na.AbstractArray):
            # the lines of a plain quantity go along an axis of their own
            wavelength_rest = na.ScalarArray(
                ndarray=wavelength_rest,
                axes=("_line",) * np.ndim(wavelength_rest),
            )
        inputs = dataclasses.replace(inputs, wavelength_rest=wavelength_rest)

    # each line is fit on its own, along the axes of the lines
    axis_line = tuple(
        axis for axis in na.shape(inputs.wavelength_rest) if axis not in obs.shape
    )

    velocity = cast(na.AbstractScalar, inputs.velocity)
    velocity = velocity.cell_centers(
        axis=tuple(axis for axis in velocity.shape if axis not in axis_line),
    )
    velocity = velocity[where_crop]

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
    if axis_line:
        model_fit_line = model_fit_line.sum(axis=axis_line)

    obs = obs.copy_shallow()
    obs.outputs[where_crop] = data - model_fit_line

    return obs


def smooth(
    obs: SpectrographObservation,
    axis_wavelength: str,
    axis_detector_y: str,
) -> SpectrographObservation:
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
    obs: SpectrographObservation,
    axis_time: str,
    axis_wavelength: str,
    axis_detector_x: str,
    axis_detector_y: str,
    wavelength_rest: None | u.Quantity | na.AbstractScalar = None,
) -> SpectrographObservation:
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
    wavelength_rest
        The rest wavelengths of the spectral lines to remove from the
        background, see :func:`subtract_spectral_line`.
        If :obj:`None` (the default), only the line which names the spectral
        window is removed.

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
                obs.inputs.position.x[index],
                obs.inputs.position.y[index],
                C=np.nanmean(obs.outputs.value[index], axis=obs.axis_wavelength),
                ax=ax[0],
                norm=mappable.norm,
                cmap=mappable.cmap,
            )
            ax[1].set_title("corrected")
            na.plt.pcolormesh(
                obs_nobg.inputs.position.x[index],
                obs_nobg.inputs.position.y[index],
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
            na.plt.stairs(
                obs.inputs.wavelength.mean(obs.axis_time),
                np.nanmedian(obs.outputs, axis=axis_txy),
                label="original",
            )
            na.plt.stairs(
                obs_nobg.inputs.wavelength.mean(obs.axis_time),
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
        wavelength_rest=wavelength_rest,
    )
    bg_1 = smooth(
        obs=bg_0,
        axis_wavelength=axis_wavelength,
        axis_detector_y=axis_detector_y,
    )
    return bg_1
