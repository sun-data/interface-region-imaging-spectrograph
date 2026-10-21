from typing import Sequence, cast
import dataclasses
import warnings
import numpy as np
import astropy.units as u
import named_arrays as na
import ndoptimize
from ._spectrograph import SpectrographObservation

__all__ = [
    "fit_line",
]

_width = 20 * u.km / u.s
"""The width each fit starts at, unless it is given a guess."""

_kappa = 1.1
"""The :math:`\\kappa` each fit starts at, unless it is given a guess."""


def fit_line(
    obs: SpectrographObservation,
    model: ndoptimize.Model = ndoptimize.models.gaussian_linear,
    wavelength_rest: None | u.Quantity | na.AbstractScalar = None,
    velocity_max: u.Quantity | na.AbstractScalar = 150 * u.km / u.s,
    guess: None | na.AbstractVectorArray = None,
    variance: None | u.Quantity | na.AbstractScalar = None,
    where: bool | na.AbstractScalar = True,
    lower: None | na.AbstractVectorArray = None,
    upper: None | na.AbstractVectorArray = None,
    prior_mean: None | na.AbstractVectorArray = None,
    prior_sigma: None | na.AbstractVectorArray = None,
    correlation: None | Sequence[float] = None,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> na.ModelFitFunctionArray:
    r"""
    Fit a model of a spectral line to every spectrum of an IRIS spectrograph
    observation.

    Each spectrum, along :attr:`~iris.sg.SpectrographObservation.axis_wavelength`,
    is fit separately with the Levenberg-Marquardt method of
    :func:`ndoptimize.least_squares`, through
    :class:`named_arrays.ModelFitFunctionArray`, so a whole raster sequence is
    fit at once.
    The model sees the Doppler velocity of the center of each pixel,
    relative to `wavelength_rest`, and only the pixels within `velocity_max`
    of it are fit.

    Parameters
    ----------
    obs
        The observation to fit.
    model
        The model of the line, made with :func:`ndoptimize.model`, with one
        input, the velocity.
        The default is a Gaussian plus a straight line,
        :func:`ndoptimize.models.gaussian_linear`.
        :mod:`iris.sg.models` has models for IRIS lines, such as
        :func:`iris.sg.models.kappa_linear`.
    wavelength_rest
        The rest wavelength of the line, which the velocities are relative to.
        Several lines are given as a named array, along an axis which `obs`
        does not have, and are fit separately, so the results vary along that
        axis.
        If :obj:`None` (the default), the rest wavelength of the observation,
        that of the line which names its spectral window.
    velocity_max
        The largest velocity, either way, of the pixels which are fit.
    guess
        The initial guess for the parameters, a vector whose components are
        the parameters named in :attr:`ndoptimize.Model.parameters`.
        If :obj:`None` (the default), it is found from the data, for a model
        whose parameters are among ``amplitude``, ``center``, ``width``,
        ``kappa``, ``slope``, and ``intercept``: the median of the outer half
        of the fit as the intercept, and no slope; a width of
        :math:`20 \, \mathrm{km \, s^{-1}}` and a :math:`\kappa` of 1.1; the
        centroid of the inner half on the side of the intercept that most of
        it is on, above it for an emission line and below it for an
        absorption line, as the center; and the amplitude at which the line
        of the model, with that center, width, and :math:`\kappa`, has the
        area of the inner half relative to the intercept.
    variance
        The variance of the noise of each output.
        If :obj:`None` (the default), that of a single exposure in DN,
        :attr:`~iris.sg.SpectrographObservation.variance`, the read noise and
        the photon noise of the signal.
        The photon noise of the outputs is correlated with their noise, so a
        fit weighted by it would be biased low: the observation is fit twice,
        first with the photon noise of the outputs and then with that of the
        model of the first fit.
    where
        A boolean mask of the outputs to fit, of the pixels within
        `velocity_max` of `wavelength_rest`.
        Outputs which are NaN are left out of their fit.
    lower
        The lower bounds of the parameters, a vector with a component for each
        parameter that has one.
        If :obj:`None` (the default), the center is at least `-velocity_max`,
        the width at least half the spacing of the pixels in velocity, and
        :math:`\kappa` at least 0.1, for those of them that the model has.
        A line narrower than that can fall between two pixels, where nothing
        limits its amplitude.
    upper
        The upper bounds of the parameters, in the same form as `lower`.
        If :obj:`None` (the default), the center is at most `velocity_max`,
        the width at most half of `velocity_max`, and :math:`\kappa` at most
        1000.
        A line wider than that can take the place of a curve in the
        continuum.
    prior_mean
        The means of the Gaussian priors of the parameters, a vector with a
        component for each parameter that has a prior.
    prior_sigma
        The standard deviations of the Gaussian priors of the parameters, a
        vector with the same components as `prior_mean`.
    correlation
        The correlation of the noise of two outputs by their separation in
        wavelength, in pixels, starting with 1, as in
        :func:`ndoptimize.least_squares`.
        If :obj:`None` (the default), the noise is uncorrelated.
    max_iterations
        The maximum number of steps of each fit.
    tolerance
        The convergence tolerance on the Newton decrement, as in
        :func:`ndoptimize.least_squares`.

    Returns
    -------
    :
        The fit of the pixels within `velocity_max` of the line, whose inputs
        are their velocities and whose outputs are those of `obs`.
        Its parameters, their covariance, and the other results of
        :class:`named_arrays.ModelFitFunctionArray` vary along the axes of
        `obs` other than that of wavelength, and calling it evaluates the
        best-fit model at other velocities.
        Spectra with nothing to fit, such as the rows outside the slit, are
        not converged, so check
        :attr:`~named_arrays.ModelFitFunctionArray.converged`.
        A fit which ends with a parameter at one of its bounds is converged,
        so compare the parameters with the bounds too, and the uncertainty
        of a parameter which the data do not constrain, like the amplitude
        of a line centered where the spectrum is missing, is infinite or NaN.

    Raises
    ------
    ValueError
        If `wavelength_rest` is an array without names, or if no pixel is
        within `velocity_max` of it.

    Examples
    --------

    Fit a Gaussian plus a straight line to every spectrum of the
    :math:`\text{Si\,IV}\;1394\,\AA` line in a raster, and plot its
    radiance, Doppler shift, and width.

    .. jupyter-execute::

        import numpy as np
        import matplotlib.pyplot as plt
        import astropy.units as u
        import astropy.visualization
        import named_arrays as na
        import iris

        # Load a 320-step raster
        obs = iris.sg.open("2021-09-23T06:13")

        # Fit every spectrum within 100 km/s of the line
        fit = iris.sg.fit_line(obs, velocity_max=100 * u.km / u.s)

        # The area of each line, and the coordinates of each pixel of the
        # first raster
        p = fit.parameters
        area = np.sqrt(2 * np.pi) * p.amplitude * p.width
        index = {obs.axis_time: 0, obs.axis_wavelength: 0}
        position = obs.inputs.position[index]

        # Plot the area, Doppler shift, and width of the line
        with astropy.visualization.quantity_support():
            fig, ax = plt.subplots(
                ncols=3,
                figsize=(10, 4),
                sharex=True,
                sharey=True,
                constrained_layout=True,
            )
            for i, (title, value, kwargs) in enumerate([
                ("area", area, dict(vmin=0, vmax=np.nanpercentile(area, 99).value)),
                ("Doppler shift", p.center, dict(vmin=-20, vmax=20, cmap="RdBu_r")),
                ("width", p.width, dict(vmin=0, vmax=40)),
            ]):
                img = na.plt.pcolormesh(
                    position.x,
                    position.y,
                    C=value[{obs.axis_time: 0}].value,
                    ax=ax[i],
                    **kwargs,
                )
                ax[i].set_title(f"{title} ({value.unit:latex_inline})")
                ax[i].set_aspect("equal")
                fig.colorbar(img.ndarray.item(), ax=ax[i], location="bottom")
            unit_x, unit_y = ax[0].get_xlabel(), ax[0].get_ylabel()
            for axes in ax:
                axes.set_xlabel(f"helioprojective $x$ ({unit_x})")
            ax[0].set_ylabel(f"helioprojective $y$ ({unit_y})")
    """
    inputs = obs.inputs
    if wavelength_rest is not None:
        # the fits of several lines vary along the axis of the lines, which
        # an array without names does not have
        named = isinstance(wavelength_rest, na.AbstractArray)
        if not named and np.ndim(wavelength_rest) > 0:
            raise ValueError(
                f"Several rest wavelengths must be a named array, such as "
                f"`na.ScalarArray(wavelength_rest, axes='line')`, so that the "
                f"fits vary along its axis, got an array of shape "
                f"{np.shape(wavelength_rest)}."
            )
        inputs = dataclasses.replace(inputs, wavelength_rest=wavelength_rest)

    # the centers of the pixels along each axis of the detector the velocity
    # varies along, along which the inputs are the vertices of the pixels
    velocity = cast(na.AbstractScalar, inputs.velocity)
    axis_detector = (obs.axis_wavelength, obs.axis_detector_x, obs.axis_detector_y)
    velocity = velocity.cell_centers(
        axis=tuple(axis for axis in axis_detector if axis in velocity.shape),
    )

    # only the pixels within `velocity_max` of the line in some spectrum
    # are kept, so that the arrays of the fit are no larger than they need
    # to be
    inside = np.abs(velocity) <= velocity_max
    index = _crop(inside, axis=obs.axis_wavelength)
    obs = cast(SpectrographObservation, obs[index])
    velocity = cast(na.AbstractScalar, velocity[index])
    where = (where & inside)[index]
    if isinstance(variance, na.AbstractArray):
        variance = cast(na.AbstractScalar, variance[index])

    if guess is None:
        guess = _guess(
            model=model,
            velocity=velocity,
            outputs=obs.outputs,
            where=where,
            axis=obs.axis_wavelength,
            velocity_max=velocity_max,
        )

    lower_default, upper_default = _bounds(
        model=model,
        velocity=velocity,
        axis=obs.axis_wavelength,
        velocity_max=velocity_max,
    )
    if lower is None:
        lower = lower_default
    if upper is None:
        upper = upper_default

    result = na.ModelFitFunctionArray(
        inputs=velocity,
        outputs=obs.outputs,
        model=model,
        guess=guess,
        variance=obs.variance if variance is None else variance,
        axis_model=obs.axis_wavelength,
        where_model=where,
        lower=lower,
        upper=upper,
        prior_mean=prior_mean,
        prior_sigma=prior_sigma,
        correlation=correlation,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    if variance is None:
        # The photon noise of each output is correlated with the noise of
        # that output, so a fit weighted by it is biased low. The photon
        # noise of the model of the first fit is not, so the fit is done
        # again with that, starting from where the first one ended.
        expected = dataclasses.replace(obs, outputs=result.predictions)
        result = dataclasses.replace(
            result,
            guess=result.parameters,
            variance=expected.variance,
        )

    return result


def _crop(
    inside: na.AbstractScalar,
    axis: str,
) -> dict[str, slice]:
    """
    The index of the shortest range along `axis` which holds every pixel
    that is fit.

    Parameters
    ----------
    inside
        Whether each pixel is fit.
    axis
        The axis of wavelength.
    """
    # the index along `axis` of each pixel which is fit, in any spectrum
    index = na.arange(0, inside.shape[axis], axis=axis).broadcast_to(inside.shape)
    index = cast(na.ScalarArray, index[inside])
    index = index.ndarray_aligned(tuple(index.shape))
    if not index.size:
        raise ValueError("No pixel is within `velocity_max` of `wavelength_rest`.")
    return {axis: slice(int(index.min()), int(index.max()) + 1)}


def _guess(
    model: ndoptimize.Model,
    velocity: na.AbstractArray,
    outputs: na.AbstractScalar,
    where: bool | na.AbstractScalar,
    axis: str,
    velocity_max: u.Quantity | na.AbstractScalar,
) -> na.CartesianNdVectorArray:
    """
    The initial guess of :func:`fit_line` for a model whose parameters are
    among those it knows: the amplitude, center, width, kappa, slope, and
    intercept of a line.

    It is also the guess of :func:`iris.sg.background.fit`.

    Parameters
    ----------
    model
        The model to fit.
    velocity
        The velocity of each output.
    outputs
        The spectra to fit.
    where
        The outputs which are fit.
    axis
        The axis of wavelength.
    velocity_max
        The largest velocity of the outputs which are fit.
    """
    known = ("amplitude", "center", "width", "kappa", "slope", "intercept")
    unknown = tuple(name for name in model.parameters if name not in known)
    if unknown:
        raise ValueError(
            f"There is no default guess for the parameters {unknown} of the "
            f"model, so give one with `guess`."
        )

    unit_velocity = na.unit_normalized(velocity)
    unit_outputs = na.unit_normalized(outputs)
    width = _width.to(unit_velocity)

    data = np.where(where, outputs, np.nan)
    wing = np.abs(velocity) > velocity_max / 2

    intercept = 0 * unit_outputs
    if "intercept" in model.parameters:
        with warnings.catch_warnings():
            # spectra with no data in the wings, filled in below
            warnings.filterwarnings("ignore", "All-NaN slice encountered")
            intercept = np.nanmedian(np.where(wing, data, np.nan), axis=axis)
        intercept = np.where(np.isfinite(intercept), intercept, 0 * unit_outputs)

    excess = np.where(wing, 0 * unit_outputs, data - intercept)
    area = np.nansum(excess, axis=axis)

    # the centroid of the inner half on the side of the intercept that the
    # line is on, above it for an emission line and below it for an
    # absorption line, since a fit which starts too far from the line can end
    # with the straight line taking its place
    side = np.maximum(np.sign(area) * excess, 0 * unit_outputs)
    with np.errstate(divide="ignore", invalid="ignore"):
        center = np.nansum(velocity * side, axis=axis) / np.nansum(side, axis=axis)
    center = np.where(np.isfinite(center), center, 0 * unit_velocity)

    components = dict(
        amplitude=1 * unit_outputs,
        center=center,
        width=width,
        kappa=_kappa,
        slope=0 * unit_outputs / unit_velocity,
        intercept=0 * unit_outputs,
    )

    # the amplitude at which the line of the model has the area of the
    # excess, from the line with an amplitude of one, evaluated by a fit which
    # takes no steps, and summed over the same pixels as the excess
    line = na.ModelFitFunctionArray(
        inputs=velocity,
        outputs=na.ScalarArray(0 * unit_outputs),
        model=model,
        guess=na.CartesianNdVectorArray(
            {name: components[name] for name in model.parameters}
        ),
        variance=1 * unit_outputs**2,
        axis_model=axis,
        max_iterations=0,
    ).predictions
    line = line * (~wing & np.isfinite(data))
    with np.errstate(divide="ignore", invalid="ignore"):
        # zero for a spectrum with no data in the inner half
        amplitude = np.nan_to_num(area / line.sum(axis)) * unit_outputs

    components |= dict(amplitude=amplitude, intercept=intercept)
    return na.CartesianNdVectorArray(
        {name: components[name] for name in model.parameters}
    )


def _bounds(
    model: ndoptimize.Model,
    velocity: na.AbstractArray,
    axis: str,
    velocity_max: u.Quantity | na.AbstractScalar,
) -> tuple[na.CartesianNdVectorArray, na.CartesianNdVectorArray]:
    """
    The default lower and upper bounds of :func:`fit_line`, for the
    parameters which the model has: the center within `velocity_max`, the
    width from half the spacing of the pixels to half of `velocity_max`, and
    kappa from 0.1 to 1000.

    They are also the bounds of :func:`iris.sg.background.fit`.

    Parameters
    ----------
    model
        The model to fit.
    velocity
        The velocity of each output.
    axis
        The axis of wavelength.
    velocity_max
        The largest velocity of the outputs which are fit.
    """
    step = np.abs(np.diff(velocity, axis=axis)).mean(axis)
    lower = dict(
        center=-velocity_max,
        width=step / 2,
        kappa=0.1,
    )
    upper = dict(
        # a copy, so that changing the bound changes neither the argument
        # nor the default of `velocity_max`
        center=velocity_max.copy(),
        width=velocity_max / 2,
        kappa=1000,
    )
    lower_model, upper_model = (
        na.CartesianNdVectorArray(
            {name: bounds[name] for name in model.parameters if name in bounds}
        )
        for bounds in (lower, upper)
    )
    return lower_model, upper_model
