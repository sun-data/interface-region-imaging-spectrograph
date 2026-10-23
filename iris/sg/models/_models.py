import math
import numpy as np
import ndoptimize

__all__ = [
    "kappa_linear",
]


@ndoptimize.model(
    parameters=("amplitude", "center", "width", "kappa", "slope", "intercept"),
    inputs=("velocity",),
)
def kappa_linear(
    x: np.ndarray,
    parameters: np.ndarray,
    f: np.ndarray,
    jacobian: np.ndarray,
) -> None:
    r"""
    A kappa distribution plus a straight line,

    .. math::

        f(v) = A \left( 1 + \frac{(v - v_0)^2}{\kappa \sigma^2} \right)^{-\kappa - 1}
            + m v + b,

    the model of :func:`iris.sg.background.model_total`, compiled with
    :func:`ndoptimize.model` so that :func:`iris.sg.fit_line` can fit it.

    Its parameters are the amplitude :math:`A`, the center :math:`v_0`, the
    width :math:`\sigma`, :math:`\kappa`, which controls how heavy the tails
    are, the slope :math:`m`, and the intercept :math:`b`, of a single input,
    the velocity :math:`v`.
    The center and the intercept are the ``shift`` and the ``bias`` of
    :func:`iris.sg.background.model_total`.
    As :math:`\kappa` grows, the profile approaches the Gaussian
    :math:`A \exp(-(v - v_0)^2 / \sigma^2)`, whose standard deviation is
    :math:`\sigma / \sqrt{2}`.
    Its full width at half maximum,
    :math:`2 \sigma \sqrt{\kappa (2^{1 / (\kappa + 1)} - 1)}`, is
    :math:`1.67 \sigma` for a large :math:`\kappa`, but only
    :math:`0.59 \sigma` for :math:`\kappa = 0.1`.

    Parameters
    ----------
    x
        The velocities at which to evaluate the model, ``(num_points, 1)``.
    parameters
        The amplitude, center, width, kappa, slope, and intercept.
    f
        The output array for the model, ``(num_points,)``.
    jacobian
        The output array for the derivatives of the model with respect to the
        parameters, ``(num_points, 6)``.
    """
    amplitude = parameters[0]
    center = parameters[1]
    width = parameters[2]
    kappa = parameters[3]
    slope = parameters[4]
    intercept = parameters[5]
    for i in range(x.shape[0]):
        v = x[i, 0]
        d = v - center
        q = d * d / (kappa * width * width)
        base = 1 + q
        log_base = math.log1p(q)
        e = math.exp((-kappa - 1) * log_base)
        g = amplitude * (kappa + 1) * e / base
        f[i] = amplitude * e + slope * v + intercept
        jacobian[i, 0] = e
        jacobian[i, 1] = 2 * g * d / (kappa * width * width)
        jacobian[i, 2] = 2 * g * q / width
        jacobian[i, 3] = amplitude * e * ((kappa + 1) * q / (kappa * base) - log_base)
        jacobian[i, 4] = v
        jacobian[i, 5] = 1
