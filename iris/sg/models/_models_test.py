import pytest
import numpy as np
import astropy.units as u
import named_arrays as na
import iris


@pytest.mark.parametrize(
    argnames="parameters",
    argvalues=[
        np.array([20.0, 3.0, 25.0, 1.1, 0.002, 1.5]),
        np.array([5.0, -10.0, 8.0, 40.0, -0.01, 0.0]),
        np.array([1.0, 0.0, 30.0, 0.3, 0.0, -2.0]),
    ],
)
def test_kappa_linear(parameters: np.ndarray):
    """
    The compiled model is :func:`iris.sg.background.model_total`, and its
    Jacobian is the derivative of the model.
    """
    model = iris.sg.models.kappa_linear
    v = np.linspace(-150, 150, 61)
    x = v[:, np.newaxis]
    f = np.empty(v.size)
    jacobian = np.empty((v.size, parameters.size))
    model(x, parameters, f, jacobian)

    amplitude, center, width, kappa, slope, intercept = parameters
    expected = iris.sg.background.model_total(
        velocity=na.ScalarArray(v, "wavelength") * u.km / u.s,
        amplitude=amplitude * u.DN,
        shift=center * u.km / u.s,
        width=width * u.km / u.s,
        kappa=kappa,
        bias=intercept * u.DN,
        slope=slope * u.DN / (u.km / u.s),
    )
    assert np.allclose(f, expected.ndarray.to_value(u.DN))

    # compiled from its Python source, so that its body is checked for
    # coverage too
    f_python = np.empty(v.size)
    jacobian_python = np.empty((v.size, parameters.size))
    model.function.py_func(x, parameters, f_python, jacobian_python)  # type: ignore[attr-defined]
    assert np.allclose(f_python, f)
    assert np.allclose(jacobian_python, jacobian)

    step = 1e-6 * np.maximum(np.abs(parameters), 1)
    for k in range(parameters.size):
        delta = np.zeros(parameters.size)
        delta[k] = step[k]
        f_upper, f_lower = np.empty(v.size), np.empty(v.size)
        j = np.empty((v.size, parameters.size))
        model(x, parameters + delta, f_upper, j)
        model(x, parameters - delta, f_lower, j)
        derivative = (f_upper - f_lower) / (2 * step[k])
        assert np.allclose(jacobian[:, k], derivative, rtol=1e-5, atol=1e-8)
