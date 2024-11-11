Introduction
============

The `Interface Region Imaging Spectograph <https://iris.lmsal.com>`_ (IRIS) is a NASA
Small Explorer satellite which has been taking continuous ultraviolet images of
the Sun since 2013 :cite:p:`DePontieu2014`.

This Python package aims to represent IRIS imagery using :mod:`named_arrays`,
a named tensor implementation with :class:`astropy.units.Quantity` support.

Installation
============

This package is published to PyPI and can be installed using pip.

.. code-block::

    pip install interface-region-imaging-spectrograph

API Reference
=============

.. autosummary::
    :toctree: _autosummary
    :template: module_custom.rst
    :recursive:

    iris


Examples
========

Load an IRIS spectrograph raster sequence,
and display as a false-color movie.

.. jupyter-execute::

    import IPython.display
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.time
    import astropy.visualization
    import named_arrays as na
    import iris

    # Download a 320-step raster sequence
    obs = iris.sg.SpectrographObservation.from_time_range(
        time_start=astropy.time.Time("2017-02-11T04:50"),
        time_stop=astropy.time.Time("2017-02-11T05:00"),
    )

    # Calculate the mean rest wavelength of the
    # brightest spectral line
    wavelength_center = obs.wavelength_center.ndarray.mean()

    # Average wavelength over time and convert to Doppler velocity
    obs.inputs = obs.inputs.explicit
    obs.inputs.wavelength = obs.inputs.wavelength.mean(
        axis=obs.axis_time,
    ).to(
        unit=u.km / u.s,
        equivalencies=u.doppler_optical(wavelength_center),
    )

    # Define velocity bounds
    velocity_min = -100 * u.km / u.s
    velocity_max = +100 * u.km / u.s

    # Define the spectral normalization curve
    vmax = np.nanpercentile(
        a=obs.outputs,
        q=99.5,
        axis=(obs.axis_time, obs.axis_detector_x, obs.axis_detector_y),
    )

    # Display the result as an RGB movie
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(
            ncols=2,
            figsize=(6, 6),
            gridspec_kw=dict(width_ratios=[.9, .1]),
            constrained_layout=True,
        )
        ani, colorbar = na.plt.rgbmovie(
            C=obs,
            axis_time=obs.axis_time,
            axis_wavelength=obs.axis_wavelength,
            ax=ax[0],
            vmin=0 * u.DN,
            vmax=vmax,
            wavelength_min=velocity_min,
            wavelength_max=velocity_max,
            interval=500,
        )
        na.plt.pcolormesh(
            C=colorbar,
            axis_rgb=obs.axis_wavelength,
            ax=ax[1],
        )
        ax[0].set_aspect("equal")
        ax[0].set_xlabel(f"helioprojective $x$ ({ax[0].get_xlabel()})")
        ax[0].set_ylabel(f"helioprojective $y$ ({ax[0].get_ylabel()})")
        ax[1].yaxis.tick_right()
        ax[1].yaxis.set_label_position("right")
        ax[1].set_ylim(velocity_min, velocity_max)

    plt.close(fig)
    IPython.display.HTML(ani.to_jshtml())


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
