Introduction
============

The `Interface Region Imaging Spectograph <iris.lmsal.com>`_ (IRIS) is a NASA
Small Explorer satellite which has been taking continuous ultraviolet images of
the Sun since 2013.

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
    import dataclasses
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

    # Define a tuple of the temporal and spatial axes
    axis_txy = (obs.axis_time, obs.axis_detector_x, obs.axis_detector_y)

    # Take the mean of the wavelength over the spatial
    # and temporal axes since it is constant
    wavelength = obs.inputs.wavelength.mean(axis_txy)

    # Convert to Doppler velocity
    velocity = wavelength.to(
        unit=u.km / u.s,
        equivalencies=u.doppler_optical(wavelength_center),
    )

    # Define velocity bounds
    velocity_min = -100 * u.km / u.s
    velocity_max = +100 * u.km / u.s

    # Define the spectral normalization curve
    spd_max = np.nanpercentile(
        a=obs.outputs,
        q=99.5,
        axis=axis_txy,
    )

    # Convert the spectral radiance to
    # red/green/blue channels
    rgb, colorbar = na.colorsynth.rgb_and_colorbar(
        spd=obs.outputs,
        wavelength=velocity,
        axis=obs.axis_wavelength,
        spd_min=0 * u.DN,
        spd_max=spd_max,
        wavelength_min=velocity_min,
        wavelength_max=velocity_max,
    )

    # Isolate the angular position of each RGB point
    position = obs.inputs.position.mean(obs.axis_wavelength)

    # Display the result as an RGB movie
    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(
            ncols=2,
            figsize=(6, 6),
            gridspec_kw=dict(width_ratios=[.9, .1]),
            constrained_layout=True,
        )
        ani = na.plt.pcolormovie(
            obs.inputs.time,
            position.x,
            position.y,
            C=rgb,
            axis_time=obs.axis_time,
            axis_rgb=obs.axis_wavelength,
            ax=ax[0],
            kwargs_animation=dict(
                interval=500,
            )
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


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
