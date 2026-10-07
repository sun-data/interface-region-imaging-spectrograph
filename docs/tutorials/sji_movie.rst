Make a movie of slit-jaw images
===============================

This tutorial loads the Si IV 1400 Å images which the IRIS slit-jaw imager
captured while the EUV Snapshot Imaging Spectrograph (ESIS) sounding rocket
was observing the Sun on 2019 September 30,
and animates them with :func:`named_arrays.plt.pcolormovie`.


Load the images
---------------

:func:`iris.sji.open` downloads the slit-jaw images of one channel,
and keeps the frames which began during the given time range,
here the five minutes in which ESIS was taking images.

.. jupyter-execute::

    import IPython.display
    import numpy as np
    import matplotlib.pyplot as plt
    import astropy.units as u
    import astropy.visualization
    import named_arrays as na
    import iris

    obs = iris.sji.open(
        time="2019-09-30T18:06:11",
        time_stop="2019-09-30T18:11:01",
        window="SJI_1400",
    )

    obs.shape

There is one frame every 9.4 seconds.
Dividing each frame by its exposure time,
:attr:`~iris.sji.SlitJawObservation.timedelta`,
puts every frame on the same scale,
even in an observation whose exposure time changes from frame to frame.

.. jupyter-execute::

    rate = obs.outputs / obs.timedelta

    # One color scale for every frame, so that a brightening in the movie
    # is a brightening on the Sun rather than a change of scale.
    vmin = 0 * u.DN / u.s
    vmax = np.nanpercentile(rate, 99.5)


Choose a region
---------------

The whole field of view is about three minutes of arc across,
which makes a large movie,
so select a region in the bottom right corner of the field by indexing
the detector axes.
The coordinates of each pixel come along with it.

.. jupyter-execute::

    index = dict(
        detector_x=slice(330, None),
        detector_y=slice(None, 150),
    )

    region = obs[index]

    # The corners of the region in the first frame, in order around it
    corners = region.inputs.position[dict(
        time=0,
        detector_x=na.ScalarArray(np.array([0, -1, -1, 0, 0]), axes="corner"),
        detector_y=na.ScalarArray(np.array([0, 0, -1, -1, 0]), axes="corner"),
    )]

    first = obs[dict(time=0)]

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(6, 6), constrained_layout=True)
        na.plt.pcolormesh(
            first.inputs.position.x,
            first.inputs.position.y,
            C=rate[dict(time=0)],
            ax=ax,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        na.plt.plot(
            corners.x,
            corners.y,
            axis="corner",
            ax=ax,
            color="tab:red",
        )
        ax.set_aspect("equal")
        ax.set_title(first.inputs.time.ndarray)
        ax.set_xlabel(f"helioprojective $x$ ({ax.get_xlabel()})")
        ax.set_ylabel(f"helioprojective $y$ ({ax.get_ylabel()})")


Animate the region
------------------

:func:`named_arrays.plt.pcolormovie` draws one frame for each index along
the time axis, and labels each frame with its time in the lower right corner.

.. jupyter-execute::

    with astropy.visualization.quantity_support():
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)
        ani = na.plt.pcolormovie(
            region.inputs.time,
            region.inputs.position.x,
            region.inputs.position.y,
            C=rate[index],
            axis_time="time",
            ax=ax,
            cmap="gray",
            vmin=vmin,
            vmax=vmax,
        )
        # The frames are drawn later, as the movie is made, so the axes do
        # not know the unit of the coordinates yet.
        unit = region.inputs.position.x.unit
        ax.set_aspect("equal")
        ax.set_xlabel(f"helioprojective $x$ ({unit:latex_inline})")
        ax.set_ylabel(f"helioprojective $y$ ({unit:latex_inline})")
        plt.close(fig)

    IPython.display.HTML(ani.to_jshtml(fps=5))

The edges of the field and the dark line of the slit step back and forth
from frame to frame.
IRIS rasters its spectrograph by moving the image of the Sun across the slit,
and each slit-jaw image of level 2 is shifted back so that the Sun stays put,
which leaves the edges of the field and the slit to move instead.
