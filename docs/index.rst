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

    import iris

    # Download a raster sequence
    obs = iris.sg.open("2017-02-11T05:00")

    # Display the raster sequence as a false-color animation
    obs.to_jshtml()


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
