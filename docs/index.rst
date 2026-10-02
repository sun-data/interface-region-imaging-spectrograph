Introduction
============

The `Interface Region Imaging Spectrograph <https://iris.lmsal.com>`_ (IRIS) is a NASA
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


Citation
========

If you use :mod:`iris` in your research, please cite it.
The citation metadata is kept in
`CITATION.cff <https://github.com/sun-data/interface-region-imaging-spectrograph/blob/main/CITATION.cff>`_,
which the "Cite this repository" button on the
`GitHub page <https://github.com/sun-data/interface-region-imaging-spectrograph>`_
can export as BibTeX or APA.

Every release of :mod:`iris` is archived on Zenodo with its own DOI.
The concept DOI,
`10.5281/zenodo.23104892 <https://doi.org/10.5281/zenodo.23104892>`_,
always resolves to the latest version,
and the Zenodo page lists the DOI of every version.
Please include the version of :mod:`iris` that you used,
which is given by ``importlib.metadata.version("interface-region-imaging-spectrograph")``.
The BibTeX entry below uses the concept DOI.
To cite a specific version instead,
replace ``doi`` with the DOI of that version.

.. code-block:: bibtex

    @software{interface-region-imaging-spectrograph,
      author = {Smart, Roy T.},
      title = {interface-region-imaging-spectrograph},
      version = {X.Y.Z},
      doi = {10.5281/zenodo.23104892},
      url = {https://github.com/sun-data/interface-region-imaging-spectrograph},
    }


References
==========

.. bibliography::

|


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
