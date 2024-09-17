# interface-region-imaging-spectrograph

[![tests](https://github.com/sun-data/interface-region-imaging-spectrograph/actions/workflows/tests.yml/badge.svg)](https://github.com/sun-data/interface-region-imaging-spectrograph/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/sun-data/interface-region-imaging-spectrograph/graph/badge.svg?token=9VdGTSq2hT)](https://codecov.io/gh/sun-data/interface-region-imaging-spectrograph)
[![Black](https://github.com/sun-data/interface-region-imaging-spectrograph/actions/workflows/black.yml/badge.svg)](https://github.com/sun-data/interface-region-imaging-spectrograph/actions/workflows/black.yml)
[![Ruff](https://github.com/sun-data/interface-region-imaging-spectrograph/actions/workflows/ruff.yml/badge.svg)](https://github.com/sun-data/interface-region-imaging-spectrograph/actions/workflows/ruff.yml)
[![Documentation Status](https://readthedocs.org/projects/interface-region-imaging-spectrograph/badge/?version=latest)](https://interface-region-imaging-spectrograph.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/interface-region-imaging-spectrograph.svg)](https://badge.fury.io/py/interface-region-imaging-spectrograph)

A Python library for downloading and analyzing images from the [Interface Region Imaging Spectrograph (IRIS)](iris.lmsal.com), 
a NASA small explorer satellite which observes the Sun in ultraviolet.

## Installation

This library is published on PyPI and is installed using pip
```
pip install interface-region-imaging-spectrograph
```

## Gallery

Download spectrograph rasters from a specified time range and plot the Si IV 1403 Angstrom
spectral line as a false-color image.

[![obs](https://interface-region-imaging-spectrograph.readthedocs.io/en/latest/_images/iris.sg.SpectrographObservation_0_1.png)](https://interface-region-imaging-spectrograph.readthedocs.io/en/latest/_autosummary/iris.sg.SpectrographObservation.html#iris.sg.SpectrographObservation)
