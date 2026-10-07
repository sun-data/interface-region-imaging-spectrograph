"""
Represent and manipulate images captured by the IRIS slit-jaw imager
"""

from ._slit_jaw import SlitJawObservation
from ._sji import open

__all__ = [
    "SlitJawObservation",
    "open",
]
