"""Python Interface for the Infineon XENSIV Avian sensors

The Python package allows an easy and pythonic way to access the Avian radar
sensors like BGT60TR13C, BGT60UTR11AIP, BGT60ATR24, or BGT60TR13D.
"""

from .common import exceptions
from .common.sdk_base import get_version, get_version_full

__all__ = ['exceptions', 'get_version', 'get_version_full']
