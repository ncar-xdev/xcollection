#!/usr/bin/env python3
# flake8: noqa
""" Top-level module for xcollection. """
from pkg_resources import DistributionNotFound, get_distribution

from .main import Collection, open_collection

try:
    __version__ = get_distribution('xcollection').version
except DistributionNotFound:  # pragma: no cover
    __version__ = 'unknown'  # pragma: no cover
