"""Typed exceptions raised by :mod:`tspfs`.

Each subclasses the built-in it replaces, so existing ``except ValueError`` /
``except TypeError`` handlers keep working unchanged.
"""

from __future__ import annotations


class TSPError(Exception):
    """Base class for all errors raised by tspfs."""


class TSPValueError(TSPError, ValueError):
    """Invalid value passed to a tspfs estimator (e.g. bad parameter or data)."""


class TSPTypeError(TSPError, TypeError):
    """Invalid type passed to a tspfs estimator."""
