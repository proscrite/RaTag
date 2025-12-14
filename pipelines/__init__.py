"""
RaTag pipelines module.

High-level pipeline functions for composing workflow steps.
Pipelines use functional composition to define analysis sequences.
"""

from .alpha_calibration import (
    alpha_calibration,
    alpha_calibration_replot,
    alpha_calibration_quick,
    prepare_isotope_separation,  # Deprecated alias
)

__all__ = [
    'alpha_calibration',
    'alpha_calibration_replot',
    'alpha_calibration_quick',
    'prepare_isotope_separation',  # Deprecated
]
