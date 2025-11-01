# units.py
"""
Lightweight unit conversion helpers for canonical internal representation.
The goal is to minimize floating-point clutter (e.g. use 1.5 µs instead of 1.5e-6 s),
while ensuring internal consistency across the analysis pipeline.
"""

from dataclasses import dataclass

# -------------------------------
# Canonical units documentation
# -------------------------------

@dataclass(frozen=True)
class CanonicalUnits:
    """
    Defines canonical internal units for all magnitudes in the package.
    """
    time: str = "µs"
    voltage: str = "mV"
    drift_field: str = "V/cm"
    reduced_field: str = "Td"     # Townsend
    drift_speed: str = "mm/µs"


# -------------------------------
# Base conversion helpers
# -------------------------------

def us(x: float) -> float:
    """Convert µs → s (if needed externally), or just tag for readability."""
    return x * 1e-6

def ms(x: float) -> float:
    """Convert ms → s."""
    return x * 1e-3

def ns(x: float) -> float:
    """Convert ns → s."""
    return x * 1e-9

def mV(x: float) -> float:
    """Convert mV → V (if needed externally), or just tag for readability."""
    return x * 1e-3

def V(x: float) -> float:
    """Identity function for clarity when specifying volts explicitly."""
    return x * 1.0

def cm(x: float) -> float:
    """Convert cm → m (if needed externally)."""
    return x * 1e-2

def mm(x: float) -> float:
    """Convert mm → m (if needed externally)."""
    return x * 1e-3


# -------------------------------
# Derived physics quantities
# -------------------------------

def to_Td(reduced_field: float) -> float:
    """
    Convert reduced field [V·cm²] to Townsend [Td].
    """
    return reduced_field * 1e17

def Td_to_Vpcm(reduced_field_Td: float) -> float:
    """
    Convert reduced field [Td] to [V·cm²].
    """
    return reduced_field_Td * 1e-17


def cm_to_mm(cm_value: float) -> float:
    """
    Convert cm to mm.
    """
    return cm_value * 10.0

def s_to_us(s_value: float) -> float:
    """
    Convert seconds to microseconds.
    """
    return s_value * 1e6

def V_to_mV(V_value: float) -> float:
    """
    Convert volts to millivolts.
    """
    return V_value * 1e3

def to_mm_per_us(speed: float) -> float:
    """
    Convert drift speed to mm/µs (if model returns SI).

    Parameters
    ----------
    speed : float
        Speed in m/s

    Returns
    -------
    float
        Speed in mm/µs
    """
    return speed * 1e3 * 1e-6  # m/s → mm/µs
