from dataclasses import dataclass
from typing import List, Optional, Dict

@dataclass(frozen=True)
class TransmissionPoint:
    """A single processed measurement point."""
    v_cathode: float
    v_gate: float
    v_anode: float
    i_mean_pA: float
    i_std_pA: float
    e_drift_V_cm: float
    e_el_V_cm: float
    r_factor: float
    transmission: float = 0.0
    transmission_err: float = 0.0

@dataclass(frozen=True)
class TransmissionRun:
    """All points belonging to a specific physical run."""
    run_id: str
    description: str
    points: List[TransmissionPoint]
    drift_field_V_cm: float = 0.0
    fit_params: Optional[Dict[str, float]] = None