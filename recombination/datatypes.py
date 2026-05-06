from dataclasses import dataclass, field
from typing import Dict, Any, Callable

@dataclass(frozen=True)
class CorrectionFactor:
    """Immutable data contract for any physical artifact injected into the engine."""
    name: str
    value: float
    uncertainty: float
    units: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass(frozen=True)
class CorrectionModel:
    """Immutable contract for physical models that evaluate dynamically per data point."""
    name: str
    evaluate: Callable[..., float]  # pure mathematical function
    metadata: Dict[str, Any] = field(default_factory=dict)

def get_identity_factor(name: str) -> CorrectionFactor:
    """Returns a mathematical identity (1.0 ± 0.0) to pass through a pipeline step cleanly."""
    return CorrectionFactor(
        name=name,
        value=1.0,
        uncertainty=0.01,  # small uncertainty to avoid zero-division issues in error propagation
        units="fraction",
        metadata={"note": "Identity applied for linear pipeline flow."}
    )