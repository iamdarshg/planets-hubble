"""Model-building slice for the dependency-light AstroMamba-H scaffold."""

from .astromamba_h import AstroMambaH, AstroMambaHConfig, AstroMambaHInputs
from .candidate_classifier import CandidateRecord, HeatmapCandidateClassifier
from .configurations import research_config

__all__ = [
    "AstroMambaH",
    "AstroMambaHConfig",
    "AstroMambaHInputs",
    "CandidateRecord",
    "HeatmapCandidateClassifier",
    "research_config",
]
