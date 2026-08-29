"""Bounded local training and GPU smoke-test utilities."""

from .adapters import (
    AstroMambaHTrainingAdapter,
    AstroMambaHTrainingBatch,
    TinyAstroAdapter,
    TinyAstroAdapterBatch,
    make_tiny_adapter_batch,
    make_tiny_astromamba_batch,
    tiny_astromamba_config,
)
from .harness import (
    BoundedTrainer,
    CheckpointReport,
    EpochReport,
    NonFiniteTrainingError,
    TrainingConfig,
    TrainingState,
    DEFAULT_RSS_CAP_BYTES,
    DEFAULT_STORAGE_CAP_BYTES,
    amp_supported,
    default_loss_fn,
    process_rss_bytes,
    resolve_device,
)

__all__ = [
    "BoundedTrainer",
    "DEFAULT_RSS_CAP_BYTES",
    "DEFAULT_STORAGE_CAP_BYTES",
    "AstroMambaHTrainingAdapter",
    "AstroMambaHTrainingBatch",
    "CheckpointReport",
    "EpochReport",
    "NonFiniteTrainingError",
    "TinyAstroAdapter",
    "TinyAstroAdapterBatch",
    "TrainingConfig",
    "TrainingState",
    "amp_supported",
    "default_loss_fn",
    "make_tiny_adapter_batch",
    "make_tiny_astromamba_batch",
    "process_rss_bytes",
    "resolve_device",
    "tiny_astromamba_config",
]
