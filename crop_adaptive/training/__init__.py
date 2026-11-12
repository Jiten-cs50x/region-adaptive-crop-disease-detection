"""Model training utilities for the region-adaptive crop disease pipeline."""

__all__ = ["train_from_config"]


def __getattr__(name: str):
    """Load the optional PyTorch training engine only when explicitly requested."""
    if name == "train_from_config":
        from crop_adaptive.training.engine import train_from_config
        return train_from_config
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
