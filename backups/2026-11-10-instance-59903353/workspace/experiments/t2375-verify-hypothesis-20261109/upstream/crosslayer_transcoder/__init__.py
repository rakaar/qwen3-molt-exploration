"""
Crosslayer Transcoder Package

An implementation of Anthropic's crosslayer transcoders for neural network interpretability research.
"""

# Keep inference-only dashboard imports independent of the training stack.
def __getattr__(name):
    from importlib import import_module
    modules = {
        "ActivationDataModule": ".data",
        "ReplacementModelAccuracy": ".metrics",
        "CrossLayerTranscoderModule": ".model",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

# Version info
__version__ = "0.1.0"

__all__ = [
    "ActivationDataModule",
    "CrossLayerTranscoderModule",
    "ReplacementModelAccuracy",
]
