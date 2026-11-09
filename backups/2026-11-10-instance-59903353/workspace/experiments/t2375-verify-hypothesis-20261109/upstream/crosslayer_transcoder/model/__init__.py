"""
Cross-layer transcoder model components.
"""

def __getattr__(name):
    from importlib import import_module
    modules = {
        "CrossLayerTranscoder": ".clt",
        "CrossLayerTranscoderModule": ".clt_lightning",
        "Molt": ".molt",
        "BatchTopK": ".topk",
        "PerLayerBatchTopK": ".topk",
        "PerLayerTopK": ".topk",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(modules[name], __name__), name)
    globals()[name] = value
    return value

__all__ = [
    "CrossLayerTranscoder",
    "CrossLayerTranscoderModule",
    "Molt",
    "BatchTopK",
    "PerLayerTopK",
    "PerLayerBatchTopK",
]
