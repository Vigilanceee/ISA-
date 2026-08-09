"""Model package with lazy exports to avoid device/model import cycles."""

__all__ = ["VGG8", "TIA_Layer"]


def __getattr__(name):
    if name == "VGG8":
        from .vgg8 import VGG8

        return VGG8
    if name == "TIA_Layer":
        from .tia_layer import TIA_Layer

        return TIA_Layer
    raise AttributeError(name)
