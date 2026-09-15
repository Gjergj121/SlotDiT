from typing import Literal, List

from functools import partial

import importlib


Module = Literal[
    "",
    "Conv2d",
    "PaddedConv3D",
    "AttnBlock",
    "AttnBlock3D",
    "ResnetBlock3D",
    "Upsample",
    "Downsample",
    "SpatialUpsample2x",
    "SpatialDownsample2x",
    "Spatial2xTime2x3DUpsample",
    "Spatial2xTime2x3DDownsample",
]

MODULES_3D: List[Module] = [
    "PaddedConv3D",
    "AttnBlock3D",
    "ResnetBlock3D",
    "SpatialUpsample2x",
    "SpatialDownsample2x",
    "Spatial2xTime2x3DUpsample",
    "Spatial2xTime2x3DDownsample",
]

MODULES_BASE_CANDIDATES = ("vae_modules", "src.vae_modules")

def resolve_str_to_module(name: Module, is_causal: bool) -> type:
    if name == "":
        raise ValueError("Empty string is not a valid module name.")
    module = None
    last_error = None
    for module_base in MODULES_BASE_CANDIDATES:
        try:
            module = importlib.import_module(module_base)
            break
        except ModuleNotFoundError as err:
            last_error = err
    if module is None:
        raise ModuleNotFoundError(
            f"Could not import any module base from {MODULES_BASE_CANDIDATES}"
        ) from last_error
    module_cls = getattr(module, name)
    if name in MODULES_3D:
        module_cls = partial(module_cls, is_causal=is_causal)
    return module_cls
