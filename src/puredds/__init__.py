"""puredds - High-performance DDS texture file reader and converter"""

__version__ = "0.4.0"

# Main DDS class
from .dds import DDS

# Header structures
from .headers import (
    DDS_HEADER,
    DDS_HEADER_DXT10,
    DDS_PIXELFORMAT,
)

# Enumerations and flags
from .enums import (
    DDSD,
    DDPF,
    DDSCAPS,
    DDSCAPS2,
    FourCC,
    DXGI_FORMAT,
    D3D10_RESOURCE_DIMENSION,
    DDS_RESOURCE_MISC,
)

# CLI entry point
from .cli import main

__all__ = [
    '__version__',
    'DDS',
    'DDS_HEADER',
    'DDS_HEADER_DXT10',
    'DDS_PIXELFORMAT',
    'DDSD',
    'DDPF',
    'DDSCAPS',
    'DDSCAPS2',
    'FourCC',
    'DXGI_FORMAT',
    'D3D10_RESOURCE_DIMENSION',
    'DDS_RESOURCE_MISC',
    'main',
]
