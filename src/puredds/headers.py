"""DDS header structures"""
import struct
from typing import List
from .enums import DDPF, DDSD, DDSCAPS, DDSCAPS2, DXGI_FORMAT, D3D10_RESOURCE_DIMENSION, DDS_RESOURCE_MISC


class DDS_PIXELFORMAT:
    """DDS Pixel Format structure (32 bytes)"""
    def __init__(self) -> None:
        self.dwSize: int = 32  # Size of structure (always 32)
        self.dwFlags: DDPF = DDPF(0)  # Flags to indicate which members are valid
        self.dwFourCC: int = 0  # FourCC code (can be any FourCC value)
        self.dwRGBBitCount: int = 0  # Number of bits per pixel
        self.dwRBitMask: int = 0  # Red bit mask
        self.dwGBitMask: int = 0  # Green bit mask
        self.dwBBitMask: int = 0  # Blue bit mask
        self.dwABitMask: int = 0  # Alpha bit mask

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DDS_PIXELFORMAT':
        """Read DDS_PIXELFORMAT from 32 bytes of data"""
        if len(data) < 32:
            raise ValueError(f"Expected 32 bytes for DDS_PIXELFORMAT, got {len(data)}")

        pixelformat = cls()
        values = struct.unpack('<8I', data[:32])
        pixelformat.dwSize = values[0]
        pixelformat.dwFlags = DDPF(values[1])
        pixelformat.dwFourCC = values[2]
        pixelformat.dwRGBBitCount = values[3]
        pixelformat.dwRBitMask = values[4]
        pixelformat.dwGBitMask = values[5]
        pixelformat.dwBBitMask = values[6]
        pixelformat.dwABitMask = values[7]

        return pixelformat


class DDS_HEADER:
    """DDS Header structure (124 bytes)"""
    def __init__(self) -> None:
        self.dwSize: int = 124  # Size of structure (always 124)
        self.dwFlags: DDSD = DDSD(0)  # Flags to indicate which members are valid
        self.dwHeight: int = 0  # Height of surface in pixels
        self.dwWidth: int = 0  # Width of surface in pixels
        self.dwPitchOrLinearSize: int = 0  # Pitch or linear size of data
        self.dwDepth: int = 0  # Depth of volume texture
        self.dwMipMapCount: int = 0  # Number of mipmap levels
        self.dwReserved1: List[int] = [0] * 11  # Reserved (11 DWORDs)
        self.ddspf: DDS_PIXELFORMAT = DDS_PIXELFORMAT()  # Pixel format
        self.dwCaps: DDSCAPS = DDSCAPS(0)  # Surface complexity flags
        self.dwCaps2: DDSCAPS2 = DDSCAPS2(0)  # Additional surface flags
        self.dwCaps3: int = 0  # Reserved
        self.dwCaps4: int = 0  # Reserved
        self.dwReserved2: int = 0  # Reserved

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DDS_HEADER':
        """Read DDS_HEADER from 124 bytes of data"""
        if len(data) < 124:
            raise ValueError(f"Expected 124 bytes for DDS_HEADER, got {len(data)}")

        header = cls()

        # Read the first part (7 DWORDs + 11 reserved DWORDs)
        values = struct.unpack('<18I', data[:72])
        header.dwSize = values[0]
        header.dwFlags = DDSD(values[1])
        header.dwHeight = values[2]
        header.dwWidth = values[3]
        header.dwPitchOrLinearSize = values[4]
        header.dwDepth = values[5]
        header.dwMipMapCount = values[6]
        header.dwReserved1 = list(values[7:18])

        # Read pixel format (32 bytes starting at offset 72)
        header.ddspf = DDS_PIXELFORMAT.from_bytes(data[72:104])

        # Read caps (5 DWORDs starting at offset 104)
        caps = struct.unpack('<5I', data[104:124])
        header.dwCaps = DDSCAPS(caps[0])
        header.dwCaps2 = DDSCAPS2(caps[1])
        header.dwCaps3 = caps[2]
        header.dwCaps4 = caps[3]
        header.dwReserved2 = caps[4]

        return header


class DDS_HEADER_DXT10:
    """DDS DX10 Extended Header structure (20 bytes)"""
    def __init__(self) -> None:
        self.dxgiFormat: DXGI_FORMAT = DXGI_FORMAT.UNKNOWN  # DXGI format
        self.resourceDimension: D3D10_RESOURCE_DIMENSION = D3D10_RESOURCE_DIMENSION.UNKNOWN  # Resource dimension
        self.miscFlag: DDS_RESOURCE_MISC = DDS_RESOURCE_MISC(0)  # Miscellaneous flags
        self.arraySize: int = 0  # Array size
        self.miscFlags2: int = 0  # Additional miscellaneous flags

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DDS_HEADER_DXT10':
        """Read DDS_HEADER_DXT10 from 20 bytes of data"""
        if len(data) < 20:
            raise ValueError(f"Expected 20 bytes for DDS_HEADER_DXT10, got {len(data)}")

        header10 = cls()
        values = struct.unpack('<5I', data[:20])
        header10.dxgiFormat = DXGI_FORMAT(values[0])
        header10.resourceDimension = D3D10_RESOURCE_DIMENSION(values[1])
        header10.miscFlag = DDS_RESOURCE_MISC(values[2])
        header10.arraySize = values[3]
        header10.miscFlags2 = values[4]

        return header10
