import struct
from typing import Optional, List, BinaryIO
from enum import IntEnum, IntFlag
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image
from numba import jit


__all__ = [
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


class DDSD(IntFlag):
    """DDS Header Flags (dwFlags)"""
    CAPS = 0x1
    HEIGHT = 0x2
    WIDTH = 0x4
    PITCH = 0x8
    PIXELFORMAT = 0x1000
    MIPMAPCOUNT = 0x20000
    LINEARSIZE = 0x80000
    DEPTH = 0x800000


class DDPF(IntFlag):
    """DDS Pixel Format Flags (dwFlags)"""
    ALPHAPIXELS = 0x1
    ALPHA = 0x2
    FOURCC = 0x4
    RGB = 0x40
    YUV = 0x200
    LUMINANCE = 0x20000


class DDSCAPS(IntFlag):
    """DDS Caps (dwCaps)"""
    COMPLEX = 0x8
    MIPMAP = 0x400000
    TEXTURE = 0x1000


class DDSCAPS2(IntFlag):
    """DDS Caps2 (dwCaps2)"""
    CUBEMAP = 0x200
    CUBEMAP_POSITIVEX = 0x400
    CUBEMAP_NEGATIVEX = 0x800
    CUBEMAP_POSITIVEY = 0x1000
    CUBEMAP_NEGATIVEY = 0x2000
    CUBEMAP_POSITIVEZ = 0x4000
    CUBEMAP_NEGATIVEZ = 0x8000
    VOLUME = 0x200000


class FourCC(IntEnum):
    """FourCC codes"""
    DXT1 = 0x31545844
    DXT2 = 0x32545844
    DXT3 = 0x33545844
    DXT4 = 0x34545844
    DXT5 = 0x35545844
    DX10 = 0x30315844
    BC4U = 0x55344342
    BC4S = 0x53344342
    BC5U = 0x55354342
    BC5S = 0x53354342
    ATI1 = 0x31495441
    ATI2 = 0x32495441
    RXGB = 0x42475852

class DXGI_FORMAT(IntEnum):
    """DXGI Format Enum"""
    UNKNOWN = 0
    R32G32B32A32_TYPELESS = 1
    R32G32B32A32_FLOAT = 2
    R32G32B32A32_UINT = 3
    R32G32B32A32_SINT = 4
    R32G32B32_TYPELESS = 5
    R32G32B32_FLOAT = 6
    R32G32B32_UINT = 7
    R32G32B32_SINT = 8
    R16G16B16A16_TYPELESS = 9
    R16G16B16A16_FLOAT = 10
    R16G16B16A16_UNORM = 11
    R16G16B16A16_UINT = 12
    R16G16B16A16_SNORM = 13
    R16G16B16A16_SINT = 14
    R32G32_TYPELESS = 15
    R32G32_FLOAT = 16
    R32G32_UINT = 17
    R32G32_SINT = 18
    R32G8X24_TYPELESS = 19
    D32_FLOAT_S8X24_UINT = 20
    R32_FLOAT_X8X24_TYPELESS = 21
    X32_TYPELESS_G8X24_UINT = 22
    R10G10B10A2_TYPELESS = 23
    R10G10B10A2_UNORM = 24
    R10G10B10A2_UINT = 25
    R11G11B10_FLOAT = 26
    R8G8B8A8_TYPELESS = 27
    R8G8B8A8_UNORM = 28
    R8G8B8A8_UNORM_SRGB = 29
    R8G8B8A8_UINT = 30
    R8G8B8A8_SNORM = 31
    R8G8B8A8_SINT = 32
    R16G16_TYPELESS = 33
    R16G16_FLOAT = 34
    R16G16_UNORM = 35
    R16G16_UINT = 36
    R16G16_SNORM = 37
    R16G16_SINT = 38
    R32_TYPELESS = 39
    D32_FLOAT = 40
    R32_FLOAT = 41
    R32_UINT = 42
    R32_SINT = 43
    R24G8_TYPELESS = 44
    D24_UNORM_S8_UINT = 45
    R24_UNORM_X8_TYPELESS = 46
    X24_TYPELESS_G8_UINT = 47
    R8G8_TYPELESS = 48
    R8G8_UNORM = 49
    R8G8_UINT = 50
    R8G8_SNORM = 51
    R8G8_SINT = 52
    R16_TYPELESS = 53
    R16_FLOAT = 54
    D16_UNORM = 55
    R16_UNORM = 56
    R16_UINT = 57
    R16_SNORM = 58
    R16_SINT = 59
    R8_TYPELESS = 60
    R8_UNORM = 61
    R8_UINT = 62
    R8_SNORM = 63
    R8_SINT = 64
    A8_UNORM = 65
    R1_UNORM = 66
    R9G9B9E5_SHAREDEXP = 67
    R8G8_B8G8_UNORM = 68
    G8R8_G8B8_UNORM = 69
    BC1_TYPELESS = 70
    BC1_UNORM = 71
    BC1_UNORM_SRGB = 72
    BC2_TYPELESS = 73
    BC2_UNORM = 74
    BC2_UNORM_SRGB = 75
    BC3_TYPELESS = 76
    BC3_UNORM = 77
    BC3_UNORM_SRGB = 78
    BC4_TYPELESS = 79
    BC4_UNORM = 80
    BC4_SNORM = 81
    BC5_TYPELESS = 82
    BC5_UNORM = 83
    BC5_SNORM = 84
    B5G6R5_UNORM = 85
    B5G5R5A1_UNORM = 86
    B8G8R8A8_UNORM = 87
    B8G8R8X8_UNORM = 88
    R10G10B10_XR_BIAS_A2_UNORM = 89
    B8G8R8A8_TYPELESS = 90
    B8G8R8A8_UNORM_SRGB = 91
    B8G8R8X8_TYPELESS = 92
    B8G8R8X8_UNORM_SRGB = 93
    BC6H_TYPELESS = 94
    BC6H_UF16 = 95
    BC6H_SF16 = 96
    BC7_TYPELESS = 97
    BC7_UNORM = 98
    BC7_UNORM_SRGB = 99
    AYUV = 100
    Y410 = 101
    Y416 = 102
    NV12 = 103
    P010 = 104
    P016 = 105
    OPAQUE_420 = 106
    YUY2 = 107
    Y210 = 108
    Y216 = 109
    NV11 = 110
    AI44 = 111
    IA44 = 112
    P8 = 113
    A8P8 = 114
    B4G4R4A4_UNORM = 115


class D3D10_RESOURCE_DIMENSION(IntEnum):
    """D3D10 Resource Dimension"""
    UNKNOWN = 0
    BUFFER = 1
    TEXTURE1D = 2
    TEXTURE2D = 3
    TEXTURE3D = 4


class DDS_RESOURCE_MISC(IntFlag):
    """DX10 Misc Flags"""
    TEXTURECUBE = 0x4


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


class TextureDecompressor(ABC):
    """Base class for texture decompression"""
    @abstractmethod
    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress texture data to RGBA8 format

        Args:
            data: Compressed texture data
            width: Texture width in pixels
            height: Texture height in pixels

        Returns:
            numpy array of shape (height, width, 4) with dtype uint8 (RGBA)
        """
        pass


class BC1Decompressor(TextureDecompressor):
    """BC1 (DXT1) texture decompressor - NumPy vectorization + Numba JIT"""

    @staticmethod
    @jit(nopython=True, cache=True)
    def _process_blocks_jit(colors, indices, output, blocks_x, blocks_y, width, height):
        """JIT-compiled block processing for BC1 decompression"""
        num_blocks = blocks_x * blocks_y
        for block_idx in range(num_blocks):
            block_x = block_idx % blocks_x
            block_y = block_idx // blocks_x

            # Extract 16 2-bit indices for this block
            idx_bits = indices[block_idx]

            # Calculate pixel coordinates
            y_start = block_y * 4
            y_end = min(y_start + 4, height)
            x_start = block_x * 4
            x_end = min(x_start + 4, width)

            # Process each pixel in the 4x4 block
            for pixel_idx in range(16):
                pixel_y = pixel_idx // 4
                pixel_x = pixel_idx % 4

                # Extract 2-bit color index for this pixel
                color_idx = (idx_bits >> (pixel_idx * 2)) & 0x3

                # Calculate output position
                out_y = y_start + pixel_y
                out_x = x_start + pixel_x

                # Copy color to output if within bounds
                if out_y < y_end and out_x < x_end:
                    output[out_y, out_x, 0] = colors[block_idx, color_idx, 0]
                    output[out_y, out_x, 1] = colors[block_idx, color_idx, 1]
                    output[out_y, out_x, 2] = colors[block_idx, color_idx, 2]
                    output[out_y, out_x, 3] = colors[block_idx, color_idx, 3]

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress BC1 texture data to RGBA8 using vectorized NumPy operations

        BC1 stores 4x4 pixel blocks in 8 bytes each:
        - 2 bytes: color0 (RGB565)
        - 2 bytes: color1 (RGB565)
        - 4 bytes: 16 2-bit indices (one per pixel)
        """
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        # Read all blocks at once
        blocks = np.frombuffer(data[:num_blocks * 8], dtype=np.uint8).reshape(-1, 8)

        # Extract color0 and color1 (RGB565) for all blocks
        c0_packed = blocks[:, 0].astype(np.uint16) | (blocks[:, 1].astype(np.uint16) << 8)
        c1_packed = blocks[:, 2].astype(np.uint16) | (blocks[:, 3].astype(np.uint16) << 8)

        # Unpack RGB565 to RGB888 for all blocks at once
        c0_r = ((c0_packed >> 11) & 0x1F) << 3
        c0_r |= c0_r >> 5
        c0_g = ((c0_packed >> 5) & 0x3F) << 2
        c0_g |= c0_g >> 6
        c0_b = (c0_packed & 0x1F) << 3
        c0_b |= c0_b >> 5

        c1_r = ((c1_packed >> 11) & 0x1F) << 3
        c1_r |= c1_r >> 5
        c1_g = ((c1_packed >> 5) & 0x3F) << 2
        c1_g |= c1_g >> 6
        c1_b = (c1_packed & 0x1F) << 3
        c1_b |= c1_b >> 5

        # Build color palettes for all blocks (num_blocks, 4, 4) - 4 colors, RGBA
        colors = np.zeros((num_blocks, 4, 4), dtype=np.uint8)
        colors[:, 0, 0] = c0_r
        colors[:, 0, 1] = c0_g
        colors[:, 0, 2] = c0_b
        colors[:, 0, 3] = 255

        colors[:, 1, 0] = c1_r
        colors[:, 1, 1] = c1_g
        colors[:, 1, 2] = c1_b
        colors[:, 1, 3] = 255

        # Determine mode (4-color vs 3-color) for all blocks
        four_color_mode = c0_packed > c1_packed

        # Calculate color2 and color3 for all blocks
        colors[:, 2, 0] = np.where(four_color_mode, (2 * c0_r + c1_r) // 3, (c0_r + c1_r) // 2)
        colors[:, 2, 1] = np.where(four_color_mode, (2 * c0_g + c1_g) // 3, (c0_g + c1_g) // 2)
        colors[:, 2, 2] = np.where(four_color_mode, (2 * c0_b + c1_b) // 3, (c0_b + c1_b) // 2)
        colors[:, 2, 3] = 255

        colors[:, 3, 0] = np.where(four_color_mode, (c0_r + 2 * c1_r) // 3, 0)
        colors[:, 3, 1] = np.where(four_color_mode, (c0_g + 2 * c1_g) // 3, 0)
        colors[:, 3, 2] = np.where(four_color_mode, (c0_b + 2 * c1_b) // 3, 0)
        colors[:, 3, 3] = np.where(four_color_mode, 255, 0)

        # Extract indices for all blocks
        indices = blocks[:, 4].astype(np.uint32) | \
                  (blocks[:, 5].astype(np.uint32) << 8) | \
                  (blocks[:, 6].astype(np.uint32) << 16) | \
                  (blocks[:, 7].astype(np.uint32) << 24)

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.uint8)

        # Process all blocks using JIT-compiled function
        self._process_blocks_jit(colors, indices, output, blocks_x, blocks_y, width, height)

        return output


class BC2Decompressor(TextureDecompressor):
    """
    BC2 (DXT2/DXT3) texture decompressor - NumPy vectorization + Numba JIT

    Handles both DXT2 (premultiplied alpha) and DXT3 (straight alpha).
    The decompressor extracts raw RGBA values; premultiplied alpha
    conversion is handled separately by the DDS class if needed.
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _process_blocks_jit(colors, alphas, indices, output, blocks_x, blocks_y, width, height):
        """JIT-compiled block processing for BC2 decompression"""
        num_blocks = blocks_x * blocks_y
        for block_idx in range(num_blocks):
            block_x = block_idx % blocks_x
            block_y = block_idx // blocks_x

            idx_bits = indices[block_idx]

            y_start = block_y * 4
            y_end = min(y_start + 4, height)
            x_start = block_x * 4
            x_end = min(x_start + 4, width)

            for pixel_idx in range(16):
                pixel_y = pixel_idx // 4
                pixel_x = pixel_idx % 4

                color_idx = (idx_bits >> (pixel_idx * 2)) & 0x3

                out_y = y_start + pixel_y
                out_x = x_start + pixel_x

                if out_y < y_end and out_x < x_end:
                    output[out_y, out_x, 0] = colors[block_idx, color_idx, 0]
                    output[out_y, out_x, 1] = colors[block_idx, color_idx, 1]
                    output[out_y, out_x, 2] = colors[block_idx, color_idx, 2]
                    output[out_y, out_x, 3] = alphas[block_idx, pixel_idx]

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress BC2 texture data to RGBA8 using vectorized NumPy operations

        BC2 stores 4x4 pixel blocks in 16 bytes each:
        - 8 bytes: explicit alpha (4 bits per pixel, 16 pixels)
        - 2 bytes: color0 (RGB565)
        - 2 bytes: color1 (RGB565)
        - 4 bytes: 16 2-bit indices (one per pixel)
        """
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        # Read all blocks at once
        blocks = np.frombuffer(data[:num_blocks * 16], dtype=np.uint8).reshape(-1, 16)

        # Extract alpha data (first 8 bytes) for all blocks
        alpha_data = blocks[:, :8].copy().view(np.uint64).flatten()

        # Expand 4-bit alpha to 8-bit for all blocks (vectorized)
        alphas = np.zeros((num_blocks, 16), dtype=np.uint8)
        for i in range(16):
            alpha4 = (alpha_data >> (i * 4)) & 0xF
            alphas[:, i] = (alpha4 << 4) | alpha4

        # Extract color0 and color1 (RGB565) for all blocks
        c0_packed = blocks[:, 8].astype(np.uint16) | (blocks[:, 9].astype(np.uint16) << 8)
        c1_packed = blocks[:, 10].astype(np.uint16) | (blocks[:, 11].astype(np.uint16) << 8)

        # Unpack RGB565 to RGB888 for all blocks at once
        c0_r = ((c0_packed >> 11) & 0x1F) << 3
        c0_r |= c0_r >> 5
        c0_g = ((c0_packed >> 5) & 0x3F) << 2
        c0_g |= c0_g >> 6
        c0_b = (c0_packed & 0x1F) << 3
        c0_b |= c0_b >> 5

        c1_r = ((c1_packed >> 11) & 0x1F) << 3
        c1_r |= c1_r >> 5
        c1_g = ((c1_packed >> 5) & 0x3F) << 2
        c1_g |= c1_g >> 6
        c1_b = (c1_packed & 0x1F) << 3
        c1_b |= c1_b >> 5

        # Build color palettes for all blocks (BC2 always uses 4-color mode)
        colors = np.zeros((num_blocks, 4, 3), dtype=np.uint8)
        colors[:, 0, 0] = c0_r
        colors[:, 0, 1] = c0_g
        colors[:, 0, 2] = c0_b

        colors[:, 1, 0] = c1_r
        colors[:, 1, 1] = c1_g
        colors[:, 1, 2] = c1_b

        colors[:, 2, 0] = (2 * c0_r + c1_r) // 3
        colors[:, 2, 1] = (2 * c0_g + c1_g) // 3
        colors[:, 2, 2] = (2 * c0_b + c1_b) // 3

        colors[:, 3, 0] = (c0_r + 2 * c1_r) // 3
        colors[:, 3, 1] = (c0_g + 2 * c1_g) // 3
        colors[:, 3, 2] = (c0_b + 2 * c1_b) // 3

        # Extract indices for all blocks
        indices = blocks[:, 12].astype(np.uint32) | \
                  (blocks[:, 13].astype(np.uint32) << 8) | \
                  (blocks[:, 14].astype(np.uint32) << 16) | \
                  (blocks[:, 15].astype(np.uint32) << 24)

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.uint8)

        # Process all blocks using JIT-compiled function
        self._process_blocks_jit(colors, alphas, indices, output, blocks_x, blocks_y, width, height)

        return output


class BC3Decompressor(TextureDecompressor):
    """
    BC3 (DXT4/DXT5) texture decompressor - NumPy vectorization + Numba JIT

    Handles both DXT4 (premultiplied alpha) and DXT5 (straight alpha).
    BC3 uses interpolated alpha, similar to BC1's color interpolation.
    The decompressor extracts raw RGBA values; premultiplied alpha
    conversion is handled separately by the DDS class if needed.
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _process_blocks_jit(colors, alpha_palettes, alpha_indices, color_indices, output, blocks_x, blocks_y, width, height):
        """JIT-compiled block processing for BC3 decompression"""
        num_blocks = blocks_x * blocks_y
        for block_idx in range(num_blocks):
            block_x = block_idx % blocks_x
            block_y = block_idx // blocks_x

            alpha_idx_bits = alpha_indices[block_idx]
            color_idx_bits = color_indices[block_idx]

            y_start = block_y * 4
            y_end = min(y_start + 4, height)
            x_start = block_x * 4
            x_end = min(x_start + 4, width)

            for pixel_idx in range(16):
                pixel_y = pixel_idx // 4
                pixel_x = pixel_idx % 4

                alpha_idx = (alpha_idx_bits >> (pixel_idx * 3)) & 0x7
                color_idx = (color_idx_bits >> (pixel_idx * 2)) & 0x3

                out_y = y_start + pixel_y
                out_x = x_start + pixel_x

                if out_y < y_end and out_x < x_end:
                    output[out_y, out_x, 0] = colors[block_idx, color_idx, 0]
                    output[out_y, out_x, 1] = colors[block_idx, color_idx, 1]
                    output[out_y, out_x, 2] = colors[block_idx, color_idx, 2]
                    output[out_y, out_x, 3] = alpha_palettes[block_idx, alpha_idx]

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress BC3 texture data to RGBA8 using vectorized NumPy operations

        BC3 stores 4x4 pixel blocks in 16 bytes each:
        - 1 byte: alpha0 endpoint
        - 1 byte: alpha1 endpoint
        - 6 bytes: 16 3-bit alpha indices (48 bits total)
        - 2 bytes: color0 (RGB565)
        - 2 bytes: color1 (RGB565)
        - 4 bytes: 16 2-bit color indices (one per pixel)
        """
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        # Read all blocks at once
        blocks = np.frombuffer(data[:num_blocks * 16], dtype=np.uint8).reshape(-1, 16)

        # Extract alpha endpoints for all blocks
        alpha0 = blocks[:, 0].astype(np.uint16)
        alpha1 = blocks[:, 1].astype(np.uint16)

        # Build alpha palettes for all blocks (vectorized)
        alpha_palettes = np.zeros((num_blocks, 8), dtype=np.uint8)
        alpha_palettes[:, 0] = alpha0
        alpha_palettes[:, 1] = alpha1

        # Determine mode (8-alpha vs 6-alpha) for all blocks
        eight_alpha_mode = alpha0 > alpha1

        # Calculate interpolated alpha values for all blocks
        alpha_palettes[:, 2] = np.where(eight_alpha_mode, (6 * alpha0 + alpha1) // 7, (4 * alpha0 + alpha1) // 5)
        alpha_palettes[:, 3] = np.where(eight_alpha_mode, (5 * alpha0 + 2 * alpha1) // 7, (3 * alpha0 + 2 * alpha1) // 5)
        alpha_palettes[:, 4] = np.where(eight_alpha_mode, (4 * alpha0 + 3 * alpha1) // 7, (2 * alpha0 + 3 * alpha1) // 5)
        alpha_palettes[:, 5] = np.where(eight_alpha_mode, (3 * alpha0 + 4 * alpha1) // 7, (alpha0 + 4 * alpha1) // 5)
        alpha_palettes[:, 6] = np.where(eight_alpha_mode, (2 * alpha0 + 5 * alpha1) // 7, 0)
        alpha_palettes[:, 7] = np.where(eight_alpha_mode, (alpha0 + 6 * alpha1) // 7, 255)

        # Extract alpha indices (48 bits, 3 bits per pixel)
        alpha_indices_bytes = np.pad(blocks[:, 2:8], ((0, 0), (0, 2)), constant_values=0)
        alpha_indices = alpha_indices_bytes.copy().view(np.int64).flatten()

        # Extract color0 and color1 (RGB565) for all blocks
        c0_packed = blocks[:, 8].astype(np.uint16) | (blocks[:, 9].astype(np.uint16) << 8)
        c1_packed = blocks[:, 10].astype(np.uint16) | (blocks[:, 11].astype(np.uint16) << 8)

        # Unpack RGB565 to RGB888 for all blocks at once
        c0_r = ((c0_packed >> 11) & 0x1F) << 3
        c0_r |= c0_r >> 5
        c0_g = ((c0_packed >> 5) & 0x3F) << 2
        c0_g |= c0_g >> 6
        c0_b = (c0_packed & 0x1F) << 3
        c0_b |= c0_b >> 5

        c1_r = ((c1_packed >> 11) & 0x1F) << 3
        c1_r |= c1_r >> 5
        c1_g = ((c1_packed >> 5) & 0x3F) << 2
        c1_g |= c1_g >> 6
        c1_b = (c1_packed & 0x1F) << 3
        c1_b |= c1_b >> 5

        # Build color palettes for all blocks (BC3 always uses 4-color mode)
        colors = np.zeros((num_blocks, 4, 3), dtype=np.uint8)
        colors[:, 0, 0] = c0_r
        colors[:, 0, 1] = c0_g
        colors[:, 0, 2] = c0_b

        colors[:, 1, 0] = c1_r
        colors[:, 1, 1] = c1_g
        colors[:, 1, 2] = c1_b

        colors[:, 2, 0] = (2 * c0_r + c1_r) // 3
        colors[:, 2, 1] = (2 * c0_g + c1_g) // 3
        colors[:, 2, 2] = (2 * c0_b + c1_b) // 3

        colors[:, 3, 0] = (c0_r + 2 * c1_r) // 3
        colors[:, 3, 1] = (c0_g + 2 * c1_g) // 3
        colors[:, 3, 2] = (c0_b + 2 * c1_b) // 3

        # Extract color indices for all blocks
        color_indices = blocks[:, 12].astype(np.uint32) | \
                       (blocks[:, 13].astype(np.uint32) << 8) | \
                       (blocks[:, 14].astype(np.uint32) << 16) | \
                       (blocks[:, 15].astype(np.uint32) << 24)

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.uint8)

        # Process all blocks using JIT-compiled function
        self._process_blocks_jit(colors, alpha_palettes, alpha_indices, color_indices, output, blocks_x, blocks_y, width, height)

        return output


class BC4Decompressor(TextureDecompressor):
    """
    BC4 texture decompressor - NumPy vectorization + Numba JIT

    BC4 stores a single interpolated channel (typically used for grayscale).
    The decompressor outputs the channel to all RGB components with full alpha.
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _process_blocks_jit(palettes, indices, output, blocks_x, blocks_y, width, height):
        """JIT-compiled block processing for BC4 decompression"""
        num_blocks = blocks_x * blocks_y
        for block_idx in range(num_blocks):
            block_x = block_idx % blocks_x
            block_y = block_idx // blocks_x

            idx_bits = indices[block_idx]

            y_start = block_y * 4
            y_end = min(y_start + 4, height)
            x_start = block_x * 4
            x_end = min(x_start + 4, width)

            for pixel_idx in range(16):
                pixel_y = pixel_idx // 4
                pixel_x = pixel_idx % 4

                value_idx = (idx_bits >> (pixel_idx * 3)) & 0x7
                value = palettes[block_idx, value_idx]

                out_y = y_start + pixel_y
                out_x = x_start + pixel_x

                if out_y < y_end and out_x < x_end:
                    output[out_y, out_x, 0] = value
                    output[out_y, out_x, 1] = value
                    output[out_y, out_x, 2] = value
                    output[out_y, out_x, 3] = 255

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress BC4 texture data to RGBA8 using vectorized NumPy operations

        BC4 stores 4x4 pixel blocks in 8 bytes each:
        - 1 byte: value0 endpoint
        - 1 byte: value1 endpoint
        - 6 bytes: 16 3-bit indices (48 bits total)
        """
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        # Read all blocks at once
        blocks = np.frombuffer(data[:num_blocks * 8], dtype=np.uint8).reshape(-1, 8)

        # Extract endpoints for all blocks
        value0 = blocks[:, 0].astype(np.uint16)
        value1 = blocks[:, 1].astype(np.uint16)

        # Build value palettes for all blocks (vectorized)
        palettes = np.zeros((num_blocks, 8), dtype=np.uint8)
        palettes[:, 0] = value0
        palettes[:, 1] = value1

        # Determine mode (8-value vs 6-value) for all blocks
        eight_value_mode = value0 > value1

        # Calculate interpolated values for all blocks
        palettes[:, 2] = np.where(eight_value_mode, (6 * value0 + value1) // 7, (4 * value0 + value1) // 5)
        palettes[:, 3] = np.where(eight_value_mode, (5 * value0 + 2 * value1) // 7, (3 * value0 + 2 * value1) // 5)
        palettes[:, 4] = np.where(eight_value_mode, (4 * value0 + 3 * value1) // 7, (2 * value0 + 3 * value1) // 5)
        palettes[:, 5] = np.where(eight_value_mode, (3 * value0 + 4 * value1) // 7, (value0 + 4 * value1) // 5)
        palettes[:, 6] = np.where(eight_value_mode, (2 * value0 + 5 * value1) // 7, 0)
        palettes[:, 7] = np.where(eight_value_mode, (value0 + 6 * value1) // 7, 255)

        # Extract indices (48 bits, 3 bits per pixel)
        indices_bytes = np.pad(blocks[:, 2:8], ((0, 0), (0, 2)), constant_values=0)
        indices = indices_bytes.copy().view(np.int64).flatten()

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.uint8)

        # Process all blocks using JIT-compiled function
        self._process_blocks_jit(palettes, indices, output, blocks_x, blocks_y, width, height)

        return output


class BC5Decompressor(TextureDecompressor):
    """
    BC5 texture decompressor - NumPy vectorization + Numba JIT

    BC5 stores two interpolated channels (typically used for normal maps).
    The format is essentially two BC4 blocks side by side.
    The decompressor outputs red and green channels with blue=0 and alpha=255.
    For normal maps, blue can be reconstructed as sqrt(1 - R^2 - G^2).
    """

    @staticmethod
    @jit(nopython=True, cache=True)
    def _process_blocks_jit(red_palettes, green_palettes, red_indices, green_indices, output, blocks_x, blocks_y, width, height):
        """JIT-compiled block processing for BC5 decompression"""
        num_blocks = blocks_x * blocks_y
        for block_idx in range(num_blocks):
            block_x = block_idx % blocks_x
            block_y = block_idx // blocks_x

            red_idx_bits = red_indices[block_idx]
            green_idx_bits = green_indices[block_idx]

            y_start = block_y * 4
            y_end = min(y_start + 4, height)
            x_start = block_x * 4
            x_end = min(x_start + 4, width)

            for pixel_idx in range(16):
                pixel_y = pixel_idx // 4
                pixel_x = pixel_idx % 4

                red_idx = (red_idx_bits >> (pixel_idx * 3)) & 0x7
                green_idx = (green_idx_bits >> (pixel_idx * 3)) & 0x7

                out_y = y_start + pixel_y
                out_x = x_start + pixel_x

                if out_y < y_end and out_x < x_end:
                    output[out_y, out_x, 0] = red_palettes[block_idx, red_idx]
                    output[out_y, out_x, 1] = green_palettes[block_idx, green_idx]
                    output[out_y, out_x, 2] = 0
                    output[out_y, out_x, 3] = 255

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress BC5 texture data to RGBA8 using vectorized NumPy operations

        BC5 stores 4x4 pixel blocks in 16 bytes each:
        - 8 bytes: red channel (BC4 block)
        - 8 bytes: green channel (BC4 block)
        """
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        # Read all blocks at once
        blocks = np.frombuffer(data[:num_blocks * 16], dtype=np.uint8).reshape(-1, 16)

        # Process red channel (first 8 bytes)
        red_blocks = blocks[:, :8]
        red_value0 = red_blocks[:, 0].astype(np.uint16)
        red_value1 = red_blocks[:, 1].astype(np.uint16)

        # Build red palettes
        red_palettes = np.zeros((num_blocks, 8), dtype=np.uint8)
        red_palettes[:, 0] = red_value0
        red_palettes[:, 1] = red_value1

        red_eight_mode = red_value0 > red_value1
        red_palettes[:, 2] = np.where(red_eight_mode, (6 * red_value0 + red_value1) // 7, (4 * red_value0 + red_value1) // 5)
        red_palettes[:, 3] = np.where(red_eight_mode, (5 * red_value0 + 2 * red_value1) // 7, (3 * red_value0 + 2 * red_value1) // 5)
        red_palettes[:, 4] = np.where(red_eight_mode, (4 * red_value0 + 3 * red_value1) // 7, (2 * red_value0 + 3 * red_value1) // 5)
        red_palettes[:, 5] = np.where(red_eight_mode, (3 * red_value0 + 4 * red_value1) // 7, (red_value0 + 4 * red_value1) // 5)
        red_palettes[:, 6] = np.where(red_eight_mode, (2 * red_value0 + 5 * red_value1) // 7, 0)
        red_palettes[:, 7] = np.where(red_eight_mode, (red_value0 + 6 * red_value1) // 7, 255)

        # Extract red indices
        red_indices_bytes = np.pad(red_blocks[:, 2:8], ((0, 0), (0, 2)), constant_values=0)
        red_indices = red_indices_bytes.copy().view(np.int64).flatten()

        # Process green channel (second 8 bytes)
        green_blocks = blocks[:, 8:16]
        green_value0 = green_blocks[:, 0].astype(np.uint16)
        green_value1 = green_blocks[:, 1].astype(np.uint16)

        # Build green palettes
        green_palettes = np.zeros((num_blocks, 8), dtype=np.uint8)
        green_palettes[:, 0] = green_value0
        green_palettes[:, 1] = green_value1

        green_eight_mode = green_value0 > green_value1
        green_palettes[:, 2] = np.where(green_eight_mode, (6 * green_value0 + green_value1) // 7, (4 * green_value0 + green_value1) // 5)
        green_palettes[:, 3] = np.where(green_eight_mode, (5 * green_value0 + 2 * green_value1) // 7, (3 * green_value0 + 2 * green_value1) // 5)
        green_palettes[:, 4] = np.where(green_eight_mode, (4 * green_value0 + 3 * green_value1) // 7, (2 * green_value0 + 3 * green_value1) // 5)
        green_palettes[:, 5] = np.where(green_eight_mode, (3 * green_value0 + 4 * green_value1) // 7, (green_value0 + 4 * green_value1) // 5)
        green_palettes[:, 6] = np.where(green_eight_mode, (2 * green_value0 + 5 * green_value1) // 7, 0)
        green_palettes[:, 7] = np.where(green_eight_mode, (green_value0 + 6 * green_value1) // 7, 255)

        # Extract green indices
        green_indices_bytes = np.pad(green_blocks[:, 2:8], ((0, 0), (0, 2)), constant_values=0)
        green_indices = green_indices_bytes.copy().view(np.int64).flatten()

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.uint8)

        # Process all blocks using JIT-compiled function
        self._process_blocks_jit(red_palettes, green_palettes, red_indices, green_indices, output, blocks_x, blocks_y, width, height)

        return output


class DDS:
    """DirectDraw Surface container"""
    def __init__(self) -> None:
        self.magic: bytes = b'DDS '  # Magic number (always "DDS ")
        self.header: DDS_HEADER = DDS_HEADER()
        self.header10: Optional[DDS_HEADER_DXT10] = None  # Optional DX10 extended header
        self.data: bytes = b''  # Image data

    def __str__(self) -> str:
        """Return debug string representation of DDS file"""
        lines = ["DDS File Information:"]
        lines.append(f"  Magic: {self.magic}")
        lines.append(f"  Dimensions: {self.header.dwWidth}x{self.header.dwHeight}")

        if self.header.dwDepth > 0:
            lines.append(f"  Depth: {self.header.dwDepth}")

        if self.header.dwMipMapCount > 0:
            lines.append(f"  Mipmap Levels: {self.header.dwMipMapCount}")

        lines.append(f"  Flags: {self.header.dwFlags}")

        # Format information
        if self.header10:
            lines.append(f"  Format: DX10")
            lines.append(f"    DXGI Format: {self.header10.dxgiFormat.name} ({self.header10.dxgiFormat.value})")
            lines.append(f"    Resource Dimension: {self.header10.resourceDimension.name}")
            if self.header10.arraySize > 1:
                lines.append(f"    Array Size: {self.header10.arraySize}")
            if self.header10.miscFlag:
                lines.append(f"    Misc Flags: {self.header10.miscFlag}")
        else:
            # Try to decode FourCC if present
            if self.header.ddspf.dwFlags & DDPF.FOURCC:
                fourcc = self.header.ddspf.dwFourCC
                try:
                    fourcc_enum = FourCC(fourcc)
                    fourcc_str = fourcc_enum.name
                except ValueError:
                    # Unknown FourCC, display as bytes
                    fourcc_bytes = fourcc.to_bytes(4, 'little')
                    fourcc_str = fourcc_bytes.decode('ascii', errors='replace')
                lines.append(f"  Format: FourCC '{fourcc_str}' (0x{fourcc:08X})")
            elif self.header.ddspf.dwFlags & DDPF.RGB:
                lines.append(f"  Format: Uncompressed RGB ({self.header.ddspf.dwRGBBitCount}-bit)")
                if self.header.ddspf.dwFlags & DDPF.ALPHAPIXELS:
                    lines.append(f"    With Alpha")
            else:
                lines.append(f"  Format: Other (flags: {self.header.ddspf.dwFlags})")

        lines.append(f"  Caps: {self.header.dwCaps}")
        if self.header.dwCaps2:
            lines.append(f"  Caps2: {self.header.dwCaps2}")

        lines.append(f"  Data Size: {len(self.data)} bytes")

        return "\n".join(lines)

    @classmethod
    def from_bytes(cls, data: bytes) -> 'DDS':
        """Read DDS from bytes"""
        if len(data) < 128:  # Minimum: 4 (magic) + 124 (header)
            raise ValueError(f"Data too small for DDS file: {len(data)} bytes")

        dds = cls()
        offset = 0

        # Read and validate magic number
        dds.magic = data[offset:offset+4]
        if dds.magic != b'DDS ':
            raise ValueError(f"Invalid DDS magic number: {dds.magic}")
        offset += 4

        # Read DDS header
        dds.header = DDS_HEADER.from_bytes(data[offset:offset+124])
        offset += 124

        # Check if DX10 extended header is present
        if dds.header.ddspf.dwFourCC == FourCC.DX10:
            if len(data) < offset + 20:
                raise ValueError(f"Incomplete DX10 header")
            dds.header10 = DDS_HEADER_DXT10.from_bytes(data[offset:offset+20])
            offset += 20

        # Read image data
        dds.data = data[offset:]

        return dds

    def _get_mipmap_size(self, width: int, height: int, fourcc: Optional[int], dxgi_format: Optional[DXGI_FORMAT]) -> int:
        """
        Calculate the size in bytes of a single mipmap level for the given format.

        Args:
            width: Width of mipmap level in pixels
            height: Height of mipmap level in pixels
            fourcc: FourCC format code (if applicable)
            dxgi_format: DXGI format (if applicable)

        Returns:
            Size of the mipmap level in bytes
        """
        # BC1/DXT1 - 8 bytes per 4x4 block
        if fourcc == FourCC.DXT1 or dxgi_format in (DXGI_FORMAT.BC1_UNORM, DXGI_FORMAT.BC1_UNORM_SRGB, DXGI_FORMAT.BC1_TYPELESS):
            blocks_x = (width + 3) // 4
            blocks_y = (height + 3) // 4
            return blocks_x * blocks_y * 8

        # BC2/DXT3 - 16 bytes per 4x4 block
        if fourcc in (FourCC.DXT2, FourCC.DXT3) or dxgi_format in (DXGI_FORMAT.BC2_UNORM, DXGI_FORMAT.BC2_UNORM_SRGB, DXGI_FORMAT.BC2_TYPELESS):
            blocks_x = (width + 3) // 4
            blocks_y = (height + 3) // 4
            return blocks_x * blocks_y * 16

        # BC3/DXT5 - 16 bytes per 4x4 block
        if fourcc in (FourCC.DXT4, FourCC.DXT5) or dxgi_format in (DXGI_FORMAT.BC3_UNORM, DXGI_FORMAT.BC3_UNORM_SRGB, DXGI_FORMAT.BC3_TYPELESS):
            blocks_x = (width + 3) // 4
            blocks_y = (height + 3) // 4
            return blocks_x * blocks_y * 16

        # BC4 - 8 bytes per 4x4 block
        if fourcc in (FourCC.BC4U, FourCC.BC4S, FourCC.ATI1) or dxgi_format in (DXGI_FORMAT.BC4_UNORM, DXGI_FORMAT.BC4_SNORM, DXGI_FORMAT.BC4_TYPELESS):
            blocks_x = (width + 3) // 4
            blocks_y = (height + 3) // 4
            return blocks_x * blocks_y * 8

        # BC5 - 16 bytes per 4x4 block
        if fourcc in (FourCC.BC5U, FourCC.BC5S, FourCC.ATI2) or dxgi_format in (DXGI_FORMAT.BC5_UNORM, DXGI_FORMAT.BC5_SNORM, DXGI_FORMAT.BC5_TYPELESS):
            blocks_x = (width + 3) // 4
            blocks_y = (height + 3) // 4
            return blocks_x * blocks_y * 16

        raise NotImplementedError(f"Mipmap size calculation not implemented for {self._get_format_str()}")

    def _unpremultiply_alpha(rgba_data: np.ndarray) -> np.ndarray:
        """
        Convert premultiplied alpha to straight alpha.

        In premultiplied alpha, RGB values are stored as RGB × Alpha.
        This function converts them back to the original RGB values.

        Args:
            rgba_data: numpy array of shape (height, width, 4) with dtype uint8 (RGBA)
                    where RGB values are premultiplied by alpha

        Returns:
            numpy array of shape (height, width, 4) with dtype uint8 (RGBA)
            with straight (non-premultiplied) alpha
        """
        # Work with float to avoid precision loss
        result = rgba_data.astype(np.float32)

        # Get alpha channel
        alpha = result[:, :, 3]

        # Create a mask for non-zero alpha to avoid division by zero
        # Where alpha > 0, we'll un-premultiply
        non_zero_mask = alpha > 0

        # Un-premultiply RGB channels where alpha > 0
        # RGB_original = RGB_premultiplied / (Alpha / 255)
        for channel in range(3):  # R, G, B channels
            result[:, :, channel] = np.where(
                non_zero_mask,
                np.clip(result[:, :, channel] * 255.0 / alpha, 0, 255),
                result[:, :, channel]
            )

        # Convert back to uint8
        return result.astype(np.uint8)

    def _get_format_str(self) -> str:
        # Determine format
        fourcc = None
        dxgi_format = None

        if self.header10:
            dxgi_format = self.header10.dxgiFormat
        elif self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

        if fourcc:
            try:
                fourcc_enum = FourCC(fourcc)
                format_str = f"FourCC {fourcc_enum.name}"
            except ValueError:
                fourcc_bytes = fourcc.to_bytes(4, 'little')
                format_str = f"FourCC 0x{fourcc:08X} ({fourcc_bytes})"
        elif dxgi_format:
            format_str = f"{dxgi_format.name}"
        else:
            format_str = "Unknown format"
        return format_str


    def to_image(self, mipmap_level: int = 0) -> Image.Image:
        """
        Convert DDS texture to PIL Image

        Args:
            mipmap_level: Mipmap level to extract (0 = full resolution)

        Returns:
            PIL Image in RGBA mode

        Raises:
            ValueError: If the format is not supported or mipmap level is invalid
            NotImplementedError: If the format decompressor is not implemented yet
        """
        # Determine format
        fourcc = None
        dxgi_format = None

        if self.header10:
            dxgi_format = self.header10.dxgiFormat
        elif self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

        # Validate mipmap level
        mipmap_count = self.header.dwMipMapCount if self.header.dwMipMapCount > 0 else 1
        if mipmap_level < 0 or mipmap_level >= mipmap_count:
            raise ValueError(f"Invalid mipmap level {mipmap_level}. Texture has {mipmap_count} mipmap level(s).")

        # Calculate dimensions and offset for the requested mipmap level
        base_width = self.header.dwWidth
        base_height = self.header.dwHeight

        # Calculate offset by summing sizes of all previous mipmap levels
        data_offset = 0
        for level in range(mipmap_level):
            level_width = max(1, base_width >> level)
            level_height = max(1, base_height >> level)
            data_offset += self._get_mipmap_size(level_width, level_height, fourcc, dxgi_format)

        # Calculate dimensions for the requested mipmap level
        width = max(1, base_width >> mipmap_level)
        height = max(1, base_height >> mipmap_level)

        # Calculate size of the requested mipmap level
        mipmap_size = self._get_mipmap_size(width, height, fourcc, dxgi_format)

        # Validate that we have enough data
        if data_offset + mipmap_size > len(self.data):
            raise ValueError(f"Not enough data for mipmap level {mipmap_level}. Expected {data_offset + mipmap_size} bytes, got {len(self.data)} bytes.")

        # Extract data for this mipmap level
        mipmap_data = self.data[data_offset:data_offset + mipmap_size]

        # Select appropriate decompressor
        decompressor: Optional[TextureDecompressor] = None

        if fourcc == FourCC.DXT1:
            decompressor = BC1Decompressor()
        elif dxgi_format in (DXGI_FORMAT.BC1_UNORM, DXGI_FORMAT.BC1_UNORM_SRGB, DXGI_FORMAT.BC1_TYPELESS):
            decompressor = BC1Decompressor()
        elif fourcc in (FourCC.DXT2, FourCC.DXT3):
            decompressor = BC2Decompressor()
        elif dxgi_format in (DXGI_FORMAT.BC2_UNORM, DXGI_FORMAT.BC2_UNORM_SRGB, DXGI_FORMAT.BC2_TYPELESS):
            decompressor = BC2Decompressor()
        elif fourcc in (FourCC.DXT4, FourCC.DXT5):
            decompressor = BC3Decompressor()
        elif dxgi_format in (DXGI_FORMAT.BC3_UNORM, DXGI_FORMAT.BC3_UNORM_SRGB, DXGI_FORMAT.BC3_TYPELESS):
            decompressor = BC3Decompressor()
        elif fourcc in (FourCC.BC4U, FourCC.BC4S, FourCC.ATI1):
            decompressor = BC4Decompressor()
        elif dxgi_format in (DXGI_FORMAT.BC4_UNORM, DXGI_FORMAT.BC4_SNORM, DXGI_FORMAT.BC4_TYPELESS):
            decompressor = BC4Decompressor()
        elif fourcc in (FourCC.BC5U, FourCC.BC5S, FourCC.ATI2):
            decompressor = BC5Decompressor()
        elif dxgi_format in (DXGI_FORMAT.BC5_UNORM, DXGI_FORMAT.BC5_SNORM, DXGI_FORMAT.BC5_TYPELESS):
            decompressor = BC5Decompressor()
        else:
            raise NotImplementedError(f"Decompression not yet implemented for {self._get_format_str()}")

        # Decompress texture data for this mipmap level
        rgba_data = decompressor.decompress(mipmap_data, width, height)

        # Handle premultiplied alpha formats
        # DXT2 and DXT4 use premultiplied alpha and need conversion
        is_premultiplied = fourcc in (FourCC.DXT2, FourCC.DXT4)
        if is_premultiplied:
            rgba_data = self._unpremultiply_alpha(rgba_data)

        # Create PIL Image from numpy array
        return Image.fromarray(rgba_data, mode='RGBA')


def main():
    """Command-line interface for puredds"""
    import sys
    import os
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description='Read and convert DDS (DirectDraw Surface) texture files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  puredds texture.dds                          # Display DDS file info
  puredds texture.dds -o output.png            # Convert to PNG
  puredds texture.dds -o output.png -m 1       # Extract mipmap level 1
  puredds texture.dds -o output.png --mipmap 2 # Extract mipmap level 2
        """
    )

    parser.add_argument('input', help='Input DDS file path')
    parser.add_argument('-o', '--output', help='Output image file path (e.g., output.png)')
    parser.add_argument('-m', '--mipmap', type=int, default=0,
                        help='Mipmap level to extract (default: 0 = full resolution)')

    args = parser.parse_args()

    try:
        with open(args.input, 'rb') as f:
            data = f.read()

        dds = DDS.from_bytes(data)
        print(dds)

        # Try to convert to image if output path is specified
        if args.output:
            try:
                print(f"\nConverting to image (mipmap level {args.mipmap})...")

                # Time decompression
                start_decompress = time.perf_counter()
                image = dds.to_image(args.mipmap)
                end_decompress = time.perf_counter()
                decompress_time = end_decompress - start_decompress

                # Time saving
                start_save = time.perf_counter()
                image.save(args.output)
                end_save = time.perf_counter()
                save_time = end_save - start_save

                print(f"Saved to: {args.output}")
                print(f"Image size: {image.width}x{image.height}")
                print(f"Decompression time: {decompress_time*1000:.2f} ms")
                print(f"Save time: {save_time*1000:.2f} ms")
                print(f"Total conversion time: {(decompress_time + save_time)*1000:.2f} ms")
            except NotImplementedError as e:
                print(f"Cannot convert to image: {e}")
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error converting to image: {e}")
                import traceback
                traceback.print_exc()

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing DDS file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
