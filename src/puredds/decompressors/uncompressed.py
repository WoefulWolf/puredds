"""Generic decompressor for uncompressed DXGI formats"""
from dataclasses import dataclass
from typing import Tuple, Callable
import numpy as np
from .base import TextureDecompressor
from ..enums import DXGI_FORMAT


@dataclass
class FormatDescriptor:
    """Descriptor for uncompressed format properties"""
    dtype: np.dtype  # NumPy dtype for reading raw data
    channels: str  # Channel layout: 'R', 'RG', 'RGB', 'RGBA', 'BGRA', etc.
    normalization: str  # 'none', 'unorm', 'snorm'
    srgb: bool = False  # Apply sRGB gamma correction
    special_decoder: Callable[[bytes, int, int], np.ndarray] = None  # For special formats


class UncompressedDecompressor(TextureDecompressor):
    """Generic decompressor for uncompressed DXGI formats"""

    # Format descriptor mapping
    FORMAT_DESCRIPTORS = {
        # 128-bit formats (16 bytes per pixel)
        DXGI_FORMAT.R32G32B32A32_FLOAT: FormatDescriptor(np.dtype('<f4'), 'RGBA', 'none'),
        DXGI_FORMAT.R32G32B32A32_UINT: FormatDescriptor(np.dtype('<u4'), 'RGBA', 'none'),
        DXGI_FORMAT.R32G32B32A32_SINT: FormatDescriptor(np.dtype('<i4'), 'RGBA', 'none'),

        # 96-bit formats (12 bytes per pixel)
        DXGI_FORMAT.R32G32B32_FLOAT: FormatDescriptor(np.dtype('<f4'), 'RGB', 'none'),
        DXGI_FORMAT.R32G32B32_UINT: FormatDescriptor(np.dtype('<u4'), 'RGB', 'none'),
        DXGI_FORMAT.R32G32B32_SINT: FormatDescriptor(np.dtype('<i4'), 'RGB', 'none'),

        # 64-bit formats (8 bytes per pixel)
        DXGI_FORMAT.R16G16B16A16_FLOAT: FormatDescriptor(np.dtype('<f2'), 'RGBA', 'none'),
        DXGI_FORMAT.R16G16B16A16_UNORM: FormatDescriptor(np.dtype('<u2'), 'RGBA', 'unorm'),
        DXGI_FORMAT.R16G16B16A16_UINT: FormatDescriptor(np.dtype('<u2'), 'RGBA', 'none'),
        DXGI_FORMAT.R16G16B16A16_SNORM: FormatDescriptor(np.dtype('<i2'), 'RGBA', 'snorm'),
        DXGI_FORMAT.R16G16B16A16_SINT: FormatDescriptor(np.dtype('<i2'), 'RGBA', 'none'),

        DXGI_FORMAT.R32G32_FLOAT: FormatDescriptor(np.dtype('<f4'), 'RG', 'none'),
        DXGI_FORMAT.R32G32_UINT: FormatDescriptor(np.dtype('<u4'), 'RG', 'none'),
        DXGI_FORMAT.R32G32_SINT: FormatDescriptor(np.dtype('<i4'), 'RG', 'none'),

        # 32-bit formats (4 bytes per pixel)
        DXGI_FORMAT.R8G8B8A8_UNORM: FormatDescriptor(np.dtype('u1'), 'RGBA', 'unorm'),
        DXGI_FORMAT.R8G8B8A8_UNORM_SRGB: FormatDescriptor(np.dtype('u1'), 'RGBA', 'unorm', srgb=True),
        DXGI_FORMAT.R8G8B8A8_UINT: FormatDescriptor(np.dtype('u1'), 'RGBA', 'none'),
        DXGI_FORMAT.R8G8B8A8_SNORM: FormatDescriptor(np.dtype('i1'), 'RGBA', 'snorm'),
        DXGI_FORMAT.R8G8B8A8_SINT: FormatDescriptor(np.dtype('i1'), 'RGBA', 'none'),

        DXGI_FORMAT.B8G8R8A8_UNORM: FormatDescriptor(np.dtype('u1'), 'BGRA', 'unorm'),
        DXGI_FORMAT.B8G8R8A8_UNORM_SRGB: FormatDescriptor(np.dtype('u1'), 'BGRA', 'unorm', srgb=True),
        DXGI_FORMAT.B8G8R8X8_UNORM: FormatDescriptor(np.dtype('u1'), 'BGRX', 'unorm'),
        DXGI_FORMAT.B8G8R8X8_UNORM_SRGB: FormatDescriptor(np.dtype('u1'), 'BGRX', 'unorm', srgb=True),

        DXGI_FORMAT.R16G16_FLOAT: FormatDescriptor(np.dtype('<f2'), 'RG', 'none'),
        DXGI_FORMAT.R16G16_UNORM: FormatDescriptor(np.dtype('<u2'), 'RG', 'unorm'),
        DXGI_FORMAT.R16G16_UINT: FormatDescriptor(np.dtype('<u2'), 'RG', 'none'),
        DXGI_FORMAT.R16G16_SNORM: FormatDescriptor(np.dtype('<i2'), 'RG', 'snorm'),
        DXGI_FORMAT.R16G16_SINT: FormatDescriptor(np.dtype('<i2'), 'RG', 'none'),

        DXGI_FORMAT.R32_FLOAT: FormatDescriptor(np.dtype('<f4'), 'R', 'none'),
        DXGI_FORMAT.R32_UINT: FormatDescriptor(np.dtype('<u4'), 'R', 'none'),
        DXGI_FORMAT.R32_SINT: FormatDescriptor(np.dtype('<i4'), 'R', 'none'),
        DXGI_FORMAT.D32_FLOAT: FormatDescriptor(np.dtype('<f4'), 'R', 'none'),  # Depth

        # 16-bit formats (2 bytes per pixel)
        DXGI_FORMAT.R8G8_UNORM: FormatDescriptor(np.dtype('u1'), 'RG', 'unorm'),
        DXGI_FORMAT.R8G8_UINT: FormatDescriptor(np.dtype('u1'), 'RG', 'none'),
        DXGI_FORMAT.R8G8_SNORM: FormatDescriptor(np.dtype('i1'), 'RG', 'snorm'),
        DXGI_FORMAT.R8G8_SINT: FormatDescriptor(np.dtype('i1'), 'RG', 'none'),

        DXGI_FORMAT.R16_FLOAT: FormatDescriptor(np.dtype('<f2'), 'R', 'none'),
        DXGI_FORMAT.R16_UNORM: FormatDescriptor(np.dtype('<u2'), 'R', 'unorm'),
        DXGI_FORMAT.R16_UINT: FormatDescriptor(np.dtype('<u2'), 'R', 'none'),
        DXGI_FORMAT.R16_SNORM: FormatDescriptor(np.dtype('<i2'), 'R', 'snorm'),
        DXGI_FORMAT.R16_SINT: FormatDescriptor(np.dtype('<i2'), 'R', 'none'),
        DXGI_FORMAT.D16_UNORM: FormatDescriptor(np.dtype('<u2'), 'R', 'unorm'),  # Depth

        # 8-bit formats (1 byte per pixel)
        DXGI_FORMAT.R8_UNORM: FormatDescriptor(np.dtype('u1'), 'R', 'unorm'),
        DXGI_FORMAT.R8_UINT: FormatDescriptor(np.dtype('u1'), 'R', 'none'),
        DXGI_FORMAT.R8_SNORM: FormatDescriptor(np.dtype('i1'), 'R', 'snorm'),
        DXGI_FORMAT.R8_SINT: FormatDescriptor(np.dtype('i1'), 'R', 'none'),
        DXGI_FORMAT.A8_UNORM: FormatDescriptor(np.dtype('u1'), 'A', 'unorm'),
    }

    def __init__(self, dxgi_format: DXGI_FORMAT):
        """
        Initialize uncompressed decompressor

        Args:
            dxgi_format: DXGI format enum value
        """
        self.dxgi_format = dxgi_format

        # Handle special packed formats
        if dxgi_format == DXGI_FORMAT.R10G10B10A2_UNORM:
            self.descriptor = None  # Special handling
        elif dxgi_format == DXGI_FORMAT.R10G10B10A2_UINT:
            self.descriptor = None  # Special handling
        elif dxgi_format == DXGI_FORMAT.R11G11B10_FLOAT:
            self.descriptor = None  # Special handling
        elif dxgi_format == DXGI_FORMAT.B5G6R5_UNORM:
            self.descriptor = None  # Special handling
        elif dxgi_format == DXGI_FORMAT.B5G5R5A1_UNORM:
            self.descriptor = None  # Special handling
        elif dxgi_format == DXGI_FORMAT.B4G4R4A4_UNORM:
            self.descriptor = None  # Special handling
        else:
            self.descriptor = self.FORMAT_DESCRIPTORS.get(dxgi_format)
            if self.descriptor is None:
                raise ValueError(f"Unsupported uncompressed format: {dxgi_format}")

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress uncompressed texture data to RGBA8 format

        Args:
            data: Raw texture data
            width: Texture width in pixels
            height: Texture height in pixels

        Returns:
            numpy array of shape (height, width, 4) with dtype uint8 (RGBA)
        """
        # Handle special packed formats
        if self.descriptor is None:
            return self._decompress_packed(data, width, height)

        # Standard format decompression
        return self._decompress_standard(data, width, height)

    def _decompress_standard(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress standard uncompressed formats"""
        desc = self.descriptor
        num_channels = len(desc.channels)

        # Read raw pixel data
        pixels = np.frombuffer(data, dtype=desc.dtype)
        pixels = pixels.reshape(height, width, num_channels)

        # Convert to float [0, 1] range
        if desc.normalization == 'unorm':
            # Unsigned normalized: map [0, max] to [0, 1]
            if desc.dtype == np.dtype('u1'):
                pixels_float = pixels.astype(np.float32) / 255.0
            elif desc.dtype == np.dtype('<u2'):
                pixels_float = pixels.astype(np.float32) / 65535.0
            else:
                pixels_float = pixels.astype(np.float32)
        elif desc.normalization == 'snorm':
            # Signed normalized: map [-max, max] to [-1, 1], then clamp to [0, 1]
            if desc.dtype == np.dtype('i1'):
                pixels_float = np.maximum(pixels.astype(np.float32) / 127.0, -1.0)
            elif desc.dtype == np.dtype('<i2'):
                pixels_float = np.maximum(pixels.astype(np.float32) / 32767.0, -1.0)
            else:
                pixels_float = pixels.astype(np.float32)
            # Map [-1, 1] to [0, 1] for display
            pixels_float = (pixels_float + 1.0) * 0.5
        else:
            # No normalization (float or integer types)
            pixels_float = pixels.astype(np.float32)

            # For integer types, normalize to reasonable range
            if np.issubdtype(desc.dtype, np.integer):
                if desc.dtype == np.dtype('u1'):
                    pixels_float /= 255.0
                elif desc.dtype == np.dtype('<u2'):
                    pixels_float /= 65535.0
                elif desc.dtype == np.dtype('<u4'):
                    pixels_float /= 4294967295.0
                elif desc.dtype == np.dtype('i1'):
                    pixels_float = (pixels_float + 128.0) / 255.0
                elif desc.dtype == np.dtype('<i2'):
                    pixels_float = (pixels_float + 32768.0) / 65535.0
                elif desc.dtype == np.dtype('<i4'):
                    pixels_float = (pixels_float + 2147483648.0) / 4294967295.0

        # Clamp to [0, 1] range
        pixels_float = np.clip(pixels_float, 0.0, 1.0)

        # Apply sRGB to linear conversion if needed
        if desc.srgb:
            pixels_float = self._srgb_to_linear(pixels_float)

        # Swizzle channels to RGBA
        output = self._swizzle_to_rgba(pixels_float, desc.channels)

        # Convert to uint8 [0, 255]
        return (output * 255.0).astype(np.uint8)

    def _decompress_packed(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress packed pixel formats"""
        if self.dxgi_format == DXGI_FORMAT.R10G10B10A2_UNORM:
            return self._decompress_r10g10b10a2_unorm(data, width, height)
        elif self.dxgi_format == DXGI_FORMAT.R10G10B10A2_UINT:
            return self._decompress_r10g10b10a2_uint(data, width, height)
        elif self.dxgi_format == DXGI_FORMAT.R11G11B10_FLOAT:
            return self._decompress_r11g11b10_float(data, width, height)
        elif self.dxgi_format == DXGI_FORMAT.B5G6R5_UNORM:
            return self._decompress_b5g6r5_unorm(data, width, height)
        elif self.dxgi_format == DXGI_FORMAT.B5G5R5A1_UNORM:
            return self._decompress_b5g5r5a1_unorm(data, width, height)
        elif self.dxgi_format == DXGI_FORMAT.B4G4R4A4_UNORM:
            return self._decompress_b4g4r4a4_unorm(data, width, height)
        else:
            raise ValueError(f"Unsupported packed format: {self.dxgi_format}")

    def _decompress_r10g10b10a2_unorm(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress R10G10B10A2_UNORM format"""
        pixels = np.frombuffer(data, dtype=np.uint32).reshape(height, width)

        r = ((pixels >> 0) & 0x3FF).astype(np.float32) / 1023.0
        g = ((pixels >> 10) & 0x3FF).astype(np.float32) / 1023.0
        b = ((pixels >> 20) & 0x3FF).astype(np.float32) / 1023.0
        a = ((pixels >> 30) & 0x3).astype(np.float32) / 3.0

        output = np.stack([r, g, b, a], axis=-1)
        return (output * 255.0).astype(np.uint8)

    def _decompress_r10g10b10a2_uint(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress R10G10B10A2_UINT format"""
        pixels = np.frombuffer(data, dtype=np.uint32).reshape(height, width)

        r = ((pixels >> 0) & 0x3FF).astype(np.float32) / 1023.0
        g = ((pixels >> 10) & 0x3FF).astype(np.float32) / 1023.0
        b = ((pixels >> 20) & 0x3FF).astype(np.float32) / 1023.0
        a = ((pixels >> 30) & 0x3).astype(np.float32) / 3.0

        output = np.stack([r, g, b, a], axis=-1)
        return (output * 255.0).astype(np.uint8)

    def _decompress_r11g11b10_float(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress R11G11B10_FLOAT format (packed unsigned floats)"""
        pixels = np.frombuffer(data, dtype=np.uint32).reshape(height, width)

        # Extract packed components
        r_bits = (pixels >> 0) & 0x7FF
        g_bits = (pixels >> 11) & 0x7FF
        b_bits = (pixels >> 22) & 0x3FF

        # Decode 11-bit float (6 exponent, 5 mantissa)
        r = self._decode_float11(r_bits)
        g = self._decode_float11(g_bits)

        # Decode 10-bit float (5 exponent, 5 mantissa)
        b = self._decode_float10(b_bits)

        # Stack and clamp
        output = np.stack([r, g, b, np.ones_like(r)], axis=-1)
        output = np.clip(output, 0.0, 1.0)

        return (output * 255.0).astype(np.uint8)

    def _decompress_b5g6r5_unorm(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress B5G6R5_UNORM format"""
        pixels = np.frombuffer(data, dtype=np.uint16).reshape(height, width)

        b = ((pixels >> 11) & 0x1F).astype(np.uint8)
        g = ((pixels >> 5) & 0x3F).astype(np.uint8)
        r = ((pixels >> 0) & 0x1F).astype(np.uint8)

        # Expand to 8-bit by replicating high bits
        r = (r << 3) | (r >> 2)
        g = (g << 2) | (g >> 4)
        b = (b << 3) | (b >> 2)
        a = np.full_like(r, 255, dtype=np.uint8)

        return np.stack([r, g, b, a], axis=-1)

    def _decompress_b5g5r5a1_unorm(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress B5G5R5A1_UNORM format"""
        pixels = np.frombuffer(data, dtype=np.uint16).reshape(height, width)

        b = ((pixels >> 10) & 0x1F).astype(np.uint8)
        g = ((pixels >> 5) & 0x1F).astype(np.uint8)
        r = ((pixels >> 0) & 0x1F).astype(np.uint8)
        a = ((pixels >> 15) & 0x1).astype(np.uint8)

        # Expand to 8-bit
        r = (r << 3) | (r >> 2)
        g = (g << 3) | (g >> 2)
        b = (b << 3) | (b >> 2)
        a = a * 255

        return np.stack([r, g, b, a], axis=-1)

    def _decompress_b4g4r4a4_unorm(self, data: bytes, width: int, height: int) -> np.ndarray:
        """Decompress B4G4R4A4_UNORM format"""
        pixels = np.frombuffer(data, dtype=np.uint16).reshape(height, width)

        b = ((pixels >> 8) & 0xF).astype(np.uint8)
        g = ((pixels >> 4) & 0xF).astype(np.uint8)
        r = ((pixels >> 0) & 0xF).astype(np.uint8)
        a = ((pixels >> 12) & 0xF).astype(np.uint8)

        # Expand to 8-bit by replicating the 4 bits
        r = (r << 4) | r
        g = (g << 4) | g
        b = (b << 4) | b
        a = (a << 4) | a

        return np.stack([r, g, b, a], axis=-1)

    @staticmethod
    def _decode_float11(bits: np.ndarray) -> np.ndarray:
        """Decode 11-bit unsigned float (6-bit exponent, 5-bit mantissa)"""
        exponent = (bits >> 5) & 0x3F
        mantissa = bits & 0x1F

        # Handle special cases
        result = np.zeros_like(bits, dtype=np.float32)

        # Normal numbers
        mask = (exponent != 0) & (exponent != 0x3F)
        result[mask] = np.ldexp((mantissa[mask] / 32.0 + 1.0).astype(np.float32),
                                (exponent[mask] - 31).astype(np.int32))

        # Subnormal numbers
        mask = (exponent == 0) & (mantissa != 0)
        result[mask] = np.ldexp((mantissa[mask] / 32.0).astype(np.float32), -30)

        # Infinity/NaN (exponent == 0x3F) - clamp to 1.0
        mask = (exponent == 0x3F)
        result[mask] = 1.0

        return result

    @staticmethod
    def _decode_float10(bits: np.ndarray) -> np.ndarray:
        """Decode 10-bit unsigned float (5-bit exponent, 5-bit mantissa)"""
        exponent = (bits >> 5) & 0x1F
        mantissa = bits & 0x1F

        # Handle special cases
        result = np.zeros_like(bits, dtype=np.float32)

        # Normal numbers
        mask = (exponent != 0) & (exponent != 0x1F)
        result[mask] = np.ldexp((mantissa[mask] / 32.0 + 1.0).astype(np.float32),
                                (exponent[mask] - 15).astype(np.int32))

        # Subnormal numbers
        mask = (exponent == 0) & (mantissa != 0)
        result[mask] = np.ldexp((mantissa[mask] / 32.0).astype(np.float32), -14)

        # Infinity/NaN (exponent == 0x1F) - clamp to 1.0
        mask = (exponent == 0x1F)
        result[mask] = 1.0

        return result

    @staticmethod
    def _srgb_to_linear(srgb: np.ndarray) -> np.ndarray:
        """Convert sRGB to linear color space"""
        linear = np.where(
            srgb <= 0.04045,
            srgb / 12.92,
            np.power((srgb + 0.055) / 1.055, 2.4)
        )
        return linear

    @staticmethod
    def _swizzle_to_rgba(pixels: np.ndarray, channels: str) -> np.ndarray:
        """Swizzle channel layout to RGBA"""
        height, width = pixels.shape[:2]
        output = np.zeros((height, width, 4), dtype=np.float32)

        channel_map = {'R': 0, 'G': 1, 'B': 2, 'A': 3}

        for i, ch in enumerate(channels):
            if ch == 'X':
                # X means ignore/opaque alpha
                continue
            target_idx = channel_map.get(ch)
            if target_idx is not None:
                output[:, :, target_idx] = pixels[:, :, i]

        # Set default alpha to 1.0 if not specified
        if 'A' not in channels:
            output[:, :, 3] = 1.0

        return output
