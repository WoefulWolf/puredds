"""Main DDS file handler"""
from typing import Optional, List
import numpy as np

from .enums import DDPF, DDSCAPS, DDSCAPS2, DDSD, FourCC, DXGI_FORMAT, DDS_RESOURCE_MISC, D3D10_RESOURCE_DIMENSION
from .headers import DDS_HEADER, DDS_HEADER_DXT10
from .decompressors import (
    TextureDecompressor,
    BC1Decompressor,
    BC2Decompressor,
    BC3Decompressor,
    BC4Decompressor,
    BC5Decompressor,
    BC7Decompressor,
    UncompressedDecompressor,
)


def _format_flags(value: int, flag_enum) -> str:
    """
    Format an integer flag value as a list of flag names separated by ' | '.

    Args:
        value: The integer flag value
        flag_enum: The IntFlag enum class to use for decoding

    Returns:
        String with flag names separated by ' | ', or '0' if no flags are set
    """
    if value == 0:
        return '0'

    # Get all set flags
    flags = []
    for flag in flag_enum:
        if value & flag:
            flags.append(flag.name)

    if not flags:
        return f'0x{value:X}'

    return ' | '.join(flags)


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


class DDS:
    """DirectDraw Surface container"""
    def __init__(self) -> None:
        self.magic: bytes = b'DDS '  # Magic number (always "DDS ")
        self.header: DDS_HEADER = DDS_HEADER()
        self.header10: Optional[DDS_HEADER_DXT10] = None  # Optional DX10 extended header
        self.data: List[bytes] = []  # List of subresources (mipmap levels, array slices, cubemap faces)

    def __str__(self) -> str:
        """Return debug string representation of DDS file"""
        lines = ["DDS File Information:"]
        lines.append(f"  Magic: {self.magic}")
        lines.append(f"  Dimensions: {self.header.dwWidth}x{self.header.dwHeight}")
        lines.append(f"  Depth: {self.header.dwDepth}")

        if self.header.dwMipMapCount > 0:
            lines.append(f"  Mipmap Levels: {self.header.dwMipMapCount}")

        lines.append(f"  Flags: {_format_flags(self.header.dwFlags, DDSD)}")


        # Format information
        if self.header10:
            lines.append(f"  Format: DX10")
            lines.append(f"    DXGI Format: {self.header10.dxgiFormat.name} ({self.header10.dxgiFormat.value})")
            lines.append(f"    Resource Dimension: {self.header10.resourceDimension.name}")
            if self.header10.arraySize > 1:
                lines.append(f"    Array Size: {self.header10.arraySize}")
            if self.header10.miscFlag:
                lines.append(f"    Misc Flags: {_format_flags(self.header10.miscFlag, DDS_RESOURCE_MISC)}")
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
                lines.append(f"  Format: Other (flags: {_format_flags(self.header.ddspf.dwFlags, DDPF)})")

        lines.append(f"  Caps: {_format_flags(self.header.dwCaps, DDSCAPS)}")
        if self.header.dwCaps2:
            lines.append(f"  Caps2: {_format_flags(self.header.dwCaps2, DDSCAPS2)}")

        total_data_size = sum(len(subresource) for subresource in self.data)
        lines.append(f"  Subresources: {len(self.data)}")
        lines.append(f"  Total Data Size: {total_data_size} bytes")

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

        # Split image data into subresources
        dds._split_subresources(data[offset:])

        return dds

    def is_volume(self) -> bool:
        """
        Check if this is a volume (3D) texture.
        Checks both the legacy DDSCAPS2.VOLUME flag and DX10 resourceDimension.

        Returns:
            True if this is a volume texture, False otherwise
        """
        # Check DX10 header resourceDimension (modern method)
        if self.header10 and self.header10.resourceDimension == D3D10_RESOURCE_DIMENSION.TEXTURE3D:
            return True

        # Check legacy DDSCAPS2.VOLUME flag (older method)
        if self.header.dwCaps2 & DDSCAPS2.VOLUME:
            return True

        return False

    def _split_subresources(self, raw_data: bytes) -> None:
        """
        Split raw image data into subresources.

        Subresources are ordered as:
        - For volume textures (3D):
          - For each array slice:
            - For each face (6 for cubemap, 1 otherwise):
              - For each mipmap level (from largest to smallest):
                - For each depth slice:
                  - Depth slice data
        - For non-volume textures:
          - For each array slice:
            - For each face (6 for cubemap, 1 otherwise):
              - For each mipmap level (from largest to smallest):
                - Mipmap data

        Args:
            raw_data: Raw image data bytes to split
        """
        # Determine format
        fourcc = None
        dxgi_format = None

        if self.header10:
            dxgi_format = self.header10.dxgiFormat
        elif self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

        # Determine number of subresources
        mipmap_count = self.header.dwMipMapCount if self.header.dwMipMapCount > 0 else 1

        # Determine if this is a cubemap
        is_cubemap = bool(self.header.dwCaps2 & DDSCAPS2.CUBEMAP)
        num_faces = 6 if is_cubemap else 1

        # Determine if this is a volume texture
        is_volume = self.is_volume()
        base_depth = self.header.dwDepth if is_volume and self.header.dwDepth > 0 else 1

        # Determine array size (DX10 only, otherwise 1)
        array_size = self.header10.arraySize if self.header10 and self.header10.arraySize > 0 else 1

        # Base dimensions
        base_width = self.header.dwWidth
        base_height = self.header.dwHeight

        # Split data into subresources
        self.data = []
        offset = 0

        for array_idx in range(array_size):
            for face_idx in range(num_faces):
                for mip_idx in range(mipmap_count):
                    # Calculate dimensions for this mipmap level
                    width = max(1, base_width >> mip_idx)
                    height = max(1, base_height >> mip_idx)
                    depth = max(1, base_depth >> mip_idx) if is_volume else 1

                    # For volume textures, split by depth slices
                    for depth_idx in range(depth):
                        # Calculate size for this depth slice
                        slice_size = self._get_mipmap_size(width, height, fourcc, dxgi_format)

                        # Extract subresource data
                        if offset + slice_size > len(raw_data):
                            raise ValueError(
                                f"Not enough data for subresource (array={array_idx}, face={face_idx}, mip={mip_idx}, depth={depth_idx}). "
                                f"Expected {slice_size} bytes at offset {offset}, but only {len(raw_data) - offset} bytes remaining."
                            )

                        subresource = raw_data[offset:offset + slice_size]
                        self.data.append(subresource)
                        offset += slice_size

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
        # Some DDS files store DXGI format codes directly in dwFourCC without DX10 header
        # Try to interpret fourcc as DXGI format if it's a small integer
        if fourcc is not None and dxgi_format is None and fourcc < 256:
            try:
                dxgi_format = DXGI_FORMAT(fourcc)
            except ValueError:
                pass
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

        # BC7 - 16 bytes per 4x4 block
        if dxgi_format in (DXGI_FORMAT.BC7_UNORM, DXGI_FORMAT.BC7_UNORM_SRGB, DXGI_FORMAT.BC7_TYPELESS):
            blocks_x = (width + 3) // 4
            blocks_y = (height + 3) // 4
            return blocks_x * blocks_y * 16

        # Uncompressed formats - calculate based on bytes per pixel
        if dxgi_format is not None:
            bytes_per_pixel = self._get_bytes_per_pixel(dxgi_format)
            if bytes_per_pixel > 0:
                return width * height * bytes_per_pixel

        raise NotImplementedError(f"Mipmap size calculation not implemented for {self.get_format_str()}")

    def _get_bytes_per_pixel(self, dxgi_format: DXGI_FORMAT) -> int:
        """
        Get bytes per pixel for uncompressed DXGI formats

        Args:
            dxgi_format: DXGI format enum value

        Returns:
            Bytes per pixel, or 0 if format is not recognized as uncompressed
        """
        # 128-bit formats (16 bytes per pixel)
        if dxgi_format in (
            DXGI_FORMAT.R32G32B32A32_TYPELESS,
            DXGI_FORMAT.R32G32B32A32_FLOAT,
            DXGI_FORMAT.R32G32B32A32_UINT,
            DXGI_FORMAT.R32G32B32A32_SINT,
        ):
            return 16

        # 96-bit formats (12 bytes per pixel)
        if dxgi_format in (
            DXGI_FORMAT.R32G32B32_TYPELESS,
            DXGI_FORMAT.R32G32B32_FLOAT,
            DXGI_FORMAT.R32G32B32_UINT,
            DXGI_FORMAT.R32G32B32_SINT,
        ):
            return 12

        # 64-bit formats (8 bytes per pixel)
        if dxgi_format in (
            DXGI_FORMAT.R16G16B16A16_TYPELESS,
            DXGI_FORMAT.R16G16B16A16_FLOAT,
            DXGI_FORMAT.R16G16B16A16_UNORM,
            DXGI_FORMAT.R16G16B16A16_UINT,
            DXGI_FORMAT.R16G16B16A16_SNORM,
            DXGI_FORMAT.R16G16B16A16_SINT,
            DXGI_FORMAT.R32G32_TYPELESS,
            DXGI_FORMAT.R32G32_FLOAT,
            DXGI_FORMAT.R32G32_UINT,
            DXGI_FORMAT.R32G32_SINT,
        ):
            return 8

        # 32-bit formats (4 bytes per pixel)
        if dxgi_format in (
            DXGI_FORMAT.R10G10B10A2_TYPELESS,
            DXGI_FORMAT.R10G10B10A2_UNORM,
            DXGI_FORMAT.R10G10B10A2_UINT,
            DXGI_FORMAT.R11G11B10_FLOAT,
            DXGI_FORMAT.R8G8B8A8_TYPELESS,
            DXGI_FORMAT.R8G8B8A8_UNORM,
            DXGI_FORMAT.R8G8B8A8_UNORM_SRGB,
            DXGI_FORMAT.R8G8B8A8_UINT,
            DXGI_FORMAT.R8G8B8A8_SNORM,
            DXGI_FORMAT.R8G8B8A8_SINT,
            DXGI_FORMAT.R16G16_TYPELESS,
            DXGI_FORMAT.R16G16_FLOAT,
            DXGI_FORMAT.R16G16_UNORM,
            DXGI_FORMAT.R16G16_UINT,
            DXGI_FORMAT.R16G16_SNORM,
            DXGI_FORMAT.R16G16_SINT,
            DXGI_FORMAT.R32_TYPELESS,
            DXGI_FORMAT.D32_FLOAT,
            DXGI_FORMAT.R32_FLOAT,
            DXGI_FORMAT.R32_UINT,
            DXGI_FORMAT.R32_SINT,
            DXGI_FORMAT.B8G8R8A8_UNORM,
            DXGI_FORMAT.B8G8R8X8_UNORM,
            DXGI_FORMAT.B8G8R8A8_TYPELESS,
            DXGI_FORMAT.B8G8R8A8_UNORM_SRGB,
            DXGI_FORMAT.B8G8R8X8_TYPELESS,
            DXGI_FORMAT.B8G8R8X8_UNORM_SRGB,
        ):
            return 4

        # 16-bit formats (2 bytes per pixel)
        if dxgi_format in (
            DXGI_FORMAT.R8G8_TYPELESS,
            DXGI_FORMAT.R8G8_UNORM,
            DXGI_FORMAT.R8G8_UINT,
            DXGI_FORMAT.R8G8_SNORM,
            DXGI_FORMAT.R8G8_SINT,
            DXGI_FORMAT.R16_TYPELESS,
            DXGI_FORMAT.R16_FLOAT,
            DXGI_FORMAT.D16_UNORM,
            DXGI_FORMAT.R16_UNORM,
            DXGI_FORMAT.R16_UINT,
            DXGI_FORMAT.R16_SNORM,
            DXGI_FORMAT.R16_SINT,
            DXGI_FORMAT.B5G6R5_UNORM,
            DXGI_FORMAT.B5G5R5A1_UNORM,
            DXGI_FORMAT.B4G4R4A4_UNORM,
        ):
            return 2

        # 8-bit formats (1 byte per pixel)
        if dxgi_format in (
            DXGI_FORMAT.R8_TYPELESS,
            DXGI_FORMAT.R8_UNORM,
            DXGI_FORMAT.R8_UINT,
            DXGI_FORMAT.R8_SNORM,
            DXGI_FORMAT.R8_SINT,
            DXGI_FORMAT.A8_UNORM,
            DXGI_FORMAT.P8,
        ):
            return 1

        # Format not recognized or compressed
        return 0

    def get_format_str(self) -> str:
        """Get a human-readable format string"""
        # Try to get DXGI format (works for both DX10 header and FourCC)
        dxgi_format_value = self.get_dxgi_format()

        if dxgi_format_value is not None:
            try:
                dxgi_format = DXGI_FORMAT(dxgi_format_value)
                return f"{dxgi_format.name}"
            except ValueError:
                pass

        # No DXGI equivalent - check if we have a FourCC
        if self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC
            try:
                fourcc_enum = FourCC(fourcc)
                return f"FourCC {fourcc_enum.name}"
            except ValueError:
                fourcc_bytes = fourcc.to_bytes(4, 'little')
                return f"FourCC 0x{fourcc:08X} ({fourcc_bytes})"

        return "Unknown format"

    def get_width(self) -> int:
        """Get the width of the texture in pixels"""
        return self.header.dwWidth

    def get_height(self) -> int:
        """Get the height of the texture in pixels"""
        return self.header.dwHeight

    def get_depth(self) -> int:
        """Get the depth of the texture (for volume textures), or 0 if not a volume texture"""
        return self.header.dwDepth

    def get_mip_count(self) -> int:
        """Get the number of mipmap levels"""
        return self.header.dwMipMapCount if self.header.dwMipMapCount > 0 else 1

    def get_size(self) -> int:
        """Get the total size of all subresource data in bytes"""
        return sum(len(subresource) for subresource in self.data)

    def get_dxgi_format(self) -> Optional[int]:
        """
        Get the DXGI format enum value (integer).
        Returns the DXGI format from DX10 header if present,
        otherwise returns the equivalent DXGI format for FourCC codes.
        Returns None if no equivalent DXGI format exists.
        """
        # If DX10 header exists, return its DXGI format
        if self.header10:
            return self.header10.dxgiFormat.value

        # Check if we have a FourCC code
        if self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

            # Map FourCC codes to equivalent DXGI formats
            fourcc_to_dxgi = {
                FourCC.DXT1: DXGI_FORMAT.BC1_UNORM,
                FourCC.DXT2: DXGI_FORMAT.BC2_UNORM,  # Premultiplied alpha
                FourCC.DXT3: DXGI_FORMAT.BC2_UNORM,
                FourCC.DXT4: DXGI_FORMAT.BC3_UNORM,  # Premultiplied alpha
                FourCC.DXT5: DXGI_FORMAT.BC3_UNORM,
                FourCC.BC4U: DXGI_FORMAT.BC4_UNORM,
                FourCC.BC4S: DXGI_FORMAT.BC4_SNORM,
                FourCC.BC5U: DXGI_FORMAT.BC5_UNORM,
                FourCC.BC5S: DXGI_FORMAT.BC5_SNORM,
                FourCC.ATI1: DXGI_FORMAT.BC4_UNORM,
                FourCC.ATI2: DXGI_FORMAT.BC5_UNORM,
            }

            # Try to find mapping for this FourCC
            if fourcc in fourcc_to_dxgi:
                return fourcc_to_dxgi[fourcc].value

            # Some DDS files store DXGI format codes directly in dwFourCC
            # Try to interpret as DXGI format if it's a small integer
            if fourcc < 256:
                try:
                    dxgi_format = DXGI_FORMAT(fourcc)
                    return dxgi_format.value
                except ValueError:
                    pass

        # No equivalent DXGI format found
        return None

    def get_subresource_count(self) -> int:
        """Get the total number of subresources (mipmap levels * array slices * faces)"""
        return len(self.data)

    def _get_subresource_indices(self, subresource_index: int) -> tuple[int, int, int, int]:
        """
        Convert a flat subresource index to (array_index, face_index, mip_level, depth_index).

        Args:
            subresource_index: Flat subresource index

        Returns:
            Tuple of (array_index, face_index, mip_level, depth_index)
        """
        mipmap_count = self.get_mip_count()
        is_cubemap = bool(self.header.dwCaps2 & DDSCAPS2.CUBEMAP)
        num_faces = 6 if is_cubemap else 1
        array_size = self.header10.arraySize if self.header10 and self.header10.arraySize > 0 else 1
        is_volume = self.is_volume()

        # Reverse the calculation from _split_subresources
        # For volume textures: array -> face -> mip -> depth
        # For non-volume: array -> face -> mip (depth is always 0)

        if is_volume:
            # Need to calculate depth slice count for each mip level
            remaining = subresource_index
            for array_idx in range(array_size):
                for face_idx in range(num_faces):
                    for mip_idx in range(mipmap_count):
                        depth_count = max(1, self.header.dwDepth >> mip_idx)
                        if remaining < depth_count:
                            return (array_idx, face_idx, mip_idx, remaining)
                        remaining -= depth_count
            # Should not reach here for valid index
            raise ValueError(f"Invalid subresource index {subresource_index}")
        else:
            # Simple calculation for non-volume textures
            mip_level = subresource_index % mipmap_count
            remaining = subresource_index // mipmap_count
            face_index = remaining % num_faces
            array_index = remaining // num_faces
            return (array_index, face_index, mip_level, 0)

    def get_subresource_width(self, subresource_index: int) -> int:
        """
        Get the width of a specific subresource in pixels.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Width in pixels
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        _, _, mip_level, _ = self._get_subresource_indices(subresource_index)
        return max(1, self.header.dwWidth >> mip_level)

    def get_subresource_height(self, subresource_index: int) -> int:
        """
        Get the height of a specific subresource in pixels.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Height in pixels
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        _, _, mip_level, _ = self._get_subresource_indices(subresource_index)
        return max(1, self.header.dwHeight >> mip_level)

    def get_subresource_depth(self, subresource_index: int) -> int:
        """
        Get the depth of a specific subresource (for volume textures).
        Since subresources are now split by depth slice, this always returns 1.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Always 1 (each subresource is a single depth slice)
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        # Since we now split volume textures by depth slices, each subresource is a single 2D slice
        return 1

    def get_subresource_size(self, subresource_index: int) -> int:
        """
        Get the size of a specific subresource in bytes.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Size in bytes
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        return len(self.data[subresource_index])

    def get_subresource_row_pitch(self, subresource_index: int) -> int:
        """
        Get the row pitch (bytes per row) for a specific subresource.
        For block-compressed formats, this is the number of bytes per row of blocks.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Row pitch in bytes
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        width = self.get_subresource_width(subresource_index)
        height = self.get_subresource_height(subresource_index)

        # Determine format
        fourcc = None
        dxgi_format = None

        if self.header10:
            dxgi_format = self.header10.dxgiFormat
        elif self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

        # Try to interpret fourcc as DXGI format if it's a small integer
        if fourcc is not None and dxgi_format is None and fourcc < 256:
            try:
                dxgi_format = DXGI_FORMAT(fourcc)
            except ValueError:
                pass

        # For block-compressed formats, calculate row pitch based on blocks
        # BC1/DXT1 - 8 bytes per 4x4 block
        if fourcc == FourCC.DXT1 or dxgi_format in (DXGI_FORMAT.BC1_UNORM, DXGI_FORMAT.BC1_UNORM_SRGB, DXGI_FORMAT.BC1_TYPELESS):
            blocks_x = (width + 3) // 4
            return blocks_x * 8

        # BC2/DXT3, BC3/DXT5, BC5, BC7 - 16 bytes per 4x4 block
        if (fourcc in (FourCC.DXT2, FourCC.DXT3, FourCC.DXT4, FourCC.DXT5, FourCC.BC5U, FourCC.BC5S, FourCC.ATI2) or
            dxgi_format in (DXGI_FORMAT.BC2_UNORM, DXGI_FORMAT.BC2_UNORM_SRGB, DXGI_FORMAT.BC2_TYPELESS,
                           DXGI_FORMAT.BC3_UNORM, DXGI_FORMAT.BC3_UNORM_SRGB, DXGI_FORMAT.BC3_TYPELESS,
                           DXGI_FORMAT.BC5_UNORM, DXGI_FORMAT.BC5_SNORM, DXGI_FORMAT.BC5_TYPELESS,
                           DXGI_FORMAT.BC7_UNORM, DXGI_FORMAT.BC7_UNORM_SRGB, DXGI_FORMAT.BC7_TYPELESS)):
            blocks_x = (width + 3) // 4
            return blocks_x * 16

        # BC4 - 8 bytes per 4x4 block
        if (fourcc in (FourCC.BC4U, FourCC.BC4S, FourCC.ATI1) or
            dxgi_format in (DXGI_FORMAT.BC4_UNORM, DXGI_FORMAT.BC4_SNORM, DXGI_FORMAT.BC4_TYPELESS)):
            blocks_x = (width + 3) // 4
            return blocks_x * 8

        # Uncompressed formats - bytes per pixel * width
        if dxgi_format is not None:
            bytes_per_pixel = self._get_bytes_per_pixel(dxgi_format)
            if bytes_per_pixel > 0:
                return width * bytes_per_pixel

        raise NotImplementedError(f"Row pitch calculation not implemented for {self.get_format_str()}")

    def get_subresource_row_count(self, subresource_index: int) -> int:
        """
        Get the number of rows for a specific subresource.
        For block-compressed formats, this is the number of rows of blocks.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Number of rows (or rows of blocks for compressed formats)
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        height = self.get_subresource_height(subresource_index)

        # Determine format
        fourcc = None
        dxgi_format = None

        if self.header10:
            dxgi_format = self.header10.dxgiFormat
        elif self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

        # Try to interpret fourcc as DXGI format if it's a small integer
        if fourcc is not None and dxgi_format is None and fourcc < 256:
            try:
                dxgi_format = DXGI_FORMAT(fourcc)
            except ValueError:
                pass

        # For block-compressed formats, calculate row count based on blocks
        is_block_compressed = (
            fourcc in (FourCC.DXT1, FourCC.DXT2, FourCC.DXT3, FourCC.DXT4, FourCC.DXT5,
                      FourCC.BC4U, FourCC.BC4S, FourCC.ATI1, FourCC.BC5U, FourCC.BC5S, FourCC.ATI2) or
            dxgi_format in (DXGI_FORMAT.BC1_UNORM, DXGI_FORMAT.BC1_UNORM_SRGB, DXGI_FORMAT.BC1_TYPELESS,
                           DXGI_FORMAT.BC2_UNORM, DXGI_FORMAT.BC2_UNORM_SRGB, DXGI_FORMAT.BC2_TYPELESS,
                           DXGI_FORMAT.BC3_UNORM, DXGI_FORMAT.BC3_UNORM_SRGB, DXGI_FORMAT.BC3_TYPELESS,
                           DXGI_FORMAT.BC4_UNORM, DXGI_FORMAT.BC4_SNORM, DXGI_FORMAT.BC4_TYPELESS,
                           DXGI_FORMAT.BC5_UNORM, DXGI_FORMAT.BC5_SNORM, DXGI_FORMAT.BC5_TYPELESS,
                           DXGI_FORMAT.BC7_UNORM, DXGI_FORMAT.BC7_UNORM_SRGB, DXGI_FORMAT.BC7_TYPELESS)
        )

        if is_block_compressed:
            # For block-compressed formats, return number of block rows
            return (height + 3) // 4

        # For uncompressed formats, return the height
        return height

    def get_subresource_offset(self, subresource_index: int) -> int:
        """
        Get the offset of a specific subresource from the start of the image data.
        This does not include the header size - it's the offset within the image data itself.

        Args:
            subresource_index: Index of the subresource

        Returns:
            Offset in bytes from the start of the image data (after headers)
        """
        if subresource_index < 0 or subresource_index >= len(self.data):
            raise ValueError(f"Invalid subresource index {subresource_index}. Texture has {len(self.data)} subresource(s).")

        # Sum sizes of all previous subresources
        offset = 0
        for i in range(subresource_index):
            offset += len(self.data[i])

        return offset

    def to_image(self, mipmap_level: int = 0, array_index: int = 0, face_index: int = 0, depth_index: int = 0) -> np.ndarray:
        """
        Convert DDS texture to numpy array

        Args:
            mipmap_level: Mipmap level to extract (0 = full resolution)
            array_index: Array slice index (0 for non-arrays)
            face_index: Cubemap face index (0 for non-cubemaps)
            depth_index: Depth slice index for volume textures (0 for non-volume textures)

        Returns:
            numpy array of shape (height, width, 4) with dtype uint8 (RGBA values 0-255)

            Can be saved with imageio:
            - imageio.imwrite('output.png', array)
            - imageio.imwrite('output.jpg', array)

        Raises:
            ValueError: If the format is not supported or indices are invalid
            NotImplementedError: If the format decompressor is not implemented yet
        """
        # Determine format
        fourcc = None
        dxgi_format = None

        if self.header10:
            dxgi_format = self.header10.dxgiFormat
        elif self.header.ddspf.dwFlags & DDPF.FOURCC:
            fourcc = self.header.ddspf.dwFourCC

        # Some DDS files store DXGI format codes directly in dwFourCC without DX10 header
        # Try to interpret fourcc as DXGI format if it's a small integer
        if fourcc is not None and dxgi_format is None and fourcc < 256:
            try:
                dxgi_format = DXGI_FORMAT(fourcc)
                fourcc = None  # Clear fourcc since we're using dxgi_format instead
            except ValueError:
                pass

        # Validate indices
        mipmap_count = self.header.dwMipMapCount if self.header.dwMipMapCount > 0 else 1
        if mipmap_level < 0 or mipmap_level >= mipmap_count:
            raise ValueError(f"Invalid mipmap level {mipmap_level}. Texture has {mipmap_count} mipmap level(s).")

        is_cubemap = bool(self.header.dwCaps2 & DDSCAPS2.CUBEMAP)
        num_faces = 6 if is_cubemap else 1
        if face_index < 0 or face_index >= num_faces:
            raise ValueError(f"Invalid face index {face_index}. Texture has {num_faces} face(s).")

        array_size = self.header10.arraySize if self.header10 and self.header10.arraySize > 0 else 1
        if array_index < 0 or array_index >= array_size:
            raise ValueError(f"Invalid array index {array_index}. Texture has {array_size} array slice(s).")

        is_volume = self.is_volume()
        if is_volume:
            depth_count = max(1, self.header.dwDepth >> mipmap_level)
            if depth_index < 0 or depth_index >= depth_count:
                raise ValueError(f"Invalid depth index {depth_index}. Texture has {depth_count} depth slice(s) at mipmap level {mipmap_level}.")
        elif depth_index != 0:
            raise ValueError(f"Depth index must be 0 for non-volume textures.")

        # Calculate subresource index
        # For volume textures: array -> face -> mip -> depth
        # For non-volume: array -> face -> mip
        if is_volume:
            # Need to count depth slices for all previous mip levels
            subresource_index = 0
            for a_idx in range(array_index + 1):
                for f_idx in range(num_faces):
                    if a_idx < array_index or f_idx < face_index:
                        # Add all mipmap levels for this array/face
                        for m_idx in range(mipmap_count):
                            d_count = max(1, self.header.dwDepth >> m_idx)
                            subresource_index += d_count
                    elif a_idx == array_index and f_idx == face_index:
                        # Add previous mipmap levels
                        for m_idx in range(mipmap_level):
                            d_count = max(1, self.header.dwDepth >> m_idx)
                            subresource_index += d_count
                        # Add depth slices for current mipmap level
                        subresource_index += depth_index
                        break
        else:
            subresource_index = (array_index * num_faces + face_index) * mipmap_count + mipmap_level

        # Validate subresource index
        if subresource_index >= len(self.data):
            raise ValueError(
                f"Invalid subresource index {subresource_index} "
                f"(array={array_index}, face={face_index}, mip={mipmap_level}, depth={depth_index}). "
                f"Texture has {len(self.data)} subresource(s)."
            )

        # Calculate dimensions for the requested mipmap level
        base_width = self.header.dwWidth
        base_height = self.header.dwHeight
        width = max(1, base_width >> mipmap_level)
        height = max(1, base_height >> mipmap_level)

        # Get data for this subresource
        mipmap_data = self.data[subresource_index]

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
        elif dxgi_format in (DXGI_FORMAT.BC7_UNORM, DXGI_FORMAT.BC7_UNORM_SRGB, DXGI_FORMAT.BC7_TYPELESS):
            decompressor = BC7Decompressor()
        elif dxgi_format in UncompressedDecompressor.FORMAT_DESCRIPTORS or dxgi_format in (
            DXGI_FORMAT.R10G10B10A2_UNORM,
            DXGI_FORMAT.R10G10B10A2_UINT,
            DXGI_FORMAT.R11G11B10_FLOAT,
            DXGI_FORMAT.B5G6R5_UNORM,
            DXGI_FORMAT.B5G5R5A1_UNORM,
            DXGI_FORMAT.B4G4R4A4_UNORM,
        ):
            decompressor = UncompressedDecompressor(dxgi_format)
        else:
            raise NotImplementedError(f"Decompression not yet implemented for {self.get_format_str()}")

        # Decompress texture data for this mipmap level
        rgba_data = decompressor.decompress(mipmap_data, width, height)

        # Handle premultiplied alpha formats
        # DXT2 and DXT4 use premultiplied alpha and need conversion
        is_premultiplied = fourcc in (FourCC.DXT2, FourCC.DXT4)
        if is_premultiplied:
            rgba_data = _unpremultiply_alpha(rgba_data)

        # Return numpy array directly
        # BC6H returns float32 HDR data, other formats return uint8
        return rgba_data
