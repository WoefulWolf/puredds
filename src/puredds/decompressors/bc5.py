"""BC5 texture decompressor"""
import numpy as np
from numba import jit
from .base import TextureDecompressor


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
