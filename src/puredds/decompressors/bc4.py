"""BC4 texture decompressor"""
import numpy as np
from numba import jit
from .base import TextureDecompressor


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
