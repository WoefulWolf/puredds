"""BC3 (DXT4/DXT5) texture decompressor"""
import numpy as np
from numba import jit
from .base import TextureDecompressor


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
