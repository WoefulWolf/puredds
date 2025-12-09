"""BC1 (DXT1) texture decompressor"""
import numpy as np
from numba import jit
from .base import TextureDecompressor


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
