"""BC7 texture decompressor"""
import numpy as np
from numba import jit
from .base import TextureDecompressor


# Module-level JIT-compiled helper functions
@jit(nopython=True, cache=True)
def _read_bits(data, offset, count):
    """Read count bits from data starting at bit offset"""
    byte_offset = offset // 8
    bit_offset = offset % 8

    result = 0
    bits_read = 0

    while bits_read < count:
        if byte_offset >= len(data):
            # Return what we have so far if we hit end of data
            break

        byte_val = data[byte_offset]
        bits_available = 8 - bit_offset
        bits_to_read = min(count - bits_read, bits_available)

        mask = (1 << bits_to_read) - 1
        bits = (byte_val >> bit_offset) & mask
        result |= bits << bits_read

        bits_read += bits_to_read
        byte_offset += 1
        bit_offset = 0

    return result


@jit(nopython=True, cache=True)
def _interpolate(e0, e1, index, index_bits):
    """Interpolate between two endpoints"""
    if index_bits == 2:
        w = np.array([0, 21, 43, 64], dtype=np.int32)
    elif index_bits == 3:
        w = np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.int32)
    elif index_bits == 4:
        w = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int32)
    else:
        return e0

    if index >= len(w):
        index = len(w) - 1

    return (e0 * (64 - w[index]) + e1 * w[index] + 32) >> 6


@jit(nopython=True, cache=True)
def _expand_quantized(value, from_bits, to_bits):
    """Expand a quantized value to full bit depth using proper bit replication"""
    if from_bits == 0:
        return 0
    if from_bits >= to_bits:
        return value >> (from_bits - to_bits)

    # BC7 spec method: replicate source bits to fill target
    # This ensures 0->0 and max->max mapping
    # E.g., 3-bit 0b111 (7) -> 8-bit 0b11111111 (255)
    #       5-bit 0b10000 (16) -> 8-bit 0b10000100 (132)

    # Shift value to MSB position
    result = value << (to_bits - from_bits)

    # Replicate bits by repeatedly ORing with right-shifted result
    # Keep going until we've filled all the lower bits
    shift = from_bits
    while shift < to_bits:
        result |= result >> shift
        shift *= 2

    return result


@jit(nopython=True, cache=True)
def _get_anchor_index_for_subset(partition_table, partition_id, subset_index, num_subsets):
    """
    Find the anchor pixel index for a given subset by finding the first pixel
    in raster order that belongs to that subset.
    """
    if subset_index == 0:
        return 0  # Subset 0 always has anchor at pixel 0

    # Find first pixel that belongs to this subset
    for pixel_idx in range(16):
        if partition_table[partition_id, pixel_idx] == subset_index:
            return pixel_idx

    return 0  # Fallback


@jit(nopython=True, cache=True)
def _get_anchor_index(partition_set_id, subset, num_subsets):
    """Get the anchor pixel index for a given subset in a partition"""

    # For subset 0, anchor is always pixel 0
    if subset == 0:
        return 0

    # We need to look up the partition table to find the anchor
    # For now, use precomputed tables (these are derived from the partition patterns)

    # Anchor indices for 2-subset partitions (subset 1 anchor for each of 64 patterns)
    # Source: Khronos OpenGL BPTC specification Table.A2
    anchor_indices_2 = np.array([
        15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
        15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
        15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
         6, 2, 6, 8,15,15, 2, 2,15,15,15,15,15, 2, 2,15,
    ], dtype=np.uint8)

    # Anchor indices for 3-subset partitions
    # Subset 1 anchors - Source: Khronos OpenGL BPTC specification Table.A3a
    anchor_indices_3_subset1 = np.array([
         3,  3,15,15,  8,  3,15,15,  8,  8,  6,  6,  6,  5,  3,  3,
         3,  3,  8,15,  3,  3,  6,10,  5,  8,  8,  6,  8,  5,15,15,
         8,15,  3,  5,  6,10,  8,15,15,  3,15,  5,15,15,15,15,
         3,15,  5,  5,  5,  8,  5,10,  5,10,  8,13,15,12,  3,  3,
    ], dtype=np.uint8)

    # Subset 2 anchors - Source: Khronos OpenGL BPTC specification Table.A3b
    anchor_indices_3_subset2 = np.array([
        15,  8,  8,  3,15,15,  3,  8,15,15,15,15,15,15,15,  8,
        15,  8,15,  3,15,  8,15,  8,  3,15,  6,10,15,15,10,  8,
        15,  3,15,10,10,  8,  9,10,  6,15,  8,15,  3,  6,  6,  8,
        15,  3,15,15,15,15,15,15,15,15,15,15,  3,15,15,  8,
    ], dtype=np.uint8)

    if partition_set_id >= 64:
        partition_set_id = 0

    if num_subsets == 2:
        if subset <= 0:
            return 0
        else:
            return anchor_indices_2[partition_set_id]
    elif num_subsets == 3:
        if subset <= 0:
            return 0
        elif subset == 1:
            return anchor_indices_3_subset1[partition_set_id]
        else:  # subset == 2
            return anchor_indices_3_subset2[partition_set_id]
    return 0


# Partition tables as module-level constants
# Source: Khronos OpenGL BPTC specification (EXT_texture_compression_bptc)
# Table.P2: 64 partition patterns for 2-subset compression
PARTITION_TABLE_2 = np.array([
        [0,0,1,1,0,0,1,1,0,0,1,1,0,0,1,1], [0,0,0,1,0,0,0,1,0,0,0,1,0,0,0,1],
        [0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1], [0,0,0,1,0,0,1,1,0,0,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,1,0,0,0,1,0,0,1,1], [0,0,1,1,0,1,1,1,0,1,1,1,1,1,1,1],
        [0,0,0,1,0,0,1,1,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,1,0,0,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,1], [0,0,1,1,0,1,1,1,1,1,1,1,1,1,1,1],
        [0,0,0,0,0,0,0,1,0,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0,0,1,0,1,1,1],
        [0,0,0,1,0,1,1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1],
        [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1],
        [0,0,0,0,1,0,0,0,1,1,1,0,1,1,1,1], [0,1,1,1,0,0,0,1,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,1,1,1,0], [0,1,1,1,0,0,1,1,0,0,0,1,0,0,0,0],
        [0,0,1,1,0,0,0,1,0,0,0,0,0,0,0,0], [0,0,0,0,1,0,0,0,1,1,0,0,1,1,1,0],
        [0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0], [0,1,1,1,0,0,1,1,0,0,1,1,0,0,0,1],
        [0,0,1,1,0,0,0,1,0,0,0,1,0,0,0,0], [0,0,0,0,1,0,0,0,1,0,0,0,1,1,0,0],
        [0,1,1,0,0,1,1,0,0,1,1,0,0,1,1,0], [0,0,1,1,0,1,1,0,0,1,1,0,1,1,0,0],
        [0,0,0,1,0,1,1,1,1,1,1,0,1,0,0,0], [0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0],
        [0,1,1,1,0,0,0,1,1,0,0,0,1,1,1,0], [0,0,1,1,1,0,0,1,1,0,0,1,1,1,0,0],
        [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1], [0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1],
        [0,1,0,1,1,0,1,0,0,1,0,1,1,0,1,0], [0,0,1,1,0,0,1,1,1,1,0,0,1,1,0,0],
        [0,0,1,1,1,1,0,0,0,0,1,1,1,1,0,0], [0,1,0,1,0,1,0,1,1,0,1,0,1,0,1,0],
        [0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1], [0,1,0,1,1,0,1,0,1,0,1,0,0,1,0,1],
        [0,1,1,1,0,0,1,1,1,1,0,0,1,1,1,0], [0,0,0,1,0,0,1,1,1,1,0,0,1,0,0,0],
        [0,0,1,1,0,0,1,0,0,1,0,0,1,1,0,0], [0,0,1,1,1,0,1,1,1,1,0,1,1,1,0,0],
        [0,1,1,0,1,0,0,1,1,0,0,1,0,1,1,0], [0,0,1,1,1,1,0,0,1,1,0,0,0,0,1,1],
        [0,1,1,0,0,1,1,0,1,0,0,1,1,0,0,1], [0,0,0,0,0,1,1,0,0,1,1,0,0,0,0,0],
        [0,1,0,0,1,1,1,0,0,1,0,0,0,0,0,0], [0,0,1,0,0,1,1,1,0,0,1,0,0,0,0,0],
        [0,0,0,0,0,0,1,0,0,1,1,1,0,0,1,0], [0,0,0,0,0,1,0,0,1,1,1,0,0,1,0,0],
        [0,1,1,0,1,1,0,0,1,0,0,1,0,0,1,1], [0,0,1,1,0,1,1,0,1,1,0,0,1,0,0,1],
        [0,1,1,0,0,0,1,1,1,0,0,1,1,1,0,0], [0,0,1,1,1,0,0,1,1,1,0,0,0,1,1,0],
        [0,1,1,0,1,1,0,0,1,1,0,0,1,0,0,1], [0,1,1,0,0,0,1,1,0,0,1,1,1,0,0,1],
        [0,1,1,1,1,1,1,0,1,0,0,0,0,0,0,1], [0,0,0,1,1,0,0,0,1,1,1,0,0,1,1,1],
        [0,0,0,0,1,1,1,1,0,0,1,1,0,0,1,1], [0,0,1,1,0,0,1,1,1,1,1,1,0,0,0,0],
        [0,0,1,0,0,0,1,0,1,1,1,0,1,1,1,0], [0,1,0,0,0,1,0,0,0,1,1,1,0,1,1,1],
], dtype=np.uint8)

# Table.P3: 64 partition patterns for 3-subset compression
PARTITION_TABLE_3 = np.array([
        [0,0,1,1,0,0,1,1,0,2,2,1,2,2,2,2], [0,0,0,1,0,0,1,1,2,2,1,1,2,2,2,1],
        [0,0,0,0,2,0,0,1,2,2,1,1,2,2,1,1], [0,2,2,2,0,0,2,2,0,0,1,1,0,1,1,1],
        [0,0,0,0,0,0,0,0,1,1,2,2,1,1,2,2], [0,0,1,1,0,0,1,1,0,0,2,2,0,0,2,2],
        [0,0,2,2,0,0,2,2,1,1,1,1,1,1,1,1], [0,0,1,1,0,0,1,1,2,2,1,1,2,2,1,1],
        [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2], [0,0,0,0,1,1,1,1,1,1,1,1,2,2,2,2],
        [0,0,0,0,1,1,1,1,2,2,2,2,2,2,2,2], [0,0,1,2,0,0,1,2,0,0,1,2,0,0,1,2],
        [0,1,1,2,0,1,1,2,0,1,1,2,0,1,1,2], [0,1,2,2,0,1,2,2,0,1,2,2,0,1,2,2],
        [0,0,1,1,0,1,1,2,1,1,2,2,1,2,2,2], [0,0,1,1,2,0,0,1,2,2,0,0,2,2,2,0],
        [0,0,0,1,0,0,1,1,0,1,1,2,1,1,2,2], [0,1,1,1,0,0,1,1,2,0,0,1,2,2,0,0],
        [0,0,0,0,1,1,2,2,1,1,2,2,1,1,2,2], [0,0,2,2,0,0,2,2,0,0,2,2,1,1,1,1],
        [0,1,1,1,0,1,1,1,0,2,2,2,0,2,2,2], [0,0,0,1,0,0,0,1,2,2,2,1,2,2,2,1],
        [0,0,0,0,0,0,1,1,0,1,2,2,0,1,2,2], [0,0,0,0,1,1,0,0,2,2,1,0,2,2,1,0],
        [0,1,2,2,0,1,2,2,0,0,1,1,0,0,0,0], [0,0,1,2,0,0,1,2,1,1,2,2,2,2,2,2],
        [0,1,1,0,1,2,2,1,1,2,2,1,0,1,1,0], [0,0,0,0,0,1,1,0,1,2,2,1,1,2,2,1],
        [0,0,2,2,1,1,0,2,1,1,0,2,0,0,2,2], [0,1,1,0,0,1,1,0,2,0,0,2,2,2,2,2],
        [0,0,1,1,0,1,2,2,0,1,2,2,0,0,1,1], [0,0,0,0,2,0,0,0,2,2,1,1,2,2,2,1],
        [0,0,0,0,0,0,0,2,1,1,2,2,1,2,2,2], [0,2,2,2,0,0,2,2,0,0,1,2,0,0,1,1],
        [0,0,1,1,0,0,1,2,0,0,2,2,0,2,2,2], [0,1,2,0,0,1,2,0,0,1,2,0,0,1,2,0],
        [0,0,0,0,1,1,1,1,2,2,2,2,0,0,0,0], [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0],
        [0,1,2,0,2,0,1,2,1,2,0,1,0,1,2,0], [0,0,1,1,2,2,0,0,1,1,2,2,0,0,1,1],
        [0,0,1,1,1,1,2,2,2,2,0,0,0,0,1,1], [0,1,0,1,0,1,0,1,2,2,2,2,2,2,2,2],
        [0,0,0,0,0,0,0,0,2,1,2,1,2,1,2,1], [0,0,2,2,1,1,2,2,0,0,2,2,1,1,2,2],
        [0,0,2,2,0,0,1,1,0,0,2,2,0,0,1,1], [0,2,2,0,1,2,2,1,0,2,2,0,1,2,2,1],
        [0,1,0,1,2,2,2,2,2,2,2,2,0,1,0,1], [0,0,0,0,2,1,2,1,2,1,2,1,2,1,2,1],
        [0,1,0,1,0,1,0,1,0,1,0,1,2,2,2,2], [0,2,2,2,0,1,1,1,0,2,2,2,0,1,1,1],
        [0,0,0,2,1,1,1,2,0,0,0,2,1,1,1,2], [0,0,0,0,2,1,1,2,2,1,1,2,2,1,1,2],
        [0,2,2,2,0,1,1,1,0,1,1,1,0,2,2,2], [0,0,0,2,1,1,1,2,1,1,1,2,0,0,0,2],
        [0,1,1,0,0,1,1,0,0,1,1,0,2,2,2,2], [0,0,0,0,0,0,0,0,2,1,1,2,2,1,1,2],
        [0,1,1,0,0,1,1,0,2,2,2,2,2,2,2,2], [0,0,2,2,0,0,1,1,0,0,1,1,0,0,2,2],
        [0,0,2,2,1,1,2,2,1,1,2,2,0,0,2,2], [0,0,0,0,0,0,0,0,0,0,0,0,2,1,1,2],
        [0,0,0,2,0,0,0,1,0,0,0,2,0,0,0,1], [0,2,2,2,1,2,2,2,0,2,2,2,1,2,2,2],
        [0,1,0,1,2,2,2,2,2,2,2,2,2,2,2,2], [0,1,1,1,2,0,1,1,2,2,0,1,2,2,2,0],
    ], dtype=np.uint8)


@jit(nopython=True, cache=True)
def _get_partition_index(partition_set_id, pixel_index, num_subsets):
    """Get which subset a pixel belongs to based on partition pattern"""
    if num_subsets == 2:
        if partition_set_id >= 64:
            partition_set_id = 0
        return PARTITION_TABLE_2[partition_set_id, pixel_index]
    elif num_subsets == 3:
        if partition_set_id >= 64:
            partition_set_id = 0
        return PARTITION_TABLE_3[partition_set_id, pixel_index]
    return 0


@jit(nopython=True, cache=True)
def _decode_mode_0(block_data):
    """Decode BC7 mode 0: 3 subsets, 4-bit partition, 4-bit RGB, p-bit per endpoint, 3-bit indices"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 1  # Skip mode bit

    # Read partition
    partition = _read_bits(block_data, bit_pos, 4)
    bit_pos += 4

    # Read 6 RGB endpoints (3 subsets × 2 endpoints)
    endpoints_r = np.zeros(6, dtype=np.uint8)
    endpoints_g = np.zeros(6, dtype=np.uint8)
    endpoints_b = np.zeros(6, dtype=np.uint8)

    for i in range(6):
        endpoints_r[i] = _read_bits(block_data, bit_pos, 4)
        bit_pos += 4
    for i in range(6):
        endpoints_g[i] = _read_bits(block_data, bit_pos, 4)
        bit_pos += 4
    for i in range(6):
        endpoints_b[i] = _read_bits(block_data, bit_pos, 4)
        bit_pos += 4

    # Read p-bits (one per endpoint)
    p_bits = np.zeros(6, dtype=np.uint8)
    for i in range(6):
        p_bits[i] = _read_bits(block_data, bit_pos, 1)
        bit_pos += 1

    # Expand endpoints (4-bit + p-bit → 8-bit)
    for i in range(6):
        endpoints_r[i] = _expand_quantized((endpoints_r[i] << 1) | p_bits[i], 5, 8)
        endpoints_g[i] = _expand_quantized((endpoints_g[i] << 1) | p_bits[i], 5, 8)
        endpoints_b[i] = _expand_quantized((endpoints_b[i] << 1) | p_bits[i], 5, 8)

    # Read indices and interpolate
    for i in range(16):
        subset = _get_partition_index(partition, i, 3)
        # Clamp subset to valid range
        if subset < 0 or subset > 2:
            subset = 0
        # Check if this pixel is an anchor
        anchor_idx = _get_anchor_index(partition, subset, 3)
        is_anchor = (i == anchor_idx)
        index_bits = 2 if is_anchor else 3
        index = _read_bits(block_data, bit_pos, index_bits)
        bit_pos += index_bits

        e0 = subset * 2
        e1 = subset * 2 + 1

        output[i, 0] = _interpolate(endpoints_r[e0], endpoints_r[e1], index, 3)
        output[i, 1] = _interpolate(endpoints_g[e0], endpoints_g[e1], index, 3)
        output[i, 2] = _interpolate(endpoints_b[e0], endpoints_b[e1], index, 3)
        output[i, 3] = 255

    return output


@jit(nopython=True, cache=True)
def _decode_mode_1(block_data):
    """Decode BC7 mode 1: 2 subsets, 6-bit partition, 6-bit RGB, shared p-bit, 3-bit indices"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 2  # Skip mode bits

    # Read partition
    partition = _read_bits(block_data, bit_pos, 6)
    bit_pos += 6

    # Read 4 RGB endpoints
    endpoints_r = np.zeros(4, dtype=np.uint8)
    endpoints_g = np.zeros(4, dtype=np.uint8)
    endpoints_b = np.zeros(4, dtype=np.uint8)

    for i in range(4):
        endpoints_r[i] = _read_bits(block_data, bit_pos, 6)
        bit_pos += 6
    for i in range(4):
        endpoints_g[i] = _read_bits(block_data, bit_pos, 6)
        bit_pos += 6
    for i in range(4):
        endpoints_b[i] = _read_bits(block_data, bit_pos, 6)
        bit_pos += 6

    # Read shared p-bits (one per subset)
    p0 = _read_bits(block_data, bit_pos, 1)
    bit_pos += 1
    p1 = _read_bits(block_data, bit_pos, 1)
    bit_pos += 1

    # Expand endpoints
    for i in range(2):
        p = p0 if i == 0 else p1
        endpoints_r[i*2] = _expand_quantized((endpoints_r[i*2] << 1) | p, 7, 8)
        endpoints_r[i*2+1] = _expand_quantized((endpoints_r[i*2+1] << 1) | p, 7, 8)
        endpoints_g[i*2] = _expand_quantized((endpoints_g[i*2] << 1) | p, 7, 8)
        endpoints_g[i*2+1] = _expand_quantized((endpoints_g[i*2+1] << 1) | p, 7, 8)
        endpoints_b[i*2] = _expand_quantized((endpoints_b[i*2] << 1) | p, 7, 8)
        endpoints_b[i*2+1] = _expand_quantized((endpoints_b[i*2+1] << 1) | p, 7, 8)

    # Read indices
    for i in range(16):
        subset = _get_partition_index(partition, i, 2)
        # Clamp subset to valid range (0-1 for 2 subsets)
        if subset < 0 or subset > 1:
            subset = 0
        anchor_idx = _get_anchor_index(partition, subset, 2)
        is_anchor = (i == anchor_idx)
        index_bits = 2 if is_anchor else 3
        index = _read_bits(block_data, bit_pos, index_bits)
        bit_pos += index_bits

        e0 = subset * 2
        e1 = subset * 2 + 1

        output[i, 0] = _interpolate(endpoints_r[e0], endpoints_r[e1], index, 3)
        output[i, 1] = _interpolate(endpoints_g[e0], endpoints_g[e1], index, 3)
        output[i, 2] = _interpolate(endpoints_b[e0], endpoints_b[e1], index, 3)
        output[i, 3] = 255

    return output


@jit(nopython=True, cache=True)
def _decode_mode_2(block_data):
    """Decode BC7 mode 2: 3 subsets, 6-bit partition, 5-bit RGB, 2-bit indices"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 3  # Skip mode bits

    partition = _read_bits(block_data, bit_pos, 6)
    bit_pos += 6

    # Read 6 RGB endpoints
    endpoints_r = np.zeros(6, dtype=np.uint8)
    endpoints_g = np.zeros(6, dtype=np.uint8)
    endpoints_b = np.zeros(6, dtype=np.uint8)

    for i in range(6):
        endpoints_r[i] = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
        bit_pos += 5
    for i in range(6):
        endpoints_g[i] = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
        bit_pos += 5
    for i in range(6):
        endpoints_b[i] = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
        bit_pos += 5

    # Read indices
    for i in range(16):
        subset = _get_partition_index(partition, i, 3)
        # Clamp subset to valid range (0-2 for 3 subsets)
        if subset < 0 or subset > 2:
            subset = 0
        anchor_idx = _get_anchor_index(partition, subset, 3)
        is_anchor = (i == anchor_idx)
        index_bits = 1 if is_anchor else 2
        index = _read_bits(block_data, bit_pos, index_bits)
        bit_pos += index_bits

        e0 = subset * 2
        e1 = subset * 2 + 1

        output[i, 0] = _interpolate(endpoints_r[e0], endpoints_r[e1], index, 2)
        output[i, 1] = _interpolate(endpoints_g[e0], endpoints_g[e1], index, 2)
        output[i, 2] = _interpolate(endpoints_b[e0], endpoints_b[e1], index, 2)
        output[i, 3] = 255

    return output


@jit(nopython=True, cache=True)
def _decode_mode_3(block_data):
    """Decode BC7 mode 3: 2 subsets, 6-bit partition, 7-bit RGB, p-bit per endpoint, 2-bit indices"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 4  # Skip mode bits

    partition = _read_bits(block_data, bit_pos, 6)
    bit_pos += 6

    # Read 4 RGB endpoints
    endpoints_r = np.zeros(4, dtype=np.uint8)
    endpoints_g = np.zeros(4, dtype=np.uint8)
    endpoints_b = np.zeros(4, dtype=np.uint8)

    for i in range(4):
        endpoints_r[i] = _read_bits(block_data, bit_pos, 7)
        bit_pos += 7
    for i in range(4):
        endpoints_g[i] = _read_bits(block_data, bit_pos, 7)
        bit_pos += 7
    for i in range(4):
        endpoints_b[i] = _read_bits(block_data, bit_pos, 7)
        bit_pos += 7

    # Read p-bits
    p_bits = np.zeros(4, dtype=np.uint8)
    for i in range(4):
        p_bits[i] = _read_bits(block_data, bit_pos, 1)
        bit_pos += 1

    # Expand endpoints
    for i in range(4):
        endpoints_r[i] = _expand_quantized((endpoints_r[i] << 1) | p_bits[i], 8, 8)
        endpoints_g[i] = _expand_quantized((endpoints_g[i] << 1) | p_bits[i], 8, 8)
        endpoints_b[i] = _expand_quantized((endpoints_b[i] << 1) | p_bits[i], 8, 8)

    # Read indices
    for i in range(16):
        subset = _get_partition_index(partition, i, 2)
        # Clamp subset to valid range (0-1 for 2 subsets)
        if subset < 0 or subset > 1:
            subset = 0
        anchor_idx = _get_anchor_index(partition, subset, 2)
        is_anchor = (i == anchor_idx)
        index_bits = 1 if is_anchor else 2
        index = _read_bits(block_data, bit_pos, index_bits)
        bit_pos += index_bits

        e0 = subset * 2
        e1 = subset * 2 + 1

        output[i, 0] = _interpolate(endpoints_r[e0], endpoints_r[e1], index, 2)
        output[i, 1] = _interpolate(endpoints_g[e0], endpoints_g[e1], index, 2)
        output[i, 2] = _interpolate(endpoints_b[e0], endpoints_b[e1], index, 2)
        output[i, 3] = 255

    return output


@jit(nopython=True, cache=True)
def _decode_mode_4(block_data):
    """Decode BC7 mode 4: 1 subset, rotation, index selection, 5-bit RGB + 6-bit A"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 5  # Skip mode bits

    rotation = _read_bits(block_data, bit_pos, 2)
    bit_pos += 2
    index_selection = _read_bits(block_data, bit_pos, 1)
    bit_pos += 1

    # Read RGB endpoints (5 bits)
    r0 = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
    bit_pos += 5
    r1 = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
    bit_pos += 5
    g0 = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
    bit_pos += 5
    g1 = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
    bit_pos += 5
    b0 = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
    bit_pos += 5
    b1 = _expand_quantized(_read_bits(block_data, bit_pos, 5), 5, 8)
    bit_pos += 5

    # Read alpha endpoints (6 bits)
    a0 = _expand_quantized(_read_bits(block_data, bit_pos, 6), 6, 8)
    bit_pos += 6
    a1 = _expand_quantized(_read_bits(block_data, bit_pos, 6), 6, 8)
    bit_pos += 6

    # Read color indices (2-bit)
    color_indices = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        color_index_bits = 1 if i == 0 else 2
        color_indices[i] = _read_bits(block_data, bit_pos, color_index_bits)
        bit_pos += color_index_bits

    # Read alpha indices (3-bit)
    alpha_indices = np.zeros(16, dtype=np.uint8)
    for i in range(16):
        alpha_index_bits = 2 if i == 0 else 3
        alpha_indices[i] = _read_bits(block_data, bit_pos, alpha_index_bits)
        bit_pos += alpha_index_bits

    # Interpolate for each pixel
    for i in range(16):
        if index_selection == 0:
            # Normal: color indices for RGB, alpha indices for A
            r = _interpolate(r0, r1, color_indices[i], 2)
            g = _interpolate(g0, g1, color_indices[i], 2)
            b = _interpolate(b0, b1, color_indices[i], 2)
            a = _interpolate(a0, a1, alpha_indices[i], 3)
        else:
            # Swapped: alpha indices for RGB, color indices for A
            r = _interpolate(r0, r1, alpha_indices[i], 3)
            g = _interpolate(g0, g1, alpha_indices[i], 3)
            b = _interpolate(b0, b1, alpha_indices[i], 3)
            a = _interpolate(a0, a1, color_indices[i], 2)

        # Apply rotation (swap channels)
        if rotation == 0:
            output[i, 0] = r
            output[i, 1] = g
            output[i, 2] = b
            output[i, 3] = a
        elif rotation == 1:  # Swap A and R
            output[i, 0] = a
            output[i, 1] = g
            output[i, 2] = b
            output[i, 3] = r
        elif rotation == 2:  # Swap A and G
            output[i, 0] = r
            output[i, 1] = a
            output[i, 2] = b
            output[i, 3] = g
        elif rotation == 3:  # Swap A and B
            output[i, 0] = r
            output[i, 1] = g
            output[i, 2] = a
            output[i, 3] = b

    return output


@jit(nopython=True, cache=True)
def _decode_mode_5(block_data):
    """Decode BC7 mode 5: 1 subset, rotation, 7-bit RGB + 8-bit A"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 6  # Skip mode bits

    rotation = _read_bits(block_data, bit_pos, 2)
    bit_pos += 2

    # Read RGB endpoints (7 bits)
    r0 = _expand_quantized(_read_bits(block_data, bit_pos, 7), 7, 8)
    bit_pos += 7
    r1 = _expand_quantized(_read_bits(block_data, bit_pos, 7), 7, 8)
    bit_pos += 7
    g0 = _expand_quantized(_read_bits(block_data, bit_pos, 7), 7, 8)
    bit_pos += 7
    g1 = _expand_quantized(_read_bits(block_data, bit_pos, 7), 7, 8)
    bit_pos += 7
    b0 = _expand_quantized(_read_bits(block_data, bit_pos, 7), 7, 8)
    bit_pos += 7
    b1 = _expand_quantized(_read_bits(block_data, bit_pos, 7), 7, 8)
    bit_pos += 7

    # Read alpha endpoints (8 bits)
    a0 = _read_bits(block_data, bit_pos, 8)
    bit_pos += 8
    a1 = _read_bits(block_data, bit_pos, 8)
    bit_pos += 8

    # Read color indices (2 bits)
    for i in range(16):
        color_index_bits = 1 if i == 0 else 2
        color_index = _read_bits(block_data, bit_pos, color_index_bits)
        bit_pos += color_index_bits

        output[i, 0] = _interpolate(r0, r1, color_index, 2)
        output[i, 1] = _interpolate(g0, g1, color_index, 2)
        output[i, 2] = _interpolate(b0, b1, color_index, 2)
        output[i, 3] = 255

    # Read alpha indices (2 bits)
    for i in range(16):
        alpha_index_bits = 1 if i == 0 else 2
        alpha_index = _read_bits(block_data, bit_pos, alpha_index_bits)
        bit_pos += alpha_index_bits

        output[i, 3] = _interpolate(a0, a1, alpha_index, 2)

    # Apply rotation (swap channels if needed)
    if rotation > 0:
        for i in range(16):
            a = output[i, 3]
            if rotation == 1:  # Swap A and R
                output[i, 3] = output[i, 0]
                output[i, 0] = a
            elif rotation == 2:  # Swap A and G
                output[i, 3] = output[i, 1]
                output[i, 1] = a
            elif rotation == 3:  # Swap A and B
                output[i, 3] = output[i, 2]
                output[i, 2] = a

    return output


@jit(nopython=True, cache=True)
def _decode_mode_6(block_data):
    """Decode BC7 mode 6 (1 subset, 7.7.7.7 bits, p-bit per endpoint)"""
    output = np.zeros((16, 4), dtype=np.uint8)

    # Mode 6: 1 subset, 4-bit indices, 7-bit RGBA endpoints with p-bits
    bit_pos = 7  # Skip mode bits (7 bits for mode 6: 0b1000000)

    # Read endpoints (7 bits each, RGBA)
    r0 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    r1 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    g0 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    g1 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    b0 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    b1 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    a0 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7
    a1 = _read_bits(block_data, bit_pos, 7)
    bit_pos += 7

    # Read p-bits
    p0 = _read_bits(block_data, bit_pos, 1)
    bit_pos += 1
    p1 = _read_bits(block_data, bit_pos, 1)
    bit_pos += 1

    # Expand endpoints to 8 bits with p-bit
    r0 = _expand_quantized((r0 << 1) | p0, 8, 8)
    r1 = _expand_quantized((r1 << 1) | p1, 8, 8)
    g0 = _expand_quantized((g0 << 1) | p0, 8, 8)
    g1 = _expand_quantized((g1 << 1) | p1, 8, 8)
    b0 = _expand_quantized((b0 << 1) | p0, 8, 8)
    b1 = _expand_quantized((b1 << 1) | p1, 8, 8)
    a0 = _expand_quantized((a0 << 1) | p0, 8, 8)
    a1 = _expand_quantized((a1 << 1) | p1, 8, 8)

    # Read indices (4 bits each, 16 pixels)
    for i in range(16):
        index_bits = 3 if i == 0 else 4  # Anchor has implicit 0 MSB

        index = _read_bits(block_data, bit_pos, index_bits)
        bit_pos += index_bits

        # Interpolate
        output[i, 0] = _interpolate(r0, r1, index, 4)
        output[i, 1] = _interpolate(g0, g1, index, 4)
        output[i, 2] = _interpolate(b0, b1, index, 4)
        output[i, 3] = _interpolate(a0, a1, index, 4)

    return output


@jit(nopython=True, cache=True)
def _decode_mode_7(block_data):
    """Decode BC7 mode 7: 2 subsets, 6-bit partition, 5-bit RGBA, p-bit per endpoint, 2-bit indices"""
    output = np.zeros((16, 4), dtype=np.uint8)
    bit_pos = 8  # Skip mode bits

    partition = _read_bits(block_data, bit_pos, 6)
    bit_pos += 6

    # Read 4 RGBA endpoints
    endpoints_r = np.zeros(4, dtype=np.uint8)
    endpoints_g = np.zeros(4, dtype=np.uint8)
    endpoints_b = np.zeros(4, dtype=np.uint8)
    endpoints_a = np.zeros(4, dtype=np.uint8)

    for i in range(4):
        endpoints_r[i] = _read_bits(block_data, bit_pos, 5)
        bit_pos += 5
    for i in range(4):
        endpoints_g[i] = _read_bits(block_data, bit_pos, 5)
        bit_pos += 5
    for i in range(4):
        endpoints_b[i] = _read_bits(block_data, bit_pos, 5)
        bit_pos += 5
    for i in range(4):
        endpoints_a[i] = _read_bits(block_data, bit_pos, 5)
        bit_pos += 5

    # Read p-bits
    p_bits = np.zeros(4, dtype=np.uint8)
    for i in range(4):
        p_bits[i] = _read_bits(block_data, bit_pos, 1)
        bit_pos += 1

    # Expand endpoints
    for i in range(4):
        endpoints_r[i] = _expand_quantized((endpoints_r[i] << 1) | p_bits[i], 6, 8)
        endpoints_g[i] = _expand_quantized((endpoints_g[i] << 1) | p_bits[i], 6, 8)
        endpoints_b[i] = _expand_quantized((endpoints_b[i] << 1) | p_bits[i], 6, 8)
        endpoints_a[i] = _expand_quantized((endpoints_a[i] << 1) | p_bits[i], 6, 8)

    # Read indices
    for i in range(16):
        subset = _get_partition_index(partition, i, 2)
        # Clamp subset to valid range (0-1 for 2 subsets)
        if subset < 0 or subset > 1:
            subset = 0
        anchor_idx = _get_anchor_index(partition, subset, 2)
        is_anchor = (i == anchor_idx)
        index_bits = 1 if is_anchor else 2
        index = _read_bits(block_data, bit_pos, index_bits)
        bit_pos += index_bits

        e0 = subset * 2
        e1 = subset * 2 + 1

        output[i, 0] = _interpolate(endpoints_r[e0], endpoints_r[e1], index, 2)
        output[i, 1] = _interpolate(endpoints_g[e0], endpoints_g[e1], index, 2)
        output[i, 2] = _interpolate(endpoints_b[e0], endpoints_b[e1], index, 2)
        output[i, 3] = _interpolate(endpoints_a[e0], endpoints_a[e1], index, 2)

    return output


@jit(nopython=True, cache=True)
def _decode_block_jit(block_data):
    """Decode a single BC7 block - JIT compiled"""
    # Determine mode from first bits
    mode = -1
    mode_bits = block_data[0]

    for m in range(8):
        if mode_bits & (1 << m):
            mode = m
            break

    # If no valid mode found, return black
    if mode == -1:
        return np.zeros((16, 4), dtype=np.uint8)

    # Decode based on mode
    if mode == 0:
        return _decode_mode_0(block_data)
    elif mode == 1:
        return _decode_mode_1(block_data)
    elif mode == 2:
        return _decode_mode_2(block_data)
    elif mode == 3:
        return _decode_mode_3(block_data)
    elif mode == 4:
        return _decode_mode_4(block_data)
    elif mode == 5:
        return _decode_mode_5(block_data)
    elif mode == 6:
        return _decode_mode_6(block_data)
    elif mode == 7:
        return _decode_mode_7(block_data)

    # Fallback (shouldn't happen)
    return np.zeros((16, 4), dtype=np.uint8)


@jit(nopython=True, cache=True)
def _process_blocks_jit(blocks, output, blocks_x, blocks_y, width, height):
    """JIT-compiled block processing for BC7 decompression"""
    num_blocks = blocks_x * blocks_y

    for block_idx in range(num_blocks):
        block_x = block_idx % blocks_x
        block_y = block_idx // blocks_x

        # Decode block
        block_colors = _decode_block_jit(blocks[block_idx])

        # Copy to output
        y_start = block_y * 4
        y_end = min(y_start + 4, height)
        x_start = block_x * 4
        x_end = min(x_start + 4, width)

        for pixel_idx in range(16):
            pixel_y = pixel_idx // 4
            pixel_x = pixel_idx % 4

            out_y = y_start + pixel_y
            out_x = x_start + pixel_x

            if out_y < y_end and out_x < x_end:
                output[out_y, out_x, 0] = block_colors[pixel_idx, 0]
                output[out_y, out_x, 1] = block_colors[pixel_idx, 1]
                output[out_y, out_x, 2] = block_colors[pixel_idx, 2]
                output[out_y, out_x, 3] = block_colors[pixel_idx, 3]


class BC7Decompressor(TextureDecompressor):
    """
    BC7 texture decompressor - NumPy vectorization + Numba JIT

    BC7 is a high-quality block compression format with 8 modes (0-7).
    Each 4x4 block is 16 bytes (128 bits) and can use different encoding modes
    optimized for different content types.
    """

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress BC7 texture data to RGBA8

        BC7 stores 4x4 pixel blocks in 16 bytes each with 8 different modes
        optimized for different content types.
        """
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        # Read all blocks
        blocks = np.frombuffer(data[:num_blocks * 16], dtype=np.uint8).reshape(-1, 16)

        # Create output array
        output = np.zeros((height, width, 4), dtype=np.uint8)

        # Process all blocks using JIT-compiled function
        _process_blocks_jit(blocks, output, blocks_x, blocks_y, width, height)

        return output
