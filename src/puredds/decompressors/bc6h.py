"""BC6H texture decompressor"""
import numpy as np
from numba import jit
from .base import TextureDecompressor


# Interpolation weights (same as BC7)
WEIGHTS_3 = np.array([0, 9, 18, 27, 37, 46, 55, 64], dtype=np.int32)
WEIGHTS_4 = np.array([0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64], dtype=np.int32)

# 2-subset partition table (same as BC7, 64 entries but BC6H only uses first 32)
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

# Anchor index for subset 1 in 2-subset partitions
ANCHOR_TABLE_2 = np.array([
    15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,
    15, 2, 8, 2, 2, 8, 8,15, 2, 8, 2, 2, 8, 8, 2, 2,
    15,15, 6, 8, 2, 8,15,15, 2, 8, 2, 2, 2,15,15, 6,
     6, 2, 6, 8,15,15, 2, 2,15,15,15,15,15, 2, 2,15,
], dtype=np.uint8)

# Mode info: [base_bits, delta_r, delta_g, delta_b, num_subsets, transformed, index_bits]
MODE_INFO = np.array([
    [10, 5, 5, 5, 2, 1, 3],   # 0:  code 00
    [ 7, 6, 6, 6, 2, 1, 3],   # 1:  code 01
    [11, 5, 4, 4, 2, 1, 3],   # 2:  code 00010
    [11, 4, 5, 4, 2, 1, 3],   # 3:  code 00110
    [11, 4, 4, 5, 2, 1, 3],   # 4:  code 01010
    [ 9, 5, 5, 5, 2, 1, 3],   # 5:  code 01110
    [ 8, 6, 5, 5, 2, 1, 3],   # 6:  code 10010
    [ 8, 5, 6, 5, 2, 1, 3],   # 7:  code 10110
    [ 8, 5, 5, 6, 2, 1, 3],   # 8:  code 11010
    [ 6, 6, 6, 6, 2, 0, 3],   # 9:  code 11110
    [10,10,10,10, 1, 0, 4],   # 10: code 00011
    [11, 9, 9, 9, 1, 1, 4],   # 11: code 00111
    [12, 8, 8, 8, 1, 1, 4],   # 12: code 01011
    [16, 4, 4, 4, 1, 1, 4],   # 13: code 01111
], dtype=np.int32)


@jit(nopython=True, cache=True)
def _rb(data, offset, count):
    """Read count bits from byte array at bit offset (LSB first)"""
    result = 0
    for i in range(count):
        bp = offset + i
        result |= ((data[bp >> 3] >> (bp & 7)) & 1) << i
    return result


@jit(nopython=True, cache=True)
def _rb_r(data, offset, count):
    """Read count bits in reverse order (for modes 12, 13)"""
    val = _rb(data, offset, count)
    result = 0
    for i in range(count):
        result = (result << 1) | (val & 1)
        val >>= 1
    return result


@jit(nopython=True, cache=True)
def _sign_extend(value, num_bits):
    """Sign extend a value from num_bits to int32"""
    if num_bits >= 32 or num_bits <= 0:
        return value
    if value & (1 << (num_bits - 1)):
        value |= -(1 << num_bits)
    return value


@jit(nopython=True, cache=True)
def _unquantize(val, bits, signed):
    """Unquantize endpoint value to full range"""
    if not signed:
        if bits >= 15:
            return val
        if val == 0:
            return 0
        if val == ((1 << bits) - 1):
            return 0xFFFF
        return ((val << 16) + 0x8000) >> bits
    else:
        if bits >= 16:
            return val
        s = 0
        if val < 0:
            s = 1
            val = -val
        if val == 0:
            unq = 0
        elif val >= ((1 << (bits - 1)) - 1):
            unq = 0x7FFF
        else:
            unq = ((val << 15) + 0x4000) >> (bits - 1)
        if s:
            unq = -unq
        return unq


@jit(nopython=True, cache=True)
def _finish_unquantize(val, signed):
    """Final unquantize + pack as half-float uint16 bit pattern"""
    if not signed:
        return (val * 31) >> 6
    else:
        if val < 0:
            val = -((-val * 31) >> 5)
        else:
            val = (val * 31) >> 5
        s = 0
        if val < 0:
            s = 0x8000
            val = -val
        return s | val


@jit(nopython=True, cache=True)
def _half_to_float(h):
    """Convert uint16 half-float bit pattern to float32"""
    sign = (h >> 15) & 1
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF
    if exp == 0:
        if mant == 0:
            return np.float32(-0.0) if sign else np.float32(0.0)
        # Denormalized
        val = np.float32(mant) / np.float32(1024.0) * np.float32(2.0 ** -14)
        return -val if sign else val
    elif exp == 31:
        if mant == 0:
            return np.float32(-np.inf) if sign else np.float32(np.inf)
        return np.float32(np.nan)
    else:
        val = (np.float32(1.0) + np.float32(mant) / np.float32(1024.0)) * np.float32(2.0 ** (exp - 15))
        return -val if sign else val


@jit(nopython=True, cache=True)
def _extract_mode0(data, p, r, g, b):
    """Mode 0: 10.555 (code 00)"""
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 5); p += 5
    g[3] |= _rb(data, p, 1) << 4; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1); p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode1(data, p, r, g, b):
    """Mode 1: 7666 (code 01)"""
    g[2] |= _rb(data, p, 1) << 5; p += 1
    g[3] |= _rb(data, p, 1) << 4; p += 1
    g[3] |= _rb(data, p, 1) << 5; p += 1
    r[0] |= _rb(data, p, 7); p += 7
    b[3] |= _rb(data, p, 1); p += 1
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[0] |= _rb(data, p, 7); p += 7
    b[2] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 2; p += 1
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[0] |= _rb(data, p, 7); p += 7
    b[3] |= _rb(data, p, 1) << 3; p += 1
    b[3] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[1] |= _rb(data, p, 6); p += 6
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 6); p += 6
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 6); p += 6
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 6); p += 6
    r[3] |= _rb(data, p, 6); p += 6
    return p


@jit(nopython=True, cache=True)
def _extract_mode2(data, p, r, g, b):
    """Mode 2: 11.555/11.444/11.444 (code 00010)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 5); p += 5
    r[0] |= _rb(data, p, 1) << 10; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 4); p += 4
    g[0] |= _rb(data, p, 1) << 10; p += 1
    b[3] |= _rb(data, p, 1); p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 4); p += 4
    b[0] |= _rb(data, p, 1) << 10; p += 1
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode3(data, p, r, g, b):
    """Mode 3: 11.444/11.555/11.444 (code 00110)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 4); p += 4
    r[0] |= _rb(data, p, 1) << 10; p += 1
    g[3] |= _rb(data, p, 1) << 4; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 5); p += 5
    g[0] |= _rb(data, p, 1) << 10; p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 4); p += 4
    b[0] |= _rb(data, p, 1) << 10; p += 1
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 4); p += 4
    b[3] |= _rb(data, p, 1); p += 1
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 4); p += 4
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode4(data, p, r, g, b):
    """Mode 4: 11.444/11.444/11.555 (code 01010)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 4); p += 4
    r[0] |= _rb(data, p, 1) << 10; p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 4); p += 4
    g[0] |= _rb(data, p, 1) << 10; p += 1
    b[3] |= _rb(data, p, 1); p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 5); p += 5
    b[0] |= _rb(data, p, 1) << 10; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 4); p += 4
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 4); p += 4
    b[3] |= _rb(data, p, 1) << 4; p += 1
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode5(data, p, r, g, b):
    """Mode 5: 9555 (code 01110)"""
    r[0] |= _rb(data, p, 9); p += 9
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[0] |= _rb(data, p, 9); p += 9
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[0] |= _rb(data, p, 9); p += 9
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[1] |= _rb(data, p, 5); p += 5
    g[3] |= _rb(data, p, 1) << 4; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1); p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode6(data, p, r, g, b):
    """Mode 6: 8666/8555/8555 (code 10010)"""
    r[0] |= _rb(data, p, 8); p += 8
    g[3] |= _rb(data, p, 1) << 4; p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[0] |= _rb(data, p, 8); p += 8
    b[3] |= _rb(data, p, 1) << 2; p += 1
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[0] |= _rb(data, p, 8); p += 8
    b[3] |= _rb(data, p, 1) << 3; p += 1
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[1] |= _rb(data, p, 6); p += 6
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1); p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 6); p += 6
    r[3] |= _rb(data, p, 6); p += 6
    return p


@jit(nopython=True, cache=True)
def _extract_mode7(data, p, r, g, b):
    """Mode 7: 8555/8666/8555 (code 10110)"""
    r[0] |= _rb(data, p, 8); p += 8
    b[3] |= _rb(data, p, 1); p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[0] |= _rb(data, p, 8); p += 8
    g[2] |= _rb(data, p, 1) << 5; p += 1
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[0] |= _rb(data, p, 8); p += 8
    g[3] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[1] |= _rb(data, p, 5); p += 5
    g[3] |= _rb(data, p, 1) << 4; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 6); p += 6
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode8(data, p, r, g, b):
    """Mode 8: 8555/8555/8666 (code 11010)"""
    r[0] |= _rb(data, p, 8); p += 8
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[0] |= _rb(data, p, 8); p += 8
    b[2] |= _rb(data, p, 1) << 5; p += 1
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[0] |= _rb(data, p, 8); p += 8
    b[3] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[1] |= _rb(data, p, 5); p += 5
    g[3] |= _rb(data, p, 1) << 4; p += 1
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1); p += 1
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 6); p += 6
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 2; p += 1
    r[3] |= _rb(data, p, 5); p += 5
    b[3] |= _rb(data, p, 1) << 3; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode9(data, p, r, g, b):
    """Mode 9: 6666 (code 11110) - NOT transformed"""
    r[0] |= _rb(data, p, 6); p += 6
    g[3] |= _rb(data, p, 1) << 4; p += 1
    b[3] |= _rb(data, p, 1); p += 1
    b[3] |= _rb(data, p, 1) << 1; p += 1
    b[2] |= _rb(data, p, 1) << 4; p += 1
    g[0] |= _rb(data, p, 6); p += 6
    g[2] |= _rb(data, p, 1) << 5; p += 1
    b[2] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 2; p += 1
    g[2] |= _rb(data, p, 1) << 4; p += 1
    b[0] |= _rb(data, p, 6); p += 6
    g[3] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 3; p += 1
    b[3] |= _rb(data, p, 1) << 5; p += 1
    b[3] |= _rb(data, p, 1) << 4; p += 1
    r[1] |= _rb(data, p, 6); p += 6
    g[2] |= _rb(data, p, 4); p += 4
    g[1] |= _rb(data, p, 6); p += 6
    g[3] |= _rb(data, p, 4); p += 4
    b[1] |= _rb(data, p, 6); p += 6
    b[2] |= _rb(data, p, 4); p += 4
    r[2] |= _rb(data, p, 6); p += 6
    r[3] |= _rb(data, p, 6); p += 6
    return p


@jit(nopython=True, cache=True)
def _extract_mode10(data, p, r, g, b):
    """Mode 10: 10.10 single region, NOT transformed (code 00011)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 10); p += 10
    g[1] |= _rb(data, p, 10); p += 10
    b[1] |= _rb(data, p, 10); p += 10
    return p


@jit(nopython=True, cache=True)
def _extract_mode11(data, p, r, g, b):
    """Mode 11: 11.9 single region (code 00111)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 9); p += 9
    r[0] |= _rb(data, p, 1) << 10; p += 1
    g[1] |= _rb(data, p, 9); p += 9
    g[0] |= _rb(data, p, 1) << 10; p += 1
    b[1] |= _rb(data, p, 9); p += 9
    b[0] |= _rb(data, p, 1) << 10; p += 1
    return p


@jit(nopython=True, cache=True)
def _extract_mode12(data, p, r, g, b):
    """Mode 12: 12.8 single region (code 01011)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 8); p += 8
    r[0] |= _rb_r(data, p, 2) << 10; p += 2
    g[1] |= _rb(data, p, 8); p += 8
    g[0] |= _rb_r(data, p, 2) << 10; p += 2
    b[1] |= _rb(data, p, 8); p += 8
    b[0] |= _rb_r(data, p, 2) << 10; p += 2
    return p


@jit(nopython=True, cache=True)
def _extract_mode13(data, p, r, g, b):
    """Mode 13: 16.4 single region (code 01111)"""
    r[0] |= _rb(data, p, 10); p += 10
    g[0] |= _rb(data, p, 10); p += 10
    b[0] |= _rb(data, p, 10); p += 10
    r[1] |= _rb(data, p, 4); p += 4
    r[0] |= _rb_r(data, p, 6) << 10; p += 6
    g[1] |= _rb(data, p, 4); p += 4
    g[0] |= _rb_r(data, p, 6) << 10; p += 6
    b[1] |= _rb(data, p, 4); p += 4
    b[0] |= _rb_r(data, p, 6) << 10; p += 6
    return p


@jit(nopython=True, cache=True)
def _decode_bc6h_block(data, signed):
    """Decode a single BC6H block to 16 float32 RGB pixels"""
    output = np.zeros((16, 3), dtype=np.float32)

    r = np.zeros(4, dtype=np.int32)
    g = np.zeros(4, dtype=np.int32)
    b = np.zeros(4, dtype=np.int32)

    # Mode detection
    mode_val = _rb(data, 0, 2)
    if mode_val < 2:
        bit_pos = 2
    else:
        mode_val = _rb(data, 0, 5)
        bit_pos = 5

    # Map raw mode value to mode index
    if mode_val == 0:
        mode = 0
    elif mode_val == 1:
        mode = 1
    elif mode_val == 2:
        mode = 2
    elif mode_val == 6:
        mode = 3
    elif mode_val == 10:
        mode = 4
    elif mode_val == 14:
        mode = 5
    elif mode_val == 18:
        mode = 6
    elif mode_val == 22:
        mode = 7
    elif mode_val == 26:
        mode = 8
    elif mode_val == 30:
        mode = 9
    elif mode_val == 3:
        mode = 10
    elif mode_val == 7:
        mode = 11
    elif mode_val == 11:
        mode = 12
    elif mode_val == 15:
        mode = 13
    else:
        return output  # Reserved mode

    # Get mode info
    base_bits = MODE_INFO[mode, 0]
    delta_r = MODE_INFO[mode, 1]
    delta_g = MODE_INFO[mode, 2]
    delta_b = MODE_INFO[mode, 3]
    num_subsets = MODE_INFO[mode, 4]
    transformed = MODE_INFO[mode, 5]
    index_bits = MODE_INFO[mode, 6]

    # Extract endpoints (scattered bit extraction per mode)
    if mode == 0:
        bit_pos = _extract_mode0(data, bit_pos, r, g, b)
    elif mode == 1:
        bit_pos = _extract_mode1(data, bit_pos, r, g, b)
    elif mode == 2:
        bit_pos = _extract_mode2(data, bit_pos, r, g, b)
    elif mode == 3:
        bit_pos = _extract_mode3(data, bit_pos, r, g, b)
    elif mode == 4:
        bit_pos = _extract_mode4(data, bit_pos, r, g, b)
    elif mode == 5:
        bit_pos = _extract_mode5(data, bit_pos, r, g, b)
    elif mode == 6:
        bit_pos = _extract_mode6(data, bit_pos, r, g, b)
    elif mode == 7:
        bit_pos = _extract_mode7(data, bit_pos, r, g, b)
    elif mode == 8:
        bit_pos = _extract_mode8(data, bit_pos, r, g, b)
    elif mode == 9:
        bit_pos = _extract_mode9(data, bit_pos, r, g, b)
    elif mode == 10:
        bit_pos = _extract_mode10(data, bit_pos, r, g, b)
    elif mode == 11:
        bit_pos = _extract_mode11(data, bit_pos, r, g, b)
    elif mode == 12:
        bit_pos = _extract_mode12(data, bit_pos, r, g, b)
    elif mode == 13:
        bit_pos = _extract_mode13(data, bit_pos, r, g, b)

    # Read partition (2-subset modes only)
    partition = 0
    if num_subsets == 2:
        partition = _rb(data, bit_pos, 5)
        bit_pos += 5

    num_ep = 4 if num_subsets == 2 else 2

    # ---- Sign extension ----
    # Base endpoint: sign extend only if signed format
    if signed:
        r[0] = _sign_extend(r[0], base_bits)
        g[0] = _sign_extend(g[0], base_bits)
        b[0] = _sign_extend(b[0], base_bits)

    # Other endpoints: sign extend if transformed OR signed
    if transformed or signed:
        for i in range(1, num_ep):
            r[i] = _sign_extend(r[i], delta_r)
            g[i] = _sign_extend(g[i], delta_g)
            b[i] = _sign_extend(b[i], delta_b)
    # For non-transformed unsigned (mode 9, 10): no sign extension on secondary endpoints

    # ---- Transform inversion (delta to absolute) ----
    if transformed:
        mask = (1 << base_bits) - 1
        for i in range(1, num_ep):
            r[i] = (r[0] + r[i]) & mask
            g[i] = (g[0] + g[i]) & mask
            b[i] = (b[0] + b[i]) & mask
            if signed:
                r[i] = _sign_extend(r[i], base_bits)
                g[i] = _sign_extend(g[i], base_bits)
                b[i] = _sign_extend(b[i], base_bits)

    # ---- Unquantize all endpoints ----
    for i in range(num_ep):
        r[i] = _unquantize(r[i], base_bits, signed)
        g[i] = _unquantize(g[i], base_bits, signed)
        b[i] = _unquantize(b[i], base_bits, signed)

    # ---- Read indices, interpolate, finish unquantize ----
    for pixel in range(16):
        if num_subsets == 2:
            subset = PARTITION_TABLE_2[partition, pixel]
        else:
            subset = 0

        # Determine if this pixel is an anchor (loses 1 index bit)
        is_anchor = False
        if subset == 0 and pixel == 0:
            is_anchor = True
        elif subset == 1 and pixel == ANCHOR_TABLE_2[partition]:
            is_anchor = True

        n_bits = index_bits - (1 if is_anchor else 0)
        idx = _rb(data, bit_pos, n_bits)
        bit_pos += n_bits

        # Get weight
        if index_bits == 3:
            weight = WEIGHTS_3[idx]
        else:
            weight = WEIGHTS_4[idx]

        # Endpoint pair for this subset
        e0 = subset * 2
        e1 = subset * 2 + 1

        # Interpolate
        ri = (r[e0] * (64 - weight) + r[e1] * weight + 32) >> 6
        gi = (g[e0] * (64 - weight) + g[e1] * weight + 32) >> 6
        bi = (b[e0] * (64 - weight) + b[e1] * weight + 32) >> 6

        # Finish unquantize (produces half-float uint16 bit pattern)
        rh = _finish_unquantize(ri, signed)
        gh = _finish_unquantize(gi, signed)
        bh = _finish_unquantize(bi, signed)

        # Convert half-float to float32
        output[pixel, 0] = _half_to_float(rh)
        output[pixel, 1] = _half_to_float(gh)
        output[pixel, 2] = _half_to_float(bh)

    return output


@jit(nopython=True, cache=True)
def _process_blocks_jit(blocks, output, blocks_x, blocks_y, width, height, signed):
    """Process all BC6H blocks"""
    num_blocks = blocks_x * blocks_y
    for block_idx in range(num_blocks):
        block_x = block_idx % blocks_x
        block_y = block_idx // blocks_x

        block_colors = _decode_bc6h_block(blocks[block_idx], signed)

        y_start = block_y * 4
        x_start = block_x * 4

        for pixel_idx in range(16):
            py = pixel_idx // 4
            px = pixel_idx % 4
            out_y = y_start + py
            out_x = x_start + px
            if out_y < height and out_x < width:
                output[out_y, out_x, 0] = block_colors[pixel_idx, 0]
                output[out_y, out_x, 1] = block_colors[pixel_idx, 1]
                output[out_y, out_x, 2] = block_colors[pixel_idx, 2]


class BC6HDecompressor(TextureDecompressor):
    """
    BC6H texture decompressor (HDR block compression)

    BC6H compresses HDR RGB data into 128-bit blocks.
    Supports both unsigned (UF16) and signed (SF16) formats.
    Returns float32 RGB data with shape (height, width, 3).
    """

    def __init__(self, signed=False):
        self.signed = signed

    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        blocks_x = (width + 3) // 4
        blocks_y = (height + 3) // 4
        num_blocks = blocks_x * blocks_y

        blocks = np.frombuffer(data[:num_blocks * 16], dtype=np.uint8).reshape(-1, 16)
        output = np.zeros((height, width, 3), dtype=np.float32)

        _process_blocks_jit(blocks, output, blocks_x, blocks_y, width, height, self.signed)

        return output
