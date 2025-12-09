# puredds

A pure Python library for reading and converting DDS (DirectDraw Surface) texture files.

## Features

- Read DDS texture files with various compression formats
- Support for BC1/DXT1, BC2/DXT3, BC3/DXT5, BC4, BC5, and BC7 compression
- Convert DDS textures to numpy arrays compatible with imageio
- Extract specific mipmap levels
- Vectorized NumPy-based decompression with Numba JIT for high performance
- Command-line interface for quick conversions
- Export to any format supported by imageio (PNG, JPEG, TIFF, etc.)

## Installation

```bash
pip install puredds
```

For development:

```bash
pip install puredds[dev]
```

From local wheel file:
```bash
# First build using instructions below
pip install ./dist/puredds-version-pyX-none-any.whl
```

## Usage

### As a Library

```python
from puredds import DDS
import imageio.v3 as iio

# Read a DDS file
with open('texture.dds', 'rb') as f:
    data = f.read()

dds = DDS.from_bytes(data)

# Print information about the texture
print(dds)

# Convert to numpy array and save
# Returns uint8 RGBA array
image_array = dds.to_image()
iio.imwrite('output.png', image_array)

# Extract a specific mipmap level
mipmap1 = dds.to_image(mipmap_level=1)
iio.imwrite('mipmap1.png', mipmap1)
```

### Command Line

```bash
# Display DDS file information
puredds texture.dds

# Convert to PNG
puredds texture.dds -o output.png

# Extract mipmap level 1
puredds texture.dds -o output.png -m 1
```

## Supported Formats

- **BC1 (DXT1)**: RGB compression with optional 1-bit alpha
- **BC2 (DXT2/DXT3)**: RGB compression with explicit 4-bit alpha
- **BC3 (DXT4/DXT5)**: RGB compression with interpolated alpha
- **BC4**: Single-channel compression (grayscale)
- **BC5**: Two-channel compression (typically for normal maps)
- **BC7**: High-quality RGB/RGBA compression with multiple encoding modes

Both legacy FourCC formats and DX10 DXGI formats are supported.

## Performance Note

**First-time decompression**: The first time you decompress a texture in each format, Numba will compile the decompression functions to machine code. This compilation takes a few seconds but only happens once. The compiled code is cached, so subsequent runs will be significantly faster.

For optimal performance in production:
- Warm up the cache by decompressing a sample texture of each format you'll use
- Distribute the Numba cache directory (`__pycache__/*.nbc` files) with your application if needed

## Building from Source

```bash
# Install build dependencies
pip install build

# Build wheel
python -m build

# The wheel will be in the dist/ directory
```

## Development

```bash
# Install in editable mode with dev dependencies
pip install -e .[dev]

# Run tests (if available)
pytest
```

## Contributing

Please feel free to submit a Pull Request.
