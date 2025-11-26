# puredds

A pure Python library for reading and converting DDS (DirectDraw Surface) texture files.

## Features

- Read DDS texture files with various compression formats
- Support for BC1/DXT1, BC2/DXT3, BC3/DXT5, BC4, and BC5 compression
- Convert DDS textures to PIL Images
- Extract specific mipmap levels
- Vectorized NumPy-based decompression for performance
- Command-line interface for quick conversions

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
from PIL import Image

# Read a DDS file
with open('texture.dds', 'rb') as f:
    data = f.read()

dds = DDS.from_bytes(data)

# Print information about the texture
print(dds)

# Convert to PIL Image
image = dds.to_image()
image.save('output.png')

# Extract a specific mipmap level
mipmap1 = dds.to_image(mipmap_level=1)
mipmap1.save('mipmap1.png')
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

Both legacy FourCC formats and DX10 DXGI formats are supported.

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
