"""Base class for texture decompression"""
from abc import ABC, abstractmethod
import numpy as np


class TextureDecompressor(ABC):
    """Base class for texture decompression"""
    @abstractmethod
    def decompress(self, data: bytes, width: int, height: int) -> np.ndarray:
        """
        Decompress texture data to RGBA8 format

        Args:
            data: Compressed texture data
            width: Texture width in pixels
            height: Texture height in pixels

        Returns:
            numpy array of shape (height, width, 4) with dtype uint8 (RGBA)
        """
        pass
