"""Command-line interface for puredds"""
import sys
import argparse
import time
import imageio.v3 as iio
from .dds import DDS


def main():
    """Command-line interface for puredds"""
    parser = argparse.ArgumentParser(
        description='Read and convert DDS (DirectDraw Surface) texture files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  puredds texture.dds                          # Display DDS file info
  puredds texture.dds -o output.png            # Convert to PNG
  puredds texture.dds -o output.png -m 1       # Extract mipmap level 1
  puredds texture.dds -o output.png --mipmap 2 # Extract mipmap level 2
        """
    )

    parser.add_argument('input', help='Input DDS file path')
    parser.add_argument('-o', '--output', help='Output image file path (e.g., output.png)')
    parser.add_argument('-m', '--mipmap', type=int, default=0,
                        help='Mipmap level to extract (default: 0 = full resolution)')

    args = parser.parse_args()

    try:
        with open(args.input, 'rb') as f:
            data = f.read()

        dds = DDS.from_bytes(data)
        print(dds)

        # Try to convert to image if output path is specified
        if args.output:
            try:
                print(f"\nConverting to image (mipmap level {args.mipmap})...")

                # Time decompression
                start_decompress = time.perf_counter()
                image_array = dds.to_image(args.mipmap)
                end_decompress = time.perf_counter()
                decompress_time = end_decompress - start_decompress

                # Time saving
                start_save = time.perf_counter()
                iio.imwrite(args.output, image_array)
                end_save = time.perf_counter()
                save_time = end_save - start_save

                print(f"Saved to: {args.output}")
                print(f"Image size: {image_array.shape[1]}x{image_array.shape[0]}")
                print(f"Image format: {image_array.dtype} ({image_array.shape[2]} channels)")
                print(f"Decompression time: {decompress_time*1000:.2f} ms")
                print(f"Save time: {save_time*1000:.2f} ms")
                print(f"Total conversion time: {(decompress_time + save_time)*1000:.2f} ms")
            except NotImplementedError as e:
                print(f"Cannot convert to image: {e}")
            except ValueError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error converting to image: {e}")
                import traceback
                traceback.print_exc()

    except FileNotFoundError:
        print(f"Error: File '{args.input}' not found")
        sys.exit(1)
    except ValueError as e:
        print(f"Error parsing DDS file: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
