"""Command-line interface for puredds"""
import sys
import argparse
import time
import os
import imageio.v3 as iio
from .dds import DDS


def main():
    """Command-line interface for puredds"""
    parser = argparse.ArgumentParser(
        description='Read and convert DDS (DirectDraw Surface) texture files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  puredds texture.dds                               # Display DDS file info
  puredds texture.dds -o output.png                 # Convert to PNG
  puredds texture.dds -o output.png -m 1            # Extract mipmap level 1
  puredds texture.dds -o output.png --mipmap 2      # Extract mipmap level 2
  puredds cubemap.dds -o output.png --all           # Extract all 6 cubemap faces
  puredds cubemap.dds -o output.png -f 2            # Extract face 2 of cubemap
  puredds array.dds -o output.png -a 0 --all        # Extract all mipmap levels for array slice 0
  puredds volume.dds -o output.png --all            # Extract all depth slices from volume texture
  puredds volume.dds -o output.png -d 5             # Extract depth slice 5 from volume texture
  puredds texture.dds -o output.png --slices 0,2,4  # Extract specific slices
  puredds texture.dds -o output.png --slices 0-5    # Extract slice range
        """
    )

    parser.add_argument('input', help='Input DDS file path')
    parser.add_argument('-o', '--output', help='Output image file path (e.g., output.png)')
    parser.add_argument('-m', '--mipmap', type=int, default=0,
                        help='Mipmap level to extract (default: 0 = full resolution)')
    parser.add_argument('-a', '--array-index', type=int, default=0,
                        help='Array slice index to extract (default: 0)')
    parser.add_argument('-f', '--face', type=int, default=0,
                        help='Cubemap face index to extract (default: 0, range: 0-5)')
    parser.add_argument('-d', '--depth', type=int, default=0,
                        help='Depth slice index for volume textures (default: 0)')
    parser.add_argument('--all', action='store_true',
                        help='Extract all faces (cubemaps), all array slices, or all depth slices (volume textures)')
    parser.add_argument('--slices', type=str,
                        help='Specify slices to extract (e.g., "0,2,4" or "0-5")')

    args = parser.parse_args()

    try:
        with open(args.input, 'rb') as f:
            data = f.read()

        dds = DDS.from_bytes(data)
        print(dds)

        # Try to convert to image if output path is specified
        if args.output:
            try:
                # Import DDSCAPS2 for cubemap and volume texture detection
                from .enums import DDSCAPS2

                # Determine texture type
                is_cubemap = bool(dds.header.dwCaps2 & DDSCAPS2.CUBEMAP)
                is_volume = dds.is_volume()

                # Determine which slices to extract
                slices_to_extract = []
                slice_type = "default"  # Types: "default", "face", "array", "depth"

                if args.slices:
                    # Parse slice specification
                    slices_to_extract = parse_slice_spec(args.slices)
                    # Determine slice type based on texture type
                    if is_cubemap:
                        slice_type = "face"
                    elif is_volume:
                        slice_type = "depth"
                    else:
                        slice_type = "array"
                elif args.all:
                    # Extract all faces, array slices, or depth slices
                    if is_cubemap:
                        # Extract all 6 faces
                        slices_to_extract = list(range(6))
                        slice_type = "face"
                    elif is_volume:
                        # Extract all depth slices for the specified mipmap level
                        depth_count = max(1, dds.header.dwDepth >> args.mipmap)
                        slices_to_extract = list(range(depth_count))
                        slice_type = "depth"
                    else:
                        # Extract all array slices
                        array_size = dds.header10.arraySize if dds.header10 and dds.header10.arraySize > 0 else 1
                        slices_to_extract = list(range(array_size))
                        slice_type = "array"
                else:
                    # Extract single slice (default behavior)
                    if is_cubemap:
                        slices_to_extract = [args.face]
                        slice_type = "face"
                    elif is_volume:
                        slices_to_extract = [args.depth]
                        slice_type = "depth"
                    else:
                        slices_to_extract = [args.array_index]
                        slice_type = "array"

                # Split output filename into base and extension
                output_base, output_ext = os.path.splitext(args.output)

                total_decompress_time = 0
                total_save_time = 0

                for i, slice_idx in enumerate(slices_to_extract):
                    # Generate output filename
                    if len(slices_to_extract) > 1:
                        if slice_type == "face":
                            output_file = f"{output_base}_face{slice_idx}{output_ext}"
                        elif slice_type == "depth":
                            output_file = f"{output_base}_depth{slice_idx}{output_ext}"
                        else:  # array
                            output_file = f"{output_base}_array{slice_idx}{output_ext}"
                    else:
                        output_file = args.output

                    # Determine parameters for to_image
                    if slice_type == "face":
                        face_index = slice_idx
                        array_index = args.array_index
                        depth_index = args.depth
                    elif slice_type == "depth":
                        face_index = args.face
                        array_index = args.array_index
                        depth_index = slice_idx
                    else:  # array
                        face_index = args.face
                        array_index = slice_idx
                        depth_index = args.depth

                    print(f"\nConverting to image (mipmap level {args.mipmap}, array {array_index}, face {face_index}, depth {depth_index})...")

                    # Time decompression
                    start_decompress = time.perf_counter()
                    image_array = dds.to_image(mipmap_level=args.mipmap, array_index=array_index, face_index=face_index, depth_index=depth_index)
                    end_decompress = time.perf_counter()
                    decompress_time = end_decompress - start_decompress
                    total_decompress_time += decompress_time

                    # Time saving
                    start_save = time.perf_counter()
                    iio.imwrite(output_file, image_array)
                    end_save = time.perf_counter()
                    save_time = end_save - start_save
                    total_save_time += save_time

                    print(f"Saved to: {output_file}")
                    print(f"Image size: {image_array.shape[1]}x{image_array.shape[0]}")
                    print(f"Image format: {image_array.dtype} ({image_array.shape[2]} channels)")
                    print(f"Decompression time: {decompress_time*1000:.2f} ms")
                    print(f"Save time: {save_time*1000:.2f} ms")

                if len(slices_to_extract) > 1:
                    print(f"\nTotal images exported: {len(slices_to_extract)}")
                    print(f"Total decompression time: {total_decompress_time*1000:.2f} ms")
                    print(f"Total save time: {total_save_time*1000:.2f} ms")
                    print(f"Total conversion time: {(total_decompress_time + total_save_time)*1000:.2f} ms")

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


def parse_slice_spec(spec: str) -> list[int]:
    """
    Parse a slice specification string into a list of slice indices.

    Args:
        spec: Slice specification (e.g., "0,2,4" or "0-5")

    Returns:
        List of slice indices

    Examples:
        "0,2,4" -> [0, 2, 4]
        "0-5" -> [0, 1, 2, 3, 4, 5]
        "0,2-4,6" -> [0, 2, 3, 4, 6]
    """
    slices = []
    parts = spec.split(',')

    for part in parts:
        part = part.strip()
        if '-' in part:
            # Range specification
            start_str, end_str = part.split('-', 1)
            start = int(start_str.strip())
            end = int(end_str.strip())
            slices.extend(range(start, end + 1))
        else:
            # Single index
            slices.append(int(part))

    return slices


if __name__ == "__main__":
    main()
