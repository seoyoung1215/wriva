"""
Copyright © 2022-2023 The Johns Hopkins University Applied Physics Laboratory LLC

Permission is hereby granted, free of charge, to any person obtaining a copy 
of this software and associated documentation files (the “Software”), to 
deal in the Software without restriction, including without limitation the 
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or 
sell copies of the Software, and to permit persons to whom the Software is 
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in 
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, 
WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR 
IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
from pathlib import Path

from wrivalib.metadata.read_write_model import read_model
from wrivalib.metadata.utils import (
    read_metadata,
    model_to_metadata,
    read_origin,
    write_metadata,
)


def colmap_to_wriva(root_dir, origin=None):
    """
    Converts COLMAP's bin files to WRIVA's image json files.
    Expects bin files in sparse/0/ of root directory.
    Expects and overwrites json files in images/ of root directory.

    :param root_dir: path to root directory
    :param origin: origin (lat, lon, alt) of ENU coordinate system, defaults to reading from sparse/0/origin.txt
    """
    # define directories
    sparse_0_dir = Path(root_dir) / "sparse" / "0"
    images_dir = Path(root_dir) / "images"

    # read COLMAP files
    model = read_model(sparse_0_dir, ".bin")

    # read WRIVA files
    metadata_dicts = read_metadata(images_dir)

    # convert metadata
    if not origin:
        origin = read_origin(sparse_0_dir)
    metadata_dicts = model_to_metadata(model, origin, metadata_dicts)

    # write files
    write_metadata(metadata_dicts, images_dir)


def main():
    # define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root_dir", type=str, required=True, help="root directory of dataset"
    )
    parser.add_argument(
        "--origin",
        type=float,
        nargs=3,
        help="origin (lat, lon, alt) of ENU coordinate system",
        default=None,
    )
    args = parser.parse_args()

    # convert metadata
    colmap_to_wriva(args.root_dir, args.origin)


if __name__ == "__main__":
    main()
