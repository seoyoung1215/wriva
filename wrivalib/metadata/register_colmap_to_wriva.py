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

from wrivalib.metadata.read_write_model import read_model, write_model
from wrivalib.metadata.utils import (
    compute_centroid,
    metadata_to_model,
    read_metadata,
    register_colmap_models,
    write_origin,
)


def register_colmap_to_wriva(root_dir, origin=None):
    """
    Registers the coordinate system of COLMAP's bin files to that of WRIVA's image json files.
    Expects and overwrites bin files in sparse/0/ of root directory.
    Expects json files in images/ of root directory.
    Also stores origin.txt in sparse/0/.

    :param root_dir: path to root directory
    :param origin: origin (lat, lon, alt) of ENU coordinate system, defaults to centroid of camera positions
    """
    # define directories
    sparse_0_dir = Path(root_dir) / "sparse" / "0"
    images_dir = Path(root_dir) / "images"

    # read WRIVA files
    metadata_dicts = read_metadata(images_dir)

    # convert metadata
    if not origin:
        origin = compute_centroid(metadata_dicts)
    model_m = metadata_to_model(metadata_dicts, origin)

    # read COLMAP files
    model_c = read_model(sparse_0_dir, ".bin")

    # register COLMAP models
    model_t = register_colmap_models(model_c, model_m)

    # write files
    write_model(*model_t, sparse_0_dir, ".bin")
    write_origin(origin, sparse_0_dir)


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
    register_colmap_to_wriva(args.root_dir, args.origin)


if __name__ == "__main__":
    main()
