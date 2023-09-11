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


from pathlib import Path
import math
import json
import argparse

import numpy as np
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
import cv2
from tqdm import tqdm

from utils import read_metadata, compute_centroid
from read_write_model import qvec2rotmat

# TODO add support for fisheye cams

DEFAULT_AABB_SCALE = 16     # used by instant-ngp
CAMERA_MODEL = "OPENCV"     # used by nerf-studio
CALC_SHARPNESS = False   # slows down conversion, but used by instant-ngp (maybe not required?)

# adapted from instant-ngp/scripts/colmap2nerf.py
def variance_of_laplacian(image):
    return cv2.Laplacian(image, cv2.CV_64F).var()


# adapted from instant-ngp/scripts/colmap2nerf.py
def sharpness(imagePath):
    image = cv2.imread(str(imagePath))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm


def wriva_to_nerf_synth(root_dir, origin=None):
    """
    Converts WRIVA's image json files to NeRF-Synthetic transforms.json style
    Expects json files in images/ of root directory.
    Outputs a transforms.json file in the same root directory

    :param root_dir: path to root directory
    :param origin: origin (lat, lon, alt) of ENU coordinate system, defaults to centroid of camera positions
    """
    # define directories
    images_dir = Path(root_dir) #/ "images"

    # read WRIVA files
    metadata_dicts = read_metadata(images_dir)

    # convert metadata
    if not origin:
        origin = compute_centroid(metadata_dicts)
        print("Calculated origin:", f"lat: {origin[0]}, lon: {origin[1]}, alt: {origin[2]}")
    
    # save origin
    origin_path = Path(root_dir) / "colmap" / "sparse" / "0" / "origin.txt"
    origin_path.parent.mkdir(exist_ok=True, parents=True)
    with open(origin_path, 'w') as fp:
        fp.write(f"lat: {origin[0]}, lon: {origin[1]}, alt: {origin[2]}\n")

    transforms_dict_test = {'aabb_scale': DEFAULT_AABB_SCALE, 'frames': []}
    transforms_dict_train = {'aabb_scale': DEFAULT_AABB_SCALE, 'frames': []}
    for i, metadata_dict in enumerate(tqdm(metadata_dicts)):
        frame = {}
        frame['file_path'] = "images/" + metadata_dict["fname"]
        if CALC_SHARPNESS:
            frame['sharpness'] = sharpness(Path(root_dir) / frame['file_path'])

        # intrinsics
        frame['camera_model'] = CAMERA_MODEL
        frame['fl_x'] = metadata_dict['intrinsics']['fx']
        frame['fl_y'] = metadata_dict['intrinsics']['fy']
        frame['cx'] = metadata_dict['intrinsics']['cx']
        frame['cy'] = metadata_dict['intrinsics']['cy']
        frame['w'] = metadata_dict['intrinsics']['columns']
        frame['h'] = metadata_dict['intrinsics']['rows']
        frame['k1'] = metadata_dict['intrinsics']['k1']
        frame['k2'] = metadata_dict['intrinsics']['k2']
        frame['k3'] = metadata_dict['intrinsics']['k3']
        frame['p1'] = metadata_dict['intrinsics']['p1']
        frame['p2'] = metadata_dict['intrinsics']['p2']
        frame['camera_angle_x'] = math.atan(frame['w'] / (frame['fl_x'] * 2)) * 2
        frame['camera_angle_y'] = math.atan(frame['h'] / (frame['fl_y'] * 2)) * 2

        # extrinsics
        d = metadata_dict["extrinsics"]
        r = (
            R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            * R.from_euler(
                "zyx",
                [d["kappa"], d["phi"], d["omega"]],
                degrees=True,
            ).inv()
        )
        qvec = np.roll(r.as_quat(), 1)
        tvec = r.apply(-np.array(pm.geodetic2enu(d["lat"], d["lon"], d["alt"], *origin)))

        rotation = qvec2rotmat(qvec)
        translation = tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system to ours
        c2w[0:3, 1:3] *= -1
        c2w = c2w[np.array([1, 0, 2, 3]), :]
        c2w[2, :] *= -1
        frame['transform_matrix'] = c2w.tolist()
        if i%8==0:
            transforms_dict_test['frames'].append(frame)
        else:
            transforms_dict_train['frames'].append(frame)
    json.dump(transforms_dict_test, open(Path(root_dir) / "transforms_test.json", 'w'), indent=4)
    json.dump(transforms_dict_train, open(Path(root_dir) / "transforms_train.json", 'w'), indent=4)



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

    assert Path(args.root_dir).exists(), "--root_dir does not exist!"

    # convert metadata
    wriva_to_nerf_synth(args.root_dir, None)


if __name__ == "__main__":
    main()
