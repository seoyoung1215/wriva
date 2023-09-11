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
import glob
import json
import logging
import math
import os
import pickle
from copy import deepcopy
from typing import Tuple

import cv2
import numpy as np
from scipy.interpolate import griddata
from tqdm import tqdm


def project_intrinsics(
    xyz_camera: Tuple[float, float, float], intrinsics: dict, projection: str
) -> Tuple[int, int]:
    """
    project xyz camera coordinates to image coordinates using intrinsic parameters
    :param xyz_camera: xyz in camera coordinates, after applying extrinsic transform
    :param intrinsics: camera intrinsic parameters
    :param projection: camera projection model (e.g., 'fisheye_pix4d')
    :return column, row: projected image coordinates
    """
    if projection == "fisheye_pix4d":
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        p2 = intrinsics["p2"]
        p3 = intrinsics["p3"]
        p4 = intrinsics["p4"]
        C = intrinsics["C"]
        D = intrinsics["D"]
        E = intrinsics["E"]
        F = intrinsics["F"]
        theta = abs(
            (2.0 / math.pi)
            * math.atan(
                math.sqrt(xyz_camera[0] * xyz_camera[0] + xyz_camera[1] * xyz_camera[1])
                / xyz_camera[2]
            )
        )
        rho = (
            theta
            + p2 * theta * theta
            + p3 * theta * theta * theta
            + p4 * theta * theta * theta * theta
        )
        xh = (
            rho
            * xyz_camera[0]
            / math.sqrt(xyz_camera[0] * xyz_camera[0] + xyz_camera[1] * xyz_camera[1])
        )
        yh = (
            rho
            * xyz_camera[1]
            / math.sqrt(xyz_camera[0] * xyz_camera[0] + xyz_camera[1] * xyz_camera[1])
        )
        column = C * xh + D * yh + cx
        row = E * xh + F * yh + cy
    else:
        fx = intrinsics["fx"]
        fy = intrinsics["fy"]
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        k1 = intrinsics["k1"]
        k2 = intrinsics["k2"]
        k3 = intrinsics["k3"]
        p1 = intrinsics["p1"]
        p2 = intrinsics["p2"]
        xh = xyz_camera[0] / xyz_camera[2]
        yh = xyz_camera[1] / xyz_camera[2]
        r2 = xh * xh + yh * yh
        gamma = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        xhd = xh * gamma + 2.0 * p1 * xh * yh + p2 * (r2 + 2 * xh * xh)
        yhd = yh * gamma + p1 * (r2 + 2 * yh * yh) + 2.0 * p2 * xh * yh
        K = [[-fx, 0, cx], [0, -fy, cy], [0, 0, 1]]
        ijk = np.matmul(K, [xhd, yhd, 1.0])
        column = ijk[0] / ijk[2]
        row = ijk[1] / ijk[2]

    return column, row


def get_interp_map(
    input_intrinsics: dict, output_intrinsics: dict, input_proj: str, output_proj: str
) -> Tuple[np.ndarray, np.ndarray]:
    xmax = ymax = z = 1.0

    if input_proj == "fisheye_pix4d":
        xyz: Tuple[float, float, float] = (1.0, 1.0, z)
    else:
        xyz: Tuple[float, float, float] = (-1.0, -1.0, z)

    rows: int = input_intrinsics["rows"]
    columns: int = input_intrinsics["columns"]
    cx = columns / 2
    cy = rows / 2

    for _ in range(1000):
        col, row = project_intrinsics(xyz, input_intrinsics, input_proj)
        error = math.sqrt((columns - col) ** 2 + (rows - row) ** 2)
        if error < 0.5:
            break

        xmax = xyz[0] + (xyz[0] / col) * (columns - col)
        ymax = xyz[1] + (xyz[1] / row) * (rows - row)
        xyz = (xmax, ymax, 1.0)
        dx = xmax / cx
        dy = ymax / cy

    # logger.info(f"Refined xyz grid for # iterations, and error (pixels): {iter} {error}")

    # Add a buffer to xyz grid to avoid NaNs in the interpolation grid
    xmax = 1.1 * xmax
    ymax = 1.1 * ymax
    dx = xmax / cx
    dy = ymax / cy

    # Define the output image grid
    columns, rows = np.meshgrid(
        np.arange(0, input_intrinsics["columns"]),
        np.arange(0, input_intrinsics["rows"]),
    )

    # Project range of xyz values to pixel coordinates to fill in a grid
    # Project same xyz values using output intrinsics to get pixel shifts
    xrange = np.arange(-xmax, xmax, dx)
    yrange = np.arange(-ymax, ymax, dy)
    points = []
    xs = []
    ys = []

    for x in tqdm(xrange):
        for y in yrange:
            c1, r1 = project_intrinsics((x, y, z), input_intrinsics, input_proj)
            c2, r2 = project_intrinsics((x, y, z), output_intrinsics, output_proj)
            points.append((c1, r1))
            xs.append(c2)
            ys.append(r2)

    map_x = griddata(points, xs, (columns, rows), method="linear")
    map_y = griddata(points, ys, (columns, rows), method="linear")
    interp_map = (map_x, map_y)
    num_nans = np.sum(np.isnan(interp_map))

    if num_nans > 0:
        print(f"NaNs in interpolation map: {num_nans}")

    return interp_map


def get_undistortion_map(metadata: dict):
    # Get intrinsics from input metadata
    input_intrinsics = metadata["intrinsics"]

    # Get camera projection
    projection = metadata.get("projection", "perspective")

    # Copy input intrinsics to pinhole intrinsics variable
    pinhole_intrinsics = deepcopy(input_intrinsics)

    # Set pinhole distortion coefficients to 0
    pinhole_intrinsics["k1"] = 0.0
    pinhole_intrinsics["k2"] = 0.0
    pinhole_intrinsics["k3"] = 0.0
    pinhole_intrinsics["p1"] = 0.0
    pinhole_intrinsics["p2"] = 0.0

    # Reconfigure intrinsics for fisheye camera models
    if projection == "fisheye_pix4d":
        # Set focal length based on fisheye parameters
        pinhole_intrinsics["fx"] = -2.0 * metadata["intrinsics"]["C"] / math.pi
        pinhole_intrinsics["fy"] = -2.0 * metadata["intrinsics"]["F"] / math.pi

        # Pop extra fisheye intrinsic parameters from pinhole intrinsics
        pinhole_intrinsics.pop("p3")
        pinhole_intrinsics.pop("p4")
        pinhole_intrinsics.pop("C")
        pinhole_intrinsics.pop("D")
        pinhole_intrinsics.pop("E")
        pinhole_intrinsics.pop("F")

    # Get interpolation map from input image to perspective model
    src2dest = get_interp_map(
        input_intrinsics, pinhole_intrinsics, projection, "perspective"
    )

    # Get interpolation map from perspective model to input image
    dest2src = get_interp_map(
        pinhole_intrinsics, input_intrinsics, "perspective", projection
    )

    # Copy entire input metadata to pinhole metadata variable
    pinhole_metadata = deepcopy(metadata)

    # Re-assign intrinsics with pinhole intrinsics
    pinhole_metadata["intrinsics"] = pinhole_intrinsics

    return src2dest, dest2src, pinhole_metadata


def remap_image(input_image, interp_map) -> Tuple[np.ndarray, np.ndarray]:
    """
    remap pixels in image using an interpolation grid
    :param input_image: input image to be remapped
    :param interp_map: interpolation grid generated using camera parameters
    :return output_image, output_mask: remapped output image and mask indicating invalid pixels in the output image
    """
    # Create a copy of the input image and replace any black pixels with white
    filtered_image = np.copy(input_image)
    ind = np.where(filtered_image[:, :] == [0, 0, 0])
    filtered_image[ind[0], ind[1]] = [1, 1, 1]

    # Remap the filtered image using the interpolation map
    output_image = cv2.remap(
        filtered_image,
        np.float32(interp_map[0]),
        np.float32(interp_map[1]),
        cv2.INTER_LINEAR,
    )

    # Generate a mask indicating which pixels in the output image are invalid (i.e. black)
    output_mask = np.uint8(255 * (output_image != [0, 0, 0]))

    return output_image, output_mask


def undistort_truth_images(
    output_dir, metadata_dir, image_dir, interp_map_dir
) -> list[np.ndarray]:
    """
    Undistort ground truth images based on configuration dictionary containing
    information about the directories where images are located and where the
    undistorted images should be saved. The function creates a directory to store
    undistorted images if one does not already exist.

    Args:
        :param interp_map_dir: The directory that interpolation maps get cached to
        :param image_dir: The directory of images to be undistorted
        :param metadata_dir: The metadata directory of the images to be undistorted
        :param output_dir: The directory that the undistorted imagery will be dumped to
    """
    # Create a directory to store undistorted images if it does not already exist
    os.makedirs(output_dir, exist_ok=True)

    # Get paths of all json files in the ground truth directory and sort them
    gt_json_paths = sorted(glob.glob(os.path.join(metadata_dir, "*.json")))
    gt_metadata = []

    # Load metadata from each json file and append it to the gt_metadata list
    for path in gt_json_paths:
        with open(path) as f:
            gt_metadata.append(json.load(f))

    masks = []
    # Loop through each metadata and undistort its corresponding image
    for metadata in tqdm(gt_metadata, desc="Undistorted truth imagery"):
        # Get the file name and path of the ground truth image from the metadata
        gt_img_name = metadata["fname"]
        gt_img_path = os.path.join(image_dir, gt_img_name)

        # Get the interpolated map for undistorting the camera based on the sensor
        interp_map_path = os.path.join(interp_map_dir, f"{metadata['source']}.p")
        if os.path.exists(interp_map_path):
            # If the interpolated map already exists, load it from the cache
            with open(interp_map_path, "rb") as f:
                src2dest_map, dst2src_map, pinhole_params = pickle.load(f)
        else:
            # Otherwise, generate the interpolated map and save it to the cache
            os.makedirs(interp_map_dir, exist_ok=True)
            src2dest_map, dst2src_map, pinhole_params = get_undistortion_map(metadata)
            with open(interp_map_path, "wb+") as f:
                pickle.dump((src2dest_map, dst2src_map, pinhole_params), f)

        # Load the ground truth image and undistort it using the interpolated map
        gt_img = cv2.imread(gt_img_path)
        out_img, out_mask = remap_image(gt_img, dst2src_map)
        masks.append(out_mask)

        # Save the undistorted image to the undistorted directory
        cv2.imwrite(os.path.join(output_dir, gt_img_name), out_img)

    return masks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="The directory that the undistorted images get dumped into",
    )
    parser.add_argument(
        "--interp_map_dir",
        type=str,
        required=False,
        default=None,
        help="Optional; The directory where cached undistortion maps may go. Undistortion maps are cached on a sensor name basis",
    )
    parser.add_argument(
        "--metadata_dir",
        type=str,
        required=True,
        help="The directory where the image metadata is stored",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="The directory where the actual images to be undistorted are stored.",
    )
    args = parser.parse_args()

    _ = undistort_truth_images(
        output_dir=args.out_dir,
        metadata_dir=args.metadata_dir,
        image_dir=args.image_dir,
        interp_map_dir=args.interp_map_dir,
    )
