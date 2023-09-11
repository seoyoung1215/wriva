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

import json
import math
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyproj
from scipy.spatial.transform import Rotation as R
from shapely.geometry import Polygon
from skimage.io import imread
from sklearn.neighbors import KDTree
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm

from wrivalib.projections import (
    enu_opk_to_utm_opk,
    lla_to_utm,
    opk_to_rotation,
    rotation_to_opk,
    utm_epsg_from_wgs84,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg"}


def load_dataset(dataset_dirs, subdir_splits=None):
    """
    For a given dataset spec, loads .json metadata all at once
    :param dataset_dirs: list of Paths to directories with image/json pairs, or train/test sub-directories
    :param subdir_splits: (Optional) list of sub-directories to search for each dataset directory (for train and test sets, set to ['train', 'test'])
    :return: dictionary where keys are image paths, and values are metadata loaded from .json files
    """
    image_dict_per_split = {}
    splits = [""] if subdir_splits is None else subdir_splits
    for split in splits:
        im_path_list = []
        for dataset_dir in dataset_dirs:
            im_path_list += sorted(
                [
                    path
                    for path in (dataset_dir / split).rglob("*")
                    if path.suffix in IMAGE_EXTS
                ]
            )

        split_dict = {}
        for im_path in tqdm(im_path_list, desc=f"{split} Image Loop"):
            json_data = json.load(open(im_path.parent / f"{im_path.stem}.json"))
            split_dict[str(im_path)] = json_data
        image_dict_per_split[split] = split_dict

    if subdir_splits is None:
        return image_dict_per_split[""]
    else:
        return image_dict_per_split


def get_quaternion_from_euler(omega, phi, kappa):
    """
    Convert an Euler angle to a quaternion.
    :param omega, phi, kappa: angle of rotation in degrees
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
    """
    r = (
        R.from_matrix([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
        * R.from_euler(
            "zyx",
            [kappa, phi, omega],
            degrees=True,
        ).inv()
    )
    quat = r.as_quat()
    return quat


def my_get_utm_convergence_matrix(lat, lon, alt, transformer):
    """
    compute convergence matrix from true North for UTM projection to WGS84
    :params lat,lon,alt: input WGS84 coordinates
    :return: convergence matrix to convert UTM grid north to true north angle (degrees)
    """
    delta = 1e-6
    p1 = np.array(transformer.transform(lat + delta, lon, alt, radians=False))
    p2 = np.array(transformer.transform(lat, lon, alt))
    xnp = p1 - p2
    angle = math.atan2(xnp[0], xnp[1])
    R_c = np.array(
        [
            [math.cos(angle), -math.sin(angle), 0],
            [math.sin(angle), math.cos(angle), 0],
            [0, 0, 1],
        ]
    )
    return R_c


def my_enu_opk_to_utm_opk(lat, lon, alt, omega_enu, phi_enu, kappa_enu, transformer):
    """
    convert ENU OPK angles to UTM OPK angles
    :params lat, lon, alt: camera location in WGS84 coordinates
    :params omega_enu, phi_enu, kappa_enu: camera OPK angles in ENU
    :return omega_utm, phi_utm, kappa_utm: camera OPK angles in UTM projection
    """
    # build rotation matrix from OPK angles
    R = opk_to_rotation([omega_enu, phi_enu, kappa_enu])
    # get UTM convergence angle rotation matrix and map UTM OPK to ENU
    R_c = my_get_utm_convergence_matrix(lat, lon, alt, transformer)
    R = np.dot(np.linalg.inv(R_c), R)
    omega_utm, phi_utm, kappa_utm = rotation_to_opk(R)
    return omega_utm, phi_utm, kappa_utm


def get_extrinsics_matrix(lat, lon, alt, omega, phi, kappa, transformer):
    """
    helper function to return the extrinsics matrix from extrinsic parameters
    :param lat, lon, alt, omega, phi, kappa: extrinsic parameters
    :return: 4x4 extrinsics matrix
    """
    # using pre-loaded transform to save compute, should be ok?
    # x, y, z = lla_to_utm(lat, lon, alt)
    x, y, z = transformer.transform(lon, lat, alt, radians=False)

    omega, phi, kappa = my_enu_opk_to_utm_opk(
        lat, lon, alt, omega, phi, kappa, transformer
    )

    R = opk_to_rotation([omega, phi, kappa])
    T = np.array([x, y, z])
    return np.vstack((np.hstack((R, np.expand_dims(T, axis=1))), [0, 0, 0, 1]))


def get_intrinsics_matrix(fx, fy, cx, cy, columns, rows):
    """
    helper function to return the intrinsics matrix from intrinsic parameters
    :param x, fy, cx, cy, columns, rows: intrinsic parameters
    :return: 3x4 intrinsics matrix
    """
    return np.asarray(
        # [[fx / columns, 0, cx / columns, 0], [0, fy / rows, cy / rows, 0], [0, 0, 1, 0]]
        [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0]]
    )


def IOU(pol1_xy, pol2_xy):
    """
    calculates intersection-over-union between two convex polygons
    :param pol1_xy, pol2_xy: lists of xy points that define the polygon, in order of connection
    :return: intersection-over-union score
    """
    # Define each polygon
    polygon1_shape = Polygon(pol1_xy)
    polygon2_shape = Polygon(pol2_xy)

    try:
        # Calculate intersection and union, and the IOU
        polygon_intersection = polygon1_shape.intersection(polygon2_shape).area
        polygon_union = polygon1_shape.area + polygon2_shape.area - polygon_intersection
        return polygon_intersection / polygon_union
    except Exception as e:
        # fig, ax = plt.subplots()
        # plt.plot(*polygon1_shape.exterior.xy)
        # plt.show()

        # fig, ax = plt.subplots()
        # plt.plot(*polygon2_shape.exterior.xy)
        # plt.show()

        # print(e)
        # TODO fix this, apparently when comparing landscape image with portrait image, input coords are out of order (might need to rotate projection matrix?)
        return -999


def find_nearest_images(ref_im_path, images_dict, mode, k=5) -> Tuple[List, List]:
    """
    find nearest image in images_dict, in relation to a reference image
    :param ref_im_path: path to reference image
    :param images_dict: dictionary where keys are image paths, and values are metadata loaded from .json files
    :param mode: method of comparing images to reference image, i.e. what distance/similarity metric to use
    :param k: number of nearest images and distances to return
    :return: nearest k image paths, and their associated distances from the reference image
    """
    ref_json_data = json.load(
        open(Path(ref_im_path).parent / f"{Path(ref_im_path).stem}.json")
    )
    im_paths = sorted(images_dict.keys())

    # TODO keep in mind, 32618 might not be the code for every site?
    transformer = pyproj.Transformer.from_crs(4326, 32618, always_xy=True)

    if mode == "corners_projection_iou":
        ref_E = get_extrinsics_matrix(
            ref_json_data["extrinsics"]["lat"],
            ref_json_data["extrinsics"]["lon"],
            ref_json_data["extrinsics"]["alt"],
            ref_json_data["extrinsics"]["omega"],
            ref_json_data["extrinsics"]["phi"],
            ref_json_data["extrinsics"]["kappa"],
            transformer,
        )
        ref_I = get_intrinsics_matrix(
            ref_json_data["intrinsics"]["fx"],
            ref_json_data["intrinsics"]["fy"],
            ref_json_data["intrinsics"]["cx"],
            ref_json_data["intrinsics"]["cy"],
            ref_json_data["intrinsics"]["columns"],
            ref_json_data["intrinsics"]["rows"],
        )

        ref_corners = [
            [0, 0, 1],
            [ref_json_data["intrinsics"]["columns"], 0, 1],
            [
                ref_json_data["intrinsics"]["columns"],
                ref_json_data["intrinsics"]["rows"],
                1,
            ],
            [0, ref_json_data["intrinsics"]["rows"], 1],
        ]

        image_corners_list = []
        ref_corners_in_image_list = []
        for im_path in im_paths:
            if im_path == ref_im_path:
                # for debugging
                a = 0
            image_json_data = images_dict[im_path]

            E = get_extrinsics_matrix(
                image_json_data["extrinsics"]["lat"],
                image_json_data["extrinsics"]["lon"],
                image_json_data["extrinsics"]["alt"],
                image_json_data["extrinsics"]["omega"],
                image_json_data["extrinsics"]["phi"],
                image_json_data["extrinsics"]["kappa"],
                transformer,
            )
            I = get_intrinsics_matrix(
                image_json_data["intrinsics"]["fx"],
                image_json_data["intrinsics"]["fy"],
                image_json_data["intrinsics"]["cx"],
                image_json_data["intrinsics"]["cy"],
                image_json_data["intrinsics"]["columns"],
                image_json_data["intrinsics"]["rows"],
            )

            # create matrix that projects points from ref image into other images
            M = ref_I @ ref_E @ np.linalg.inv(E) @ np.linalg.pinv(I)
            # M = I @ E @ np.linalg.inv(ref_E) @ np.linalg.pinv(ref_I)

            image_corners = [
                [0, 0],
                [image_json_data["intrinsics"]["columns"], 0],
                [
                    image_json_data["intrinsics"]["columns"],
                    image_json_data["intrinsics"]["rows"],
                ],
                [0, image_json_data["intrinsics"]["rows"]],
            ]
            image_corners_list.append(image_corners)

            ref_corners_in_image = []
            for corner in ref_corners:
                coord = M @ np.asarray(corner)
                ref_corners_in_image.append([coord[0] / coord[2], coord[1] / coord[2]])
            ref_corners_in_image_list.append(ref_corners_in_image)

        ious = [
            IOU(image_corners_list[i], ref_corners_in_image_list[i])
            for i in range(len(image_corners_list))
        ]
        print(f"{len([x for x in ious if x == -999])} invalid ious out of {len(ious)}")

        sorted_indices = np.argsort(ious)[::-1][:k]
        return (
            [im_paths[index] for index in sorted_indices],
            [ious[index] for index in sorted_indices],
            [ref_corners_in_image_list[index] for index in sorted_indices],
        )
    elif mode == "extrinsics_and_intrinsics_product_feats":
        ref_E = get_extrinsics_matrix(
            ref_json_data["extrinsics"]["lat"],
            ref_json_data["extrinsics"]["lon"],
            ref_json_data["extrinsics"]["alt"],
            ref_json_data["extrinsics"]["omega"],
            ref_json_data["extrinsics"]["phi"],
            ref_json_data["extrinsics"]["kappa"],
            transformer,
        )
        ref_I = get_intrinsics_matrix(
            ref_json_data["intrinsics"]["fx"],
            ref_json_data["intrinsics"]["fy"],
            ref_json_data["intrinsics"]["cx"],
            ref_json_data["intrinsics"]["cy"],
            ref_json_data["intrinsics"]["columns"],
            ref_json_data["intrinsics"]["rows"],
        )

        feats = []
        for im_path in im_paths:
            image_json_data = images_dict[im_path]

            E = get_extrinsics_matrix(
                image_json_data["extrinsics"]["lat"],
                image_json_data["extrinsics"]["lon"],
                image_json_data["extrinsics"]["alt"],
                image_json_data["extrinsics"]["omega"],
                image_json_data["extrinsics"]["phi"],
                image_json_data["extrinsics"]["kappa"],
                transformer,
            )
            I = get_intrinsics_matrix(
                image_json_data["intrinsics"]["fx"],
                image_json_data["intrinsics"]["fy"],
                image_json_data["intrinsics"]["cx"],
                image_json_data["intrinsics"]["cy"],
                image_json_data["intrinsics"]["columns"],
                image_json_data["intrinsics"]["rows"],
            )

            feats.append((I @ E).flatten())

        scalar = MinMaxScaler(feature_range=(-1, 1))
        feats = scalar.fit_transform(np.asarray(feats))
        tree = KDTree(feats)

        ref_feat = (ref_I @ ref_E).flatten()
        scaled_ref_feat = scalar.transform(np.expand_dims(ref_feat, axis=0))[0]
        distances, indices = tree.query(np.expand_dims(scaled_ref_feat, axis=0), k)

        return [im_paths[index] for index in indices[0]], distances[0].tolist()
    elif mode == "extrinsics_pos2D_only":
        params = ["lat", "lon"]
    elif mode == "extrinsics_pos3D_only":
        params = ["lat", "lon", "alt"]
    elif mode == "extrinsics":
        params = ["lat", "lon", "alt", "omega", "phi", "kappa"]
    else:
        print(f"Invalid mode, {mode}, given")
        return -1

    ref_ext = np.asarray([ref_json_data["extrinsics"][p] for p in params])
    image_ext_arr = np.asarray(
        [
            [images_dict[train_im_path]["extrinsics"][p] for p in params]
            for train_im_path in im_paths
        ]
    )

    # convert to utm so lat and lon units are same as alt
    if ("lat" in params) and ("lon" in params) and ("alt" in params):
        x, y, z = transformer.transform(
            ref_ext[1], ref_ext[0], ref_ext[2], radians=False
        )
        ref_ext = np.concatenate((np.asarray([x, y, z]), ref_ext[3:]))
        new_image_ext = []
        for i in range(len(image_ext_arr)):
            x, y, z = transformer.transform(
                image_ext_arr[i][1],
                image_ext_arr[i][0],
                image_ext_arr[i][2],
                radians=False,
            )
            new_image_ext.append(
                np.concatenate((np.asarray([x, y, z]), image_ext_arr[i][3:]))
            )
        image_ext_arr = np.asarray(new_image_ext)

    # convert euler angles to quaternions
    if ("omega" in params) and ("phi" in params) and ("kappa" in params):
        ref_ext = np.concatenate(
            (
                ref_ext[:3],
                np.asarray(
                    get_quaternion_from_euler(ref_ext[3], ref_ext[4], ref_ext[5])
                ),
            )
        )
        new_image_ext = []
        for i in range(len(image_ext_arr)):
            new_image_ext.append(
                np.concatenate(
                    (
                        image_ext_arr[i][:3],
                        np.asarray(
                            get_quaternion_from_euler(
                                image_ext_arr[i][3],
                                image_ext_arr[i][4],
                                image_ext_arr[i][5],
                            )
                        ),
                    )
                )
            )
        image_ext_arr = np.asarray(new_image_ext)

        scalar = MinMaxScaler(feature_range=(-1, 1))
        scaled_image_ext_arr = np.concatenate(
            (scalar.fit_transform(image_ext_arr[:, :3]), image_ext_arr[:, 3:]), axis=1
        )
        scaled_ref_ext = np.concatenate(
            (scalar.transform(np.expand_dims(ref_ext[:3], axis=0))[0], ref_ext[3:])
        )
    else:
        scalar = MinMaxScaler()
        scaled_image_ext_arr = scalar.fit_transform(image_ext_arr)
        scaled_ref_ext = scalar.transform(np.expand_dims(ref_ext, axis=0))[0]

    tree = KDTree(scaled_image_ext_arr)
    distances, indices = tree.query(np.expand_dims(scaled_ref_ext, axis=0), k)

    return [im_paths[index] for index in indices[0]], distances[0].tolist()
