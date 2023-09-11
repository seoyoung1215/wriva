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
import json
import logging
from pathlib import Path
from typing import List, Tuple
from riviera.s3 import use_s3_profile
from s3path import S3Path

import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread
from tqdm import tqdm

from wrivalib.occlusion_check import DSM
from wrivalib.projections import geodetic_to_enu, lla_in_image_fov, lla_to_utm

logging.captureWarnings(True)
logger = logging.getLogger(__name__)

s3 = use_s3_profile("wriva-te")


def world_coordinate_search(
    lat: float,
    lon: float,
    alt: float,
    json_paths: list,
    use_norm_dist: bool = True,
    do_dsm_occlusion_check: bool = False,
    dsm_path: Path = None,
) -> Tuple[List, List]:
    """
    Finds and sorts by distance the images where the world coordinate is in the field of view

    :param lat: float representing latitude world coordinate
    :param lon: float representing longitude world coordinate
    :param alt: float representing altitude world coordinate
    :param json_paths: list of paths to json metadata files associated with images of interest
    :param use_norm_dist: Sort by normalized distance from image center (True) or point to camera distance (False)
    :return: sorted lists of json metadata files and associated distances where the world coordinate is in the FOV
    """
    if do_dsm_occlusion_check:
        # get scene point UTM
        x, y, z = lla_to_utm(lat, lon, alt)
        scene_point_utm = np.array([x, y, z])

        # load the DSM
        dsm = DSM(str(dsm_path))

    files = []
    distances = []

    # Iterate though all files in list of JSON paths
    for json_path in tqdm(json_paths, desc="World Coordinate Search Loop"):
        if isinstance(json_path, S3Path):
            params = json.loads(s3.read(json_path.as_uri()))
        else:
            with open(json_path) as f:
                # Load metadata
                params = json.load(f)

        # Check if world coordinate is inside image field of view as well as in front of camera
        inside, front, norm_dist = lla_in_image_fov(lat, lon, alt, params)

        # ignore if too close to image border because inaccuracy
        inside = inside and front and norm_dist < 0.8

        if inside:
            if do_dsm_occlusion_check:
                assert (
                    params["site"] == "siteA01"
                ), "Only siteA01 supported for DSM occlusion check, at the moment"

                try:
                    # convert camera coordinate to UTM
                    tlat = params["extrinsics"]["lat"]
                    tlon = params["extrinsics"]["lon"]
                    talt = params["extrinsics"]["alt"]
                    x, y, z = lla_to_utm(tlat, tlon, talt)
                    camera_point_utm = np.array([x, y, z])
                    # check for occlusion
                    if not dsm.is_visible(
                        camera_point_utm,
                        scene_point_utm,
                        camera_type=params["type"],
                    ):
                        # if occluded, move to next image
                        continue
                except Exception as e:
                    print("ERROR: Occlusion check failure for", str(json_path), "-", e)
                    continue

            files.append(json_path)
            if use_norm_dist:
                distances.append(norm_dist)
            else:
                dist = np.linalg.norm(
                    geodetic_to_enu(
                        params["extrinsics"]["lat"],
                        params["extrinsics"]["lon"],
                        params["extrinsics"]["alt"],
                        lat,
                        lon,
                        alt,
                    )
                )
                distances.append(dist)

    sort_index = np.argsort(np.array(distances))

    return [files[ind] for ind in sort_index], [distances[ind] for ind in sort_index]


def multi_world_coordinate_search(
    world_coordinates: list,
    json_paths: list,
    use_norm_dist: bool = True,
    do_dsm_occlusion_check: bool = False,
    dsm_path: Path = None,
) -> Tuple[List, List]:
    """
    Finds and sorts by distance the images where a world coordinate in a list of world coordinates is in the field of view

    :param world_coordinates: list of world coordinate lists, which are each made up latitude, longitude, and altitude
    :param json_paths: list of paths to json metadata files associated with images of interest
    :param use_norm_dist: Sort by normalized distance from image center (True) or point to camera distance (False)
    :return: sorted lists of json metadata files and associated distances where the world coordinate is in the FOV
    """

    results_dict = {}
    for world_coordinate in tqdm(
        world_coordinates, desc="Multiple World Coordinate Search Loop"
    ):
        in_view_json_paths, distances_to_world_coordinate = world_coordinate_search(
            lat=world_coordinate[0],
            lon=world_coordinate[1],
            alt=world_coordinate[2],
            json_paths=json_paths,
            use_norm_dist=use_norm_dist,
            do_dsm_occlusion_check=do_dsm_occlusion_check,
            dsm_path=dsm_path,
        )

        logger.info(
            f"Found {len(in_view_json_paths)} images with world coordinate in view: {world_coordinate}"
        )

        for i in range(len(in_view_json_paths)):
            # update distance if this one is smaller, so we keep the distance to nearest world coordinate
            if (
                in_view_json_paths[i] in results_dict
                and distances_to_world_coordinate[i]
                >= results_dict[in_view_json_paths[i]]
            ):
                continue
            else:
                results_dict[in_view_json_paths[i]] = distances_to_world_coordinate[i]

    final_in_view_json_paths, final_distances_to_world_coordinate = zip(
        *results_dict.items()
    )

    final_sorted_distances_to_world_coordinate, final_sorted_in_view_json_paths = zip(
        *sorted(zip(final_distances_to_world_coordinate, final_in_view_json_paths))
    )

    return list(final_sorted_in_view_json_paths), list(
        final_sorted_distances_to_world_coordinate
    )


if __name__ == "__main__":
    # Configure logger
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    handler.setStream(tqdm)
    handler.terminator = ""

    logging.basicConfig(level=logging.INFO, handlers=[handler])

    # Initialize argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lla",
        required=True,
        help="Tuple of (lat, lon, alt) to search",
        nargs=3,
        type=float,
    )
    parser.add_argument(
        "--dir", required=True, help="Image directory to search", type=str
    )
    args = parser.parse_args()

    # Set search variables based on arguments
    lat, lon, alt = args.lla
    images_dir = Path(args.dir)

    # Get list of JSON paths found in directory
    json_paths = sorted(list(images_dir.rglob("*.json")))

    # Find metadata files where coordinate is within FOV of image sorted by normalized distance
    files, distances = world_coordinate_search(lat, lon, alt, json_paths, True)
    print(*files, sep="\n")
    print(*distances, sep="\n")

    # Create figure and show five closest images by normalized distance
    plt.figure(figsize=[19.4, 9.6])
    for i in range(5):
        plt.subplot(2, 5, i + 1)
        plt.imshow(imread(files[i].with_suffix(".jpg")))
        plt.title(f"Distance: {distances[i]:.4f}")

    # Find metadata files where coordinate is within FOV of image sorted by camera to point distance
    files, distances = world_coordinate_search(lat, lon, alt, json_paths, False)
    print(*files, sep="\n")
    print(*distances, sep="\n")

    print(
        "{} out of {} images found with lat, lon, alt coordinates {}, {}, {}.".format(
            len(files), len(json_paths), lat, lon, alt
        )
    )

    # Show five closest images by point distance
    for i in range(5):
        plt.subplot(2, 5, i + 6)
        plt.imshow(imread(files[i].with_suffix(".jpg")))
        plt.title(f"Distance: {distances[i]:.4f}")

    plt.show()
