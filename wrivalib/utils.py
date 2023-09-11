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

import datetime
import logging
from pathlib import Path
from typing import List, Tuple

import pandas as pd

logging.captureWarnings(True)
logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]


def print_when_is_now():
    """Print string with current date and time.
    Useful for tracking when plots where generated.
    """
    dt = datetime.datetime.now()
    tz = dt.astimezone().tzname()
    print("Produced on {}, {}".format(str(dt), tz))


def get_camera_collection_paths(
    camera_path: Path, sub_dir: str = "3_finished"
) -> list[Path]:
    """Get collection paths for given camera. Ignores any directories starting with an underscore (_).

    Args:
        camera_path (Path): Path to camera directory containing collections.
        sub_dir (str, optional): Processing step sub-directory. Defaults to "3_finished".

    Returns:
        list[Path]: A list of collection paths
    """
    collection_paths: list[Path] = []

    try:
        collection_paths = sorted(
            [
                path
                for path in (camera_path / sub_dir).iterdir()
                if path.is_dir() and not path.name.startswith("_")
            ]
        )
    except FileNotFoundError:
        # print(f"{sub_dir} directory not found for {camera_path.name}")
        pass

    return collection_paths


def get_collection_paths(
    site_path: Path,
    data_type: str = "real",
    delivered: bool = False,
) -> Tuple[
    List[Path], dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]]
]:
    """Get collection paths for a given site.

    Args:
        site_path (Path): Path to site directory containing cameras and collection directories.
        data_type (str): Type of data collection (real or synthetic). Defaults to "real".
        delivered (bool): Boolean representing delivered or un-delivered data. Defaults to False.

    Returns:
        Tuple[ List[Path], dict[str, list[Path]], dict[str, list[Path]], dict[str, list[Path]] ]:
        Camera paths, finished paths, pii-detected paths, and pii-removed paths
    """
    camera_paths = sorted([dir for dir in site_path.glob("cam*") if dir.is_dir()])

    finished_collection_paths: dict[str, list[Path]] = {}
    pii_detected_collection_paths: dict[str, list[Path]] = {}
    pii_removed_collection_paths: dict[str, list[Path]] = {}

    # Get paths to collections
    for camera_path in camera_paths:
        if data_type == "synthetic" or delivered:
            finished_collection_paths[camera_path.name] = get_camera_collection_paths(
                camera_path, ""
            )
            pii_detected_collection_paths[camera_path.name] = finished_collection_paths[
                camera_path.name
            ]
            pii_removed_collection_paths[camera_path.name] = finished_collection_paths[
                camera_path.name
            ]
        elif data_type == "real":
            finished_collection_paths[camera_path.name] = get_camera_collection_paths(
                camera_path, "3_finished"
            )
            pii_detected_collection_paths[
                camera_path.name
            ] = get_camera_collection_paths(camera_path, "4_pii_detected")
            pii_removed_collection_paths[
                camera_path.name
            ] = get_camera_collection_paths(camera_path, "5_pii_removed")

    return (
        camera_paths,
        finished_collection_paths,
        pii_detected_collection_paths,
        pii_removed_collection_paths,
    )


def get_delivery_paths(site_path: Path) -> Tuple[List[Path], dict[str, list[Path]]]:
    """Get collection paths for a delivery hierarchy (only pii-removed assets).

    Args:
        site_path (Path): Path to site directory containing cameras and collections.

    Returns:
        Tuple[List[Path], dict[str, list[Path]]]: Camera paths and collection paths
    """
    camera_paths = sorted([dir for dir in site_path.glob("cam*") if dir.is_dir()])

    delivery_collection_paths: dict[str, list[Path]] = {}

    # Get paths to collections
    for camera_path in camera_paths:
        delivery_collection_paths[camera_path.name] = get_camera_collection_paths(
            camera_path, ""
        )

    return (
        camera_paths,
        delivery_collection_paths,
    )


def count_images(collection_paths: dict[str, list[Path]]) -> pd.DataFrame:
    """Counts the number of images across all collections for a given camera.

    Args:
        collection_paths (dict[str, list[Path]]): Dictionary containing collection paths for camera

    Returns:
        pd.DataFrame: DataFrame containing image counts for each collection
    """
    collection_df = pd.DataFrame()

    for camera, camera_collections in collection_paths.items():
        for camera_collection in camera_collections:
            image_paths = [
                p
                for p in camera_collection.iterdir()
                if p.suffix.lower() in IMAGE_EXTENSIONS and "segmented" not in p.name
            ]

            collection_df = pd.concat(
                [
                    collection_df,
                    pd.DataFrame(
                        {
                            "camera": camera,
                            "collection": camera_collection.name,
                            "image_count": len(image_paths),
                        },
                        index=[0],
                    ),
                ],
                ignore_index=True,
            )

    return collection_df
