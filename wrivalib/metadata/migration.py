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

import copy
import json
import logging
from pathlib import Path

import click
import numpy as np
from tqdm import tqdm
from pycocotools import mask as mask_utils
from wrivalib.utils import get_collection_paths

logging.captureWarnings(True)
logger = logging.getLogger("wrivalib.metadata.migration")


def migrate_collection_metadata(site_path: str, data_type: str, conversion_function):
    # Get collection paths in site directory
    (
        camera_paths,
        finished_collection_paths,
        pii_detected_collection_paths,
        pii_removed_collection_paths,
    ) = get_collection_paths(site_path=Path(site_path), data_type=data_type)

    # Loop through all collection dirs in the site
    tqdm_collections = tqdm(
        sorted(
            [x for v in finished_collection_paths.values() for x in v]
            + [x for v in pii_detected_collection_paths.values() for x in v]
            + [x for v in pii_removed_collection_paths.values() for x in v]
        )
    )

    for collection_dir in tqdm_collections:
        # Call the callback function
        conversion_function(collection_dir)


def v4_0__to__v4_1(collection_dir: Path):
    # Glob and sort all files that end in .json
    json_files = [f for f in sorted(Path(collection_dir).glob("*.json")) if f.is_file()]

    # Loop through the JSON files
    tqdm_jsons = tqdm(json_files, desc=f"{'/'.join(collection_dir.parts[-3:])}")

    for json_file in tqdm_jsons:
        # Open and read the JSON file
        with open(json_file, "r") as f:
            old_json_data = json.load(f)

        # If the json file has a 'version' key and the value is '4.0'
        if "version" in old_json_data and (
            isinstance(old_json_data["version"], type(None))
            or (
                isinstance(old_json_data["version"], str)
                and old_json_data["version"].startswith("4.0")
            )
        ):
            # Defaults
            json__4_0__defaults = {
                "exterior": True,
                "interior": False,
                "transient_occlusions": {
                    "people": "",
                    "vehicle": "",
                    "information_overlay": "",
                },
                "artifacts": [],
                "pii_detected": [],
                "pii_removed": [],
            }
            # Copy the defaults if do not exist
            for k in json__4_0__defaults:
                if k not in old_json_data:
                    old_json_data[k] = json__4_0__defaults[k]

            # Maintain order of keys and insert new collection field after source
            json_data = {
                "version": old_json_data.pop("version"),
                "fname": old_json_data.pop("fname"),
                "site": old_json_data.pop("site"),
                "source": old_json_data.pop("source"),
                "collection": "-".join(json_file.name.split("-")[2:-1]),
                **old_json_data,
            }

            # Change the value of the 'version' key to '4.1.0'
            json_data["version"] = "4.1.0"

            # Rename the transient occlusions as a new key 'masks'
            json_data["masks"] = copy.copy(json_data["transient_occlusions"])

            # Get the zero detection RLE string
            zero_detection_rle = mask_utils.encode(
                np.asarray(
                    np.zeros(
                        (
                            json_data["intrinsics"]["columns"],
                            json_data["intrinsics"]["rows"],
                        ),
                        dtype=np.uint8,
                    ),
                    order="F",
                )
            )["counts"].decode("utf-8")

            # Loop through the masks
            for mask_key in json_data["transient_occlusions"]:
                # If the mask key is vehicle, rename to vehicles and delete old one
                if mask_key == "vehicle":
                    json_data["masks"]["vehicles"] = json_data["masks"][mask_key]
                    del json_data["masks"][mask_key]
                    mask_key = "vehicles"

                # If the mask key within the dict is empty, remove it from new dict
                if isinstance(json_data["masks"][mask_key], str) and (
                    len(json_data["masks"][mask_key]) == 0
                    or json_data["masks"][mask_key] == zero_detection_rle
                ):
                    del json_data["masks"][mask_key]

            # Get the current keys
            current_keys = list(json_data["masks"].keys())

            # Rename mask key to rle
            for ck in current_keys:
                rle = copy.copy(json_data["masks"][ck])
                json_data["masks"][ck] = {
                    "rle": rle,
                    "method": "automated",
                }

            # Set transient occlusions to the current keys list
            json_data["transient_occlusions"] = current_keys

            # Write the new json data to the file
            with open(json_file, "w") as f:
                json.dump(json_data, f, indent=4)

        else:
            continue


@click.command()
@click.option("--site_path", type=str, required=True, help="Path to the site directory")
@click.option(
    "--data_type", type=str, default="real", help="Path to the site directory"
)
def main(site_path: str, data_type: str):
    logger.info(f"Converting metadata from v4.0 to v4.1...")
    migrate_collection_metadata(site_path, data_type, v4_0__to__v4_1)


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
    main()
