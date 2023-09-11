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
import math
import os
import sys

import numpy as np
import pyproj


def eulerAnglesToRotationMatrix(theta: list):
    """
    Helper function that calculates a rotation matrix given euler angles x,y,z (roll, pitch, yaw)
    :param theta: list of xyz angles
    :return: returns the rotation matrix Rx*Ry*Rz
    """
    R_x = np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta[0]), -math.sin(theta[0])],
            [0, math.sin(theta[0]), math.cos(theta[0])],
        ]
    )

    R_y = np.array(
        [
            [math.cos(theta[1]), 0, math.sin(theta[1])],
            [0, 1, 0],
            [-math.sin(theta[1]), 0, math.cos(theta[1])],
        ]
    )

    R_z = np.array(
        [
            [math.cos(theta[2]), -math.sin(theta[2]), 0],
            [math.sin(theta[2]), math.cos(theta[2]), 0],
            [0, 0, 1],
        ]
    )
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R


def extrinsic_matrix(pos: list, ypr: list):
    """
    Function that creates the extrinsic matrix given xyz positional arguments and yaw, pitch , roll
    angular arguments
    :param pos: list of xyz coordinates
    :param ypr: list of yaw, pitch, roll arguments
    :return: returns the 4x4 extrinsic matrix

    """
    # eulerAnglesToRotationMatrix expects xyz and the angles list parameter is in yaw, pitch, roll which is zyx
    angles = [ypr[2], ypr[1], ypr[0]]
    rot = eulerAnglesToRotationMatrix(angles)
    return [
        [[rot[0][0]], [rot[0][1]], [rot[0][2]], [pos[0]]],
        [[rot[1][0]], [rot[1][1]], [rot[1][2]], [pos[1]]],
        [[rot[2][0]], [rot[2][1]], [rot[2][2]], [pos[2]]],
        [[0], [0], [0], [1]],
    ]


def wgs84_to_ecef(lat, lon, hae):
    """
    Helper function that converts between wgs84 (latitude, longitude, height above ellipsoid) to ECEF

    :param lat: latitude in degrees (lat = 37.4001100556)
    :param lon: longitude in degrees (lon = -79.1539111111)
    :param hae: height above ellipsoid (hae = 208.38)
    :return: 3-tuple of ECEF coordinates

    """
    transformer = pyproj.Transformer.from_crs(
        {"proj": "latlong", "ellps": "WGS84", "datum": "WGS84"},
        {"proj": "geocent", "ellps": "WGS84", "datum": "WGS84"},
    )
    x, y, z = transformer.transform(lon, lat, hae, radians=False)
    return (x, y, z)


def main() -> None:
    # represent sites.json as python dictionary
    site_json_path = args.sites
    if not os.path.exists(site_json_path):
        sys.exit("sites file path doesnt exist")
    sites_file = open(site_json_path)
    sites = json.load(sites_file)

    # represent collection.json as python dictionary
    json_path = args.collection
    if not os.path.exists(json_path):
        sys.exit("file path doesnt exist")
    f = open(json_path)
    collection_dict = json.load(f)
    sensors = collection_dict["sensors"]

    # create transform matrix and add to frame dictionary using lla from sites.json
    site_id = collection_dict["site_id"]
    for sensor in sensors:
        for f in sensor["frames"]:
            try:
                pos = sites[site_id]["locations"][f["location"]]
                pos = wgs84_to_ecef(pos[0], pos[1], pos[2])
                ypr = f["ypr"]
                f["transform_matrix"] = extrinsic_matrix(pos, ypr)
            except KeyError as e:
                print(e)

    # write the newly made dictionary with a transform matrix in each "frame" to a file
    new_json = args.saved
    with open(new_json, "w+") as outfile:
        json.dump(collection_dict, outfile, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""
    This is the detect faces sample using python language
    """
    )
    parser.add_argument("--sites", required=True, help="Path to the sites json")
    parser.add_argument(
        "--collection", required=True, help="Path to the collection json"
    )
    parser.add_argument(
        "--saved", required=True, help="Path and name of the new saved json"
    )
    args = parser.parse_args()
    main()
