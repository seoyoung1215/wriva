import json
import logging
import math
import os
import shutil
from glob import glob
from pathlib import Path

import numpy as np
from osgeo import gdal, osr
from tqdm import tqdm

from wrivalib.projections import (
    lla_in_image_fov,
    lla_to_utm,
    nad83_to_wgs84,
    utm_to_lla,
)

gdal.UseExceptions()
logging.captureWarnings(True)
logger = logging.getLogger(__name__)


class DSM:
    def __init__(self, dsm_path, convert_nad83=True):
        """
        initialize class and read reference DSM
        :param dsm_path: input DSM full path name
        """
        dataset = gdal.Open(dsm_path, gdal.GA_ReadOnly)
        self.dsm = dataset.ReadAsArray()
        self.rows = self.dsm.shape[0]
        self.columns = self.dsm.shape[1]
        projection = dataset.GetProjection()
        self.epsg = int(
            osr.SpatialReference(wkt=projection).GetAttrValue("AUTHORITY", 1)
        )
        metadata = dataset.GetMetadata()
        xform = dataset.GetGeoTransform()
        dataset = None
        xscale = xform[1]
        yscale = -xform[5]
        self.gsd = (xscale + yscale) / 2.0
        self.easting = xform[0]
        self.northing = xform[3] - self.rows * yscale
        dataset = None
        # convert NAD83 to WGS84
        if convert_nad83:
            x1, y1, z1 = self.ortho_to_xyz(int(self.columns / 2), int(self.rows / 2))
            lon, lat, alt = utm_to_lla(x1, y1, z1, self.epsg)
            lat, lon, alt = nad83_to_wgs84(lat, lon, alt)
            x2, y2, z2 = lla_to_utm(lat, lon, alt)
            self.easting = self.easting + x2 - x1
            self.northing = self.northing + y2 - y1
            self.dsm = self.dsm + z2 - z1
            logger.info(f"NAD83 to WGS84 offset = {z2 - z1}")

    def ortho_to_xyz(self, column, row):
        """
        compute UTM world coordinates for input image coordinate
        :params column, row: input image coordinate
        :return x,y,z: output UTM coordinate
        """
        x = column * self.gsd + self.easting
        y = self.northing + (self.rows - row - 1) * self.gsd
        z = self.dsm[row, column]
        return x, y, z

    def xy_to_ortho(self, x, y):
        """
        compute image coordinates for input UTM xyz coordinate
        :params x,y: input UTM coordinate
        :return column,row: output image coordinate
        """
        column = int((x - self.easting) / self.gsd + 0.5)
        row = self.rows - int((y - self.northing) / self.gsd + 0.5) - 1
        return column, row

    def is_visible(
        self,
        camera_point_utm,
        scene_point_utm,
        keep_if_outside_dsm=True,
        camera_type="ground",
    ):
        """
        determine if scene point is visible in image from camera point using DSM
        :param camera_point_utm: camera canter of projection in UTM coordinates
        :param scene_point_utm: scene point in UTM coordinates
        :param keep_if_outside_dsm: set to True to keep images collected by camera outside DSM
        :return visible: bool is true if scene point is visible in image from camera point
        """
        # get start point
        col1, row1 = self.xy_to_ortho(camera_point_utm[0], camera_point_utm[1])
        #        if col1 < 0 or col1 > self.columns-1: return keep_if_outside_dsm
        #        if row1 < 0 or row1 > self.rows-1: return keep_if_outside_dsm
        if col1 < 0:
            col1 = 0
        if col1 > self.columns - 1:
            col1 = self.columns - 1
        if row1 < 0:
            row1 = 0
        if row1 > self.rows - 1:
            row1 = self.rows - 1

        #        logger.info('\n')
        #        logger.info('camera_utm:', camera_point_utm)
        #        logger.info('camera col/row:', col1, row1)
        x1, y1, z1 = self.ortho_to_xyz(col1, row1)
        # logger.info('camera xyz in dsm:',x1,y1,z1)

        # get end point
        col2, row2 = self.xy_to_ortho(scene_point_utm[0], scene_point_utm[1])
        if col2 < 0 or col2 > self.columns - 1:
            return False
        if row2 < 0 or row2 > self.rows - 1:
            return False

        #        logger.info('scene_utm:', scene_point_utm)
        #        logger.info('scene col/row:', col2, row2)
        x2, y2, z2 = self.ortho_to_xyz(col2, row2)
        #        logger.info('scene xyz in dsm:',x2,y2,z2)

        # assume that camera location can be off by as much as several meters with GPS
        # so use DSM for all z values and assume visibility is at eye level
        # unless the camera is airborne and well above ground level
        eye_level = 2.0  # high enough to clear ground level noise, vehicles, etc.
        if camera_type == "airborne":
            z1 = camera_point_utm[2] + eye_level
            z2 = scene_point_utm[2] + eye_level
        #            zmax = max(z1,z2)
        else:
            z1 = z1 + eye_level
            z2 = z2 + eye_level
        #            zmax = min(z1 + eye_level, z2)
        #        logger.info('zmax:',zmax)

        # """ warning: this logic will fail if the point of interest is uphill from the camera
        # so improve it later. it's probably ok for now. we should be computing the zmax
        # for each point along the line according to the z slope
        # """

        ignore_meters = 1.0  # 3.0

        col_min = min(col1, col2)
        col_max = max(col1, col2)
        row_min = min(row1, row2)
        row_max = max(row1, row2)

        #        logger.info('#1')

        # if camera and scene point are in same dsm column, then search the line only
        # slope is not defined below in this case
        if col2 == col1:
            #            for row in range(row1,row2):
            for row in range(row_min, row_max):
                # do not penalize high z values very close to the scene point
                if abs(row - row2) < ignore_meters / self.gsd:
                    continue
                # compute zmax along line of sight
                full_distance = math.sqrt(
                    np.square(col2 - col1) + np.square(row2 - row1)
                )
                current_distance = math.sqrt(
                    np.square(col2 - col1) + np.square(row2 - row)
                )
                zmax = z2 * current_distance / full_distance + z1 * (
                    1.0 - current_distance / full_distance
                )
                # check DSM z versus zmax for occlusion
                x, y, z = self.ortho_to_xyz(col1, row)

                #                logger.info('1: z, zmax:', z, zmax)

                if z > zmax:
                    return False
            return True

        # define line equation to look for z values along that line
        m = (row2 - row1) / float(col2 - col1)
        b = row1 - m * col1

        #        logger.info('#2')

        # search along columns and rows separately
        #        for col in range(col1,col2):
        for col in range(col_min, col_max):
            row = round(m * col + b)
            # do not penalize high z values very close to the scene point
            if abs(col - col2) < ignore_meters / self.gsd:
                continue
            # compute zmax along line of sight
            full_distance = math.sqrt(np.square(col2 - col1) + np.square(row2 - row1))
            current_distance = math.sqrt(np.square(col2 - col) + np.square(row2 - row))
            zmax = z2 * current_distance / full_distance + z1 * (
                1.0 - current_distance / full_distance
            )
            # check DSM z versus zmax for occlusion
            x, y, z = self.ortho_to_xyz(col, row)

            #            logger.info('2: z, zmax:', z, zmax)

            if z > zmax:
                return False

        #        logger.info('#3')

        #        for row in range(row1,row2):
        for row in range(row_min, row_max):
            col = round((row - b) / m)
            # do not penalize high z values very close to the scene point
            if abs(row - row2) < ignore_meters / self.gsd:
                continue
            # compute zmax along line of sight
            full_distance = math.sqrt(np.square(col2 - col1) + np.square(row2 - row1))
            current_distance = math.sqrt(np.square(col2 - col) + np.square(row2 - row))
            zmax = z2 * current_distance / full_distance + z1 * (
                1.0 - current_distance / full_distance
            )
            # check DSM z versus zmax for occlusion
            x, y, z = self.ortho_to_xyz(col, row)

            #            logger.info('3: z, zmax:', z, zmax)

            if z > zmax:
                return False
        return True


def check_for_temp(path):
    found = False
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:
            break
        elif parts[1] == path:
            break
        else:
            path = parts[0]
            if parts[1][0] == "_":
                found = True
                break
    return found


def make_image_pool(
    point, dsm_path, img_path, output_path, dsm=None, norm_dist_threshold=0.8
):
    lat, lon, alt = point
    # get scene point UTM
    x, y, z = lla_to_utm(lat, lon, alt)
    scene_point_utm = np.array([x, y, z])
    # load the DSM
    if dsm is None:
        dsm = DSM(str(dsm_path))
    # loop on images in path, checking for visibility
    inside_count = 0
    visible_count = 0
    json_paths = sorted(list(img_path.rglob("*.json")))
    for file in tqdm(json_paths):
        if check_for_temp(file):
            continue
        with open(file) as f:
            params = json.load(f)
        inside, front, norm_dist = lla_in_image_fov(lat, lon, alt, params)

        #        """ debug
        #        """
        #        if params['fname'] != 'siteA01-camA012-2023-05-01-19-40-02-000001.jpg':
        #            continue

        # ignore if too close to image border because inaccuracy
        # of coordinates can result in false positives
        inside = inside and front and norm_dist < norm_dist_threshold
        #        logger.info('\ninside, front, norm_dist = ', inside, front, norm_dist)
        if inside:
            inside_count = inside_count + 1
            # convert camera coordinate to UTM
            tlat = params["extrinsics"]["lat"]
            tlon = params["extrinsics"]["lon"]
            talt = params["extrinsics"]["alt"]
            x, y, z = lla_to_utm(tlat, tlon, talt)
            camera_point_utm = np.array([x, y, z])
            # check for occlusion
            if params["type"] == "airborne":
                keep_if_outside_dsm = True
            else:
                keep_if_outside_dsm = False
            visible = dsm.is_visible(
                camera_point_utm,
                scene_point_utm,
                keep_if_outside_dsm=keep_if_outside_dsm,
                camera_type=params["type"],
            )
        else:
            visible = False
        #        logger.info(os.path.basename(file), ':', inside, front, norm_dist, visible)
        if inside and visible:
            visible_count = visible_count + 1
            #            logger.info(os.path.basename(file), ':', norm_dist)
            #            shutil.copyfile(file, os.path.join(str(output_path), os.path.basename(file)))
            #            image_file = str(file).replace(".json", ".jpg")
            #            shutil.copyfile(file, os.path.join(str(output_path), os.path.basename(file)))

            json_dirname = os.path.dirname(file)
            image_file = os.path.join(json_dirname, params["fname"])

            shutil.copyfile(
                image_file, os.path.join(str(output_path), os.path.basename(image_file))
            )
    logger.info("Inside count:", inside_count)
    logger.info("Visible count:", visible_count)


def make_image_pool_all_cameras(
    point_nad83, dsm_path, collect_path, output_path, norm_dist_threshold=0.8
):
    os.makedirs(output_path, exist_ok=True)
    lat, lon, alt = point_nad83
    point = nad83_to_wgs84(lat, lon, alt)
    logger.info("WGS84 point = ", point)
    dsm = DSM(str(dsm_path))
    img_paths = glob(os.path.join(collect_path, "*"))
    for img_path in img_paths:
        #        selected_img_path = os.path.join(img_path,'3_finished')
        selected_img_path = os.path.join(img_path, "5_pii_removed")
        if os.path.exists(selected_img_path):
            logger.info(selected_img_path)
            make_image_pool(
                point,
                dsm_path,
                Path(selected_img_path),
                output_path,
                dsm=dsm,
                norm_dist_threshold=norm_dist_threshold,
            )
