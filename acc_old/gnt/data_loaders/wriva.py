import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
from scipy.spatial.transform import Rotation as R
import pymap3d as pm
import sys
import json
from pathlib import Path
sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids
from skimage.transform import resize

def qvec2rotmat(qvec):
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )


def read_metadata(json_paths):
    metadata_dicts = []
    for json_path in json_paths:
        with open(json_path) as fid:
            metadata_dict = json.load(fid)
            metadata_dicts.append(metadata_dict)
    return metadata_dicts


def compute_centroid(metadata_dicts):
    """
    Compute coordinates (lat, lon, alt) of the centroid for an image collection.

    :param metadata_dicts: list of metadata json dictionaries
    :return: numpy array representing centroid (lat, lon, alt)
    """
    return np.array(
        [
            (
                [
                    metadata_dict["extrinsics"]["lat"],
                    metadata_dict["extrinsics"]["lon"],
                    metadata_dict["extrinsics"]["alt"],
                ]
            )
            for metadata_dict in metadata_dicts
        ]
    ).mean(axis=0)


def read_cameras(pose_files):
    metadata_dicts = read_metadata(pose_files)
    origin = compute_centroid(metadata_dicts)
    # print("Calculated origin:", f"lat: {origin[0]}, lon: {origin[1]}, alt: {origin[2]}")

    rgb_files = []
    c2w_mats = []
    for pose_file in sorted(pose_files):
        with open(pose_file, "r") as fp:
            meta = json.load(fp)
        intrinsics = meta["intrinsics"]
        intrinsics_matrix = np.array([
            [intrinsics["fx"]/2.0, 0, intrinsics["cx"]/2.0, 0],
            [0, intrinsics["fy"]/2.0, intrinsics["cy"]/2.0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        rgb_file = os.path.join(os.path.dirname(pose_file), meta["fname"])
        rgb_files.append(rgb_file)
        
        d = meta["extrinsics"]
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
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics_matrix] * len(pose_files)), c2w_mats


def get_intrinsics_from_hwf(h, w, focal):
    return np.array(
        [[focal, 0, 1.0 * w / 2, 0], [0, focal, 1.0 * h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


class WRIVADataset(Dataset):
    def __init__(
        self,
        args,
        mode,
        scenes=(),
        **kwargs
    ):
        self.folder_path = "../dataset/siteS01-carla-01/" 
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == "validation":
            mode = "val"
        assert mode in ["train", "val", "test"]
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip
        self.image_size = np.array([540, 960])
        # all_scenes = ["camA501-road-001/2023-03-14-11-36-34"]
        # all_scenes = ["camA502-security-001/2023-03-14-13-14-01"]
        # all_scenes = ["camA503-drone-001/2023-03-14-11-36-46"]
        all_scenes = []
        except_scenes = ["camA501-road-001/2023-03-14-11-36-34", "camA501-road-001/2023-03-14-11-36-39", "camA501-road-001/2023-03-14-11-36-45",
                         "camA502-security-001/2023-03-14-13-14-01", "camA502-security-001/2023-03-14-14-09-02", "camA502-security-001/2023-03-14-14-09-30",
                         "camA503-drone-001/2023-03-14-11-36-46", "camA503-drone-001/2023-03-14-11-36-50", "camA503-drone-001/2023-03-14-11-36-55"]
        all_dirs = ["camA501-road-001", "camA503-drone-001","camA502-security-001"]
        for a_dir in all_dirs:
            for d in os.listdir(os.path.join(self.folder_path, a_dir)):
                candidate = os.path.join(a_dir, d)
                if candidate in except_scenes:
                    continue
                all_scenes.append(candidate)
        # print(all_scenes)
        # import pdb; pdb.set_trace()
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} for {}".format(scenes, mode))

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.train_rgb_files = []
        self.train_intrinsics= []
        self.train_poses = []
        self.render_train_set_ids = []
        cntr=0

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene)
            pose_files = [os.path.join(root, file) for root, dirs, files in os.walk(self.scene_path) for file in files if file.endswith('.json')]
            print(self.scene_path, len(pose_files))
            rgb_files, intrinsics, poses = read_cameras(pose_files)
            i_all = np.arange(len(rgb_files))
            if self.mode != "train":
                i_render = i_all[:: self.testskip]
                i_train = i_all 
            else:
                aa = i_all[:: self.testskip]
                i_train = i_render = np.array([idx for idx in i_all if not idx in aa])
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in poses[i_render]])
            num_render = len(i_render)

            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            self.train_intrinsics.append(np.array(intrinsics)[i_train])
            self.train_poses.append(np.array(poses)[i_train])
            self.render_train_set_ids.extend([cntr] * num_render)
            cntr += 1
        print(len(self.train_rgb_files))
        print("loading {} images for {}".format(len(self.render_rgb_files), mode))

    def __len__(self):
        return len(self.render_rgb_files)

    def __getitem__(self, idx):
        rgb_file = self.render_rgb_files[idx]
        render_pose = self.render_poses[idx]
        render_intrinsics = self.render_intrinsics[idx]
        # import pdb; pdb.set_trace()

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files, train_intrinsics, train_poses = self.train_rgb_files[train_set_id],self.train_intrinsics[train_set_id],self.train_poses[train_set_id]

        if self.mode == "train":
            # id_render = int(os.path.basename(rgb_file)[:-4].split("-")[-1])
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            id_render = int(os.path.basename(rgb_file)[:-4].split("-")[-1])
            subsample_factor = 1
            num_select = self.num_source_views
        # import pdb; pdb.set_trace()
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = resize(rgb, self.image_size)
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 28),
            tar_id=id_render,
            angular_dist_method="matrix",
        )
        nearest_pose_ids = np.random.choice(nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        src_rgb_paths=[]
        for id in nearest_pose_ids:
            # print(rgb_file.split("/")[-1], "ref_images", train_rgb_files[id].split("/")[-1])
            src_rgb_paths.append(train_rgb_files[id])
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            src_rgb = resize(src_rgb, self.image_size)
            src_rgb = src_rgb[..., [-1]] * src_rgb[..., :3] + 1 - src_rgb[..., [-1]]
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]
            if self.rectify_inplane_rotation:
                train_pose, src_rgb = rectify_inplane_rotation(train_pose, render_pose, src_rgb)

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)

        near_depth = 2.0
        far_depth = 6.0

        depth_range = torch.tensor([near_depth, far_depth])

        # output_file_path = rgb_file[:-4].split("/")[-1]+".txt"
        # with open(output_file_path, "w") as file:
        #     file.write(rgb_file + "\n")
        #     for filename in src_rgb_paths:
        #         file.write(filename + "\n")
        # np.save(rgb_file[:-4].split("/")[-1]+".npy", camera)
        # np.save(rgb_file[:-4].split("/")[-1]+"_sources.npy", src_cameras)

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
