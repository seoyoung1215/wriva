import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json

sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def read_cameras(pose_file):
    basedir = os.path.dirname(pose_file)
    with open(pose_file, "r") as fp:
        meta = json.load(fp)

    camera_angle_x = float(meta["camera_angle_x"])
    rgb_files = []
    c2w_mats = []

    img = imageio.imread(os.path.join(basedir, "images", meta["frames"][0]["file_path"] + ".png"))
    H, W = img.shape[:2]
    focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
    intrinsics = get_intrinsics_from_hwf(H, W, focal)

    for i, frame in enumerate(meta["frames"]):
        rgb_file = os.path.join(basedir, "images", meta["frames"][i]["file_path"] + ".png")
        rgb_files.append(rgb_file)
        c2w = np.array(frame["transform_matrix"])
        w2c_blender = np.linalg.inv(c2w)
        w2c_opencv = w2c_blender
        w2c_opencv[1:3] *= -1
        c2w_opencv = np.linalg.inv(w2c_opencv)
        c2w_mats.append(c2w_opencv)
    c2w_mats = np.array(c2w_mats)
    return rgb_files, np.array([intrinsics] * len(meta["frames"])), c2w_mats


def get_intrinsics_from_hwf(h, w, focal):
    return np.array(
        [[focal, 0, 1.0 * w / 2, 0], [0, focal, 1.0 * h / 2, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    )


class OmniObject3DDataset(Dataset):
    def __init__(
        self,
        args,
        mode,
        scenes=(),
        **kwargs
    ):
        self.folder_path = "/dataset/omniobject3d/OpenXD-OmniObject3D-New/raw/blender_renders/" #os.path.join(args.rootdir, "../../GNT/data/nerf_synthetic/")
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        if mode == "validation":
            mode = "val"
        assert mode in ["train", "val", "test"]
        self.mode = mode  # train / test / val
        self.num_source_views = args.num_source_views
        self.testskip = args.testskip

        all_scenes = []
        for nm in os.listdir(self.folder_path):
            if "tar.gz" in nm or "txt" in nm:
                continue
            else:
                all_scenes.append(nm)
        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes

        print("loading {} scenes for {}".format(len(scenes), mode))
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.train_rgb_files = []
        self.train_intrinsics= []
        self.train_poses = []
        self.render_train_set_ids = []
        cntr=0

        for scene in scenes:
            self.scene_path = os.path.join(self.folder_path, scene, "render")
            pose_file = os.path.join(self.scene_path, "transforms.json")
            rgb_files, intrinsics, poses = read_cameras(pose_file)
            i_all = np.arange(len(rgb_files))
            if self.mode != "train":
                i_render = i_all[:: self.testskip]
                i_train = i_all #np.array([idx for idx in i_all if not idx in i_render])
            else:
                i_train = i_render = i_all
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

        # train_pose_file = os.path.join("/".join(rgb_file.split("/")[:-2]), "transforms.json")
        # train_rgb_files, train_intrinsics, train_poses = read_cameras(train_pose_file)
        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files, train_intrinsics, train_poses = self.train_rgb_files[train_set_id],self.train_intrinsics[train_set_id],self.train_poses[train_set_id]
        # import pdb; pdb.set_trace()

        if self.mode == "train":
            id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.3, 0.5, 0.2])
        else:
            id_render = int(os.path.basename(rgb_file)[:-4].split("_")[1])
            subsample_factor = 1

        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        rgb = rgb[..., [-1]] * rgb[..., :3] + 1 - rgb[..., [-1]]
        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), render_intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            int(self.num_source_views * subsample_factor),
            tar_id=id_render,
            angular_dist_method="vector",
        )
        nearest_pose_ids = np.random.choice(nearest_pose_ids, self.num_source_views, replace=False)

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            # print(rgb_file, "ref_images", train_rgb_files[id])
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
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
        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
