import os
import numpy as np
import imageio
import torch
from torch.utils.data import Dataset
import sys
import json
from tqdm import tqdm
sys.path.append("../")
from .data_utils import rectify_inplane_rotation, get_nearest_pose_ids


def similarity_from_cameras(c2w, fix_rot=False):
    """
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c)
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array([[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    if fix_rot:
        R_align = np.eye(3)
        R = np.eye(3)
    else:
        R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale = 1.0 / np.median(np.linalg.norm(t + translate, axis=-1))
    return transform, scale


class NeRDS360Dataset(Dataset):
    def __init__(
        self,
        args,
        mode,
        scenes=(),
        **kwargs
    ):
        self.folder_path =  "/mnt/vita-nas/wenyan/wriva/dataset_NERDS360/"
        assert mode in ["train", "validation", "test"]
        self.mode = mode
        self.num_source_views = args.num_source_views

        if self.mode=="train":
            split="PDMultiObjv6/train/"
        elif self.mode=="validation":
            split="PD_v6_test/test_novel_objs/"
        self.folder_path = os.path.join(self.folder_path, split)

        # all_scenes = ['SF_6thAndMission_medium0/']
        all_scenes = os.listdir(self.folder_path)
        print(all_scenes)

        if len(scenes) > 0:
            if isinstance(scenes, str):
                scenes = [scenes]
        else:
            scenes = all_scenes
        cam_scale_factor = 1.5

        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []
        print("loading {} for {}".format(scenes, mode))
        for i, scene in enumerate(scenes):
            scene_path = os.path.join(self.folder_path, scene, self.mode if self.mode!="validation" else "val")
            img_files = sorted(os.listdir(os.path.join(scene_path, 'rgb')))
            json_path = os.path.join(scene_path,"pose/pose.json")
            # print(scene_path, len(img_files), json_path)
            with open(json_path, 'r') as file:
                data = json.load(file)

            f = data["focal"]
            cx = data["img_size"][0] / 2.0
            cy = data["img_size"][1] / 2.0
            intri = np.array([[f, 0, cx, 0], [0, f, cy, 0], [0, 0, 1, 0],[0, 0, 0, 1]])

            c2w_mats = []
            intrinsics = []
            rgb_files = []
            near_far = []

            for img_fname in tqdm(img_files):#,desc=f'loading train pose {mode} ({len(img_files)})'):
                # depth_map = np.load(os.path.join(scene_path, "depth", img_fname[:-4]+".npz"))["arr_0"]
                # nearest_depth = np.nanmin(depth_map)
                # farthest_depth = np.nanmax(depth_map)
                # near_far.append([nearest_depth, farthest_depth])
                c2w = np.array(data["transform"][img_fname[:-4]])
                c2w[0:3, 1:3] *= -1
                c2w = c2w[np.array([1, 0, 2, 3]), :]
                c2w[2, :] *= -1
                w2c_blender = np.linalg.inv(c2w)
                w2c_opencv = w2c_blender
                w2c_opencv[1:3] *= -1
                c2w_opencv = np.linalg.inv(w2c_opencv)
                c2w_mats.append(c2w_opencv)
                intrinsics.append(intri)
                rgb_files.append(os.path.join(scene_path, 'rgb',img_fname))
            c2w_mats = np.stack(c2w_mats)
            intrinsics = np.stack(intrinsics)

            T, sscale = similarity_from_cameras(c2w_mats)
            c2w_mats = T @ c2w_mats
            c2w_mats[:, :3, 3] *= sscale * cam_scale_factor

            num_render = len(rgb_files)
            self.render_rgb_files.extend(np.array(rgb_files).tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats])
            self.render_train_set_ids.extend([i] * num_render)
            # self.render_depth_range.extend([dr for dr in near_far])
            
            self.train_intrinsics.append(intrinsics)
            self.train_poses.append(c2w_mats)
            self.train_rgb_files.append(np.array(rgb_files).tolist())
        print("loading {} images for {}".format(len(self.render_rgb_files), mode))
            # if mode=='train':
            #     self.train_intrinsics.append(intrinsics)
            #     self.train_poses.append(c2w_mats)
            #     self.train_rgb_files.append(np.array(rgb_files).tolist())
            # else:
            #     train_scene_path = os.path.join(self.folder_path, scene, "train")
            #     train_img_files = sorted(os.listdir(os.path.join(train_scene_path, 'rgb')))
            #     train_json_path = os.path.join(train_scene_path,"pose/pose.json")
            #     with open(train_json_path, 'r') as file:
            #         train_data = json.load(file)
            #     intri = np.array([[train_data["focal"], 0, train_data["img_size"][0] / 2.0, 0], [0, train_data["focal"], train_data["img_size"][1] / 2.0, 0], [0, 0, 1, 0],[0, 0, 0, 1]])

            #     train_c2w_mats = []
            #     train_intrinsics = []
            #     train_rgb_files = []

            #     for img_fname in tqdm(train_img_files,desc=f'loading train pose {mode} ({len(train_img_files)})'):
            #         c2w = np.array(train_data["transform"][img_fname[:-4]])
            #         c2w[0:3, 1:3] *= -1
            #         c2w = c2w[np.array([1, 0, 2, 3]), :]
            #         c2w[2, :] *= -1
            #         w2c_blender = np.linalg.inv(c2w)
            #         w2c_opencv = w2c_blender
            #         w2c_opencv[1:3] *= -1
            #         c2w_opencv = np.linalg.inv(w2c_opencv)
            #         train_c2w_mats.append(c2w_opencv)
            #         train_intrinsics.append(intri)
            #         train_rgb_files.append(os.path.join(train_scene_path, 'rgb',img_fname))
            #     train_c2w_mats = np.stack(train_c2w_mats)
            #     train_intrinsics = np.stack(train_intrinsics)

            #     T, sscale = similarity_from_cameras(train_c2w_mats)
            #     train_c2w_mats = T @ train_c2w_mats
            #     train_c2w_mats[:, :3, 3] *= sscale * cam_scale_factor

            #     self.train_intrinsics.append(train_intrinsics)
            #     self.train_poses.append(train_c2w_mats)
            #     self.train_rgb_files.append(np.array(train_rgb_files).tolist())

    def __len__(self):
        return len(self.render_rgb_files)

    
    def __getitem__(self,idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        # depth_range = self.render_depth_range[idx]

        train_set_id = self.render_train_set_ids[idx]
        train_rgb_files = self.train_rgb_files[train_set_id]
        train_poses = self.train_poses[train_set_id]
        train_intrinsics = self.train_intrinsics[train_set_id]

        img_size = rgb.shape[:2]
        camera = np.concatenate(
            (list(img_size), intrinsics.flatten(), render_pose.flatten())
        ).astype(np.float32)

        if self.mode == "train":
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = np.random.choice(np.arange(1, 4), p=[0.2, 0.45, 0.35])
            num_select = self.num_source_views + np.random.randint(low=-2, high=2)
        else:
            if rgb_file in train_rgb_files:
                id_render = train_rgb_files.index(rgb_file)
            else:
                id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 34),
            tar_id=id_render,
            angular_dist_method="dist",
        )
        nearest_pose_ids = np.random.choice(
            nearest_pose_ids, min(num_select, len(nearest_pose_ids)), replace=False
        )

        assert id_render not in nearest_pose_ids
        # occasionally include input image
        if np.random.choice([0, 1], p=[0.995, 0.005]) and self.mode == "train":
            nearest_pose_ids[np.random.choice(len(nearest_pose_ids))] = id_render

        src_rgbs = []
        src_cameras = []
        for id in nearest_pose_ids:
            # print(rgb_file, train_rgb_files[id])
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
            train_pose = train_poses[id]
            train_intrinsics_ = train_intrinsics[id]

            src_rgbs.append(src_rgb)
            img_size = src_rgb.shape[:2]
            src_camera = np.concatenate(
                (list(img_size), train_intrinsics_.flatten(), train_pose.flatten())
            ).astype(np.float32)
            src_cameras.append(src_camera)

        src_rgbs = np.stack(src_rgbs, axis=0)
        src_cameras = np.stack(src_cameras, axis=0)  

        near_depth = 0.2
        # near_depth = 0.2 # change to 0
        far_depth = 1
        depth_range = torch.tensor([near_depth, far_depth])
        # depth_range = torch.tensor([depth_range[0], depth_range[1]])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }