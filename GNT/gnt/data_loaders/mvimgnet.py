import os
import numpy as np
import imageio
import torch
import sys

sys.path.append("../")
from torch.utils.data import Dataset
from .data_utils import rectify_inplane_rotation, random_crop, get_nearest_pose_ids, random_flip
from .llff_data_utils import load_llff_data, batch_parse_llff_poses
from .pose_utils import gen_poses
import time
import datetime

class MVImgnetDataset(Dataset):
    def __init__(self, args, mode, scenes=(), random_crop=True, **kwargs):
        self.folder_path = "/data/mvimgnet/"
        self.args = args
        self.rectify_inplane_rotation = args.rectify_inplane_rotation
        self.mode = mode  # train / test / validation
        self.num_source_views = args.num_source_views
        self.random_crop = random_crop
        self.render_rgb_files = []
        self.render_intrinsics = []
        self.render_poses = []
        self.render_train_set_ids = []
        self.render_depth_range = []

        self.train_intrinsics = []
        self.train_poses = []
        self.train_rgb_files = []

        # all_cats_candi = []
        # for nm in os.listdir(self.folder_path):
        #   if ("mv" not in nm) and ("MV" not in nm) and  ("md" not in nm):
        #        all_cats_candi.append(nm)
        # if len(scenes) > 0:
        #     if isinstance(scenes, str):
        #         all_cats = [scenes]
        # else:
        #     all_cats = all_cats_candi
        # all_scenes = [os.listdir(os.path.join(self.folder_path, cat)) for cat in all_cats]
        all_scenes = ["0/0000be4b",  "10/000170c6", "100/0300106d", \
                  "101/0000ecd3", "119/00011279", "180/0000a226", "51/23000184"]


        print("loading {} for {}".format(all_scenes, mode))
        start_t = time.time()
        for i, scene in enumerate(all_scenes):
            scene_path = os.path.join(self.folder_path, scene)
            if not os.path.exists(os.path.join(scene_path, 'poses_bounds.npy')):
                print(scene_path, "generate poses_bounds.npy")
                gen_poses(scene_path)
            _, poses, bds, render_poses, i_test, rgb_files = load_llff_data(
                scene_path, load_imgs=False, factor =4
            )
            near_depth = np.min(bds)
            far_depth = np.max(bds)
            intrinsics, c2w_mats = batch_parse_llff_poses(poses)

            i_test = np.arange(poses.shape[0])[:: self.args.llffhold]
            i_train = np.array(
                [
                    j
                    for j in np.arange(int(poses.shape[0]))
                    if (j not in i_test and j not in i_test)
                ]
            )

            if mode == "train":
                i_render = i_train
            else:
                i_render = i_test

            self.train_intrinsics.append(intrinsics[i_train])
            self.train_poses.append(c2w_mats[i_train])
            self.train_rgb_files.append(np.array(rgb_files)[i_train].tolist())
            num_render = len(i_render)
            self.render_rgb_files.extend(np.array(rgb_files)[i_render].tolist())
            self.render_intrinsics.extend([intrinsics_ for intrinsics_ in intrinsics[i_render]])
            self.render_poses.extend([c2w_mat for c2w_mat in c2w_mats[i_render]])
            self.render_depth_range.extend([[near_depth, far_depth]] * num_render)
            self.render_train_set_ids.extend([i] * num_render)
        print('INFO: load all scenes cost {:0>8} seconds'.format(str(datetime.timedelta(seconds=round(time.time() - start_t)))))

    def __len__(self):
        return (
            len(self.render_rgb_files) * 100000
            if self.mode == "train"
            else len(self.render_rgb_files)
        )

    def __getitem__(self, idx):
        idx = idx % len(self.render_rgb_files)
        rgb_file = self.render_rgb_files[idx]
        rgb = imageio.imread(rgb_file).astype(np.float32) / 255.0
        render_pose = self.render_poses[idx]
        intrinsics = self.render_intrinsics[idx]
        depth_range = self.render_depth_range[idx]
        mean_depth = np.mean(depth_range)
        world_center = (render_pose.dot(np.array([[0, 0, mean_depth, 1]]).T)).flatten()[:3]

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
            id_render = -1
            subsample_factor = 1
            num_select = self.num_source_views

        nearest_pose_ids = get_nearest_pose_ids(
            render_pose,
            train_poses,
            min(self.num_source_views * subsample_factor, 28),
            tar_id=id_render,
            angular_dist_method="dist",
            scene_center=world_center
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
        # print("######nearest images######")
        for id in nearest_pose_ids:
            # print(train_rgb_files[id])
            src_rgb = imageio.imread(train_rgb_files[id]).astype(np.float32) / 255.0
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
        if self.mode == "train" and self.random_crop:
            crop_h = np.random.randint(low=250, high=750)
            crop_h = crop_h + 1 if crop_h % 2 == 1 else crop_h
            crop_w = int(400 * 600 / crop_h)
            crop_w = crop_w + 1 if crop_w % 2 == 1 else crop_w
            rgb, camera, src_rgbs, src_cameras = random_crop(
                rgb, camera, src_rgbs, src_cameras, (crop_h, crop_w)
            )
        if self.mode == 'train' and np.random.choice([0, 1], p=[0.5, 0.5]):
            rgb, camera, src_rgbs, src_cameras = random_flip(rgb, camera, src_rgbs, src_cameras)

        depth_range = torch.tensor([depth_range[0] * 0.9, depth_range[1] * 1.6])

        return {
            "rgb": torch.from_numpy(rgb[..., :3]),
            "camera": torch.from_numpy(camera),
            "rgb_path": rgb_file,
            "src_rgbs": torch.from_numpy(src_rgbs[..., :3]),
            "src_cameras": torch.from_numpy(src_cameras),
            "depth_range": depth_range,
        }
