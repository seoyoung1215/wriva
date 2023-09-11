import os
import numpy as np
import imageio
import sys
import json
import gzip
import cv2
import json
import random

folder_path =  "/mnt/vita-nas/wenyan/co3d/"
all_cats = []
for nm in os.listdir(folder_path):
    if "json" in nm:
        continue
    else:
        all_cats.append(nm)
cnt = 0
small_sz_cnt = 0
all_scenes=[]
cat_00 =[]
for i, cat in enumerate(all_cats):
    print(cat)
    json_path = os.path.join(folder_path, cat, "frame_annotations.jgz")
    with gzip.open(json_path, "r") as fp:
        all_frames_data = json.load(fp)
        for frame in all_frames_data:
            # import pdb; pdb.set_trace()
            data_split = frame['meta']['frame_type']
            if data_split=="train_known":
                cnt+=1
                seq_name = frame['sequence_name']
                if seq_name not in all_scenes:
                    all_scenes.append(seq_name)
                    cat_00.append(cat)
                # H, W = frame["image"]["size"]
                # if H < 50 or W < 50:
                #     small_sz_cnt+=1
# print("train_known ", cnt, "scenes ", len(all_scenes), "size smaller than 50 ", small_sz_cnt )
i_all = np.arange(len(list(all_scenes)))
# import pdb; pdb.set_trace()
i_selected = random.sample(list(i_all), 100)
for ii in i_selected:
    print(cat_00[ii], all_scenes[ii])

# for i, cat in enumerate(all_cats):
#     data_list = json.load(open(os.path.join(folder_path, cat,"set_lists/set_lists_fewview_train.json"), "r"))
#     for scene in data_list["train"]:
#         scene_id, _, img_f = scene
#         cnt+=1
# print("train_known ", cnt)   