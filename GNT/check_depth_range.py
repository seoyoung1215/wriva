import numpy as np
import os

# Define the directory path
# dir_path = '/mnt/vita-nas/wenyan/wriva/dataset_NERDS360/PD_v6_test/test_novel_objs/SF_6thAndMission_medium0/val/depth'
dir_path = '/mnt/vita-nas/wenyan/wriva/dataset_NERDS360/PD_v6_test/test_novel_objs/SF_6thAndMission_medium0/train/depth'
# Initialize variables to store global nearest and farthest depths
global_nearest = float('inf')
global_farthest = -float('inf')

# Iterate over each file in the directory
for filename in os.listdir(dir_path):
    # Check if the file has a .npz extension
    if filename.endswith(".npz"):
        # Construct the full file path
        file_path = os.path.join(dir_path, filename)
        # Load the depth map from the .npz file
        with np.load(file_path) as data:
            # Assume 'depth' is the key for the depth map in the .npz file
            # Modify this if the key is different in your files
            depth_map = data
            # print("Keys in the NPZ file:", data.files)
            depth_map = depth_map["arr_0"].astype(float)
            # print(depth_map.shape)
            # Compute the nearest and farthest depth for this file
            nearest_depth = np.nanmin(depth_map)
            farthest_depth = np.nanmax(depth_map)

            # Update global nearest and farthest values
            global_nearest = min(global_nearest, nearest_depth)
            global_farthest = max(global_farthest, farthest_depth)

            print(f"For file {filename}:")
            print(f"Nearest Depth: {nearest_depth}")
            print(f"Farthest Depth: {farthest_depth}\n")

print(f"Overall Nearest Depth: {global_nearest}")
print(f"Overall Farthest Depth: {global_farthest}")
