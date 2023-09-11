import numpy as np

path = "dataset/siteS01-carla-01/camA501-road-001/2023-03-14-11-36-34/depth/siteS01-camA501-2023-03-14-11-36-34-000000-depth.npy"

depth_map = np.load(path)
nearest_depth = np.nanmin(depth_map)
farthest_depth = np.nanmax(depth_map)
print(nearest_depth, farthest_depth)