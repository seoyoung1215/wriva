import cv2
import os
import numpy as np

def images_to_video(directory, output_filename, fps=30):
    # Get all jpg files from the directory
    images = [img for img in os.listdir(directory) if img.endswith(".jpg") or img.endswith(".png") ]
    
    # Sort the images by name
    images.sort(key=lambda x: (x.split('-')[2], x.split('-')[3], int(x.split('-')[-1].split('.')[0])))

    # Read the first image to get the dimensions
    frame = cv2.imread(os.path.join(directory, images[0]))
    h, w, layers = frame.shape
    size = (int(w//4), int(h//4))

    # Create a video writer object with the 'mp4v' codec for MP4 format
    out = cv2.VideoWriter(output_filename, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    for image in images:
        img_path = os.path.join(directory, image)
        print(img_path)
        img = cv2.imread(img_path)
        new_width = int(img.shape[1]//4)
        new_height = int(img.shape[0]//4)
        resized_img = cv2.resize(img, (new_width, new_height))
        # print(resized_img.shape, np.unique(resized_img))
        out.write(resized_img)

    out.release()

# Example usage
directory = '/mnt/vita-nas/wenyan/wriva/'+'dataset/siteA01-apl-office-buildings/camA009-gopro-1/2022-12-01-16-11-13'
output_filename = 'gopro.mp4'
images_to_video(directory, output_filename)
