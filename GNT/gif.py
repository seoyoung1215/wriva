from PIL import Image

def create_gif(frame1_path, frame2_path, output_gif_path, duration=500):
    """
    Create a GIF from two image frames.

    Parameters:
    - frame1_path: Path to the first image frame.
    - frame2_path: Path to the second image frame.
    - output_gif_path: Path where the GIF will be saved.
    - duration: Duration each frame should be displayed (in milliseconds).
    """
    
    # Open the images using Pillow
    frame1 = Image.open(frame1_path)
    frame2 = Image.open(frame2_path)
    
    # Create the GIF
    frame1.save(output_gif_path, save_all=True, append_images=[frame2], duration=duration, loop=0)

path1 = "/mnt/vita-nas/wenyan/wriva/dataset/siteS01-carla-01/camA501-road-001/2023-03-14-11-36-34/siteS01-camA501-2023-03-14-11-36-34-000009.png"
path2 = "/mnt/vita-nas/wenyan/wriva/dataset/siteS01-carla-01/camA501-road-001/2023-03-14-11-36-34/siteS01-camA501-2023-03-14-11-36-34-000010.png"
# Usage:
create_gif(path1, path2, "output.gif")
