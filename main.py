"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

Ben: ffmpeg code
cd  PycharmProjects\scikit\to_image

ffmpeg -i 50_kHz.mp4 -vf fps=1 image-%03d.png
convert to images until 1 second into video
ffmpeg -ss 0 -t 1 -i 50_kHz.mp4 image-%03d.png

ffmpeg -framerate 7 -i canny_image-%03d.png canny.mp4

"""
import cv2 as cv
from nd2reader import ND2Reader
import matplotlib.pyplot as plt
import os

from tracking import tracking, export_to_csv


def get_frames(parent_dir):
    # Retrieves paths of frames from a directory as a list
    input_path_list = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
    frames = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in input_path_list]
    return frames


def nd2_to_png(nd2_file_path):
    with ND2Reader(nd2_file_path) as nd2_file:
        # Print metadata
        print("Metadata:")
        print(nd2_file.metadata)
        for frame_index in range(len(nd2_file)):
            # Read frame data
            frame_data = nd2_file[frame_index]
            normalized_frame = cv.normalize(frame_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            image_path = f"nd2_to_png/frame_{frame_index:03d}.png"
            cv.imwrite(image_path, normalized_frame)
            print(f"{image_path} saved", end='\r')
            frames.append(normalized_frame)
        print("nd2 converted to png")

# Make ROI
roi_x = 0
roi_y = 0
roi_h = 2048
roi_w = 2048
ROI = (roi_x, roi_y, roi_h, roi_w)

# spot in spots = (x,y,w,h)
# spots = [(266, 673, 20, 20), (291, 184, 25, 25), (250, 824, 10, 10)]
spots = []

# Convert nd2 file to png files
# nd2_to_png(nd2_file_path="nd2_files/23Feb2024 Non RosetteSep 5kHz.nd2")

# Load frames
frames = get_frames(parent_dir="nd2_to_png/")
# Declare output path
output_path = "nd2_results/frame_"

# Canny Thresholding values:
canny_lower = 200
canny_upper = 600

# Perform tracking
print("Tracking")
overlay_frames, object_final_position, active_id_trajectory = tracking(frames, output_path, ROI, spots, canny_upper,
                                                                       canny_lower, draw_ROI=True,
                                                                       save_overlay=True)
print("Creating csv files")
# Create csv file from tracking info
csv_filename = "results/final_position_results.csv"
export_to_csv(object_final_position, csv_filename)

# Create csv file from tracking info
csv_filename = "results/active_id_trajectory.csv"
export_to_csv(active_id_trajectory, csv_filename)
print("done")
# ffmpeg to video code
# ffmpeg -framerate 7 -i frame_%d.png tracking.mp4

# TODO: workout inconsistencies in labeling
# TODO: ROI, contour ROI inconsistency
# TODO: Work on fine tuning parameters
