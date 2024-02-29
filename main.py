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
import os

from tracking import tracking, export_to_csv


def get_frames():
    # Retrieves paths of frames from a directory as a list
    parent_dir = "to_image/img"
    input_path_list = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
    frames = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in input_path_list]
    return frames


def get_frames_from_nd2(nd2_file_path):
    # Retrieves paths of frames from a nd2 file as a list
    with ND2Reader(nd2_file_path) as nd2_frames:
        converted_frames = [(image / image.max() * 255).astype('uint8') for image in nd2_frames]
        return converted_frames


# Make ROI
roi_x = 30
roi_y = 70
roi_h = 710
roi_w = 310
ROI = (roi_x, roi_y, roi_h, roi_w)

# spot in spots = (x,y,w,h)
spots = [(266, 673, 20, 20), (291, 184, 25, 25), (250, 824, 10, 10)]

# Load frames
# frames = get_frames()
frames = get_frames_from_nd2(nd2_file_path="nd2_files/23Feb2024 Non RosetteSep 5kHz.nd2")

# Declare output path
output_path = "to_image/tracking/frame_"

# Canny Thresholding values:
canny_lower = 200
canny_upper = 600

# Perform tracking
overlay_frames, object_final_position, active_id_trajectory = tracking(frames, output_path, ROI, spots, canny_upper,
                                                                       canny_lower, draw_ROI=False,
                                                                       save_overlay=False)

# Create csv file from tracking info
csv_filename = "results/final_position_results.csv"
export_to_csv(object_final_position, csv_filename)

# Create csv file from tracking info
csv_filename = "results/active_id_trajectory.csv"
export_to_csv(active_id_trajectory, csv_filename)

# ffmpeg to video code
# ffmpeg -framerate 7 -i frame_%d.png tracking.mp4

# TODO: workout inconsistencies in labeling
# TODO: ROI, contour ROI inconsistency
# TODO: Work on fine tuning parameters
