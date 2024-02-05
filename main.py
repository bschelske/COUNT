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
import os

from tracking import tracking

# Utilizing functions here and below:

# parent_dir = r"C:\Users\bensc\PycharmProjects\scikit\to_image\img\\"
# input_path_list = [parent_dir + f for f in os.listdir(parent_dir)]
# output_path_list = [r"C:\Users\bensc\PycharmProjects\scikit\to_image\bounding_boxes\overlay_" + f for f in
#                     os.listdir(parent_dir)]
#
# for input, output in zip(input_path_list, output_path_list):
#     canny_contours_overlay(input, output, save_overlay=True)
#

# Make ROI
roi_x = 30
roi_y = 70
roi_h = 710
roi_w = 310
ROI = (roi_x, roi_y, roi_h, roi_w)

# spot in spots = (x,y,w,h)
spots = [(266, 673, 20, 20), (291, 184, 25, 25), (250, 824, 10, 10)]
# Attempt Tracking:

# Load frames
parent_dir = r"C:\Users\bensc\PycharmProjects\scikit\to_image\img\\"
input_path_list = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
frames = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in input_path_list]



# # Compare frames
# spots = []
# # compare_frames()
# input = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\count_von_count.png'
# # input = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\image-001.png'
# # output = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\image-001_50_150.png'
# output = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\count_von_count_out50_150.png'
# img = cv.imread(input, cv.IMREAD_GRAYSCALE)
# img_copy = img.copy()
# img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2RGB)
# canny_img = cv.Canny(img, 50, 150)
# cv.imwrite(output, canny_img)
# # original_vs_canny()

input = r'C:\Users\bensc\PycharmProjects\scikit\to_image\img\image-001.png'
# _ = canny_contours_overlay(input, None, ROI, preview=True)

# # # Call function to overlay bounding boxes on frames
output_path = r"C:\Users\bensc\PycharmProjects\scikit\to_image\tracking\frame_"
overlayed_frames = tracking(frames, output_path, ROI, spots, save_overlay=True)

# ffmpeg to video code
# ffmpeg -framerate 7 -i frame_%d.png tracking.mp4

# This is a git change
# TODO: workout inconsistencies in labeling
# TODO: ROI, contour ROI inconsistency
# TODO: Work on fine tuning parameters
# TODO: Make slides