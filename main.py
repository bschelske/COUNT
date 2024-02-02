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
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
import os

def spot_correction(input_edges, spots):
    for spot in spots:
        x, y, w, h = spot
        input_edges[y:y + h, x:x + w] = 0
    return input_edges

def open_cv_canny_overlay():
    # Overlays edge detection onto original image and writes to a file
    img = cv.imread(r'C:\Users\bensc\PycharmProjects\scikit\to_image\img\image-001.png', cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 50, 400)

    img = cv.merge((img, img, img))  # create RGB image from grayscale
    img2 = img.copy()
    img2[edges == 255] = [0, 0, 255]  # turn edges to red (bgr)
    cv.imwrite(r'C:\Users\bensc\PycharmProjects\scikit\to_image\overlay\overlay_image-001.png', img2)

def original_vs_canny():
    # Creates a plot comparing original image to edges (in red)
    img = cv.imread(r'C:\Users\bensc\PycharmProjects\scikit\to_image\img\image-001.png', cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 50, 400)

    img = cv.merge((img, img, img))  # create RGB image from grayscale
    img2 = img.copy()
    img2[edges == 255] = [255, 0, 0]  # turn edges to red

    plt.subplot(121), plt.imshow(img)
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(img2)
    plt.title('Edge Highlighted'), plt.xticks([]), plt.yticks([])

    plt.show()


def canny_contours_overlay(input, output, ROI, spots, preview=False, save_overlay=False):
    # Creates a plot comparing original image to edges (in red)
    img = cv.imread(input, cv.IMREAD_GRAYSCALE)
    img_copy = img.copy()
    img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2RGB)
    canny_img = cv.Canny(img, 40, 500)
    corrected_image = spot_correction(canny_img, spots)
    contours, hierarchy = cv.findContours(corrected_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    area_contours = []
    font = cv.FONT_HERSHEY_SIMPLEX
    roi_x, roi_y, roi_h, roi_w = ROI

    # Get contours in analysis region with area
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if (roi_x < x < roi_x+roi_w) and (roi_y < y < roi_y+roi_h):
            if cv.contourArea(cnt, True) > 0:
                area_contours.append(cnt)
            else:
                pass
        else:
            pass

    # Draw contours and labels
    for idx, cnt in enumerate(area_contours):
        x, y, w, h = cv.boundingRect(cnt)
        if int(y) > 441:
            color = (0,150,255) # orange
        if int(y) < 441:
            color = (255,150,0) # blue
        cv.rectangle(img_copy, (x, y), (x + w, y + h), color, 2)
        cv.putText(img_copy, str(idx), (x, y), font, 1, (255, 255, 255), 1, cv.LINE_AA)
    else:
        pass

    # Draw ROI
    cv.rectangle(img_copy, (roi_x, roi_y), ((roi_x+roi_w), (roi_y+roi_h)), (0, 0, 255), 1)

    if preview:
        cv.imshow('preview', img_copy)
        cv.waitKey(0)
        cv.destroyAllWindows()
    if save_overlay:
        cv.imwrite(output, img_copy)
        return None
    else:
        return img_copy


def detect_objects(frame, frame_index, ROI, spots):
    """
    Detects objects from inputted frame using canny edge detection and contour calculations.

    Objects are deteced within the ROI bounds. Spots are regions that are removed from edge detection

    Parameters:
    - frame: image from a movie.
    - frame_index: index of frame from movie.
    - ROI: tuple (roi_x, roi_y, roi_h, roi_w)
    - spots: list of tuples where spot in spots = (x,y,w,h)

    Returns:
    The detected objects as contours: area_contours, img_copy a copy of the input frame, which may be the frame itself or contours only.
    """

    canny_lower = 200
    canny_upper = 600
    roi_x, roi_y, roi_h, roi_w = ROI
    img_copy = frame.copy()
    img_copy = cv.cvtColor(img_copy, cv.COLOR_GRAY2RGB)
    canny_img = cv.Canny(frame, canny_lower, canny_upper, 5)
    corrected_image = spot_correction(canny_img, spots)
    img_copy[corrected_image == 255] = [0, 0, 255]  # turn edges to red (bgr)
    img_copy = corrected_image  # avoids overlay on regular image, instead visualizes contours
    contours, hierarchy = cv.findContours(corrected_image, mode=cv.RETR_TREE, method=cv.CHAIN_APPROX_NONE)
    area_contours = []
    # Get contours in analysis region with area
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        if (roi_x < x < (roi_x + roi_w)) and (roi_y < y < (roi_y + roi_h)):
            if cv.contourArea(cnt, True) > 0:
                area_contours.append(DetectedObject(id=None, position=(x,y), size=(w,h), most_recent_frame=frame_index))
            else:
                pass
        else:
            pass

    return area_contours, img_copy

def determine_object_color(y, roi_y):
    if int(y) > (roi_h / 2 + roi_y):
        return (0, 150, 255)  # orange
    if int(y) < (roi_h / 2 + roi_y):
        return (255, 150, 0)  # blue

class DetectedObject:
    def __init__(self, id, position, size, most_recent_frame):
        self.id = id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame


    def update_position(self, new_position):
        self.position = new_position

    def enters_from_left(self, roi_x):
        if roi_x < self.position[0] < roi_x + 60:
            return True
        else:
            return False

    def exits_right(self, roi_x, roi_w):
        if roi_x + roi_w < self.position[0]:
            return True
        else:
            return False

    def color(self, roi_h, roi_y):
        if int(self.position[1]) > (roi_h / 2 + roi_y):
            return (0, 150, 255)  # orange
        if int(self.position[1]) < (roi_h / 2 + roi_y):
            return (255, 150, 0)  # blue

    def center(self):
        return self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2

def calculate_distance(detectedobject1, detectedobject2):
    possible_center = detectedobject1.center()
    tracked_center = detectedobject2.center()
    distance = int(
        (((possible_center[0] - tracked_center[0]) ** 2) + ((possible_center[1] - tracked_center[1]) ** 2)) ** 0.5)
    return distance

def tracking(frames, output_path, ROI, spots, save_overlay=False):
    font = cv.FONT_HERSHEY_SIMPLEX
    overlayed_frames = []
    roi_x, roi_y, roi_h, roi_w = ROI
    top = []
    bottom = []
    active_ids = {}
    threshold_distance = 50
    img_h, img_w = frames[0].shape
    timeout_threshold = 7 * 1  # Frames
    next_id = 1

    for frame_index, frame in enumerate(frames):
        objects, img_copy = detect_objects(frame, frame_index, ROI, spots)

        # Remove IDs of objects that have moved off the screen
        for obj_id, tracked_obj in list(active_ids.items()):
            if tracked_obj.position[0] > img_w:
                del active_ids[obj_id]
            elif frame_index - tracked_obj.most_recent_frame > timeout_threshold:
                del active_ids[obj_id]  # Expire IDs if no new position found

        for obj in objects:
            match_found = False
            for obj_id, tracked_obj in active_ids.items():
                distance = calculate_distance(obj, tracked_obj)
                if distance < threshold_distance and obj.position[0] > tracked_obj.position[0]:
                    tracked_obj.id = obj.id
                    tracked_obj.update_position(obj.position)
                    tracked_obj.most_recent_frame = frame_index  # Update last frame detected
                    match_found = True
                    break

            if not match_found and obj.id is None:
                if obj.enters_from_left(roi_x):
                    obj.id = next_id
                    obj.most_recent_frame = frame_index  # Set last frame detected
                    active_ids[next_id] = obj
                    next_id += 1

        # Draw IDs
        for obj_id, tracked_obj in active_ids.items():
            cv.putText(img_copy, str(obj_id), tracked_obj.position, font, 1, (255, 255, 255), 1, cv.LINE_AA)

        # Draw ROI on every frame
        # cv.rectangle(img_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255,0,0), 2)
        overlayed_frames.append(img_copy)

    print(f'Total Cells: {len(active_ids)}\nTop: {len(top)}\nBottom: {len(bottom)}')

    if save_overlay:
        for idx, overlay_frame in enumerate(overlayed_frames):
            save_path = output_path + f"{idx}.png"
            cv.imwrite(save_path, overlay_frame)
        return None
    else:
        return overlayed_frames


def extract_edges(input, output):
    # Extracts edges from input, saves to output
    img = cv.imread(input, cv.IMREAD_GRAYSCALE)
    edges = cv.Canny(img, 50, 400)
    cv.imwrite(output, edges)


def compare_frames():
    input = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\count_von_count.png'
    output = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\count_von_count_out1.png'
    img1 = canny_contours_overlay(input, output,ROI, spots, save_overlay=False)

    input2 = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\count_von_count.png'
    output2 = r'C:\Users\bensc\PycharmProjects\scikit\to_image\count von count\count_von_count_out2.png'
    img2 = canny_contours_overlay(input2, output2, ROI, spots, save_overlay=False)

    concat_image = np.concatenate((img1, img2), axis=1)
    cv.imshow('side by side', concat_image)
    cv.waitKey(0)
    cv.destroyAllWindows()


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