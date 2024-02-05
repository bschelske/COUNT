"""
For creating visuals in slideshows.

Canny edge detection and object tracking has visual appeal.
Need to come up with more ways to represent tracking and make
good slideshows. Some functions compare before and after canny edge
detection.
"""

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from main import roi_h, ROI, spots
from tracking import spot_correction


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


def determine_object_color(y, roi_y):
    if int(y) > (roi_h / 2 + roi_y):
        return (0, 150, 255)  # orange
    if int(y) < (roi_h / 2 + roi_y):
        return (255, 150, 0)  # blue


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


# Compare frames
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