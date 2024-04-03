import os
import cv2 as cv
import csv

import numpy as np
from nd2reader import ND2Reader


def tracking(frames, output_path, ROI, spots, canny_upper, canny_lower, max_centroid_distance, timeout, draw_ROI=False, save_overlay=False):
    overlay_frames = []
    active_ids = {}
    object_final_position = []
    active_id_trajectory = []
    FONT = cv.FONT_HERSHEY_SIMPLEX
    roi_x, roi_y, roi_h, roi_w = ROI
    img_h, img_w = frames[0].shape
    next_id = 1

    for frame_index, frame in enumerate(frames):
        objects, img_copy = detect_objects(frame, frame_index, ROI, spots, canny_upper, canny_lower)
        active_id_trajectory.extend(objects)
        # Remove IDs of objects that have moved off the screen
        for obj_id, tracked_obj in list(active_ids.items()):
            tracked_obj.object_id = obj_id
            if tracked_obj.position[0] > img_w:
                tracked_obj.outlet_assignment(roi_h, roi_y)  # Check outlet
                object_final_position.append(tracked_obj)
                del active_ids[obj_id]
            elif frame_index - tracked_obj.most_recent_frame > timeout:
                tracked_obj.outlet_assignment(roi_h, roi_y)  # Check outlet
                object_final_position.append(tracked_obj)
                del active_ids[obj_id]  # Expire IDs if no new position found

        for obj in objects:
            match_found = False
            # Calculate new positions for tracked objects
            for obj_id, tracked_obj in active_ids.items():
                tracked_obj.object_id = obj_id
                distance = calculate_distance(obj, tracked_obj)
                if distance < max_centroid_distance and obj.position[0] > tracked_obj.position[0]:
                    tracked_obj.object_id = obj.object_id
                    tracked_obj.most_recent_frame = frame_index  # Update last frame detected
                    match_found = True
                    break

            if not match_found and obj.object_id is None:
                if obj.enters_from_left(roi_x, roi_w):
                    obj.object_id = next_id
                    obj.most_recent_frame = frame_index  # Set last frame detected
                    active_ids[next_id] = obj
                    next_id += 1

        # Draw IDs
        for obj_id, tracked_obj in active_ids.items():
            tracked_obj.object_id = obj_id
            cv.putText(img_copy, str(obj_id), tracked_obj.position, FONT, 1, (255, 255, 255), 1, cv.LINE_AA)

        # Draw ROI on every frame
        if draw_ROI:
            cv.rectangle(img_copy, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 0, 0), 2)

        # Add frames together into a list
        overlay_frames.append(img_copy)

    if save_overlay:
        for idx, overlay_frame in enumerate(overlay_frames):
            save_path = output_path + f"{idx:03d}.png"
            cv.imwrite(save_path, overlay_frame)
    return overlay_frames, object_final_position, active_id_trajectory


def export_to_csv(object_history, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['object_id', 'x_pos', 'y_pos', 'x_size', 'y_size', 'most_recent_frame', 'DEP_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for obj in object_history:
            writer.writerow({
                'object_id': obj.object_id,
                'x_pos': obj.position[0],
                'y_pos': obj.position[1],
                'x_size': obj.size[0],
                'y_size': obj.size[1],
                'most_recent_frame': obj.most_recent_frame,
                'DEP_response': obj.DEP_outlet
            })


def spot_correction(input_edges, spots):
    for spot in spots:
        x, y, w, h = spot
        input_edges[y:y + h, x:x + w] = 0
    return input_edges


class DetectedObject:
    def __init__(self, object_id, position, size, most_recent_frame, DEP_outlet):
        self.object_id = object_id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame
        self.DEP_outlet = DEP_outlet

    def update_position(self, new_position):
        self.position = new_position

    def enters_from_left(self, roi_x, roi_w):
        if roi_x < self.position[0] < roi_x + roi_w:
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

    def outlet_assignment(self, roi_h, roi_y):
        if int(self.position[1]) <= (roi_h / 2 + roi_y):
            self.DEP_outlet = True  # DEP Responsive
        else:
            self.DEP_outlet = False  # Not DEP Responsive


def calculate_distance(detected_object1, detected_object2):
    possible_center = detected_object1.center()
    tracked_center = detected_object2.center()
    distance = int(
        (((possible_center[0] - tracked_center[0]) ** 2) + ((possible_center[1] - tracked_center[1]) ** 2)) ** 0.5)
    return distance


def detect_objects(frame, frame_index, ROI, cell_radius, spots, canny_upper, canny_lower):
    """
    Detects objects from inputted frame using canny edge detection and contour calculations.

    Objects are detected within the ROI bounds. Spots are regions that are removed from edge detection

    Parameters:
    - frame: image from a movie.
    - frame_index: index of frame from movie.
    - ROI: tuple (roi_x, roi_y, roi_h, roi_w)
    - spots: list of tuples where spot in spots = (x,y,w,h)

    Returns:
    The detected objects as contours: area_contours, img_copy a copy of the input frame, which may be the frame itself
    or contours only.

    note: the frames inputted here are png files. The assumption is that they have already been background subtracted
    """

    roi_x, roi_y, roi_h, roi_w = ROI

    # Get current frame
    frame_copy = frame.copy()
    frame_copy = cv.cvtColor(frame_copy, cv.COLOR_GRAY2RGB)
    canny_img = cv.Canny(frame, canny_lower, canny_upper, 3)

    # Spot correction likely unnecessary following background subtraction
    corrected_image = spot_correction(canny_img, spots)
    frame_copy[corrected_image == 255] = [0, 0, 255]  # turn canny edges to red (bgr)
    frame_copy = corrected_image  # avoids overlay on regular image, instead visualizes contours

    # Get contours
    contours, hierarchy = cv.findContours(corrected_image, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    # create an empty mask
    mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)

    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius + cell_radius)
        cv.circle(mask, center, radius, (255), -1)

    area_contours = []
    # find the contours on the mask (with solid drawn shapes) and draw outline on input image
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        cv.drawContours(frame_copy, [cnt], 0, (0, 0, 255), 2)
        x, y, w, h = cv.boundingRect(cnt)
        area_contours.append(
            DetectedObject(object_id=None, position=(x, y), size=(w, h), most_recent_frame=frame_index,
                           DEP_outlet=None))


    # area_contours = []
    # # Get contours in analysis region with area
    # for cnt in contours:
    #     x, y, w, h = cv.boundingRect(cnt)
    #     if (roi_x < x < (roi_x + roi_w)) and (roi_y < y < (roi_y + roi_h)):
    #         if cv.contourArea(cnt, True) > 0:
    #             area_contours.append(
    #                 DetectedObject(object_id=None, position=(x, y), size=(w, h), most_recent_frame=frame_index,
    #                                DEP_outlet=None))
    #         else:
    #             pass
    #     else:
    #         pass

    return area_contours, frame_copy


def get_frames(parent_dir):
    # Retrieves paths of frames from a directory as a list
    input_path_list = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
    frames = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in input_path_list]
    return frames


def nd2_to_png(nd2_file_path, output_path, normalize=False):
    with ND2Reader(nd2_file_path) as nd2_file:
        # Print metadata
        print("Metadata:")
        print(nd2_file.metadata)
        for frame_index in range(len(nd2_file)):
            # Read frame data
            frame_data = nd2_file[frame_index]
            if normalize:
                frame_data = cv.normalize(frame_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            converted_path = f"{output_path}frame_{frame_index:03d}.png"
            cv.imwrite(converted_path, frame_data)
            print(f"{converted_path} saved", end='\r')
        print("nd2 converted to png")