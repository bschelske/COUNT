import os
import cv2 as cv
import csv
import numpy as np
from nd2reader import ND2Reader
from typing import List, Tuple

"""
===================
T R A C K I N G 
===================
This is where most of the work is done. The main.py file will run this code using instructions from the UI. 
Read more: https://github.com/bschelske/C.O.U.N.T.
"""

def export_to_csv(object_history, csv_filename):
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['object_id', 'x_pos', 'y_pos', 'x_size', 'y_size', 'most_recent_frame', 'frames_tracked',
                      'DEP_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        count = 0

        for obj in object_history:
            if obj.frames_tracked > 1:
                count += 1
            writer.writerow({
                'object_id': obj.object_id,
                'x_pos': obj.position[0],
                'y_pos': obj.position[1],
                'x_size': obj.size[0],
                'y_size': obj.size[1],
                'most_recent_frame': obj.most_recent_frame,
                'frames_tracked': obj.frames_tracked,
                'DEP_response': obj.DEP_outlet
            })
        print(f"Cells counted: {count}")


def nd2_mog_contours(nd2_file_path, ui_app, output_path="background_subtraction/"):
    active_ids = {}
    object_final_position = []
    active_id_trajectory = []
    ROI = ui_app.get_roi()
    roi_x, roi_y, roi_h, roi_w = ROI
    cell_radius = ui_app.cell_radius.get()
    next_id = 1

    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)

    with ND2Reader(nd2_file_path) as nd2_file:
        # Print metadata
        print("Metadata:")
        print(nd2_file.metadata)
        print(nd2_file)
        img_h, img_w = nd2_file[0].shape
        overlay_frames = []
        output_path = "mog_results/frame_"
        # Loop through each frame in the nd2 file
        for frame_index in range(len(nd2_file)):
            # Get Objects
            print(f"Frame: {frame_index}/{len(nd2_file) -1}")
            objects, overlay_frame = detect_objects_mog(nd2_file_path, frame_index, backSub, ui_app)
            #     overlay_frames.append(overlay_frame)
            # for idx, overlay_frame in enumerate(overlay_frames):
            #     save_path = output_path + f"{idx:03d}.png"
            #     cv.imwrite(save_path, overlay_frame)

            active_id_trajectory.extend(objects)

            # Remove IDs of objects that have moved off the screen
            for obj_id, tracked_obj in list(active_ids.items()):
                tracked_obj.object_id = obj_id
                if tracked_obj.position[0] > (roi_x + roi_w):
                    tracked_obj.outlet_assignment(roi_h, roi_y)  # Check outlet
                    object_final_position.append(tracked_obj)
                    del active_ids[obj_id]
                elif frame_index - tracked_obj.most_recent_frame > ui_app.timeout.get():
                    tracked_obj.outlet_assignment(roi_h, roi_y)  # Check outlet
                    object_final_position.append(tracked_obj)
                    del active_ids[obj_id]  # Expire IDs if no new position found

            for obj in objects:
                match_found = False
                # Calculate new positions for tracked objects
                for obj_id, tracked_obj in active_ids.items():
                    tracked_obj.object_id = obj_id
                    distance = calculate_distance(obj, tracked_obj)
                    if distance < ui_app.max_centroid_distance.get() and obj.position[0] > tracked_obj.position[0]:
                        tracked_obj.object_id = obj.object_id
                        tracked_obj.most_recent_frame = frame_index  # Update last frame detected
                        tracked_obj.update_frames_tracked()
                        match_found = True
                        break

                if not match_found and obj.object_id is None:
                    if obj.enters_from_left(roi_x, roi_w):
                        obj.object_id = next_id
                        obj.most_recent_frame = frame_index  # Set last frame detected
                        obj.update_frames_tracked()
                        active_ids[next_id] = obj
                        next_id += 1

        # Draw IDs
        for obj_id, tracked_obj in active_ids.items():
            tracked_obj.object_id = obj_id

    return object_final_position, active_id_trajectory


def detect_objects_mog(nd2_file_path, frame_index, backSub, ui_app):
    # Detect objects on a frame following MOG background subtraction.
    ROI = ui_app.get_roi()
    roi_x, roi_y, roi_h, roi_w = ROI
    with ND2Reader(nd2_file_path) as nd2_file:
        # Get current frame
        frame_data = nd2_file[frame_index]
        frame = cv.normalize(frame_data, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        frame_copy = frame.copy()  # Copy for overlay later...
        frame_copy = cv.cvtColor(frame_copy, cv.COLOR_GRAY2BGR)

        # Do background subtraction, and normalize for canny
        foreground_mask = backSub.apply(frame)
        normalized_frame = cv.normalize(foreground_mask, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        canny_img = cv.Canny(normalized_frame, ui_app.canny_lower.get(), ui_app.canny_upper.get(), 5)
        contours, hierarchy = cv.findContours(canny_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

        # create an empty mask
        mask = np.zeros(frame_data.shape[:2], dtype=np.uint8)

        # Draw filled contours on a mask using bounding circles
        for cnt in contours:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            if (roi_x < x < (roi_x + roi_w)) and (roi_y < y < (roi_y + roi_h)) and radius > ui_app.cell_radius.get():
                center = (int(x), int(y))
                radius = int(radius + ui_app.cell_radius.get())
                cv.circle(mask, center, radius, (255), -1)

        # find the contours on the mask (with solid drawn shapes) and draw outline on input image
        objects = []
        contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            # cv.drawContours(frame_copy, [cnt], 0, (0, 0, 255), 2)
            x, y, w, h = cv.boundingRect(cnt)
            # cv.putText(frame_copy, str(f"Frame {frame_index + 1} Objects: {len(contours)}"),
            #             (int(ui_app.roi_width.get() * .2), int(ui_app.roi_width.get() * .2)), cv.FONT_HERSHEY_SIMPLEX, 2,
            #             (0, 0, 0), 3, cv.LINE_AA)
            objects.append(
                DetectedObject(object_id=None, position=(x, y), size=(w, h), most_recent_frame=frame_index,
                               DEP_outlet=None))
    return objects, frame_copy


class DetectedObject:
    def __init__(self, object_id, position, size, most_recent_frame, DEP_outlet):
        self.object_id = object_id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame
        self.DEP_outlet = DEP_outlet
        self.frames_tracked = 0

    def update_position(self, new_position):
        self.position = new_position

    def update_frames_tracked(self):
        self.frames_tracked += 1

    def enters_from_left(self, roi_x, roi_w):
        if roi_x < self.position[0] < roi_x + int(roi_w * .2):
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


def get_frames(parent_dir: str):
    # Retrieves paths of frames from a directory as a list
    input_path_list = [os.path.join(parent_dir, f) for f in os.listdir(parent_dir)]
    frames = [cv.imread(f, cv.IMREAD_GRAYSCALE) for f in input_path_list]
    return frames


def nd2_to_png(nd2_file_path: str, output_path: str, normalize=False):
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
