"""
===================
T R A C K I N G
===================
This is where most of the work is done. The main.py file will run this code using instructions from the UI.

General tracking strategy:
    Import ND2 file and perform MOG2 background subtraction
    Use canny edge detection on background subtracted frames to generate contours
    Track contours through time using minimum distance traveled between frames

Specific tracking nuance:
    Objects are tracked assuming they enter from the left of the image
    There needs to be space to the left and right of the ROI
        Necessary to check that objects enter from the left
        and disappear on the right
    Tracking would likely improve on nd2 files where
        Objects have good contrast on background
        files are high frame rate
            (High FPS = More data... try decreasing camera ROI to make files smaller at high FPS)
    If you decide to save overlays, you'll get a bunch of .png files in a folder.
        hint: ffmpeg

Read more:
https://github.com/bschelske/COUNT
"""
import cv2 as cv
import csv
import numpy as np
from pims import ND2Reader_SDK
import typing


class DetectedObject:
    """
    This class represents a detected object with various properties and methods to manipulate and track the object.

    Attributes:
    ----------
    object_id : int
        A unique identifier for the object.
    position : Tuple[float, float]
        The (x, y) position of the object.
    size : Tuple[float, float]
        The (w, h) size of the object.
    most_recent_frame : int
        The most recent frame number where the object was detected.
    DEP_outlet : Bool
        True: object is influenced by DEP, False: no DEP. Determined by obj y position
    frames_tracked : int
        The number of frames the object has been tracked.

    Methods
    -------
    update_frames_tracked() -> None
        Increments the frame tracking counter.

    enters_from_left(roi_x: float, roi_w: float) -> bool
        Checks if the object enters the region of interest (ROI) from the left.

    exits_right(roi_x: float, roi_w: float) -> bool
        Checks if the object exits the ROI from the right.

    center() -> Tuple[float, float]
        Calculates and returns the center position of the object.

    outlet_assignment(roi_h: float, roi_y: float) -> None
        Assigns the DEP outlet status based on the object's position within the ROI.
    """

    def __init__(self, object_id, position, size, most_recent_frame, DEP_outlet):
        self.object_id = object_id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame
        self.DEP_outlet = DEP_outlet
        self.frames_tracked = 1
        self.position_history = {self.most_recent_frame: self.position}

    def update(self, most_recent_frame, new_position):
        self.frames_tracked += 1
        self.most_recent_frame = most_recent_frame
        self.position = new_position
        self.position_history[self.most_recent_frame] = self.position

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

    def center(self):
        return self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2

    def outlet_assignment(self, roi_h, roi_y):
        if int(self.position[1]) <= (roi_h / 2 + roi_y):
            self.DEP_outlet = True  # DEP Responsive
        else:
            self.DEP_outlet = False  # Not DEP Responsive


def nd2_mog_contours(nd2_file_path: str, ui_app) -> typing.Tuple[
        typing.List[DetectedObject], typing.List[DetectedObject]]:

    current_objects = {}
    object_final_position = []
    active_id_list = []
    roi_x, roi_y, roi_h, roi_w = ui_app.get_roi()
    next_id = 1
    overlay_frames = []

    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)

    with ND2Reader_SDK(nd2_file_path) as nd2_file:
        total_frames = len(nd2_file)
        for frame_number, frame_data in enumerate(nd2_file):
            print(f"\rFrame: {frame_number}/{total_frames - 1}", end="")  # Track progress

            # Get Objects, add all objects to end of active_id_trajectory list
            detected_objects_list, overlay_frame = detect_objects(frame_data=frame_data, frame_index=frame_number,
                                                                  backSub=backSub, ui_app=ui_app)
            active_id_list.extend(detected_objects_list)

            # Expire objects that are currently detected
            current_objects, object_final_position = expire_objects(current_objects, object_final_position, frame_number, ui_app)

            # Track position of current objects
            for detected_object in detected_objects_list:
                match_found = False

                # Calculate new positions for tracked objects
                for obj_id, tracked_obj in current_objects.items():
                    tracked_obj.object_id = obj_id
                    distance = calculate_distance(detected_object, tracked_obj)

                    # Check if the next object position is within the centroid distance and to the right
                    if distance < ui_app.max_centroid_distance.get() and detected_object.position[0] > tracked_obj.position[0]:
                        tracked_obj.object_id = detected_object.object_id
                        tracked_obj.most_recent_frame = frame_number
                        tracked_obj.update_frames_tracked()
                        match_found = True
                        break  # The object has been tracked, move to the next in the ID list.

                # Include newly detected objects
                if not match_found and detected_object.object_id is None:
                    if detected_object.enters_from_left(roi_x, roi_w):
                        detected_object.object_id = next_id
                        detected_object.most_recent_frame = frame_number
                        current_objects[next_id] = detected_object
                        next_id += 1

            if ui_app.save_overlay.get():
                cv.putText(overlay_frame, str(f"Objects: {len(current_objects.items())} Total: {len(object_final_position)}"),
                           (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 0), 2, cv.LINE_AA)
            overlay_frames.append(overlay_frame)

        # Draw IDs
        # for obj_id, tracked_obj in current_objects.items():
        #     tracked_obj.object_id = obj_id

        if ui_app.save_overlay.get():
            for idx, overlay_frame in enumerate(overlay_frames):
                save_path = ui_app.overlay_path + f"{idx:03d}.png"
                cv.imwrite(save_path, overlay_frame)

    return object_final_position, active_id_list


def detect_objects(frame_data, frame_index, backSub, ui_app):
    frame_copy = frame_data.copy()
    if ui_app.save_overlay.get():
        frame_copy = cv.normalize(frame_copy, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        frame_copy = cv.cvtColor(frame_copy, cv.COLOR_GRAY2BGR)

    # Do background subtraction, and normalize for canny
    foreground_mask = backSub.apply(frame_copy)
    normalized_frame = cv.normalize(foreground_mask, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    canny_img = cv.Canny(normalized_frame, ui_app.canny_lower.get(), ui_app.canny_upper.get(), 5)
    contours, hierarchy = cv.findContours(canny_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    # Handle overlapping objects:
    # create an empty mask
    mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)

    # Draw filled contours on a mask using bounding circles
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        if coord_in_roi((x, y), ui_app) and radius > ui_app.cell_radius.get():
            center = (int(x), int(y))
            radius = int(radius + ui_app.cell_radius.get() // 2)
            cv.circle(mask, center, radius, 255, -1)

    # find the contours on the mask (with solid drawn shapes) and draw outline on input image
    objects = []
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if ui_app.save_overlay.get():
            cv.drawContours(frame_copy, [cnt], 0, (0, 0, 255), 2)

        x, y, w, h = cv.boundingRect(cnt)
        objects.append(
            DetectedObject(object_id=None, position=(x, y), size=(w, h), most_recent_frame=frame_index,
                           DEP_outlet=None))
    return objects, frame_copy


def calculate_distance(detected_object1: DetectedObject, detected_object2: DetectedObject) -> int:
    """Calculates the euclidean distance between two objects"""
    possible_center = detected_object1.center()
    tracked_center = detected_object2.center()
    distance = int(
        (((possible_center[0] - tracked_center[0]) ** 2) + ((possible_center[1] - tracked_center[1]) ** 2)) ** 0.5)
    return distance


def coord_in_roi(coord, ui_app):
    roi_x, roi_y, roi_h, roi_w = ui_app.get_roi()
    x, y = coord
    if (roi_x < x < (roi_x + roi_w)) and (roi_y < y < (roi_y + roi_h)):
        return True
    else:
        return False


def export_to_csv(object_history, csv_filename: str) -> None:
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
        print(f"\nCells counted: {count}")


def expire_objects(current_objects, object_final_position, frame_number, ui_app):
    roi_x, roi_y, roi_h, roi_w = ui_app.get_roi()

    for obj_id, tracked_obj in list(current_objects.items()):
        tracked_obj.object_id = obj_id

        # Check if the object is to the right of the ROI
        if tracked_obj.position[0] > (roi_x + roi_w):
            tracked_obj.outlet_assignment(roi_h, roi_y)
            object_final_position.append(tracked_obj)
            del current_objects[obj_id]

        # Check if the object has disappeared and timed out
        elif frame_number - tracked_obj.most_recent_frame > ui_app.timeout.get():
            tracked_obj.outlet_assignment(roi_h, roi_y)  # Check outlet
            object_final_position.append(tracked_obj)
            del current_objects[obj_id]  # Expire IDs if no new position found
    return current_objects, object_final_position

# def update_tracked_objects(current_objects, detected_object, frame_number, ui_app):
#     # Calculate new positions for tracked objects
#     for obj_id, tracked_obj in current_objects.items():
#         tracked_obj.object_id = obj_id
#         distance = calculate_distance(detected_object, tracked_obj)
#
#         # Check if the next object position is within the centroid distance and to the right
#         if distance < ui_app.max_centroid_distance.get() and detected_object.position[0] > tracked_obj.position[0]:
#             tracked_obj.object_id = detected_object.object_id
#             tracked_obj.most_recent_frame = frame_number
#             tracked_obj.update_frames_tracked()
#             return True

