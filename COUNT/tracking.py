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
import csv
import typing

import cv2 as cv
import numpy as np
from tqdm import tqdm
from pims import ND2Reader_SDK


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

    def __init__(self, object_id, position, size, most_recent_frame):
        self.object_id = object_id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame
        self.DEP_outlet = ''
        self.frames_tracked = 0
        self.position_history = {self.most_recent_frame: self.position}
        self.displacement_history = []
        self.avg_displacement = None

    def update(self, most_recent_frame, new_position):
        self.frames_tracked += 1
        self.most_recent_frame = most_recent_frame

        self.displacement_history.append(new_position[0] - self.position[0])
        self.calculate_avg_displacement()

        self.position = new_position
        self.position_history[self.most_recent_frame] = self.position

    def calculate_avg_displacement(self):
        self.avg_displacement = sum(self.displacement_history) / len(self.displacement_history)

    def update_frames_tracked(self):
        self.frames_tracked += 1

    def enters_from_left(self, roi_x, roi_w):
        if roi_x < self.position[0] < roi_x + int(roi_w * .2):
            return True
        else:
            return False

    def exits_right(self, roi_x, roi_w):
        if roi_x + roi_w <= self.position[0]:
            return True
        else:
            return False

    def center(self):
        return self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2

    def outlet_assignment(self, image_h):
        if int(self.position[1]) < (image_h // 2):
            self.DEP_outlet = True  # DEP Responsive
        else:
            self.DEP_outlet = False  # Not DEP Responsive


def nd2_mog_contours(nd2_file_path: str, ui_app) -> typing.Tuple[
    typing.List[DetectedObject], typing.List[DetectedObject]]:
    """
    Process ND2 file to detect and track objects across frames.

    Args:
        nd2_file_path (str): Path to the ND2 file.
        ui_app: User interface application instance with necessary methods and attributes.

    Returns:
        typing.Tuple[List[DetectedObject], List[DetectedObject]]:
            - A list of final positions of tracked objects.
            - A list of all detected objects across frames.
    """
    surviving_objects_dict = {}  # {object_id: object}
    expired_objects_dict = {}  # {object_id: object}
    object_history_list = []  # Tracking data for every object identified
    next_new_id = 1  # ID number for first object
    overlay_frames = []  # List containing frames for save overlay feature
    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)

    with ND2Reader_SDK(nd2_file_path) as nd2_file:
        # Get information about nd2 file
        image_h = nd2_file.metadata['height']
        total_frames = len(nd2_file)

        # Perform tracking on each frame
        for frame_number, frame_data in tqdm(enumerate(nd2_file)):
            # Detect Objects
            objects_in_frame_list, overlay_frame = detect_objects(frame_data=frame_data, frame_index=frame_number,
                                                                  backSub=backSub, ui_app=ui_app)
            object_history_list.extend(objects_in_frame_list)

            # Expire outgoing objects
            if surviving_objects_dict:
                surviving_objects_dict, expired_objects_dict = expire_objects(surviving_objects_dict,
                                                                              expired_objects_dict,
                                                                              frame_number, image_h, ui_app)

            # Match current objects to object history
            for object_in_frame in objects_in_frame_list:
                no_match, surviving_objects_dict = match_tracked_objects(surviving_objects_dict, object_in_frame,
                                                                         frame_number,
                                                                         ui_app)
                # Add incoming objects
                if no_match:
                    surviving_objects_dict, next_new_id = add_new_objects(object_in_frame, surviving_objects_dict,
                                                                          next_new_id,
                                                                          frame_number)
            # Add text, object ids to each frame, if chosen
            if ui_app.save_overlay.get():
                # Add text to top of frame
                cv.putText(overlay_frame,
                           str(f"In-Frame: {len(objects_in_frame_list)} Total: {len(expired_objects_dict)}"),
                           (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 0), 2, cv.LINE_AA)

                # Add IDs to each tracked object
                for object_id, tracked_object in surviving_objects_dict.items():
                    cv.putText(overlay_frame,
                               str(object_id),
                               tracked_object.position, cv.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 0, 0), 1, cv.LINE_AA)

                # Store resulting frames into a list
                overlay_frames.append(overlay_frame)

        # On the last frame, add all remaining tracked objects to expired list.
        expired_objects_dict.update(surviving_objects_dict)

        # Save overlay frames to results/overlay folder (default)
        if ui_app.save_overlay.get():
            print(f"\noverlay saving in {ui_app.overlay_path}")
            for idx, overlay_frame in enumerate(overlay_frames):
                save_path = ui_app.overlay_path + f"{idx:03d}.png"
                cv.imwrite(save_path, overlay_frame)

    return expired_objects_dict, object_history_list


def detect_objects(frame_data, frame_index, backSub, ui_app):
    frame_copy = frame_data.view()

    # If saving overlay frames, a copy of the original frame must be converted from gray to color
    if ui_app.save_overlay.get():
        overlay_frame = cv.normalize(frame_copy, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
        overlay_frame = cv.cvtColor(overlay_frame, cv.COLOR_GRAY2BGR)
    backSub = False

    if backSub:
        foreground_mask = backSub.apply(frame_copy)
    else:
        foreground_mask = frame_copy

    # Process the grayscale copy of the frame for canny edge detection
    normalized_frame = cv.normalize(foreground_mask, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    canny_img = cv.Canny(normalized_frame, ui_app.canny_lower.get(), ui_app.canny_upper.get(), 5)
    contours, hierarchy = cv.findContours(canny_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)

    objects = []
    # Create a DetectedObject instance for each contour
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        objects.append(
            DetectedObject(object_id=None, position=(x, y), size=(w, h), most_recent_frame=frame_index))

        # Draw the contour in red onto the color frame
        if ui_app.save_overlay.get():
            cv.drawContours(overlay_frame, [cnt], 0, (0, 0, 255), 2)
            frame_copy = overlay_frame

    return objects, frame_copy


def expire_objects(surviving_objects_dict, expired_objects_dict, frame_number, image_h, ui_app):
    for obj_id, tracked_obj in list(surviving_objects_dict.items()):
        if (frame_number - tracked_obj.most_recent_frame) > ui_app.timeout.get():
            # Removed expired objects that are not tracked
            if tracked_obj.frames_tracked == 0:
                del surviving_objects_dict[tracked_obj.object_id]
            # Add expired objects that were being tracked
            else:
                tracked_obj.outlet_assignment(image_h)
                expired_objects_dict[tracked_obj.object_id] = tracked_obj
                del surviving_objects_dict[tracked_obj.object_id]

    return surviving_objects_dict, expired_objects_dict


def match_tracked_objects(surviving_objects_dict, object_in_frame, frame_number, ui_app):
    if not surviving_objects_dict:
        no_match = True
        return no_match, surviving_objects_dict

    candidates = {}
    # Go through each item in the tracking queue
    for obj_id, previous_object_instance in surviving_objects_dict.items():
        distance = calculate_distance(object_in_frame, previous_object_instance)
        if (object_in_frame.position[0] > previous_object_instance.position[0]) and (
                distance < ui_app.max_centroid_distance.get()):
            candidates[previous_object_instance] = distance
    if candidates:
        matched_object = min(candidates, key=candidates.get)
        object_in_frame.object_id = matched_object.object_id
        object_in_frame.most_recent_frame = frame_number
        object_in_frame.frames_tracked = matched_object.frames_tracked + 1
        surviving_objects_dict[object_in_frame.object_id] = object_in_frame
        no_match = False
        return no_match, surviving_objects_dict
    else:
        no_match = True
        return no_match, surviving_objects_dict


def add_new_objects(new_object, objects_in_previous_frame_dict, next_new_id, frame_number):
    new_object.object_id = next_new_id
    new_object.most_recent_frame = frame_number
    objects_in_previous_frame_dict[next_new_id] = new_object
    next_new_id += 1
    return objects_in_previous_frame_dict, next_new_id


def calculate_distance(detected_object1: DetectedObject, detected_object2: DetectedObject) -> int:
    """Calculates the euclidean distance between two objects"""
    possible_center = detected_object1.center()
    tracked_center = detected_object2.center()
    distance = int(
        (((possible_center[0] - tracked_center[0]) ** 2) + ((possible_center[1] - tracked_center[1]) ** 2)) ** 0.5)
    return distance

def export_to_csv(expired_objects_dict, csv_filename: str) -> None:
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['object_id', 'x_pos', 'y_pos', 'x_size', 'y_size', 'most_recent_frame', 'frames_tracked',
                      'DEP_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        DEP_true = 0
        DEP_false = 0

        for object_id, obj in expired_objects_dict.items():
            if obj.DEP_outlet is True:
                DEP_true += 1
            else:
                DEP_false += 1

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
        print(f"\nCells counted: {len(expired_objects_dict)}")
        print(f"DEP True: {DEP_true}")
        print(f"DEP False: {DEP_false}")
