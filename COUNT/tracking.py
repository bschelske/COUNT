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
    update_position(new_position: Tuple[float, float]) -> None
        Updates the position of the object.

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

    def center(self):
        return self.position[0] + self.size[0] // 2, self.position[1] + self.size[1] // 2

    def outlet_assignment(self, roi_h, roi_y):
        if int(self.position[1]) <= (roi_h / 2 + roi_y):
            self.DEP_outlet = True  # DEP Responsive
        else:
            self.DEP_outlet = False  # Not DEP Responsive


def nd2_mog_contours(nd2_file_path: str, ui_app) -> typing.Tuple[
        typing.List[DetectedObject], typing.List[DetectedObject]]:
    active_ids = {}
    object_final_position = []
    active_id_trajectory = []
    ROI = ui_app.get_roi()
    roi_x, roi_y, roi_h, roi_w = ROI
    next_id = 1
    overlay_frames = []
    print(ui_app.save_overlay.get())

    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)

    with ND2Reader_SDK(nd2_file_path) as nd2_file:
        # Print metadata
        print("Metadata:")
        print(nd2_file.metadata)
        print(nd2_file)

        # Loop through each frame in the nd2 file
        for frame_index in range(len(nd2_file)):
            print(f"Frame: {frame_index}/{len(nd2_file) - 1}")  # Track progress

            # Get Objects
            objects, overlay_frame = detect_objects(nd2_file_path, frame_index, backSub, ui_app)

            active_id_trajectory.extend(objects)

            # Remove IDs of expired objects
            for obj_id, tracked_obj in list(active_ids.items()):
                tracked_obj.object_id = obj_id

                # Check if the object is to the right of the ROI
                if tracked_obj.position[0] > (roi_x + roi_w):
                    tracked_obj.outlet_assignment(roi_h, roi_y)
                    object_final_position.append(tracked_obj)
                    del active_ids[obj_id]

                # Check if the object has disappeared and timed out
                elif frame_index - tracked_obj.most_recent_frame > ui_app.timeout.get():
                    tracked_obj.outlet_assignment(roi_h, roi_y)  # Check outlet
                    object_final_position.append(tracked_obj)
                    del active_ids[obj_id]  # Expire IDs if no new position found

            # Track position of current objects
            for obj in objects:
                match_found = False

                # Calculate new positions for tracked objects
                for obj_id, tracked_obj in active_ids.items():
                    tracked_obj.object_id = obj_id
                    distance = calculate_distance(obj, tracked_obj)

                    # Check if the next object position is within the centroid distance and to the right
                    if distance < ui_app.max_centroid_distance.get() and obj.position[0] > tracked_obj.position[0]:
                        tracked_obj.object_id = obj.object_id
                        tracked_obj.most_recent_frame = frame_index
                        tracked_obj.update_frames_tracked()
                        match_found = True
                        break  # The object has been tracked, move to the next in the ID list.

                # Include newly detected objects
                if not match_found and obj.object_id is None:
                    if obj.enters_from_left(roi_x, roi_w):
                        obj.object_id = next_id
                        obj.most_recent_frame = frame_index
                        obj.update_frames_tracked()
                        active_ids[next_id] = obj
                        next_id += 1

            if ui_app.save_overlay.get():
                cv.putText(overlay_frame, str(f"Objects: {len(active_ids.items())} Total: {len(object_final_position)}"),
                           (10, 40), cv.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 0, 0), 2, cv.LINE_AA)
            overlay_frames.append(overlay_frame)

        # Draw IDs
        for obj_id, tracked_obj in active_ids.items():
            tracked_obj.object_id = obj_id

        if ui_app.save_overlay.get():
            for idx, overlay_frame in enumerate(overlay_frames):
                save_path = ui_app.overlay_path + f"{idx:03d}.png"
                cv.imwrite(save_path, overlay_frame)

    return object_final_position, active_id_trajectory


def detect_objects(nd2_file_path, frame_index, backSub, ui_app):
    ROI = ui_app.get_roi()
    roi_x, roi_y, roi_h, roi_w = ROI
    with ND2Reader_SDK(nd2_file_path) as nd2_file:
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
        print(f"Cells counted: {count}")
