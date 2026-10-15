"""
===================
T R A C K I N G
===================
This is where most of the work is done. The main.py file will run this code using instructions from the UI.
Input files can be Nikon .nd2 files or .avi videos.

General tracking strategy:
    Import ND2 file and perform MOG2 background subtraction
        (the background model is first primed with the opening frames, then tracking starts at frame 0)
    Use canny edge detection on background subtracted frames to generate contours
    Track contours through time: each frame, every detection is paired with at most one existing track
    (and every track with at most one detection), nearest pairs first, measured from where each track
    is predicted to be given its last velocity

Device assumptions:
    Cells flow from left to right
    Two outlets on the right side of the image, split at the vertical middle of the image
        Top outlet: positive DEP response (DEP_response = True)
        Bottom outlet: no positive DEP response (DEP_response = False)
    A cell's outlet is decided by the centre of its last detected position

Counting:
    A track is counted once it has been detected in at least `min_frames_tracked` frames.
    Shorter tracks are treated as noise and discarded.
    A track that is not seen for more than `timeout` frames is closed.

If you decide to save overlays, you'll get a .png per frame in the overlay folder.
    hint: ffmpeg

Read more:
https://github.com/bschelske/COUNT
"""
import contextlib
import csv
import os
import typing

import cv2 as cv
import numpy as np
from tqdm import tqdm

DEFAULT_MIN_FRAMES_TRACKED = 3
BACKGROUND_PRIMING_FRAMES = 30  # frames used to learn the background before tracking starts
VIDEO_EXTENSIONS = (".nd2", ".avi")  # file types COUNT can track


def is_supported_video(path: str) -> bool:
    return path.lower().endswith(VIDEO_EXTENSIONS)


class AviReader:
    """
    The frames of an AVI file as greyscale images, indexed by frame number like an ND2 reader.

    Frames are decoded in order, which is how COUNT reads them; asking for an earlier frame reopens the file.
    """

    def __init__(self, path: str):
        self.path = path
        capture = cv.VideoCapture(path)
        if not capture.isOpened():
            raise IOError(f"Could not open video file {path}")
        # Count frames by skipping through the file: the frame count stored in an AVI header can be wrong
        self._length = 0
        while capture.grab():
            self._length += 1
        capture.release()
        self._capture = None
        self._next_frame = 0
        self._open()

    def _open(self):
        if self._capture is not None:
            self._capture.release()
        self._capture = cv.VideoCapture(self.path)
        self._next_frame = 0

    def __len__(self):
        return self._length

    def __getitem__(self, index: int):
        if index < 0:
            index += self._length
        if not 0 <= index < self._length:
            raise IndexError(f"frame {index} is outside {self.path} ({self._length} frames)")
        if index < self._next_frame:
            self._open()
        while self._next_frame < index:  # skip ahead without decoding
            self._capture.grab()
            self._next_frame += 1
        ok, frame = self._capture.read()
        if not ok:
            raise IOError(f"Could not read frame {index} of {self.path}")
        self._next_frame += 1
        if frame.ndim == 3:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        return frame

    def close(self):
        if self._capture is not None:
            self._capture.release()
            self._capture = None


@contextlib.contextmanager
def open_video(path: str):
    """
    Open an .nd2 or .avi file. Use as `with open_video(path) as frames:`; `frames[i]` is frame i as a
    greyscale image and `len(frames)` is the number of frames.
    """
    extension = os.path.splitext(path)[1].lower()
    if extension == ".avi":
        reader = AviReader(path)
        try:
            yield reader
        finally:
            reader.close()
    elif extension == ".nd2":
        from pims import ND2Reader_SDK  # imported here so AVI files and the tests work without the Nikon SDK
        with ND2Reader_SDK(path) as reader:
            if 'm' in reader.sizes.keys():  # new nikon weirdness
                reader.iter_axes = 'm'
            yield reader
    else:
        raise ValueError(f"Unsupported file type {extension!r} ({path}). COUNT reads .nd2 and .avi files.")


class DetectedObject:
    """
    A detected object (a cell). The same instance is kept and updated for as long as the cell is tracked.

    Attributes:
    ----------
    object_id : int
        A unique identifier for the object (None until it is matched or added as a new track).
    position : Tuple[int, int]
        The (x, y) top-left corner of the object's bounding box in its most recent frame.
    size : Tuple[int, int]
        The (w, h) size of the bounding box in its most recent frame.
    most_recent_frame : int
        The most recent frame number where the object was detected.
    DEP_outlet : bool
        True: object left through the top outlet (positive DEP). False: bottom outlet.
        Decided from the centre of the object's last position.
    frames_tracked : int
        The number of frames in which the object was detected.
    position_history : Dict[int, Tuple[int, int]]
        {frame_number: (x, y)} for every frame the object was detected in.
    velocity : Tuple[float, float]
        Centre displacement per frame between the last two detections ((0, 0) until it has been seen twice).
    """

    def __init__(self, object_id, position, size, most_recent_frame):
        self.object_id = object_id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame
        self.DEP_outlet = ''
        self.frames_tracked = 1
        self.position_history = {self.most_recent_frame: self.position}
        self.velocity = (0.0, 0.0)

    def center(self) -> typing.Tuple[float, float]:
        return self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2

    def predicted_center(self, frame_number: int) -> typing.Tuple[float, float]:
        """Where this object should be in `frame_number` if it keeps moving at its last velocity."""
        x, y = self.center()
        frames = frame_number - self.most_recent_frame
        return x + self.velocity[0] * frames, y + self.velocity[1] * frames

    def update(self, detection: "DetectedObject", frame_number: int) -> None:
        """Move this track to a new detection in `frame_number`."""
        (old_x, old_y), (new_x, new_y) = self.center(), detection.center()
        frames = max(frame_number - self.most_recent_frame, 1)
        self.velocity = ((new_x - old_x) / frames, (new_y - old_y) / frames)
        self.position = detection.position
        self.size = detection.size
        self.most_recent_frame = frame_number
        self.frames_tracked += 1
        self.position_history[frame_number] = detection.position

    def outlet_assignment(self, image_h: int) -> None:
        """Top half of the image = top outlet = positive DEP response."""
        self.DEP_outlet = self.center()[1] < image_h / 2


def calculate_distance(detected_object1: DetectedObject, detected_object2: DetectedObject) -> float:
    """Calculates the euclidean distance between the centres of two objects"""
    x1, y1 = detected_object1.center()
    x2, y2 = detected_object2.center()
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def match_detections(tracks: typing.Dict[int, DetectedObject], detections: typing.List[DetectedObject],
                     frame_number: int, max_distance: float, backtrack_tolerance: float) -> typing.List[DetectedObject]:
    """
    Pair detections in this frame with existing tracks and update the matched tracks.

    Each track can take at most one detection and each detection can join at most one track. Candidate pairs
    are accepted closest first. Distance is measured from where the track is predicted to be in this frame
    (its last position moved on by its last velocity), so a track that lost its cell for a few frames looks
    for it further along the channel instead of grabbing a later cell passing its old position.
    A pair is only allowed if the detection is within `max_distance` of that prediction and has not moved
    left of the track's last position by more than `backtrack_tolerance` pixels (cells flow left to right,
    but a cell that pauses or whose outline jitters should still match).

    Returns the detections that did not match any track (these are new cells, or noise).
    """
    candidates = []
    for track_id, track in tracks.items():
        track_x, _ = track.center()
        predicted_x, predicted_y = track.predicted_center(frame_number)
        for index, detection in enumerate(detections):
            x, y = detection.center()
            if x < track_x - backtrack_tolerance:
                continue
            distance = ((x - predicted_x) ** 2 + (y - predicted_y) ** 2) ** 0.5
            if distance <= max_distance:
                candidates.append((distance, track_id, index))

    candidates.sort()
    used_tracks = set()
    used_detections = set()
    for distance, track_id, index in candidates:
        if track_id in used_tracks or index in used_detections:
            continue
        tracks[track_id].update(detections[index], frame_number)
        used_tracks.add(track_id)
        used_detections.add(index)

    return [detection for index, detection in enumerate(detections) if index not in used_detections]


class Tracker:
    """Keeps the active tracks for one video and decides which ones are counted."""

    def __init__(self, image_h: int, max_centroid_distance: float, timeout: int,
                 min_frames_tracked: int = DEFAULT_MIN_FRAMES_TRACKED, backtrack_tolerance: float = 0):
        self.image_h = image_h
        self.max_centroid_distance = max_centroid_distance
        self.timeout = timeout
        self.min_frames_tracked = min_frames_tracked
        self.backtrack_tolerance = backtrack_tolerance
        self.active = {}  # {object_id: DetectedObject} tracks that may still receive detections
        self.counted = {}  # {object_id: DetectedObject} closed tracks that count as cells
        self.next_id = 1

    def update(self, detections: typing.List[DetectedObject], frame_number: int) -> None:
        self._close_lost_tracks(frame_number)
        unmatched = match_detections(self.active, detections, frame_number, self.max_centroid_distance,
                                     self.backtrack_tolerance)
        for detection in unmatched:
            detection.object_id = self.next_id
            detection.most_recent_frame = frame_number
            self.active[self.next_id] = detection
            self.next_id += 1

    def finish(self) -> typing.Dict[int, DetectedObject]:
        """Close every remaining track (end of video) and return the counted objects."""
        for track_id in list(self.active):
            self._close(track_id)
        return self.counted

    def counted_so_far(self) -> int:
        """Closed tracks that counted, plus active tracks that are already long enough to count."""
        return len(self.counted) + sum(1 for t in self.active.values()
                                       if t.frames_tracked >= self.min_frames_tracked)

    def _close_lost_tracks(self, frame_number: int) -> None:
        for track_id, track in list(self.active.items()):
            if (frame_number - track.most_recent_frame) > self.timeout:
                self._close(track_id)

    def _close(self, track_id: int) -> None:
        track = self.active.pop(track_id)
        if track.frames_tracked >= self.min_frames_tracked:
            track.outlet_assignment(self.image_h)
            self.counted[track_id] = track


def _setting(ui_app, name, default=None):
    """Read a tkinter variable from the UI, or fall back to a default if the UI doesn't have it."""
    variable = getattr(ui_app, name, None)
    return variable.get() if variable is not None else default


def prime_background(backSub, frames, n_frames, n_samples=BACKGROUND_PRIMING_FRAMES):
    """
    Teach the background subtractor what the channel looks like before tracking starts.

    MOG2 needs a few dozen frames before its foreground mask is reliable; until then it marks noise as
    foreground (which becomes short noise tracks) and misses real cells. Running it over the first
    `n_samples` frames once, before going back to frame 0 to track, means detection works from the start.
    (Sampling frames spread across the whole video was tested and gave far more false detections.)
    `frames` is anything indexable by frame number (e.g. an ND2 reader).
    """
    for index in range(min(n_samples, n_frames)):
        backSub.apply(cv.normalize(frames[index], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))


def find_cell_contours(foreground_mask, canny_lower, canny_upper, cell_radius):
    """Edge-detect the background-subtracted mask and return contours that are plausibly single cells."""
    normalized_frame = cv.normalize(foreground_mask, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
    # Morphological operation to reduce noise
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
    cleaned_frame = cv.morphologyEx(normalized_frame, cv.MORPH_OPEN, kernel)
    canny_img = cv.Canny(cleaned_frame, canny_lower, canny_upper, apertureSize=3)
    contours, _ = cv.findContours(canny_img, mode=cv.RETR_EXTERNAL, method=cv.CHAIN_APPROX_SIMPLE)
    _, contours = remove_overlapped_objects(normalized_frame, contours, cell_radius)

    max_size = cell_radius * 10
    kept = []
    for cnt in contours:
        _, _, w, h = cv.boundingRect(cnt)
        if w <= max_size and h <= max_size:
            kept.append(cnt)
    return kept


def detect_objects(frame_data, frame_index, backSub_mask, ui_app):
    """
    Find cells in one frame.

    Returns (objects, contours): a DetectedObject per cell and the matching contours (for drawing overlays).
    If background subtraction found no motion in this frame, nothing is detected.
    """
    if not backSub_mask.any():
        return [], []

    contours = find_cell_contours(backSub_mask, ui_app.canny_lower.get(), ui_app.canny_upper.get(),
                                  ui_app.cell_radius.get())
    objects = []
    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        objects.append(DetectedObject(object_id=None, position=(x, y), size=(w, h), most_recent_frame=frame_index))
    return objects, contours


def remove_overlapped_objects(frame_copy, contours, cell_radius):
    mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)

    # Draw filled contours on a mask using bounding circles
    for cnt in contours:
        (x, y), radius = cv.minEnclosingCircle(cnt)
        center = (int(x), int(y))
        radius = int(radius + cell_radius)
        cv.circle(mask, center, radius, (255), -1)
    # return contours detected on overlapped objects
    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return frame_copy, contours


def draw_overlay(frame_8u, contours, tracker: Tracker, frame_number: int, objects_in_frame: int):
    """Return a colour copy of the frame with contours, the outlet dividing line, track IDs and counts."""
    overlay = cv.cvtColor(frame_8u, cv.COLOR_GRAY2BGR)
    image_h, image_w = overlay.shape[:2]

    # Dividing line between the top (DEP) and bottom (no DEP) outlets
    cv.line(overlay, (0, image_h // 2), (image_w - 1, image_h // 2), (128, 128, 128), 1)
    cv.drawContours(overlay, contours, -1, (0, 0, 255), 2)

    # Label tracks that were seen in this frame and are long enough to count
    for object_id, track in tracker.active.items():
        if track.most_recent_frame == frame_number and track.frames_tracked >= tracker.min_frames_tracked:
            x, y = track.position
            cv.putText(overlay, str(object_id), (int(x), max(int(y) - 4, 12)), cv.FONT_HERSHEY_SIMPLEX, 0.5,
                       (0, 255, 255), 1, cv.LINE_AA)

    cv.putText(overlay, f"Frame: {frame_number}  In-Frame: {objects_in_frame}  Counted: {tracker.counted_so_far()}",
               (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2, cv.LINE_AA)
    return overlay


def nd2_mog_contours(nd2_file_path: str, ui_app) -> typing.Dict[int, DetectedObject]:
    """
    Process an ND2 or AVI file to detect and track objects across frames.

    Args:
        nd2_file_path (str): Path to the .nd2 or .avi file.
        ui_app: User interface application instance with the tracking settings.

    Returns:
        {object_id: DetectedObject} for every counted cell. Each object's position_history holds its trajectory.
    """
    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
    save_overlay = bool(_setting(ui_app, "save_overlay", False))
    file_stem = os.path.splitext(os.path.basename(nd2_file_path))[0]
    tracker = None

    if save_overlay:
        os.makedirs(ui_app.overlay_path, exist_ok=True)
        print(f"\noverlay saving in {ui_app.overlay_path}")

    with open_video(nd2_file_path) as nd2_file:
        prime_background(backSub, nd2_file, len(nd2_file))

        for frame_number in tqdm(range(len(nd2_file)), desc=file_stem):
            frame_data = cv.normalize(nd2_file[frame_number], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            if tracker is None:
                tracker = Tracker(image_h=frame_data.shape[0],
                                  max_centroid_distance=ui_app.max_centroid_distance.get(),
                                  timeout=ui_app.timeout.get(),
                                  min_frames_tracked=_setting(ui_app, "min_frames_tracked",
                                                             DEFAULT_MIN_FRAMES_TRACKED),
                                  backtrack_tolerance=ui_app.cell_radius.get())

            backSub_mask = backSub.apply(frame_data)
            objects_in_frame, contours = detect_objects(frame_data=frame_data, frame_index=frame_number,
                                                        backSub_mask=backSub_mask, ui_app=ui_app)
            tracker.update(objects_in_frame, frame_number)

            if save_overlay:
                overlay = draw_overlay(frame_data, contours, tracker, frame_number, len(objects_in_frame))
                cv.imwrite(os.path.join(ui_app.overlay_path, f"{file_stem}_{frame_number:05d}.png"), overlay)

    if tracker is None:  # empty file
        return {}
    return tracker.finish()


def export_to_csv(counted_objects: typing.Dict[int, DetectedObject], csv_filename: str) -> None:
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['object_id', 'x_pos', 'y_pos', 'x_size', 'y_size', 'most_recent_frame', 'frames_tracked',
                      'DEP_response']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        DEP_true = 0
        DEP_false = 0

        for object_id, obj in counted_objects.items():
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
        print("\nObject Detection Results:")
        print(f"\tCells counted: {len(counted_objects)}")
        print(f"\tDEP True: {DEP_true}")
        print(f"\tDEP False: {DEP_false}")
        print(f"{csv_filename} saved")


def export_trajectories_to_csv(objects: typing.Iterable[DetectedObject], csv_filename: str) -> None:
    """One row per (counted object, frame it was detected in)."""
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'object_id', 'x_pos', 'y_pos', 'x_size', 'y_size']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for obj in objects:
            for frame, position in obj.position_history.items():
                writer.writerow({
                    'frame': frame,
                    'object_id': obj.object_id,
                    'x_pos': position[0],
                    'y_pos': position[1],
                    'x_size': obj.size[0],
                    'y_size': obj.size[1]
                })
    print(f"\n{csv_filename} saved")
