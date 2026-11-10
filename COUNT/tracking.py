"""
===================
T R A C K I N G
===================
This is where most of the work is done. The main.py file will run this code using instructions from the UI.
Input files can be Nikon .nd2 files or .avi videos.

Device assumptions:
    Cells flow from left to right through the channel.
    The channel splits into two outlets on the right of the image, at the outlet divider.
        Top outlet: positive DEP response (DEP_response = True)
        Bottom outlet: no positive DEP response (DEP_response = False)

How cells are found (compare each image with its neighbours):
    For every image, a background is made from the median of the `background_window` images on either side
    of it. Anything that stays in place for that long (dust, cells stuck to the device, the electrodes) is
    part of the background, so it disappears when the background is subtracted. What is left are moving
    specks: places where the image differs from its background by more than `detection_threshold` grey
    levels, with an area between `min_area` and `max_area` pixels.

How cells are tracked:
    Each image, every speck is paired with at most one existing track (and every track with at most one
    speck), closest pairs first, measured from where each track is predicted to be given its last velocity.
    A track can move at most `max_centroid_distance` pixels from that prediction between images, and is
    closed after `timeout` images without a match.

How cells are counted:
    A cell is counted once, when its track crosses the vertical counting line (x = `count_line_x`) moving
    right. If it crosses above the outlet divider (y < `outlet_divider_y`) it is going to the top outlet
    (pDEP), otherwise to the bottom outlet (nDEP). Tracks that never cross the line (cells held on the
    electrodes, noise) are not counted.

If you decide to save overlays, you'll get a .png per frame in the overlay folder.
    hint: ffmpeg

Read more:
https://github.com/bschelske/COUNT
"""
import collections
import contextlib
import csv
import os
import typing

import cv2 as cv
import numpy as np
from tqdm import tqdm

# Default value for every tracking setting. COUNT/settings.json and the settings window override these.
DEFAULT_SETTINGS = {
    "background_window": 10,      # images on each side used to build each image's background
    "detection_threshold": 8,     # grey levels a speck must differ from its background by
    "min_area": 6,                # smallest speck counted as a cell (pixels)
    "max_area": 600,              # largest speck counted as a cell (pixels)
    "max_centroid_distance": 100,  # furthest a cell can move between images (pixels, from its predicted spot)
    "timeout": 3,                 # images a track can go undetected before it is closed
    "cell_radius": 8,             # how far a cell may jitter left between images and still match (pixels)
    "count_line_x": 0,            # x of the counting line (pixels); 0 = three quarters of the image width
    "outlet_divider_y": 0,        # y of the outlet divider (pixels); 0 = the middle of the image
}
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
    position : Tuple[float, float]
        The (x, y) top-left corner of the object's bounding box in its most recent frame.
    size : Tuple[float, float]
        The (w, h) size of the bounding box in its most recent frame.
    most_recent_frame : int
        The most recent frame number where the object was detected.
    DEP_outlet : bool
        True: the cell crossed the counting line above the outlet divider (top outlet, positive DEP).
        False: below it (bottom outlet). '' until the cell is counted.
    frames_tracked : int
        The number of frames in which the object was detected.
    position_history : Dict[int, Tuple[float, float]]
        {frame_number: (x, y)} top-left corner for every frame the object was detected in.
    center_history : Dict[int, Tuple[float, float]]
        {frame_number: (x, y)} centre for every frame the object was detected in.
    velocity : Tuple[float, float]
        Centre displacement per frame between the last two detections ((0, 0) until it has been seen twice).
    count_frame, count_y :
        The frame in which the cell crossed the counting line and the y where it crossed (None until counted).
    """

    def __init__(self, object_id, position, size, most_recent_frame):
        self.object_id = object_id
        self.position = position
        self.size = size
        self.most_recent_frame = most_recent_frame
        self.DEP_outlet = ''
        self.frames_tracked = 1
        self.position_history = {self.most_recent_frame: self.position}
        self.center_history = {self.most_recent_frame: self.center()}
        self.velocity = (0.0, 0.0)
        self.count_frame = None
        self.count_y = None

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
        self.center_history[frame_number] = detection.center()

    def line_crossing(self, line_x: float) -> typing.Optional[float]:
        """
        If the last step of this track crossed the vertical line x = line_x moving right, return the y where it
        crossed. Otherwise None.
        """
        if len(self.center_history) < 2:
            return None
        (_, (x0, y0)), (_, (x1, y1)) = sorted(self.center_history.items())[-2:]
        if x0 < line_x <= x1:
            return y0 + (y1 - y0) * (line_x - x0) / (x1 - x0)
        return None


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
    """
    Links moving specks into tracks and counts each track once, when it crosses the counting line.

    `counted` holds every counted cell ({object_id: DetectedObject}); `counted_this_frame` lists the cells
    counted in the most recent update (used to draw overlays).
    """

    def __init__(self, count_line_x: float, outlet_divider_y: float, max_centroid_distance: float, timeout: int,
                 backtrack_tolerance: float = 0):
        self.count_line_x = count_line_x
        self.outlet_divider_y = outlet_divider_y
        self.max_centroid_distance = max_centroid_distance
        self.timeout = timeout
        self.backtrack_tolerance = backtrack_tolerance
        self.active = {}  # {object_id: DetectedObject} tracks that may still receive detections
        self.counted = {}  # {object_id: DetectedObject} cells that crossed the counting line
        self.counted_this_frame = []
        self.next_id = 1

    def update(self, detections: typing.List[DetectedObject], frame_number: int) -> None:
        for track_id, track in list(self.active.items()):
            if (frame_number - track.most_recent_frame) > self.timeout:
                del self.active[track_id]
        unmatched = match_detections(self.active, detections, frame_number, self.max_centroid_distance,
                                     self.backtrack_tolerance)

        self.counted_this_frame = []
        for track_id, track in self.active.items():
            if track_id in self.counted or track.most_recent_frame != frame_number:
                continue
            y = track.line_crossing(self.count_line_x)
            if y is not None:
                track.count_frame, track.count_y = frame_number, y
                track.DEP_outlet = bool(y < self.outlet_divider_y)
                self.counted[track_id] = track
                self.counted_this_frame.append(track)

        for detection in unmatched:
            detection.object_id = self.next_id
            detection.most_recent_frame = frame_number
            self.active[self.next_id] = detection
            self.next_id += 1

    def finish(self) -> typing.Dict[int, DetectedObject]:
        """End of the video: return the counted cells."""
        self.active = {}
        return self.counted

    def outlet_counts(self) -> typing.Tuple[int, int]:
        """(bottom outlet / nDEP, top outlet / pDEP) cells counted so far."""
        top = sum(1 for cell in self.counted.values() if cell.DEP_outlet is True)
        return len(self.counted) - top, top


def _setting(ui_app, name, default=None):
    """Read a setting from the UI (a tkinter variable), falling back to COUNT's default."""
    if default is None:
        default = DEFAULT_SETTINGS.get(name)
    variable = getattr(ui_app, name, None)
    return variable.get() if variable is not None else default


def count_line_and_divider(ui_app, image_shape) -> typing.Tuple[float, float]:
    """The counting line's x and the outlet divider's y in pixels (0 in the settings = automatic)."""
    height, width = image_shape[:2]
    line_x = _setting(ui_app, "count_line_x") or width * 3 / 4
    divider_y = _setting(ui_app, "outlet_divider_y") or height / 2
    return line_x, divider_y


def to_8bit(frame, scale: typing.Optional[typing.Tuple[float, float]]):
    """
    Convert a frame to 8-bit greyscale with a fixed (low, high) intensity range, so brightness is comparable
    between frames. 8-bit frames are returned unchanged.
    """
    if frame.dtype == np.uint8:
        return frame
    low, high = scale
    return np.clip((frame.astype(np.float32) - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8)


def intensity_scale(frame) -> typing.Optional[typing.Tuple[float, float]]:
    """The (low, high) range used to convert every frame of a video with more than 8 bits to 8-bit."""
    if frame.dtype == np.uint8:
        return None
    low, high = np.percentile(frame, (0.01, 99.99))
    if high - low < 1:
        low, high = float(frame.min()), float(frame.max())
    if high - low < 1:
        high = low + 1
    return float(low), float(high)


def find_moving_specks(frame, background, detection_threshold, min_area, max_area):
    """
    Specks that differ from their background: a list of (centre_x, centre_y, width, height, area, contrast).

    `frame` and `background` are 8-bit images of the same size. Anything also present in the background
    (because it stayed in place) is not found.
    """
    difference = cv.absdiff(frame, background)
    difference = cv.GaussianBlur(difference, (5, 5), 0)
    mask = (difference > detection_threshold).astype(np.uint8)
    n, labels, stats, centroids = cv.connectedComponentsWithStats(mask)
    specks = []
    for label in range(1, n):
        x, y, w, h, area = stats[label]
        if min_area <= area <= max_area:
            contrast = float(difference[y:y + h, x:x + w].max())
            specks.append((float(centroids[label][0]), float(centroids[label][1]), int(w), int(h), int(area),
                           contrast))
    return specks


def median_background(neighbours):
    """Per-pixel median (the lower one for an even count) of a list of 8-bit images."""
    stack = np.stack(neighbours)
    middle = (len(neighbours) - 1) // 2
    return np.partition(stack, middle, axis=0)[middle]


def backgrounds_by_frame(frames, n_frames: int, background_window: int, scale=None, indices=None):
    """
    For each frame (all, or just `indices`), yield (frame_number, 8-bit frame, background), where background
    is the median of the `background_window` frames on either side of it (None for a one-frame video).
    Frames are read in order and each is read once.
    """
    if scale is None and n_frames:
        scale = intensity_scale(frames[0])
    wanted = range(n_frames) if indices is None else sorted(set(indices))
    cache = collections.OrderedDict()

    def frame(j):
        if j not in cache:
            cache[j] = to_8bit(frames[j], scale)
        return cache[j]

    for i in wanted:
        lo, hi = max(0, i - background_window), min(n_frames, i + background_window + 1)
        for j in [j for j in cache if j < lo]:
            del cache[j]
        current = frame(i)
        neighbours = [frame(j) for j in range(lo, hi) if j != i]
        yield i, current, (median_background(neighbours) if neighbours else None)


def moving_specks_by_frame(frames, n_frames: int, background_window: int, detection_threshold: float,
                           min_area: int, max_area: int, scale=None, indices=None):
    """
    For each frame (all, or just `indices`), yield (frame_number, 8-bit frame, specks) where specks are the
    moving specks found by comparing the frame with the median of the `background_window` frames on either
    side of it.
    """
    for i, current, background in backgrounds_by_frame(frames, n_frames, background_window, scale, indices):
        specks = [] if background is None else find_moving_specks(current, background, detection_threshold,
                                                                  min_area, max_area)
        yield i, current, specks


def specks_to_objects(specks, frame_number: int) -> typing.List[DetectedObject]:
    return [DetectedObject(object_id=None, position=(cx - w / 2, cy - h / 2), size=(w, h),
                           most_recent_frame=frame_number) for cx, cy, w, h, _, _ in specks]


def draw_overlay(frame_8u, specks, tracker: typing.Optional[Tracker], frame_number: int, count_line_x: float,
                 outlet_divider_y: float):
    """
    A colour copy of the frame showing the counting line (blue), the outlet divider (grey), every moving speck
    (orange), cells counted in this frame (green = top/pDEP, pink = bottom/nDEP) and the running counts.
    """
    overlay = cv.cvtColor(frame_8u, cv.COLOR_GRAY2BGR)
    height, width = overlay.shape[:2]
    s = max(1, round(width / 1024))  # line thickness / font scale for the image size
    line_x, divider_y = int(round(count_line_x)), int(round(outlet_divider_y))

    cv.line(overlay, (line_x, 0), (line_x, height - 1), (255, 200, 0), s)
    cv.line(overlay, (line_x, divider_y), (width - 1, divider_y), (160, 160, 160), s)
    for cx, cy, w, h, _, _ in specks:
        cv.circle(overlay, (int(cx), int(cy)), 8 * s, (0, 165, 255), s)

    header = f"Frame {frame_number}   moving specks: {len(specks)}"
    if tracker is not None:
        for cell in tracker.counted_this_frame:
            colour = (0, 255, 0) if cell.DEP_outlet else (255, 0, 255)
            cv.circle(overlay, (line_x, int(cell.count_y)), 16 * s, colour, 2 * s)
        bottom, top = tracker.outlet_counts()
        header += f"   counted  nDEP (bottom): {bottom}   pDEP (top): {top}"
    cv.rectangle(overlay, (0, 0), (width - 1, 22 * s), (0, 0, 0), -1)
    cv.putText(overlay, header, (6 * s, 16 * s), cv.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), s, cv.LINE_AA)
    return overlay


def count_cells(video_path: str, ui_app) -> typing.Dict[int, DetectedObject]:
    """
    Count the cells in an ND2 or AVI file.

    Args:
        video_path (str): Path to the .nd2 or .avi file.
        ui_app: User interface application instance with the tracking settings.

    Returns:
        {object_id: DetectedObject} for every counted cell, with DEP_outlet set from where it crossed the
        counting line. Each object's position_history holds its trajectory.
    """
    save_overlay = bool(_setting(ui_app, "save_overlay", False))
    file_stem = os.path.splitext(os.path.basename(video_path))[0]
    if save_overlay:
        os.makedirs(ui_app.overlay_path, exist_ok=True)
        print(f"\noverlay saving in {ui_app.overlay_path}")

    tracker = None
    with open_video(video_path) as frames:
        n_frames = len(frames)
        specks_by_frame = moving_specks_by_frame(
            frames, n_frames, _setting(ui_app, "background_window"), _setting(ui_app, "detection_threshold"),
            _setting(ui_app, "min_area"), _setting(ui_app, "max_area"))
        for frame_number, frame_8u, specks in tqdm(specks_by_frame, total=n_frames, desc=file_stem):
            if tracker is None:
                line_x, divider_y = count_line_and_divider(ui_app, frame_8u.shape)
                tracker = Tracker(count_line_x=line_x, outlet_divider_y=divider_y,
                                  max_centroid_distance=_setting(ui_app, "max_centroid_distance"),
                                  timeout=_setting(ui_app, "timeout"),
                                  backtrack_tolerance=_setting(ui_app, "cell_radius"))
            tracker.update(specks_to_objects(specks, frame_number), frame_number)

            if save_overlay:
                overlay = draw_overlay(frame_8u, specks, tracker, frame_number, line_x, divider_y)
                cv.imwrite(os.path.join(ui_app.overlay_path, f"{file_stem}_{frame_number:05d}.png"), overlay)

    if tracker is None:  # empty file
        return {}
    return tracker.finish()


# Older name, kept so existing scripts keep working
nd2_mog_contours = count_cells


def preview_frames(video_path: str, ui_app, n_images: int = 2):
    """
    Overlay images (see draw_overlay) for `n_images` consecutive frames from the middle of a video, showing
    what will be detected with the current settings and where the counting line and outlet divider are.
    """
    with open_video(video_path) as frames:
        n_frames = len(frames)
        first = max(0, n_frames // 2 - n_images // 2)
        indices = list(range(first, min(n_frames, first + n_images)))
        images = []
        for frame_number, frame_8u, specks in moving_specks_by_frame(
                frames, n_frames, _setting(ui_app, "background_window"), _setting(ui_app, "detection_threshold"),
                _setting(ui_app, "min_area"), _setting(ui_app, "max_area"), indices=indices):
            line_x, divider_y = count_line_and_divider(ui_app, frame_8u.shape)
            images.append(draw_overlay(frame_8u, specks, None, frame_number, line_x, divider_y))
    return images


def export_to_csv(counted_objects: typing.Dict[int, DetectedObject], csv_filename: str) -> None:
    with open(csv_filename, 'w', newline='') as csvfile:
        fieldnames = ['object_id', 'x_pos', 'y_pos', 'x_size', 'y_size', 'most_recent_frame', 'frames_tracked',
                      'DEP_response', 'count_frame', 'y_at_count_line']
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
                'DEP_response': obj.DEP_outlet,
                'count_frame': obj.count_frame,
                'y_at_count_line': None if obj.count_y is None else round(obj.count_y, 1)
            })
        print("\nObject Detection Results:")
        print(f"\tCells counted: {len(counted_objects)}")
        print(f"\tDEP True (top outlet): {DEP_true}")
        print(f"\tDEP False (bottom outlet): {DEP_false}")
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
