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
    right and it has been detected in at least `min_detections` images (so two noise specks that happen to
    line up either side of the line are not counted). If it crosses above the outlet divider
    (y < `outlet_divider_y`) it is going to the top outlet (pDEP), otherwise to the bottom outlet (nDEP).
    Tracks that never cross the line (cells held on the electrodes, noise) are not counted.

Speed:
    Only the processing region is analysed: from `region_before_line` pixels left of the counting line to the
    right edge of the image, between rows `region_top` and `region_bottom`. Cells are only counted at the
    line, so the rest of the image (the electrodes, the area outside the channel) does not need processing.
    The video is split into chunks of frames that are processed on separate processor cores (`workers`), and
    the moving specks found are then tracked in order.

If you decide to save overlays, you'll get a .png per frame in the overlay folder.
    hint: ffmpeg

Read more:
https://github.com/bschelske/COUNT
"""
import collections
import concurrent.futures
import csv
import math
import multiprocessing
import os
import typing

import cv2 as cv
import numpy as np
from tqdm import tqdm

# File reading lives in video.py; these names are kept here so existing scripts keep working
from COUNT.video import (VIDEO_EXTENSIONS, AviReader, RawAviReader, frame_shape, is_supported_video,  # noqa: F401
                         open_video, read_region)

# Default value for every tracking setting. COUNT/settings.json and the settings window override these.
DEFAULT_SETTINGS = {
    "background_window": 10,      # images on each side used to build each image's background
    "detection_threshold": 8,     # grey levels a speck must differ from its background by
    "min_area": 6,                # smallest speck counted as a cell (pixels)
    "max_area": 600,              # largest speck counted as a cell (pixels)
    "max_centroid_distance": 100,  # furthest a cell can move between images (pixels, from its predicted spot)
    "timeout": 3,                 # images a track can go undetected before it is closed
    "cell_radius": 8,             # how far a cell may jitter left between images and still match (pixels)
    "min_detections": 3,          # a cell must be detected in at least this many images to be counted
    "count_line_x": 0,            # x of the counting line (pixels); 0 = three quarters of the image width
    "outlet_divider_y": 0,        # y of the outlet divider (pixels); 0 = the middle of the image
    "region_before_line": 600,    # process from this many pixels left of the counting line (0 = left edge)
    "region_top": 0,              # first image row to process (pixels)
    "region_bottom": 0,           # last image row to process (pixels); 0 = the bottom of the image
    "workers": 0,                 # processor cores to use; 0 = all of them
}
MIN_CHUNK_FRAMES = 40  # frames per piece of work when using several processor cores
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

    def line_crossing(self, line_x: float) -> typing.Optional[typing.Tuple[int, float]]:
        """
        If this track has crossed the vertical line x = line_x moving right, return (frame, y) for the first
        crossing: the frame of the first detection past the line, and the y where the track crossed it.
        Otherwise None.
        """
        points = sorted(self.center_history.items())
        for (_, (x0, y0)), (frame, (x1, y1)) in zip(points, points[1:]):
            if x0 < line_x <= x1:
                return frame, y0 + (y1 - y0) * (line_x - x0) / (x1 - x0)
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
                 backtrack_tolerance: float = 0, min_detections: int = 2):
        self.count_line_x = count_line_x
        self.min_detections = max(2, min_detections)
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
            if (track_id in self.counted or track.most_recent_frame != frame_number
                    or track.frames_tracked < self.min_detections):
                continue
            crossing = track.line_crossing(self.count_line_x)
            if crossing is not None:
                track.count_frame, y = crossing
                track.count_y = y
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


def processing_region(ui_app, image_shape, count_line_x: float) -> typing.Tuple[int, int, int, int]:
    """
    The part of the image to analyse, as (x0, y0, x1, y1): from `region_before_line` pixels left of the counting
    line to the right edge, between rows `region_top` and `region_bottom`.
    """
    height, width = image_shape[:2]
    before = _setting(ui_app, "region_before_line")
    x0 = int(max(0, math.floor(count_line_x - before))) if before > 0 else 0
    y0 = int(min(max(0, _setting(ui_app, "region_top")), height - 1))
    bottom = _setting(ui_app, "region_bottom")
    y1 = int(min(height, bottom)) if bottom > 0 else height
    if y1 <= y0:
        raise ValueError(f"The processing region's bottom row ({y1}) must be below its top row ({y0}).")
    if x0 >= width:
        raise ValueError(f"The processing region starts at x={x0}, beyond the image width ({width}).")
    return x0, y0, width, y1


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

    `frame` is an 8-bit image and `background` an image of the same size. Anything also present in the
    background (because it stayed in place) is not found. The difference is kept in fractions of a grey level
    (not rounded), so faint cells just above the threshold are not lost.
    """
    difference = cv.absdiff(frame.astype(np.float32), background.astype(np.float32))
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
    """Per-pixel median of a list of 8-bit images, as float32 (the average of the two middle values for an even
    count)."""
    stack = np.stack(neighbours)
    middle = (len(neighbours) - 1) // 2
    if len(neighbours) % 2:
        return np.partition(stack, middle, axis=0)[middle].astype(np.float32)
    ordered = np.partition(stack, (middle, middle + 1), axis=0)
    return (ordered[middle].astype(np.float32) + ordered[middle + 1]) / 2


def backgrounds_by_frame(frames, n_frames: int, background_window: int, scale=None, indices=None, region=None):
    """
    For each frame (all, or just `indices`), yield (frame_number, 8-bit frame, background), where background
    is the median of the `background_window` frames on either side of it (None for a one-frame video).
    With a region (x0, y0, x1, y1), only that part of each frame is read and returned.
    Frames are read in order and each is read once.
    """
    if scale is None and n_frames:
        scale = intensity_scale(read_region(frames, 0, region))
    wanted = range(n_frames) if indices is None else sorted(set(indices))
    cache = collections.OrderedDict()

    def frame(j):
        if j not in cache:
            cache[j] = to_8bit(read_region(frames, j, region), scale)
        return cache[j]

    for i in wanted:
        lo, hi = max(0, i - background_window), min(n_frames, i + background_window + 1)
        for j in [j for j in cache if j < lo]:
            del cache[j]
        current = frame(i)
        neighbours = [frame(j) for j in range(lo, hi) if j != i]
        yield i, current, (median_background(neighbours) if neighbours else None)


def moving_specks_by_frame(frames, n_frames: int, background_window: int, detection_threshold: float,
                           min_area: int, max_area: int, scale=None, indices=None, region=None):
    """
    For each frame (all, or just `indices`), yield (frame_number, 8-bit frame, specks) where specks are the
    moving specks found by comparing the frame with the median of the `background_window` frames on either
    side of it. With a region (x0, y0, x1, y1), only that part of each frame is analysed (and returned), and
    speck positions are still in whole-image pixels.
    """
    x0, y0 = (region[0], region[1]) if region else (0, 0)
    for i, current, background in backgrounds_by_frame(frames, n_frames, background_window, scale, indices, region):
        specks = [] if background is None else find_moving_specks(current, background, detection_threshold,
                                                                  min_area, max_area)
        if x0 or y0:
            specks = [(cx + x0, cy + y0, w, h, area, contrast) for cx, cy, w, h, area, contrast in specks]
        yield i, current, specks


def _specks_for_frames(path, indices, settings, region, scale):
    """Moving specks for some frames of a video: [(frame_number, specks)]. Runs in a worker process."""
    with open_video(path) as frames:
        return [(i, specks) for i, _, specks in moving_specks_by_frame(
            frames, len(frames), settings["background_window"], settings["detection_threshold"],
            settings["min_area"], settings["max_area"], scale=scale, indices=indices, region=region)]


def find_all_specks(path: str, n_frames: int, settings: dict, region, scale, workers: int = 1,
                    progress: typing.Optional[str] = None):
    """
    The moving specks in every frame of a video: a list with one list of specks per frame.

    With more than one worker, the frames are split into chunks that are processed on separate processor
    cores. The result is the same either way.
    """
    workers = workers if workers and workers > 0 else (os.cpu_count() or 1)
    n_chunks = min(workers * 3, n_frames // MIN_CHUNK_FRAMES)
    bar = tqdm(total=n_frames, desc=progress, disable=progress is None)
    specks = [None] * n_frames
    if workers <= 1 or n_chunks < 2:
        with open_video(path) as frames:
            for i, _, found in moving_specks_by_frame(
                    frames, n_frames, settings["background_window"], settings["detection_threshold"],
                    settings["min_area"], settings["max_area"], scale=scale, region=region):
                specks[i] = found
                bar.update(1)
    else:
        bounds = np.linspace(0, n_frames, n_chunks + 1).astype(int)
        chunks = [range(a, b) for a, b in zip(bounds[:-1], bounds[1:])]
        # "spawn" starts fresh processes the same way on Windows, macOS and Linux
        with concurrent.futures.ProcessPoolExecutor(max_workers=min(workers, n_chunks),
                                                    mp_context=multiprocessing.get_context("spawn")) as pool:
            futures = {pool.submit(_specks_for_frames, path, chunk, settings, region, scale): chunk
                       for chunk in chunks}
            for future in concurrent.futures.as_completed(futures):
                for i, found in future.result():
                    specks[i] = found
                bar.update(len(futures[future]))
    bar.close()
    return specks


def specks_to_objects(specks, frame_number: int) -> typing.List[DetectedObject]:
    return [DetectedObject(object_id=None, position=(cx - w / 2, cy - h / 2), size=(w, h),
                           most_recent_frame=frame_number) for cx, cy, w, h, _, _ in specks]


def draw_overlay(frame_8u, specks, frame_number: int, count_line_x: float, outlet_divider_y: float,
                 counted=(), counts=None, region=None):
    """
    A colour copy of the whole frame showing the processing region (dark green box), the counting line (blue),
    the outlet divider (grey), every moving speck (orange), the cells counted in this frame and the running
    counts.

    counted: (count_y, top_outlet, (x, y)) for each cell counted in this frame (green = top/pDEP,
    pink = bottom/nDEP). counts: (bottom, top) counted so far, or None to leave the counts out.
    """
    overlay = cv.cvtColor(frame_8u, cv.COLOR_GRAY2BGR)
    height, width = overlay.shape[:2]
    s = max(1, round(width / 1024))  # line thickness / font scale for the image size
    line_x, divider_y = int(round(count_line_x)), int(round(outlet_divider_y))

    if region is not None:
        x0, y0, x1, y1 = region
        cv.rectangle(overlay, (x0, y0), (x1 - 1, y1 - 1), (0, 140, 0), s)
    cv.line(overlay, (line_x, 0), (line_x, height - 1), (255, 200, 0), s)
    cv.line(overlay, (line_x, divider_y), (width - 1, divider_y), (160, 160, 160), s)
    for cx, cy, w, h, _, _ in specks:
        cv.circle(overlay, (int(cx), int(cy)), 8 * s, (0, 165, 255), s)

    header = f"Frame {frame_number}   moving specks: {len(specks)}"
    for count_y, top_outlet, (x, y) in counted:
        colour = (0, 255, 0) if top_outlet else (255, 0, 255)
        cv.circle(overlay, (line_x, int(count_y)), 16 * s, colour, 2 * s)
        cv.circle(overlay, (int(x), int(y)), 10 * s, colour, s)
    if counts is not None:
        header += f"   counted  nDEP (bottom): {counts[0]}   pDEP (top): {counts[1]}"
    cv.rectangle(overlay, (0, 0), (width - 1, 22 * s), (0, 0, 0), -1)
    cv.putText(overlay, header, (6 * s, 16 * s), cv.FONT_HERSHEY_SIMPLEX, 0.5 * s, (255, 255, 255), s, cv.LINE_AA)
    return overlay


def tracking_settings(ui_app) -> dict:
    """Every tracking setting's current value, as plain numbers (which can be sent to worker processes)."""
    return {name: _setting(ui_app, name) for name in DEFAULT_SETTINGS}


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
    settings = tracking_settings(ui_app)
    save_overlay = bool(_setting(ui_app, "save_overlay", False))
    file_stem = os.path.splitext(os.path.basename(video_path))[0]

    with open_video(video_path) as frames:
        n_frames = len(frames)
        if n_frames == 0:
            return {}
        shape = frame_shape(frames)
        line_x, divider_y = count_line_and_divider(ui_app, shape)
        region = processing_region(ui_app, shape, line_x)
        scale = intensity_scale(read_region(frames, 0, region))

    # 1. Find the moving specks in every frame (on several processor cores)
    specks_by_frame = find_all_specks(video_path, n_frames, settings, region, scale, settings["workers"],
                                      progress=file_stem)

    # 2. Track them in order and count cells at the line
    tracker = Tracker(count_line_x=line_x, outlet_divider_y=divider_y,
                      max_centroid_distance=settings["max_centroid_distance"], timeout=settings["timeout"],
                      backtrack_tolerance=settings["cell_radius"], min_detections=settings["min_detections"])
    overlay_notes = []
    for frame_number, specks in enumerate(specks_by_frame):
        tracker.update(specks_to_objects(specks, frame_number), frame_number)
        if save_overlay:
            overlay_notes.append(([(c.count_y, c.DEP_outlet, c.center()) for c in tracker.counted_this_frame],
                                  tracker.outlet_counts()))

    # 3. Optionally save an overlay image of every frame
    if save_overlay:
        os.makedirs(ui_app.overlay_path, exist_ok=True)
        print(f"\noverlay saving in {ui_app.overlay_path}")
        with open_video(video_path) as frames:
            for frame_number in tqdm(range(n_frames), desc=f"{file_stem} overlays"):
                counted, counts = overlay_notes[frame_number]
                overlay = draw_overlay(to_8bit(read_region(frames, frame_number), scale),
                                       specks_by_frame[frame_number], frame_number, line_x, divider_y,
                                       counted=counted, counts=counts, region=region)
                cv.imwrite(os.path.join(ui_app.overlay_path, f"{file_stem}_{frame_number:05d}.png"), overlay)

    return tracker.finish()


# Older name, kept so existing scripts keep working
nd2_mog_contours = count_cells


def preview_frames(video_path: str, ui_app, n_images: int = 2):
    """
    Overlay images (see draw_overlay) for `n_images` consecutive frames from the middle of a video, showing
    what will be detected with the current settings, the processing region, and where the counting line and
    outlet divider are.
    """
    settings = tracking_settings(ui_app)
    with open_video(video_path) as frames:
        n_frames = len(frames)
        if n_frames == 0:
            return []
        shape = frame_shape(frames)
        line_x, divider_y = count_line_and_divider(ui_app, shape)
        region = processing_region(ui_app, shape, line_x)
        scale = intensity_scale(read_region(frames, 0, region))
        first = max(0, n_frames // 2 - n_images // 2)
        indices = list(range(first, min(n_frames, first + n_images)))
        images = []
        for frame_number, _, specks in moving_specks_by_frame(
                frames, n_frames, settings["background_window"], settings["detection_threshold"],
                settings["min_area"], settings["max_area"], scale=scale, indices=indices, region=region):
            whole_frame = to_8bit(read_region(frames, frame_number), scale)
            images.append(draw_overlay(whole_frame, specks, frame_number, line_x, divider_y, region=region))
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
