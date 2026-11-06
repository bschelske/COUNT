"""
Tests for COUNT/tracking.py.

These run without the Nikon SDK, a display, or any .nd2 files: they use synthetic detections and synthetic
videos (bright discs moving left to right on a grey background).

Run from the repository root:
    python -m unittest discover COUNT/tests
"""
import csv
import os
import sys
import tempfile
import types
import unittest

import cv2 as cv
import numpy as np

from COUNT import tracking
from COUNT.tracking import DetectedObject, Tracker, calculate_distance, match_detections


class Var:
    """Stand-in for a tkinter variable."""

    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value


class FakeUI:
    """Stand-in for the tkinter UI with the settings the tracker reads."""

    def __init__(self, overlay_path="", save_overlay=False, **overrides):
        settings = dict(canny_upper=255, canny_lower=85, max_centroid_distance=70, timeout=20, cell_radius=6,
                        min_frames_tracked=3)
        settings.update(overrides)
        for name, value in settings.items():
            setattr(self, name, Var(value))
        self.save_overlay = Var(save_overlay)
        self.overlay_path = overlay_path


def obj(x, y, frame, size=(10, 10)):
    """A detection whose bounding box starts at (x, y)."""
    return DetectedObject(object_id=None, position=(x, y), size=size, most_recent_frame=frame)


def run_tracker(frames, image_h=200, **kwargs):
    """frames: list of lists of (x, y) top-left positions, one list per frame."""
    settings = dict(max_centroid_distance=70, timeout=20, min_frames_tracked=3, backtrack_tolerance=6)
    settings.update(kwargs)
    tracker = Tracker(image_h=image_h, **settings)
    for frame_number, positions in enumerate(frames):
        tracker.update([obj(x, y, frame_number) for x, y in positions], frame_number)
    return tracker, tracker.finish()


class TestDetectedObject(unittest.TestCase):
    def test_center(self):
        self.assertEqual(obj(10, 20, 0, size=(10, 6)).center(), (15, 23))

    def test_calculate_distance(self):
        self.assertEqual(calculate_distance(obj(0, 0, 0), obj(3, 4, 0)), 5)

    def test_outlet_uses_centre_not_top_edge(self):
        # Top edge (y=95) is above the midline (100) but the centre (y=105) is below it: bottom outlet.
        cell = obj(0, 95, 0, size=(10, 20))
        cell.outlet_assignment(image_h=200)
        self.assertFalse(cell.DEP_outlet)

        cell = obj(0, 80, 0, size=(10, 10))
        cell.outlet_assignment(image_h=200)
        self.assertTrue(cell.DEP_outlet)

    def test_update_records_history(self):
        cell = obj(0, 0, 0)
        cell.update(obj(20, 0, 1), 1)
        self.assertEqual(cell.frames_tracked, 2)
        self.assertEqual(cell.position_history, {0: (0, 0), 1: (20, 0)})


class TestMatching(unittest.TestCase):
    def test_each_track_takes_at_most_one_detection(self):
        tracks = {1: obj(0, 0, 0)}
        detections = [obj(20, 0, 1), obj(25, 5, 1)]
        unmatched = match_detections(tracks, detections, 1, max_distance=70, backtrack_tolerance=6)
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(tracks[1].position, (20, 0))  # the closer detection wins

    def test_closest_pairs_win(self):
        tracks = {1: obj(0, 0, 0), 2: obj(0, 50, 0)}
        detections = [obj(20, 52, 1), obj(20, 2, 1)]
        unmatched = match_detections(tracks, detections, 1, max_distance=70, backtrack_tolerance=6)
        self.assertEqual(unmatched, [])
        self.assertEqual(tracks[1].position, (20, 2))
        self.assertEqual(tracks[2].position, (20, 52))

    def test_moving_left_beyond_tolerance_does_not_match(self):
        tracks = {1: obj(50, 0, 0)}
        unmatched = match_detections(tracks, [obj(40, 0, 1)], 1, max_distance=70, backtrack_tolerance=6)
        self.assertEqual(len(unmatched), 1)

    def test_lost_track_does_not_steal_a_later_cell(self):
        # Track 1 was moving right at 30 px/frame, then was not detected for 5 frames. A different cell now
        # passes close to where track 1 was last seen. Track 1 should be predicted ~150 px further on, so it
        # must not take that detection.
        lost = obj(100, 50, 0)
        lost.update(obj(130, 50, 1), 1)
        tracks = {1: lost}
        unmatched = match_detections(tracks, [obj(140, 52, 6)], 6, max_distance=70, backtrack_tolerance=6)
        self.assertEqual(len(unmatched), 1)
        self.assertEqual(tracks[1].most_recent_frame, 1)

    def test_lost_track_is_found_further_along(self):
        lost = obj(100, 50, 0)
        lost.update(obj(130, 50, 1), 1)
        tracks = {1: lost}
        unmatched = match_detections(tracks, [obj(220, 50, 4)], 4, max_distance=70, backtrack_tolerance=6)
        self.assertEqual(unmatched, [])
        self.assertEqual(tracks[1].most_recent_frame, 4)

    def test_too_far_does_not_match(self):
        tracks = {1: obj(0, 0, 0)}
        unmatched = match_detections(tracks, [obj(80, 0, 1)], 1, max_distance=70, backtrack_tolerance=6)
        self.assertEqual(len(unmatched), 1)


class TestTracker(unittest.TestCase):
    def test_two_cells_in_same_frame_stay_separate(self):
        # Previously the second cell was merged into the first cell's brand-new track.
        frames = [[(100 + 15 * i, 50), (120 + 15 * i, 50)] for i in range(5)]
        _, counted = run_tracker(frames)
        self.assertEqual(len(counted), 2)

    def test_paused_cell_is_one_track(self):
        # Previously a cell that did not move for a frame started a duplicate track.
        frames = [[(100, 50)], [(100, 50)], [(101, 50)], [(130, 50)], [(160, 50)]]
        _, counted = run_tracker(frames)
        self.assertEqual(len(counted), 1)
        self.assertEqual(next(iter(counted.values())).frames_tracked, 5)

    def test_fast_cell_counted_even_if_seen_fewer_frames_than_timeout(self):
        # Previously a cell had to be tracked for more than `timeout` frames to be counted.
        frames = [[(10 + 60 * i, 50)] for i in range(4)]
        _, counted = run_tracker(frames, timeout=20, min_frames_tracked=3)
        self.assertEqual(len(counted), 1)

    def test_short_tracks_are_noise(self):
        frames = [[(10, 50)], [], [], [(150, 150)], [(160, 150)]]
        _, counted = run_tracker(frames, min_frames_tracked=3)
        self.assertEqual(counted, {})

    def test_track_survives_short_gap(self):
        frames = [[(10, 50)], [(40, 50)], [], [], [(70, 50)], [(100, 50)]]
        _, counted = run_tracker(frames, timeout=5)
        self.assertEqual(len(counted), 1)

    def test_track_closes_after_timeout(self):
        tracker = Tracker(image_h=200, max_centroid_distance=70, timeout=2, min_frames_tracked=3)
        for frame in range(3):
            tracker.update([obj(10 + 20 * frame, 50, frame)], frame)
        self.assertEqual(len(tracker.active), 1)
        for frame in range(3, 6):
            tracker.update([], frame)
        self.assertEqual(len(tracker.active), 0)
        self.assertEqual(len(tracker.counted), 1)

    def test_outlets_assigned_from_last_position(self):
        top = [(10 + 30 * i, 40) for i in range(5)]  # stays in the top half
        bottom = [(10 + 30 * i, 140) for i in range(5)]  # stays in the bottom half
        _, counted = run_tracker([[t, b] for t, b in zip(top, bottom)], image_h=200)
        self.assertEqual(sorted(o.DEP_outlet for o in counted.values()), [False, True])

    def test_counted_so_far(self):
        tracker = Tracker(image_h=200, max_centroid_distance=70, timeout=20, min_frames_tracked=3)
        for frame in range(2):
            tracker.update([obj(10 + 20 * frame, 50, frame)], frame)
        self.assertEqual(tracker.counted_so_far(), 0)
        tracker.update([obj(50, 50, 2)], 2)
        self.assertEqual(tracker.counted_so_far(), 1)


def synthetic_video(n_frames=30, size=(200, 300), cells=((0, 50), (-60, 150), (-120, 60)), speed=15, radius=6):
    """Bright discs moving right on a grey background. cells: (starting x, y) for each cell's centre."""
    frames = []
    for frame in range(n_frames):
        image = np.full(size, 100, dtype=np.uint16)
        for start_x, y in cells:
            x = start_x + speed * frame
            if 0 <= x < size[1]:
                cv.circle(image, (int(x), int(y)), radius, 400, -1)
        frames.append(image)
    return frames


class TestDetection(unittest.TestCase):
    def test_no_motion_means_no_detections(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        objects, contours = tracking.detect_objects(frame, 0, np.zeros_like(frame), FakeUI())
        self.assertEqual((objects, contours), ([], []))

    def test_primed_background_detects_from_the_start(self):
        # A cell already in view at frame 0 is part of the background the model first learns, so it is
        # picked up once it has moved off its starting spot (frame 1 at this speed), not after ~20 frames.
        frames = synthetic_video(n_frames=40, cells=((40, 60),), speed=10)
        backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        tracking.prime_background(backSub, frames, len(frames))
        backSub.apply(cv.normalize(frames[0], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))
        for index in (1, 2, 3):
            mask = backSub.apply(cv.normalize(frames[index], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))
            objects, _ = tracking.detect_objects(frames[index], index, mask, FakeUI())
            self.assertEqual(len(objects), 1, f"frame {index}")

    def test_detects_separate_cells(self):
        mask = np.zeros((200, 300), dtype=np.uint8)
        cv.circle(mask, (50, 50), 6, 255, -1)
        cv.circle(mask, (200, 150), 6, 255, -1)
        objects, contours = tracking.detect_objects(mask, 0, mask, FakeUI())
        self.assertEqual(len(objects), 2)
        self.assertEqual(len(contours), 2)


class FakeND2:
    """Stand-in for pims.ND2Reader_SDK backed by a list of frames."""

    frames = []

    def __init__(self, path):
        self.sizes = {'t': len(self.frames)}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, index):
        return self.frames[index]


class TestEndToEnd(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        FakeND2.frames = synthetic_video()
        self._saved_pims = sys.modules.get("pims")
        sys.modules["pims"] = types.SimpleNamespace(ND2Reader_SDK=FakeND2)

    def tearDown(self):
        if self._saved_pims is None:
            sys.modules.pop("pims", None)
        else:
            sys.modules["pims"] = self._saved_pims
        self.tmp.cleanup()

    def test_counts_and_outlets(self):
        counted = tracking.nd2_mog_contours("synthetic.nd2", FakeUI())
        self.assertEqual(len(counted), 3)
        self.assertEqual(sum(o.DEP_outlet is True for o in counted.values()), 2)
        self.assertEqual(sum(o.DEP_outlet is False for o in counted.values()), 1)

    def test_overlays_are_saved_with_annotations(self):
        overlay_path = os.path.join(self.tmp.name, "overlay")
        tracking.nd2_mog_contours(os.path.join("data", "trial_10kHz.nd2"),
                                  FakeUI(overlay_path=overlay_path, save_overlay=True))
        files = sorted(os.listdir(overlay_path))
        self.assertEqual(len(files), len(FakeND2.frames))
        self.assertEqual(files[0], "trial_10kHz_00000.png")

        # Every saved frame is the colour image (not the black-and-white mask) with the red header text on it
        for name in (files[0], files[-1]):
            image = cv.imread(os.path.join(overlay_path, name))
            red = (image[:, :, 2] > 200) & (image[:, :, 0] < 50) & (image[:, :, 1] < 50)
            self.assertTrue(red[:40].any(), name)

    def test_csv_exports(self):
        counted = tracking.nd2_mog_contours("synthetic.nd2", FakeUI())
        results = os.path.join(self.tmp.name, "results.csv")
        trajectories = os.path.join(self.tmp.name, "trajectories.csv")
        tracking.export_to_csv(counted, results)
        tracking.export_trajectories_to_csv(counted.values(), trajectories)

        with open(results, newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)

        with open(trajectories, newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), sum(o.frames_tracked for o in counted.values()))
        self.assertEqual({int(r['object_id']) for r in rows}, set(counted))


def write_avi(path, frames):
    """Save 16-bit synthetic frames as an 8-bit MJPG AVI."""
    height, width = frames[0].shape
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*"MJPG"), 30, (width, height), isColor=False)
    for frame in frames:
        writer.write(cv.normalize(frame, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U))
    writer.release()


class TestAviInput(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.frames = synthetic_video()
        self.path = os.path.join(self.tmp.name, "trial_20kHz.avi")
        write_avi(self.path, self.frames)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reader_gives_greyscale_frames_in_any_order(self):
        with tracking.open_video(self.path) as frames:
            self.assertEqual(len(frames), len(self.frames))
            first = frames[0]
            self.assertEqual(first.shape, self.frames[0].shape)  # greyscale, same size
            later = frames[10]
            self.assertTrue(np.array_equal(frames[0], first))  # going back re-reads the same frame
            self.assertTrue(np.array_equal(frames[10], later))
            self.assertTrue(np.array_equal(frames[-1], frames[len(frames) - 1]))
            with self.assertRaises(IndexError):
                frames[len(frames)]

    def test_counts_cells_in_avi(self):
        counted = tracking.nd2_mog_contours(self.path, FakeUI())
        self.assertEqual(len(counted), 3)
        self.assertEqual(sum(o.DEP_outlet is True for o in counted.values()), 2)

    def test_unsupported_file_type(self):
        with self.assertRaises(ValueError):
            with tracking.open_video(os.path.join(self.tmp.name, "movie.mp4")):
                pass

    def test_is_supported_video(self):
        self.assertTrue(tracking.is_supported_video("a.ND2"))
        self.assertTrue(tracking.is_supported_video("b.avi"))
        self.assertFalse(tracking.is_supported_video("c_results.csv"))


if __name__ == '__main__':
    unittest.main()
