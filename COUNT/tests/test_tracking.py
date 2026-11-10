"""
Tests for COUNT/tracking.py.

These run without the Nikon SDK, a display, or any .nd2 files: they use synthetic detections and synthetic
videos (discs moving left to right on a grey background, sometimes with stationary dust).

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
        settings = dict(tracking.DEFAULT_SETTINGS, max_centroid_distance=70, cell_radius=6)
        settings.update(overrides)
        for name, value in settings.items():
            setattr(self, name, Var(value))
        self.save_overlay = Var(save_overlay)
        self.overlay_path = overlay_path


def obj(x, y, frame, size=(10, 10)):
    """A detection whose bounding box starts at (x, y)."""
    return DetectedObject(object_id=None, position=(x, y), size=size, most_recent_frame=frame)


def run_tracker(frames, line_x=150, divider_y=100, **kwargs):
    """frames: list of lists of (x, y) top-left positions, one list per frame. Returns (tracker, counted)."""
    settings = dict(max_centroid_distance=70, timeout=3, backtrack_tolerance=6)
    settings.update(kwargs)
    tracker = Tracker(count_line_x=line_x, outlet_divider_y=divider_y, **settings)
    for frame_number, positions in enumerate(frames):
        tracker.update([obj(x, y, frame_number) for x, y in positions], frame_number)
    return tracker, tracker.finish()


class TestDetectedObject(unittest.TestCase):
    def test_center(self):
        self.assertEqual(obj(10, 20, 0, size=(10, 6)).center(), (15, 23))

    def test_calculate_distance(self):
        self.assertEqual(calculate_distance(obj(0, 0, 0), obj(3, 4, 0)), 5)

    def test_update_records_history(self):
        cell = obj(0, 0, 0)
        cell.update(obj(20, 0, 1), 1)
        self.assertEqual(cell.frames_tracked, 2)
        self.assertEqual(cell.position_history, {0: (0, 0), 1: (20, 0)})
        self.assertEqual(cell.center_history, {0: (5, 5), 1: (25, 5)})

    def test_line_crossing(self):
        cell = obj(90, 40, 0)  # centre (95, 45)
        self.assertIsNone(cell.line_crossing(100))  # only seen once
        cell.update(obj(110, 60, 1), 1)  # centre (115, 65)
        self.assertAlmostEqual(cell.line_crossing(100), 50)  # crossed x=100 a quarter of the way: y=50
        self.assertIsNone(cell.line_crossing(150))  # has not reached x=150
        self.assertIsNone(cell.line_crossing(90))  # was already past x=90


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
    def test_counted_once_when_crossing_the_line(self):
        tracker = Tracker(count_line_x=150, outlet_divider_y=100, max_centroid_distance=70, timeout=3)
        counts = []
        for frame, x in enumerate(range(60, 300, 30)):
            tracker.update([obj(x, 40, frame)], frame)
            counts.append(len(tracker.counted))
        self.assertEqual(counts, [0, 0, 0, 1, 1, 1, 1, 1])  # counted in the frame its centre passed x=150
        self.assertEqual(tracker.outlet_counts(), (0, 1))

    def test_outlet_decided_where_the_cell_crosses(self):
        # Starts above the divider (y=100) and ends below it, but crosses the line above it: top outlet
        frames = [[(100 + 30 * i, 60 + 20 * i)] for i in range(6)]
        _, counted = run_tracker(frames, line_x=150, divider_y=100)
        self.assertEqual([c.DEP_outlet for c in counted.values()], [True])
        self.assertEqual(next(iter(counted.values())).count_frame, 2)

    def test_two_cells_in_same_frame_stay_separate(self):
        frames = [[(100 + 15 * i, 50), (120 + 15 * i, 50)] for i in range(6)]
        _, counted = run_tracker(frames)
        self.assertEqual(len(counted), 2)

    def test_paused_cell_counted_once(self):
        frames = [[(100, 50)], [(100, 50)], [(101, 50)], [(130, 50)], [(160, 50)], [(190, 50)]]
        _, counted = run_tracker(frames)
        self.assertEqual(len(counted), 1)

    def test_track_survives_short_gap(self):
        frames = [[(80, 50)], [(110, 50)], [], [], [(200, 50)], [(230, 50)]]
        _, counted = run_tracker(frames, timeout=3, max_centroid_distance=40)
        self.assertEqual(len(counted), 1)

    def test_cells_that_never_cross_are_not_counted(self):
        # A cell that stays on the electrodes (left of the line), and one first seen right of the line
        frames = [[(40 + (i % 2), 50), (200 + 20 * i, 150)] for i in range(6)]
        _, counted = run_tracker(frames, line_x=150)
        self.assertEqual(counted, {})

    def test_moving_left_is_not_counted(self):
        frames = [[(200 - 30 * i, 50)] for i in range(6)]
        _, counted = run_tracker(frames, line_x=150, backtrack_tolerance=40)
        self.assertEqual(counted, {})

    def test_outlet_counts(self):
        top = [(60 + 30 * i, 40) for i in range(6)]
        bottom = [(60 + 30 * i, 140) for i in range(6)]
        tracker, _ = run_tracker([[t, b] for t, b in zip(top, bottom)], divider_y=100)
        self.assertEqual(tracker.outlet_counts(), (1, 1))


def synthetic_video(n_frames=30, size=(200, 300), cells=((0, 50), (-60, 150), (-120, 60)), speed=15, radius=6,
                    dust=(), cell_value=400):
    """
    Discs moving right on a grey (100) background. cells: (starting x, y) for each cell's centre.
    dust: (x, y) of stationary dark spots that never move.
    """
    frames = []
    for frame in range(n_frames):
        image = np.full(size, 100, dtype=np.uint16)
        for x, y in dust:
            cv.circle(image, (int(x), int(y)), 4, 20, -1)
        for start_x, y in cells:
            x = start_x + speed * frame
            if 0 <= x < size[1]:
                cv.circle(image, (int(x), int(y)), radius, cell_value, -1)
        frames.append(image)
    return frames


class TestDetection(unittest.TestCase):
    def test_no_difference_means_no_specks(self):
        frame = np.full((100, 100), 128, dtype=np.uint8)
        self.assertEqual(tracking.find_moving_specks(frame, frame.copy(), 8, 6, 600), [])

    def test_finds_specks_that_differ_from_background(self):
        background = np.full((200, 300), 120, dtype=np.uint8)
        frame = background.copy()
        cv.circle(frame, (50, 50), 4, 60, -1)  # darker cell
        cv.circle(frame, (200, 150), 4, 200, -1)  # brighter cell
        cv.circle(frame, (250, 30), 4, 124, -1)  # too faint to count
        specks = tracking.find_moving_specks(frame, background, 8, 6, 600)
        centres = sorted((round(cx), round(cy)) for cx, cy, *_ in specks)
        self.assertEqual(centres, [(50, 50), (200, 150)])

    def test_size_limits(self):
        background = np.full((200, 300), 120, dtype=np.uint8)
        frame = background.copy()
        cv.circle(frame, (50, 50), 4, 40, -1)
        cv.circle(frame, (200, 100), 40, 40, -1)  # far bigger than a cell
        specks = tracking.find_moving_specks(frame, background, 8, 6, 600)
        self.assertEqual(len(specks), 1)

    def test_stationary_dust_is_ignored(self):
        frames = synthetic_video(cells=((0, 50),), dust=((150, 100), (60, 170)))
        found = list(tracking.moving_specks_by_frame(frames, len(frames), 10, 8, 6, 600))
        self.assertEqual(len(found), len(frames))
        for frame_number, frame, specks in found:
            self.assertTrue(all(abs(cy - 50) < 10 for _, cy, *_ in specks), frame_number)  # only the moving cell
        self.assertGreater(sum(len(specks) for _, _, specks in found), 15)

    def test_16_bit_frames_use_one_intensity_scale(self):
        frames = synthetic_video(n_frames=3)
        scale = tracking.intensity_scale(frames[0])
        converted = [tracking.to_8bit(f, scale) for f in frames]
        self.assertTrue(all(c.dtype == np.uint8 for c in converted))
        self.assertTrue(all(c[199, 299] == converted[0][199, 299] for c in converted))  # background unchanged
        self.assertIsNone(tracking.intensity_scale(converted[0]))
        self.assertIs(tracking.to_8bit(converted[0], None), converted[0])

    def test_one_frame_video_has_no_specks(self):
        frames = synthetic_video(n_frames=1)
        self.assertEqual([specks for _, _, specks in tracking.moving_specks_by_frame(frames, 1, 10, 8, 6, 600)],
                         [[]])


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
        counted = tracking.count_cells("synthetic.nd2", FakeUI())
        self.assertEqual(len(counted), 3)
        self.assertEqual(sum(o.DEP_outlet is True for o in counted.values()), 2)
        self.assertEqual(sum(o.DEP_outlet is False for o in counted.values()), 1)

    def test_old_function_name_still_works(self):
        self.assertIs(tracking.nd2_mog_contours, tracking.count_cells)

    def test_dark_cells_with_dust(self):
        FakeND2.frames = synthetic_video(cell_value=30, dust=((150, 100), (60, 170), (240, 40)))
        counted = tracking.count_cells("synthetic.nd2", FakeUI())
        self.assertEqual(sorted(o.DEP_outlet for o in counted.values()), [False, True, True])

    def test_count_line_and_divider_settings(self):
        # Counting line beyond where the cells reach: nothing counted
        self.assertEqual(tracking.count_cells("synthetic.nd2", FakeUI(count_line_x=299)), {})
        # Divider above every cell: all counted as bottom outlet
        counted = tracking.count_cells("synthetic.nd2", FakeUI(outlet_divider_y=20))
        self.assertEqual([o.DEP_outlet for o in counted.values()], [False, False, False])

    def test_preview(self):
        images = tracking.preview_frames("synthetic.nd2", FakeUI(), n_images=2)
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0].shape, (200, 300, 3))

    def test_overlays_are_saved_with_annotations(self):
        overlay_path = os.path.join(self.tmp.name, "overlay")
        tracking.count_cells(os.path.join("data", "trial_10kHz.nd2"),
                                  FakeUI(overlay_path=overlay_path, save_overlay=True))
        files = sorted(os.listdir(overlay_path))
        self.assertEqual(len(files), len(FakeND2.frames))
        self.assertEqual(files[0], "trial_10kHz_00000.png")

        # Every saved frame is the colour image with the header text and the counting line (x=225) on it
        for name in (files[0], files[-1]):
            image = cv.imread(os.path.join(overlay_path, name))
            self.assertTrue((image[:20] > 200).all(axis=2).any(), name)  # white header text
            line = image[110:, 225]  # below the outlet divider (y=100), which crosses the line
            self.assertTrue(((line[:, 0] > 200) & (line[:, 2] < 50)).all(), name)  # blue counting line

    def test_csv_exports(self):
        counted = tracking.count_cells("synthetic.nd2", FakeUI())
        results = os.path.join(self.tmp.name, "results.csv")
        trajectories = os.path.join(self.tmp.name, "trajectories.csv")
        tracking.export_to_csv(counted, results)
        tracking.export_trajectories_to_csv(counted.values(), trajectories)

        with open(results, newline='') as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 3)
        self.assertTrue(all(r['count_frame'] and r['y_at_count_line'] for r in rows))

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
        counted = tracking.count_cells(self.path, FakeUI())
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
