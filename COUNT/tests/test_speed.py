"""
Tests for the faster processing: reading uncompressed AVI files directly, the processing region, and
spreading the work over several processor cores.

Run from the repository root:
    python -m unittest discover COUNT/tests
"""
import os
import struct
import tempfile
import unittest

import cv2 as cv
import numpy as np

from COUNT import tracking, video
from COUNT.tests.test_tracking import FakeUI, synthetic_video, write_avi


def write_raw_avi(path, frames_bgr, bits=24, bottom_up=True, empty_entries_after_each=0, palette=None):
    """
    Write an uncompressed AVI (AVI 1.0 with an idx1 index) from a list of BGR (or, for 8 bits, index) images.
    `empty_entries_after_each` adds that many zero-size index entries after each frame, as NIS-Elements does
    for images that are repeated during playback.
    """
    height, width = frames_bgr[0].shape[:2]
    channels = bits // 8
    stride = ((width * bits + 31) // 32) * 4

    def frame_bytes(image):
        if bits == 8:
            rows = image.astype(np.uint8)
        elif bits == 24:
            rows = image.reshape(height, width * 3)
        else:
            rows = np.dstack([image, np.full((height, width), 255, np.uint8)]).reshape(height, width * 4)
        padded = np.zeros((height, stride), np.uint8)
        padded[:, :width * channels] = rows
        if bottom_up:
            padded = padded[::-1]
        return padded.tobytes()

    def chunk(chunk_id, data):
        return chunk_id + struct.pack("<I", len(data)) + data + (b"\0" if len(data) % 2 else b"")

    def lst(list_type, data):
        return b"LIST" + struct.pack("<I", len(data) + 4) + list_type + data

    n = len(frames_bgr)
    avih = struct.pack("<10I4I", 33333, 0, 0, 0x10, n * (1 + empty_entries_after_each), 0, 1, 0, width, height,
                       0, 0, 0, 0)
    strh = b"vids" + b"DIB " + struct.pack("<IHHIIIIIIIIhhhh", 0, 0, 0, 0, 1, 30, 0, n, 0, 0, 0, 0, 0, width,
                                           height)
    n_colours = 256 if bits == 8 else 0
    strf = struct.pack("<IiiHHIIiiII", 40, width, height if bottom_up else -height, 1, bits, 0, stride * height,
                       0, 0, n_colours, 0)
    if bits == 8:
        pal = palette if palette is not None else np.stack([np.arange(256)] * 3, axis=1)
        strf += np.hstack([pal.astype(np.uint8), np.zeros((256, 1), np.uint8)]).tobytes()
    hdrl = lst(b"hdrl", chunk(b"avih", avih) + lst(b"strl", chunk(b"strh", strh) + chunk(b"strf", strf)))

    movi_data, index = b"", b""
    for image in frames_bgr:
        offset = 4 + len(movi_data)
        movi_data += chunk(b"00db", frame_bytes(image))
        index += b"00db" + struct.pack("<III", 0x10, offset, stride * height)
        for _ in range(empty_entries_after_each):
            index += b"00db" + struct.pack("<III", 0, offset, 0)
    body = b"AVI " + hdrl + lst(b"movi", movi_data) + chunk(b"idx1", index)
    with open(path, "wb") as f:
        f.write(b"RIFF" + struct.pack("<I", len(body)) + body)


def colour_frames(n=5, size=(37, 53)):
    """Random BGR frames with an odd width (so rows are padded) and different channels."""
    rng = np.random.default_rng(1)
    return [rng.integers(0, 256, size + (3,), dtype=np.uint8) for _ in range(n)]


class TestRawAviReader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def check_matches_opencv(self, path, expected_frames):
        with video.open_video(path) as frames:
            self.assertIsInstance(frames, video.RawAviReader)
            self.assertEqual(len(frames), len(expected_frames))
            for i in (3, 0, len(expected_frames) - 1):  # in any order
                self.assertTrue(np.array_equal(frames[i], expected_frames[i]), i)
            region = (5, 7, 40, 30)
            self.assertTrue(np.array_equal(video.read_region(frames, 2, region), expected_frames[2][7:30, 5:40]))

    def test_24_bit_bottom_up_matches_opencv_greyscale(self):
        path = os.path.join(self.tmp.name, "raw24.avi")
        bgr = colour_frames()
        write_raw_avi(path, bgr, bits=24, bottom_up=True)
        self.check_matches_opencv(path, [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in bgr])

    def test_24_bit_top_down(self):
        path = os.path.join(self.tmp.name, "raw24td.avi")
        bgr = colour_frames()
        write_raw_avi(path, bgr, bits=24, bottom_up=False)
        self.check_matches_opencv(path, [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in bgr])

    def test_32_bit(self):
        path = os.path.join(self.tmp.name, "raw32.avi")
        bgr = colour_frames()
        write_raw_avi(path, bgr, bits=32)
        self.check_matches_opencv(path, [cv.cvtColor(f, cv.COLOR_BGR2GRAY) for f in bgr])

    def test_8_bit_with_palette(self):
        path = os.path.join(self.tmp.name, "raw8.avi")
        indices = [f[:, :, 0] for f in colour_frames()]
        palette = np.random.default_rng(2).integers(0, 256, (256, 3), dtype=np.uint8)  # BGR palette
        write_raw_avi(path, indices, bits=8, palette=palette)
        grey_palette = cv.cvtColor(palette.reshape(1, -1, 3), cv.COLOR_BGR2GRAY).ravel()
        self.check_matches_opencv(path, [grey_palette[i] for i in indices])

    def test_repeated_images_are_not_extra_frames(self):
        # NIS-Elements files list each image followed by many empty entries (repeat the last image)
        path = os.path.join(self.tmp.name, "repeats.avi")
        bgr = colour_frames()
        write_raw_avi(path, bgr, empty_entries_after_each=30)
        with video.open_video(path) as frames:
            self.assertEqual(len(frames), len(bgr))

    def test_agrees_with_opencv_reader(self):
        path = os.path.join(self.tmp.name, "raw24.avi")
        write_raw_avi(path, colour_frames())
        reader = video.AviReader(path)
        with video.open_video(path) as frames:
            self.assertEqual(len(frames), len(reader))
            for i in range(len(reader)):
                self.assertTrue(np.array_equal(frames[i], reader[i]), i)
        reader.close()

    def test_compressed_avi_uses_opencv(self):
        path = os.path.join(self.tmp.name, "mjpg.avi")
        write_avi(path, synthetic_video(n_frames=3))
        with video.open_video(path) as frames:
            self.assertIsInstance(frames, video.AviReader)
            self.assertEqual(len(frames), 3)

    def test_frame_shape(self):
        path = os.path.join(self.tmp.name, "raw24.avi")
        write_raw_avi(path, colour_frames())
        with video.open_video(path) as frames:
            self.assertEqual(video.frame_shape(frames), (37, 53))


class TestProcessingRegion(unittest.TestCase):
    def test_default_region_starts_before_the_line(self):
        ui = FakeUI()  # default: 1000 px before the line
        self.assertEqual(tracking.processing_region(ui, (2048, 2048), 1500), (500, 0, 2048, 2048))
        ui = FakeUI(region_before_line=600)
        self.assertEqual(tracking.processing_region(ui, (2048, 2048), 1500), (900, 0, 2048, 2048))

    def test_rows_and_whole_width(self):
        ui = FakeUI(region_before_line=0, region_top=700, region_bottom=1750)
        self.assertEqual(tracking.processing_region(ui, (2048, 2048), 1500), (0, 700, 2048, 1750))

    def test_region_clipped_to_image(self):
        ui = FakeUI(region_before_line=600, region_bottom=5000)
        self.assertEqual(tracking.processing_region(ui, (200, 300), 225), (0, 0, 300, 200))

    def test_bad_region(self):
        with self.assertRaises(ValueError):
            tracking.processing_region(FakeUI(region_top=150, region_bottom=100), (200, 300), 225)

    def test_specks_are_in_whole_image_pixels(self):
        frames = synthetic_video(cells=((0, 150),))
        region = (100, 120, 300, 200)
        whole = {i: specks for i, _, specks in tracking.moving_specks_by_frame(frames, 30, 10, 8, 6, 600)}
        part = {i: specks for i, _, specks in tracking.moving_specks_by_frame(frames, 30, 10, 8, 6, 600,
                                                                              region=region)}
        for i in range(30):
            expected = [s for s in whole[i] if s[0] > region[0] + 10]  # well inside the region
            for cx, cy, *_ in expected:
                self.assertTrue(any(abs(cx - x) < 0.01 and abs(cy - y) < 0.01 for x, y, *_ in part[i]), i)
            self.assertTrue(all(x >= region[0] and y >= region[1] for x, y, *_ in part[i]), i)

    def test_region_does_not_change_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "video.avi")
            write_raw_avi(path, [cv.cvtColor(cv.normalize(f, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U),
                                             cv.COLOR_GRAY2BGR) for f in synthetic_video()])
            whole = tracking.count_cells(path, FakeUI(region_before_line=0))
            part = tracking.count_cells(path, FakeUI(region_before_line=120, region_top=20, region_bottom=190))
            self.assertEqual(sorted(c.DEP_outlet for c in whole.values()), [False, True, True])
            self.assertEqual(sorted(c.DEP_outlet for c in part.values()), [False, True, True])


class TestParallel(unittest.TestCase):
    def test_parallel_gives_the_same_result(self):
        cells = tuple((-40 * k, 50 if k % 2 else 150) for k in range(12))
        frames = synthetic_video(n_frames=130, cells=cells, dust=((150, 100),))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "video.avi")
            write_raw_avi(path, [cv.cvtColor(cv.normalize(f, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U),
                                             cv.COLOR_GRAY2BGR) for f in frames])
            settings = tracking.tracking_settings(FakeUI())
            region = (0, 0, 300, 200)
            one = tracking.find_all_specks(path, 130, settings, region, None, workers=1)
            several = tracking.find_all_specks(path, 130, settings, region, None, workers=2)
            self.assertEqual(one, several)
            self.assertGreater(sum(len(s) for s in one), 100)

            counted_one = tracking.count_cells(path, FakeUI(workers=1))
            counted_several = tracking.count_cells(path, FakeUI(workers=2))
            summary = lambda counted: sorted((c.count_frame, c.DEP_outlet) for c in counted.values())  # noqa: E731
            self.assertEqual(summary(counted_one), summary(counted_several))
            self.assertEqual(len(counted_one), 12)


class TestOverlaysAndPreview(unittest.TestCase):
    def test_overlays_show_whole_frame_with_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trial.avi")
            write_raw_avi(path, [cv.cvtColor(cv.normalize(f, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U),
                                             cv.COLOR_GRAY2BGR) for f in synthetic_video()])
            overlay_path = os.path.join(tmp, "overlay")
            ui = FakeUI(overlay_path=overlay_path, save_overlay=True, region_top=30, region_bottom=180)
            tracking.count_cells(path, ui)
            files = sorted(os.listdir(overlay_path))
            self.assertEqual(len(files), 30)
            image = cv.imread(os.path.join(overlay_path, files[-1]))
            self.assertEqual(image.shape, (200, 300, 3))
            self.assertTrue((image[30, 50:200] == (0, 140, 0)).all())  # region box top edge

    def test_preview_draws_region(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "trial.avi")
            write_raw_avi(path, [cv.cvtColor(cv.normalize(f, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U),
                                             cv.COLOR_GRAY2BGR) for f in synthetic_video()])
            images = tracking.preview_frames(path, FakeUI(region_top=30, region_bottom=180))
            self.assertEqual(len(images), 2)
            self.assertTrue((images[0][179, 50:200] == (0, 140, 0)).all())  # region box bottom edge


class TestHelp(unittest.TestCase):
    def test_every_setting_has_a_field_and_help(self):
        from COUNT import ui
        fields = [name for name, _ in ui.LEFT_FIELDS + ui.RIGHT_FIELDS]
        self.assertEqual(sorted(fields), sorted(tracking.DEFAULT_SETTINGS))
        self.assertEqual(sorted(ui.FIELD_HELP), sorted(tracking.DEFAULT_SETTINGS))
        text = " ".join(body for _, sections in ui.help_sections() for _, body in sections)
        for name, label in ui.LEFT_FIELDS + ui.RIGHT_FIELDS:
            self.assertIn(ui.FIELD_HELP[name], text, name)

    def test_help_window_opens_once(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception as e:  # no display available
            self.skipTest(f"no display: {e}")
        from COUNT import ui
        try:
            app = ui.ROISelectionApp(root)
            app.help_button.invoke()
            app.help_button.invoke()  # a second click brings the same window forward
            windows = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)]
            self.assertEqual(len(windows), 1)
            text = [w for w in windows[0].winfo_children() if isinstance(w, tk.Text)][0].get("1.0", "end")
            self.assertIn("Counting Line x", text)
            self.assertIn("Preview Detection", text)
        finally:
            root.destroy()


class TestSettingsWindowLayout(unittest.TestCase):
    def test_no_two_widgets_share_a_grid_cell(self):
        try:
            import tkinter as tk
            root = tk.Tk()
        except Exception as e:  # no display available
            self.skipTest(f"no display: {e}")
        from COUNT import ui
        try:
            ui.ROISelectionApp(root)
            cells = [(int(w.grid_info()["row"]), int(w.grid_info()["column"])) for w in root.grid_slaves()]
            self.assertEqual(len(cells), len(set(cells)), "two widgets are in the same place")
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
