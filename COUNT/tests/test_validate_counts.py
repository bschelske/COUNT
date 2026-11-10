"""
Tests for tools/validate_counts.py.

Run from the repository root:
    python -m unittest discover COUNT/tests
"""
import importlib.util
import os
import tempfile
import unittest

from COUNT import tracking
from COUNT.tests.test_tracking import FakeUI, synthetic_video, write_avi

TOOL = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools",
                    "validate_counts.py")
spec = importlib.util.spec_from_file_location("validate_counts", TOOL)
validate_counts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validate_counts)


class TestHandCounts(unittest.TestCase):
    def test_reads_counts_from_file_name(self):
        self.assertEqual(validate_counts.hand_counts(r"D:\COUNT test data\10V_50kHz_84nDEP_13pDEP.avi"), (84, 13))
        self.assertEqual(validate_counts.hand_counts("10V_80kHz_75nDEP_25pDEP.avi"), (75, 25))
        self.assertEqual(validate_counts.hand_counts("trial 25pDEP 75 nDEP.nd2"), (75, 25))

    def test_no_counts(self):
        self.assertIsNone(validate_counts.hand_counts("10V_50kHz.avi"))


class TestMatchesCount(unittest.TestCase):
    def test_counts_exactly_like_count_does(self):
        """The validation tool must give the same counts as COUNT itself for the same settings."""
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "video.avi")
            write_avi(path, synthetic_video(cells=((0, 50), (-60, 150), (-120, 60), (-150, 140)),
                                            dust=((150, 100), (60, 170))))
            ui = FakeUI(cell_radius=6)

            counted = tracking.count_cells(path, ui).values()
            top = sum(1 for cell in counted if cell.DEP_outlet is True)

            settings = validate_counts.Settings({name: getattr(ui, name).get() for name in tracking.DEFAULT_SETTINGS})
            setting = (settings.detection_threshold.get(), settings.min_area.get())
            detections, shape = validate_counts.detect_all(path, settings, [setting])
            line_x, divider_y = tracking.count_line_and_divider(settings, shape)
            result = validate_counts.count_from_specks(detections[setting], line_x, divider_y,
                                                       settings.max_centroid_distance.get(), settings.timeout.get(),
                                                       settings.cell_radius.get(), settings.min_detections.get())
            self.assertEqual(result, (len(counted) - top, top))
            self.assertEqual(result, (2, 2))


if __name__ == '__main__':
    unittest.main()
