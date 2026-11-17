"""Tests for what changes in the COUNT app: where files go, the progress window and counting without the window."""
import csv
import os
import sys
import tempfile
import tkinter as tk
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from COUNT import main, paths, progress_window, tracking  # noqa: E402


class TestPaths(unittest.TestCase):
    def test_from_the_code_nothing_moves(self):
        with mock.patch.object(paths, "FROZEN", False):
            self.assertEqual(paths.results_path("results/"), "results/")
            self.assertEqual(paths.default_settings_path(), paths.BUNDLED_SETTINGS_PATH)

    def test_in_the_app_relative_results_go_in_documents(self):
        with mock.patch.object(paths, "FROZEN", True):
            self.assertEqual(paths.results_path("results/"),
                             os.path.normpath(os.path.join(paths.documents_dir(), "COUNT", "results")))
            absolute = os.path.abspath("somewhere")
            self.assertEqual(paths.results_path(absolute), absolute)

    def test_in_the_app_settings_are_saved_in_the_user_folder(self):
        with mock.patch.object(paths, "FROZEN", True):
            self.assertEqual(paths.default_settings_path(), os.path.join(paths.user_data_dir(), "settings.json"))
            self.assertNotEqual(os.path.dirname(paths.default_settings_path()), paths.PACKAGE_DIR)


def _display_available():
    try:
        root = tk.Tk()
        root.destroy()
        return True
    except tk.TclError:
        return False


@unittest.skipUnless(_display_available(), "needs a display")
class TestProgressWindow(unittest.TestCase):
    def test_carriage_return_rewrites_the_line(self):
        root = tk.Tk()
        try:
            window = progress_window.ProgressWindow(root, "test")
            window.append("Processing file 1\n")
            window.append("bar  10%\rbar  50%")
            window.append("\rbar 100%\ndone\n")
            self.assertEqual(window.text.get("1.0", "end-1c"), "Processing file 1\nbar 100%\ndone\n")
        finally:
            root.destroy()

    def test_work_output_and_errors_reach_the_window(self):
        def work():
            print("hello from work")
            raise RuntimeError("boom")

        def finish_then_close(self):
            original_finish(self)
            self.master.after(10, self.master.destroy)

        original_finish = progress_window.ProgressWindow.finish
        with mock.patch.object(progress_window.ProgressWindow, "finish", finish_then_close):
            window = progress_window.run_with_progress(work)
        self.assertIsInstance(window.error, RuntimeError)
        self.assertIs(sys.stdout, sys.__stdout__)


class TestCommandLine(unittest.TestCase):
    def setUp(self):
        tools = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "tools")
        sys.path.insert(0, tools)

    def test_counts_a_synthetic_video(self):
        import make_synthetic_videos as synth
        with tempfile.TemporaryDirectory() as folder:
            name, scenario = next(iter(synth.SCENARIOS.items()))
            synth.make_video(name, scenario, folder, 0)
            out = os.path.join(folder, "results")
            code = main.command_line([
                "--count", os.path.join(folder, f"{name}.avi"), "--out", out, "--settings", paths.BUNDLED_SETTINGS_PATH,
                "--set", "count_line_x=0", "--set", "outlet_divider_y=0", "--set", "region_top=0",
                "--set", "region_bottom=0", "--set", "workers=1"])
            self.assertEqual(code, 0)
            with open(os.path.join(out, f"{name}_results.csv"), newline="") as file:
                counted = list(csv.DictReader(file))
            with open(os.path.join(folder, f"{name}_answer_key.csv"), newline="") as file:
                expected = list(csv.DictReader(file))
            self.assertEqual(len(counted), len(expected))

    def test_unknown_setting_is_an_error(self):
        with self.assertRaises(SystemExit):
            main.command_line(["--count", "x.avi", "--set", "not_a_setting=1"])


if __name__ == "__main__":
    unittest.main()
