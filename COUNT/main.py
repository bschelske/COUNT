"""
===================
C. O. U. N. T.
    Count objects until no tomorrow...
    By Ben
===================
Check the readme on gitHub for more information. https://github.com/bschelske/COUNT
    "Ah- ah- ah!"

This is the file that is meant to be run.
"""
import argparse
import json
import multiprocessing
import os
import sys
import time
import traceback

# Allow running this file directly (python COUNT/main.py, or Run in PyCharm) as well as with python -m COUNT.main
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from COUNT import paths, tracking


def count_files(files, settings, results_folder):
    """Count every video in `files` and save each one's results and trajectory CSV files in `results_folder`."""
    for index, nd2_file in enumerate(files):
        print(f"Processing file {index + 1} of {len(files)}: {nd2_file}")

        try:
            # Get the filename without extension
            file_name = os.path.splitext(os.path.basename(nd2_file))[0]

            # Track the .nd2 or .avi file using MOG2 background subtraction
            counted_objects = tracking.count_cells(nd2_file, settings)

            csv_filename = os.path.join(results_folder, f"{file_name}_results.csv")
            trajectory_csv_filename = os.path.join(results_folder, f"{file_name}_trajectory_results.csv")
            tracking.export_to_csv(counted_objects, csv_filename)
            tracking.export_trajectories_to_csv(counted_objects.values(), trajectory_csv_filename)

        except Exception as e:
            print(f"Error processing file {nd2_file}: {e}")
            traceback.print_exc()


def main():
    from COUNT import ui, progress_window

    # Create UI. Information from the UI is stored within the class-instance "app"
    # This class is initiated in ui.py
    app = ui.create_ui()
    if not app.confirmed:  # window closed without clicking Confirm
        print("No files processed.")
        return

    results_folder = app.csv_folder_path.get()

    def work():
        start_time = time.time()
        count_files(app.files, app, results_folder)
        print(f"\nDone. Time elapsed: {time.time() - start_time:.2f} s")

    # Progress is shown in a window (the app has no console) as well as in the console when there is one
    progress_window.run_with_progress(work, title="COUNT: counting", results_folder=results_folder)


class SettingValues:
    """Settings from a file or the command line, read the same way as the settings window's fields."""
    class _Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    def __init__(self, values, overlay_path=""):
        for name, value in values.items():
            setattr(self, name, self._Value(value))
        self.overlay_path = overlay_path


def command_line(argv):
    """
    Count videos without the settings window, for scripts and for checking a built app works:
        COUNT --count VIDEO [VIDEO ...] --out FOLDER [--settings FILE] [--set NAME=VALUE ...]
        COUNT --check-nd2
    """
    parser = argparse.ArgumentParser(prog="COUNT", description="Count Objects Until No Tomorrow")
    parser.add_argument("--count", nargs="+", metavar="VIDEO", help="videos (.nd2 or .avi) to count")
    parser.add_argument("--out", default="results", help="folder for the results CSV files (default: results)")
    parser.add_argument("--settings", help="settings file (default: COUNT's default settings file)")
    parser.add_argument("--set", action="append", default=[], metavar="NAME=VALUE",
                        help="change one setting, e.g. --set count_line_x=0 (can be repeated)")
    parser.add_argument("--check-nd2", action="store_true", help="check that .nd2 files can be read, then exit")
    args = parser.parse_args(argv)

    if args.check_nd2:
        try:
            from pims import ND2Reader_SDK  # noqa: F401
            from pims_nd2 import ND2SDK  # noqa: F401  (loads the Nikon SDK library)
        except Exception as e:
            print(f"ND2 files can't be read here: {e}")
            return 1
        print("ND2 files can be read (Nikon SDK loaded).")
        return 0

    values = dict(tracking.DEFAULT_SETTINGS, save_overlay=False)
    settings_file = args.settings or paths.default_settings_path()
    if not os.path.exists(settings_file) and not args.settings:
        settings_file = paths.BUNDLED_SETTINGS_PATH
    with open(settings_file) as file:
        values.update({k: v for k, v in json.load(file).items() if k in values})
    for item in args.set:
        name, _, value = item.partition("=")
        if name not in values:
            parser.error(f"unknown setting {name!r}")
        values[name] = type(values[name])(value) if not isinstance(values[name], bool) else value.lower() in (
            "1", "true", "yes")

    os.makedirs(args.out, exist_ok=True)
    start_time = time.time()
    count_files(args.count, SettingValues(values, os.path.join(args.out, "overlay")), args.out)
    print(f"Time elapsed: {time.time() - start_time:.2f} s")
    return 0


if __name__ == '__main__':
    # Must come first: in the app, the worker processes that find cells start by running this file again
    multiprocessing.freeze_support()
    # The app has no console on Windows, so there is nowhere to print to: send printing nowhere instead of failing
    for name in ("stdout", "stderr"):
        if getattr(sys, name) is None:
            setattr(sys, name, open(os.devnull, "w"))

    if len(sys.argv) > 1 and not sys.argv[1].startswith("-psn"):  # macOS may add -psn_... when opening an app
        sys.exit(command_line(sys.argv[1:]))
    main()
