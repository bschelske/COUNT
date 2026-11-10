"""
Check COUNT against hand-counted videos.

The hand counts are read from each file name: a number followed by "nDEP" (cells in the bottom outlet) and a
number followed by "pDEP" (cells in the top outlet), for example
    10V_50kHz_84nDEP_13pDEP.avi   -> 84 bottom, 13 top

Usage, from the repository folder:
    python tools/validate_counts.py "D:/COUNT test data"
        Runs COUNT with the settings in COUNT/settings.json on every .avi/.nd2 file in the folder (or on the
        files given) and compares its counts with the hand counts.

    python tools/validate_counts.py "D:/COUNT test data" --sweep
        Also tries a range of settings and lists the ones that come closest to the hand counts.
        Each video is only read once, so this takes about as long as a normal run per file.

Results are printed and saved to validation_results.csv (change with --out).
"""
import argparse
import csv
import itertools
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from COUNT import tracking  # noqa: E402

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COUNT", "settings.json")

# Settings tried by --sweep. Detection settings change which specks are found in each image; tracking settings
# change how specks are linked into cells and where they are counted. Counting line positions are offsets
# (pixels) from the line in the settings.
SWEEP_DETECTION = {
    "detection_threshold": [6, 8, 10, 12],
    "min_area": [6, 15],
}
SWEEP_TRACKING = {
    "max_centroid_distance": [70, 100, 140],
    "timeout": [2, 3, 5],
    "count_line_offset": [-200, 0, 150],
}


class Settings:
    """Settings from a JSON file, read the way COUNT reads them from the settings window."""

    class _Value:
        def __init__(self, value):
            self.value = value

        def get(self):
            return self.value

    def __init__(self, values):
        self.values = dict(tracking.DEFAULT_SETTINGS, **values)
        for name, value in self.values.items():
            setattr(self, name, self._Value(value))


def hand_counts(path):
    """(bottom, top) hand counts from a file name like ..._84nDEP_13pDEP.avi, or None."""
    name = os.path.basename(path)
    ndep = re.search(r"(\d+)\s*_?\s*ndep", name, re.IGNORECASE)
    pdep = re.search(r"(\d+)\s*_?\s*pdep", name, re.IGNORECASE)
    if not (ndep and pdep):
        return None
    return int(ndep.group(1)), int(pdep.group(1))


def find_videos(paths):
    videos = []
    for path in paths:
        if os.path.isdir(path):
            videos += [os.path.join(path, f) for f in sorted(os.listdir(path)) if tracking.is_supported_video(f)]
        else:
            videos.append(path)
    return videos


def detect_all(path, settings, detection_settings):
    """
    Read the video once and find the moving specks in every frame for each (detection_threshold, min_area)
    setting, exactly as tracking.count_cells does. Returns ({setting: [specks per frame]}, image shape).
    """
    detections = {setting: [] for setting in detection_settings}
    shape = None
    with tracking.open_video(path) as frames:
        n_frames = len(frames)
        started = time.time()
        for frame_number, frame, background in tracking.backgrounds_by_frame(
                frames, n_frames, settings.background_window.get()):
            shape = frame.shape
            for threshold, min_area in detection_settings:
                specks = [] if background is None else tracking.find_moving_specks(
                    frame, background, threshold, min_area, settings.max_area.get())
                detections[(threshold, min_area)].append(specks)
            if frame_number % 20 == 0 or frame_number == n_frames - 1:
                print(f"\r  reading frame {frame_number + 1}/{n_frames} ({time.time() - started:.0f} s)", end="")
    print()
    return detections, shape


def count_from_specks(specks_by_frame, count_line_x, outlet_divider_y, max_centroid_distance, timeout, cell_radius,
                      min_detections=tracking.DEFAULT_SETTINGS["min_detections"]):
    """Track and count stored specks with the given settings. Returns (bottom, top)."""
    tracker = tracking.Tracker(count_line_x=count_line_x, outlet_divider_y=outlet_divider_y,
                               max_centroid_distance=max_centroid_distance, timeout=timeout,
                               backtrack_tolerance=cell_radius, min_detections=min_detections)
    for frame_number, specks in enumerate(specks_by_frame):
        tracker.update(tracking.specks_to_objects(specks, frame_number), frame_number)
    tracker.finish()
    return tracker.outlet_counts()


def describe(bottom, top):
    total = bottom + top
    percent = f"{100 * top / total:.1f}%" if total else "n/a"
    return f"{total:4d} ({bottom:3d} nDEP / {top:3d} pDEP, {percent} pDEP)"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("paths", nargs="+", help=".avi/.nd2 files, or folders containing them")
    parser.add_argument("--sweep", action="store_true", help="also try a range of settings")
    parser.add_argument("--settings", default=SETTINGS_PATH, help="settings file (default: COUNT/settings.json)")
    parser.add_argument("--out", default="validation_results.csv", help="CSV file for the results")
    args = parser.parse_args()

    with open(args.settings) as f:
        settings = Settings(json.load(f))
    current_detection = (settings.detection_threshold.get(), settings.min_area.get())
    current_tracking = (settings.max_centroid_distance.get(), settings.timeout.get(), 0)

    videos = []
    for path in find_videos(args.paths):
        expected = hand_counts(path)
        if expected is None:
            print(f"Skipping {os.path.basename(path)}: no '<number>nDEP' and '<number>pDEP' in the file name")
        else:
            videos.append((path, expected))
    if not videos:
        sys.exit("No hand-counted videos found.")

    detection_settings = [current_detection]
    tracking_settings = [current_tracking]
    if args.sweep:
        detection_settings += [s for s in itertools.product(*SWEEP_DETECTION.values()) if s != current_detection]
        tracking_settings += [s for s in itertools.product(*SWEEP_TRACKING.values()) if s != current_tracking]

    # results[(detection, tracking)] = [(bottom, top) per video]
    results = {}
    for path, expected in videos:
        print(f"\n{os.path.basename(path)}: hand count {describe(*expected)}")
        detections, shape = detect_all(path, settings, detection_settings)
        line_x, divider_y = tracking.count_line_and_divider(settings, shape)
        for detection in detection_settings:
            for max_distance, timeout, offset in tracking_settings:
                counts = count_from_specks(detections[detection], line_x + offset, divider_y, max_distance, timeout,
                                           settings.cell_radius.get(), settings.min_detections.get())
                results.setdefault((detection, (max_distance, timeout, offset)), []).append(counts)

    def error(counts):
        """Total cells miscounted across all videos: |bottom - expected bottom| + |top - expected top|."""
        return sum(abs(b - eb) + abs(t - et) for (b, t), (_, (eb, et)) in zip(counts, videos))

    names = ["detection_threshold", "min_area", "max_centroid_distance", "timeout", "count_line_offset"]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        header = names[:]
        for path, _ in videos:
            stem = os.path.splitext(os.path.basename(path))[0]
            header += [f"{stem} nDEP", f"{stem} pDEP"]
        writer.writerow(header + ["total_miscounted"])
        for (detection, track), counts in sorted(results.items(), key=lambda item: error(item[1])):
            writer.writerow(list(detection) + list(track) + [n for pair in counts for n in pair] + [error(counts)])

    print("\n=== Current settings ===")
    print("  " + ", ".join(f"{name}={value}" for name, value in settings.values.items()
                           if name in tracking.DEFAULT_SETTINGS))
    for (path, expected), counts in zip(videos, results[(current_detection, current_tracking)]):
        print(f"  {os.path.basename(path)}")
        print(f"    hand count {describe(*expected)}")
        print(f"    COUNT      {describe(*counts)}")
    print(f"  Cells miscounted: {error(results[(current_detection, current_tracking)])}")

    if args.sweep:
        print(f"\n=== Closest of {len(results)} settings tried ===")
        ranked = sorted(results.items(), key=lambda item: error(item[1]))
        for (detection, track), counts in ranked[:10]:
            summary = "  |  ".join(f"{b}/{t}" for b, t in counts)
            print(f"  miscounted {error(counts):3d}   " +
                  ", ".join(f"{n}={v}" for n, v in zip(names, detection + track)) + f"   (nDEP/pDEP: {summary})")
        print("  hand counts (nDEP/pDEP): " + "  |  ".join(f"{b}/{t}" for _, (b, t) in videos))
        print("\nWith only a few hand-counted videos, the best settings may be tuned to these videos in particular.")
        print("Check any new settings on a video that was not used to choose them.")
    print(f"\nAll results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
