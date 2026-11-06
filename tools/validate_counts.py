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

import cv2 as cv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from COUNT import tracking  # noqa: E402

SETTINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "COUNT", "settings.json")

# Settings tried by --sweep. Detection settings change which objects are found in each frame; tracking settings
# change how those objects are linked into cells.
SWEEP_DETECTION = {
    "canny_lower": [40, 85, 130],
    "cell_radius": [4, 6, 8, 10],
}
SWEEP_TRACKING = {
    "max_centroid_distance": [40, 70, 100, 150],
    "timeout": [5, 10, 20],
    "min_frames_tracked": [2, 3, 5, 8],
}


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


def detect_all(path, detection_settings, canny_upper):
    """
    Read the video once and find the objects in every frame for each detection setting, exactly as
    tracking.nd2_mog_contours does. Returns ({setting: [[(position, size), ...] per frame]}, image height).
    """
    detections = {setting: [] for setting in detection_settings}
    backSub = cv.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
    image_h = None
    with tracking.open_video(path) as frames:
        tracking.prime_background(backSub, frames, len(frames))
        n_frames = len(frames)
        started = time.time()
        for frame_number in range(n_frames):
            frame = cv.normalize(frames[frame_number], None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_8U)
            image_h = frame.shape[0]
            mask = backSub.apply(frame)
            for setting in detection_settings:
                canny_lower, cell_radius = setting
                found = []
                if mask.any():
                    for cnt in tracking.find_cell_contours(mask, canny_lower, canny_upper, cell_radius):
                        x, y, w, h = cv.boundingRect(cnt)
                        found.append(((x, y), (w, h)))
                detections[setting].append(found)
            if frame_number % 200 == 0 or frame_number == n_frames - 1:
                print(f"\r  reading frame {frame_number + 1}/{n_frames} ({time.time() - started:.0f} s)", end="")
    print()
    return detections, image_h


def count_cells(frame_detections, image_h, cell_radius, max_centroid_distance, timeout, min_frames_tracked):
    """Track stored detections with the given settings. Returns (bottom, top)."""
    tracker = tracking.Tracker(image_h=image_h, max_centroid_distance=max_centroid_distance, timeout=timeout,
                               min_frames_tracked=min_frames_tracked, backtrack_tolerance=cell_radius)
    for frame_number, found in enumerate(frame_detections):
        objects = [tracking.DetectedObject(None, position, size, frame_number) for position, size in found]
        tracker.update(objects, frame_number)
    counted = tracker.finish().values()
    top = sum(1 for cell in counted if cell.DEP_outlet is True)
    return len(counted) - top, top


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
        settings = json.load(f)
    settings.setdefault("min_frames_tracked", tracking.DEFAULT_MIN_FRAMES_TRACKED)
    current_detection = (settings["canny_lower"], settings["cell_radius"])
    current_tracking = (settings["max_centroid_distance"], settings["timeout"], settings["min_frames_tracked"])

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

    # rows[(detection, tracking)] = [(bottom, top) per video]
    results = {}
    for path, expected in videos:
        print(f"\n{os.path.basename(path)}: hand count {describe(*expected)}")
        detections, image_h = detect_all(path, detection_settings, settings["canny_upper"])
        for detection in detection_settings:
            for track in tracking_settings:
                counts = count_cells(detections[detection], image_h, detection[1], *track)
                results.setdefault((detection, track), []).append(counts)

    def error(counts):
        """Total cells miscounted across all videos: |bottom - expected bottom| + |top - expected top|."""
        return sum(abs(b - eb) + abs(t - et) for (b, t), (_, (eb, et)) in zip(counts, videos))

    names = ["canny_lower", "cell_radius", "max_centroid_distance", "timeout", "min_frames_tracked"]
    with open(args.out, "w", newline="") as f:
        writer = csv.writer(f)
        header = names[:]
        for path, _ in videos:
            stem = os.path.splitext(os.path.basename(path))[0]
            header += [f"{stem} nDEP", f"{stem} pDEP"]
        writer.writerow(header + ["total_miscounted"])
        for (detection, track), counts in sorted(results.items(), key=lambda item: error(item[1])):
            writer.writerow(list(detection) + list(track) + [n for pair in counts for n in pair] + [error(counts)])

    print("\n=== Current settings (COUNT/settings.json) ===")
    print("  " + ", ".join(f"{n}={v}" for n, v in zip(names, current_detection + current_tracking)))
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
        print(f"  hand counts (nDEP/pDEP): " + "  |  ".join(f"{b}/{t}" for _, (b, t) in videos))
        print("\nWith only a few hand-counted videos, the best settings may be tuned to these videos in particular.")
        print("Check any new settings on a video that was not used to choose them.")
    print(f"\nAll results saved to {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
