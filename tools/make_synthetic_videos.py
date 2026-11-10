"""
Make synthetic microfluidic timelapses for testing COUNT.

Each video shows a channel with cells flowing left to right. Partway across, each cell is steered into the
top outlet (positive DEP) or the bottom outlet (no positive DEP), which are split by a wall along the middle
of the right side of the image. Every video comes with an answer key (CSV) listing each cell and its outlet.

The videos are 8-bit AVI files, which NIS-Elements can open and save as .nd2.

Usage:
    python make_synthetic_videos.py [output_folder]
"""
import csv
import os
import sys

import cv2 as cv
import numpy as np

W, H = 512, 256              # image size (px)
WALL_TOP, WALL_BOTTOM = 20, 236
SPLIT_X = 320                # where the outlet divider wall starts
TOP_Y, BOTTOM_Y = 72, 184    # y of the top and bottom outlet streams
STEER_START = 200            # cells start moving towards their outlet here
BACKGROUND, NOISE, CELL = 800, 25, 520
EMPTY_FRAMES = 15            # frames with no cells at the start, so background subtraction can learn


def cell_center(cell, frame):
    """(x, y) of a cell's centre in a frame, or None if it isn't in view (or is hidden that frame)."""
    t = frame - cell["enter"]
    if t < 0 or frame in cell["hidden"]:
        return None
    # time spent paused before this frame
    paused = sum(min(max(t - p_start, 0), p_len) for p_start, p_len in cell["pauses"])
    x = -cell["radius"] + cell["speed"] * (t - paused)
    if x > W + cell["radius"]:
        return None
    target = TOP_Y if cell["top"] else BOTTOM_Y
    if x <= STEER_START:
        y = cell["y0"]
    elif x >= SPLIT_X:
        y = target
    else:
        f = (x - STEER_START) / (SPLIT_X - STEER_START)
        y = cell["y0"] + (target - cell["y0"]) * (3 * f ** 2 - 2 * f ** 3)  # smooth S-curve
    return x, y


def frames_in_view(cell, n_frames):
    return [f for f in range(n_frames) if cell_center(cell, f) is not None
            and -cell["radius"] < cell_center(cell, f)[0] < W + cell["radius"]]


def background(rng):
    """Static channel image: walls, outlet divider and some fixed texture."""
    img = np.full((H, W), BACKGROUND, np.float32)
    img += cv.GaussianBlur(rng.normal(0, 40, (H, W)).astype(np.float32), (0, 0), 6)  # uneven illumination
    for y in (WALL_TOP, WALL_BOTTOM):
        cv.line(img, (0, y), (W - 1, y), 300, 4)
    cv.line(img, (SPLIT_X, H // 2), (W - 1, H // 2), 300, 4)
    for _ in range(25):  # debris stuck to the device (doesn't move, so it's background)
        cv.circle(img, (int(rng.uniform(0, W)), int(rng.uniform(WALL_TOP + 5, WALL_BOTTOM - 5))),
                  int(rng.uniform(1, 3)), 600, -1)
    return img


def render(cells, n_frames, rng):
    base = background(rng)
    for frame in range(n_frames):
        img = base.copy()
        layer = np.zeros((H, W), np.float32)
        for cell in cells:
            c = cell_center(cell, frame)
            if c is not None:
                cv.circle(layer, (int(round(c[0])), int(round(c[1]))), cell["radius"], CELL, -1, cv.LINE_AA)
        img += cv.GaussianBlur(layer, (0, 0), 1.2)
        img += rng.normal(0, NOISE, (H, W))
        yield np.clip((img - 400) * (255 / 1200), 0, 255).astype(np.uint8)


def new_cell(enter, speed, top, y0, radius=7, pauses=(), hidden=(), note=""):
    return dict(enter=enter, speed=speed, top=top, y0=y0, radius=radius, pauses=list(pauses),
                hidden=set(hidden), note=note)


def scenario_easy(rng):
    """Slow, well-spaced cells. Old and new COUNT should both get these right."""
    cells = []
    for i, top in enumerate([1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0]):
        cells.append(new_cell(EMPTY_FRAMES + 20 * i, rng.uniform(9, 12), bool(top), rng.uniform(110, 146),
                              note="slow, well spaced"))
    return cells


def scenario_fast(rng):
    """Fast cells that cross the image in 12-17 frames (fewer than the default timeout of 20)."""
    cells = []
    for i, top in enumerate([1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 1]):
        cells.append(new_cell(EMPTY_FRAMES + 9 * i, rng.uniform(32, 42), bool(top), rng.uniform(110, 146),
                              note="fast"))
    return cells


def scenario_crowded(rng):
    """Cells close together, cells that pause, and a cell that briefly disappears."""
    s = EMPTY_FRAMES
    return [
        # Trains: two cells on the same streamline, 45 px apart, entering together
        new_cell(s, 12, True, 128, note="train, leading"),
        new_cell(s + 4, 12, True, 128, note="train, trailing (45 px behind)"),
        new_cell(s + 30, 11, False, 120, note="train, leading"),
        new_cell(s + 34, 11, False, 120, note="train, trailing (44 px behind)"),
        # Side by side: two cells entering in the same frame, 40 px apart vertically
        new_cell(s + 60, 10, True, 108, note="side by side, upper"),
        new_cell(s + 60, 10, False, 148, note="side by side, lower"),
        # Pausing cells (stuck for a few frames, then continue)
        new_cell(s + 90, 10, True, 125, pauses=[(15, 4)], note="pauses 4 frames"),
        new_cell(s + 110, 10, False, 132, pauses=[(10, 3), (30, 2)], note="pauses twice"),
        # A cell that is invisible for 3 frames (out of focus), then reappears
        new_cell(s + 135, 11, True, 130, hidden=range(s + 135 + 18, s + 135 + 21), note="hidden 3 frames"),
        # A few ordinary cells to finish
        new_cell(s + 160, 9, False, 118, note="ordinary"),
        new_cell(s + 175, 12, True, 138, note="ordinary"),
        new_cell(s + 190, 10, False, 126, note="ordinary"),
    ]


SCENARIOS = {
    "synth_easy_10kHz": scenario_easy,
    "synth_fast_50kHz": scenario_fast,
    "synth_crowded_100kHz": scenario_crowded,
}


def n_frames_for(cells):
    last = 0
    for cell in cells:
        frame = cell["enter"]
        while cell_center(cell, frame) is not None or frame in cell["hidden"]:
            frame += 1
        last = max(last, frame)
    return last + 10


def make_video(name, scenario, folder, seed):
    rng = np.random.default_rng(seed)
    cells = scenario(rng)
    n_frames = n_frames_for(cells)

    path = os.path.join(folder, f"{name}.avi")
    writer = cv.VideoWriter(path, cv.VideoWriter_fourcc(*"MJPG"), 30, (W, H), isColor=False)
    writer.set(cv.VIDEOWRITER_PROP_QUALITY, 98)
    for frame in render(cells, n_frames, rng):
        writer.write(frame)
    writer.release()

    with open(os.path.join(folder, f"{name}_answer_key.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "outlet", "DEP_response", "first_frame", "last_frame", "frames_visible",
                    "speed_px_per_frame", "note"])
        for i, cell in enumerate(sorted(cells, key=lambda c: c["enter"]), 1):
            visible = frames_in_view(cell, n_frames)
            w.writerow([i, "top" if cell["top"] else "bottom", cell["top"], visible[0], visible[-1], len(visible),
                        round(cell["speed"], 1), cell["note"]])
    top = sum(c["top"] for c in cells)
    return dict(name=name, frames=n_frames, cells=len(cells), top=top, bottom=len(cells) - top)


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "synthetic_videos"
    os.makedirs(out, exist_ok=True)
    for seed, (name, scenario) in enumerate(SCENARIOS.items()):
        info = make_video(name, scenario, out, seed)
        print(f"{info['name']}: {info['frames']} frames, {info['cells']} cells "
              f"({info['top']} top / {info['bottom']} bottom)")
