# COUNT: notes for developers

This page is for people changing COUNT's code or checking how it works. For using COUNT, see the [README](README.md).

## How COUNT counts
1. **Reading frames** (`COUNT/video.py`): `.nd2` files are read with the Nikon SDK (`pims_nd2`). Uncompressed `.avi` files (as saved by NIS-Elements) are read directly from the file, so any frame can be read without the ones before it and only the needed rows are read; empty index entries (images repeated during playback) are not frames. Compressed AVIs are read with OpenCV. Frames with more than 8 bits are converted to 8-bit with one intensity range for the whole video.
2. **Finding moving cells** (`find_moving_specks`, `backgrounds_by_frame` in `COUNT/tracking.py`): each frame is compared with a centred rolling median background, the per-pixel median of the `background_window` frames before and after it (the frame itself excluded). Anything in the same place for more than about half of that window is part of the background, so dust, stuck cells and electrodes drop out. The absolute difference (kept as floats, since some cells are only ~9 grey levels from the background) is blurred (5×5 Gaussian) and thresholded at `detection_threshold`; connected blobs with an area between `min_area` and `max_area` are detections.
3. **Tracking** (`match_detections`, `Tracker`): each frame, detections and tracks are paired one-to-one, closest pairs first, measured from where each track is predicted to be from its last velocity, within `max_centroid_distance`. A detection may be up to `cell_radius` pixels left of its track (cells flow left to right). Tracks end after `timeout` frames without a match.
4. **Counting**: a track is counted once, when it has crossed the vertical line x = `count_line_x` moving right and has at least `min_detections` detections. Its outlet is decided by the y where it crossed: above `outlet_divider_y` is the top outlet (pDEP, `DEP_response = True`), below is the bottom outlet (nDEP).

Every setting, its default and a one-line description are in `DEFAULT_SETTINGS` in `COUNT/tracking.py`; the help text shown in the window is `FIELD_HELP` in `COUNT/ui.py`. A test checks that every setting has a field and help text.

## Speed
Most of the time goes into the median backgrounds. COUNT keeps this down by:
- analysing only the processing region (`processing_region`): from `region_before_line` pixels left of the counting line to the right edge, between `region_top` and `region_bottom`. Keep it starting well before the line (default 1000 px): a track needs a few frames to learn its cell's speed, and with 600 px two cells in a crowded group of the 50 kHz validation video were mixed up;
- reading only those rows from uncompressed AVI files;
- finding the moving cells in chunks of frames on separate processor cores (`find_all_specks`, `workers` setting), then tracking in order, so the result is identical to one core. Worker processes are started with "spawn" on every platform, so `main.py` must only start the program under `if __name__ == '__main__':`.

On a 2-core test machine, the two 2048 × 2048 validation videos (166 and 201 frames) take about 30–40 s each.

## Code layout
- `COUNT/main.py`: starts the settings window, then counts each chosen video and saves the CSV files
- `COUNT/ui.py`: the settings window, preview and help window
- `COUNT/tracking.py`: detection, tracking, counting, overlays and CSV export
- `COUNT/video.py`: reading `.nd2` and `.avi` files
- `COUNT/bg_plotting.py`: combining results files and plotting pDEP % against frequency
- `COUNT/box.py`: counting videos stored on Box (needs a Box app's `CLIENT_ID`/`CLIENT_SECRET` in a `.env` file)
- `COUNT/settings.json`: default settings for the current camera setup
- `tools/`: the synthetic video generator and the hand-count validation tool (below)

## Dependencies
Install with `pip install -r requirements.txt`. Keep OpenCV at the pinned 4.9: OpenCV 5.0 crashed reading uncompressed AVI files in testing (COUNT now reads those itself, but other code paths still use OpenCV). `boxsdk` is pinned below 4, because version 4 and later removed the API `box.py` uses.

## Running the tests
From the repository folder:
```
python -m unittest discover COUNT/tests
```
The tests use synthetic videos, so they don't need ND2 files or the Nikon SDK. The settings-window tests need a display (they are skipped without one).

## Checking COUNT against hand counts
Put the hand counts in each video's file name, as a number before `nDEP` (bottom outlet) and before `pDEP` (top outlet), for example `10V_50kHz_84nDEP_13pDEP.avi`, then run:
```
python tools/validate_counts.py "D:/COUNT test data"
```
This counts every video in the folder with the settings in `COUNT/settings.json` and prints COUNT's counts next to the hand counts. Add `--sweep` to also try 216 combinations of detection threshold, minimum speck size, maximum movement, timeout and counting line position, and list the closest (each video is read once). Results are saved to `validation_results.csv`. With only a few hand-counted videos, the best-scoring settings may suit those videos in particular, so check new settings on a video that wasn't used to choose them.

Results so far with the default settings:

| Video | Hand count (nDEP / pDEP) | COUNT |
|---|---|---|
| `10V_50kHz_84nDEP_13pDEP.avi` | 84 / 13 | 85 / 15 |
| `10V_80kHz_75nDEP_25pDEP.avi` | 75 / 25 | 24 / 18 (confirmed against an annotated video of the recording) |

## Synthetic test videos
`tools/make_synthetic_videos.py` makes three AVI videos (512 × 256, 12 cells each) of cells flowing into the top and bottom outlets, each with an answer key:
```
python tools/make_synthetic_videos.py synthetic_videos
```
`synth_easy_10kHz` has slow, well-spaced cells; `synth_fast_50kHz` cells crossing the image in 13–15 frames; `synth_crowded_100kHz` cells close together, side by side, pausing and briefly out of focus. Set **Counting Line x** and **Outlet Divider y** to 0 (automatic) for these, then compare each `<file>_results.csv` with its `<file>_answer_key.csv`. With the default settings COUNT counts all three exactly.

## Overlay videos
Overlay images (`overlay/<video>_<frame>.png`) can be joined into a video with ffmpeg, for example:
```
ffmpeg -framerate 3 -i "overlay/<video>_%05d.png" overlay.mp4
```

## Results file columns
`<video>_results.csv`: `object_id`, `x_pos`/`y_pos` and `x_size`/`y_size` (last bounding box), `most_recent_frame`, `frames_tracked` (frames the cell was detected in), `DEP_response` (True = top outlet), `count_frame` (frame in which it crossed the counting line) and `y_at_count_line`.
`<video>_trajectory_results.csv`: one row per counted cell per frame it was detected in.

## Contributing
Work on a branch and open a pull request into `master`. Please add or update tests for changes to counting.
