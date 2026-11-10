
# Count Objects Until No Tomorrow (C.O.U.N.T.)
🧛‍♂️: _Ah-ah-ah!_

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MIT License](https://img.shields.io/badge/license-MIT-green)
## Description
An object tracking system for analyzing cell movement and DEP response in microfluidic devices, useful in cancer diagnostics and single-cell analysis.

**General workflow:**
1. **Input Files**: Processes Nikon ND2 files directly using the `ND2Reader_SDK` library, or AVI videos
2. **Finding moving cells**: Each image is compared with the median of the images around it. Anything that stays in place (dust, cells stuck to the device, the electrodes) is part of that background and drops out, leaving only the specks that move
3. **Tracking**: Each image, every moving speck is paired with at most one existing track (closest pairs first, from where each track is predicted to be); cells are assumed to flow left to right
4. **Counting**: A cell is counted once, when it crosses a vertical counting line placed past the electrodes. If it crosses above the outlet divider it is heading for the top outlet (positive DEP response); below it, the bottom outlet (no positive DEP response). Cells that never cross the line are not counted
5. **Results**: Counts and tracks are exported to CSV files for further analysis

## Features
- Direct processing of ND2 files through the PIMS library, and of AVI videos through OpenCV
- Ignores dust and stationary cells: only cells that move are tracked
- Counts each cell once, at a counting line, and assigns its outlet where it crosses
- Preview of what will be detected, with the counting line and outlet divider drawn on the image
- Optional overlay images showing every moving cell, each count as it happens and running totals
- CSV export of counted cells including position, size and DEP response
- Settings persistence through JSON files
- A tool to check COUNT against hand-counted videos

## How to Install
1. Clone the repository to your local machine
2. Install Python 3.10 or newer, then the required dependencies:
```
pip install -r requirements.txt
```
   The Nikon ND2 SDK comes bundled with `pims_nd2` (Windows, macOS and Linux).
3. Start the application from the repository folder:
```
python COUNT/main.py
```
   (or open `COUNT/main.py` in PyCharm and click Run)

## Required Dependencies
- OpenCV (cv2)
- NumPy
- tkinter (for the UI)
- PIMS with ND2Reader_SDK (`pims_nd2`)
- tqdm (for progress bars)
- pandas, SciPy and Matplotlib (for the plotting tool)
- boxsdk 3.x and python-dotenv (only for `box.py`)

## How to Use
1. Run `main.py` to open the user interface
2. Select an individual ND2 or AVI file, or a folder containing ND2/AVI files (choosing one replaces the other)
3. Choose where to save the CSV results (optional)
4. Click **Preview Detection** to see two images from the middle of the video, with the moving specks that will be detected (orange circles), the counting line (blue) and the outlet divider (grey). Adjust the settings until the line sits in the main channel past the electrodes, the divider lines up with the tip that splits the two outlets, and the cells are circled
5. Optionally check "Save Overlay?" to save an image of every frame showing what was counted
6. Click "Confirm" to begin processing

## Understanding the Parameters
- **Max Movement Between Images (px)**: How far a cell can move between one image and the next (measured from where it is predicted to be) and still be treated as the same cell
- **Timeout (images)**: If a cell is not detected for this many images in a row, its track ends
- **Cell Radius (px)**: A cell may move left by up to this many pixels between images (outline jitter, pauses) and still be matched
- **Min Detections to Count**: A cell is only counted once it has been detected in at least this many images, so two noise specks that happen to fall either side of the counting line are not counted
- **Detection Threshold (grey levels)**: How much darker or brighter than its background a speck must be to be detected. Lower finds fainter cells but also more noise
- **Background Window (images each side)**: How many images before and after each image make up its background. Anything that stays in place for about this long is ignored. Cells must move further than their own size over this many images to be detected
- **Counting Line x (px, 0 = auto)**: Where cells are counted, as a pixel column. Place it in the main channel past the electrodes and before the outlets split. 0 puts it three quarters of the way across the image
- **Outlet Divider y (px, 0 = auto)**: Cells crossing the counting line above this pixel row are counted as top outlet (pDEP), below it as bottom outlet (nDEP). Line it up with the tip that splits the outlets. 0 uses the middle of the image
- **Min / Max Speck Area (px)**: Specks smaller or larger than this are ignored
- **Region Before Line / Region Top y / Region Bottom y (px)**: Only this part of each image is analysed: from this many pixels left of the counting line to the right edge, between the top and bottom rows (shown as a dark green box in the preview). Keep the channel and outlets inside it. Start it well before the line (the default is 1000 px) so each cell's track has learned its speed before it reaches the line; with less, crowded cells can be mixed up. Region Before Line = 0 starts at the left edge, Region Bottom y = 0 uses the bottom of the image
- **Processor Cores (0 = all)**: How many processor cores to use. Each video is split into chunks processed side by side, so more cores is faster with the same result

The defaults in `COUNT/settings.json` (counting line x = 1500, divider y = 1230, region rows 700–1750) match the 2048 × 2048 recordings used to validate COUNT. Settings can be saved with **File → Save as default settings** and loaded with **File → Load settings**.

## Speed
Most of the time goes into building each image's background. COUNT keeps this down by analysing only the processing region, reading uncompressed AVI files directly (only the rows it needs), and using all processor cores. On a 2-core test machine, the two 2048 × 2048 validation videos (166 and 201 images) take about 30–40 seconds each; a computer with more cores is faster.

## Output
For each video, in the chosen CSV folder:
- `<file>_results.csv`: one row per counted cell, with its last position and size, `frames_tracked` (frames it was detected in), `DEP_response` (True = top outlet), `count_frame` (the frame in which it crossed the counting line) and `y_at_count_line`
- `<file>_trajectory_results.csv`: one row per counted cell per frame it was detected in
- Optional overlay images

## Overlay Visualization
When the "Save Overlay?" option is selected, the program will save an image of every frame showing:
- Every moving speck detected (orange circles)
- The counting line (blue) and outlet divider (grey)
- Each cell as it is counted (green circle on the line = top outlet / pDEP, pink = bottom outlet / nDEP)
- The running nDEP and pDEP counts

These frames are saved as `<file>_<frame>.png` in the `overlay/` subdirectory of your chosen CSV output path and can be combined into a video using external tools like ffmpeg, for example:
```
ffmpeg -framerate 30 -i "overlay/<file>_%05d.png" overlay.mp4
```

## Plotting Results
Run `python COUNT/bg_plotting.py`:
1. Choose the folder of `*_results.csv` files (one per trial, with the frequency in the file name, e.g. `A375 20kHz_results.csv`). This writes `final_results/final_results.csv` inside that folder with the pDEP percentage per trial.
2. Choose a `final_results.csv` (or a folder of them) to plot pDEP % against frequency.

## Synthetic Test Videos
`tools/make_synthetic_videos.py` makes timelapses of cells flowing left to right into the top and bottom outlets, each with an answer key listing every cell and its outlet:
```
python tools/make_synthetic_videos.py synthetic_videos
```
It makes three 8-bit AVI videos (12 cells each): `synth_easy_10kHz` (slow, well spaced), `synth_fast_50kHz` (cells crossing the image in 13-15 frames) and `synth_crowded_100kHz` (cells close together, side by side, pausing, and briefly out of focus). These videos are 512 × 256 pixels, so set **Counting Line x** and **Outlet Divider y** to 0 (automatic) before running COUNT on the folder, then compare each `<file>_results.csv` with its `<file>_answer_key.csv`. To also test ND2 reading, convert an AVI in NIS-Elements (open it, then **File → Save As** `.nd2`).

## Checking COUNT Against Hand Counts
Put the hand counts in each video's file name, as a number before `nDEP` (bottom outlet) and a number before `pDEP` (top outlet), for example `10V_50kHz_84nDEP_13pDEP.avi`. Then run:
```
python tools/validate_counts.py "D:/COUNT test data"
```
This runs COUNT with the settings in `COUNT/settings.json` on every video in the folder and prints its counts next to the hand counts. Add `--sweep` to also try 216 combinations of detection threshold, minimum speck size, maximum movement, timeout and counting line position, and list the ones closest to the hand counts (each video is still only read once). All results are saved to `validation_results.csv`.

With only a few hand-counted videos, the best-scoring settings may be tuned to those videos in particular, so check new settings on a video that wasn't used to choose them.

## Running the Tests
From the repository folder:
```
python -m unittest discover COUNT/tests
```
The tests use synthetic data, so they don't need ND2 files or the Nikon SDK.

## Contributing
Contributions are welcome! Feel free to open issues or pull requests.

Contact: bschelske@iastate.edu
