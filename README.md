
# Count Objects Until No Tomorrow (C.O.U.N.T.)
🧛‍♂️: _Ah-ah-ah!_

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![MIT License](https://img.shields.io/badge/license-MIT-green)
## Description
An object tracking system for analyzing cell movement and DEP response in microfluidic devices, useful in cancer diagnostics and single-cell analysis.

**General workflow:**
1. **Input Files**: Processes Nikon ND2 files directly using the `ND2Reader_SDK` library
2. **Background Subtraction**: MOG2 foreground segmentation isolates moving objects
3. **Edge Detection**: Canny edge detection is used to find contours of objects
4. **Contour Processing**: Overlapping contours are added to a mask and processed to identify distinct objects
5. **Tracking**: Each frame, every detected object is paired with at most one existing track (closest pairs first); cells are assumed to flow left to right
6. **Classification**: Each cell is assigned to an outlet from the centre of its last detected position: top half of the image = top outlet (positive DEP response), bottom half = bottom outlet (no positive DEP response)
7. **Results**: Tracking data is exported to CSV files for further analysis

## Features
- Direct processing of ND2 files through the PIMS library
- Customizable edge detection parameters through a user-friendly interface
- Background subtraction for improved object detection
- Centroid-based tracking of objects across frames
- Optional visual overlay of tracked objects with unique IDs
- CSV export of tracking results including position, size, and DEP response data
- Configurable tracking parameters (max centroid distance, timeout threshold, etc.)
- Settings persistence through JSON files

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
2. Select an individual ND2 file or a folder containing multiple ND2 files (choosing one replaces the other)
3. Choose where to save the CSV results (optional)
4. Configure tracking parameters:
   - **Canny Upper/Lower**: Threshold values for edge detection (3:1 or 2:1 ratio recommended)
   - **Max Centroid Distance**: Maximum pixel distance an object can travel between frames
   - **Timeout Threshold**: Number of frames before a lost object is considered gone
   - **Expected Cell Radius**: Used for contour processing to improve detection
   - **Min Frames to Count**: Number of frames a cell must be detected in to be counted
5. Optionally check "Save Overlay?" to generate visualization frames with tracking data
6. Preview edge detection to confirm parameter settings
7. Click "Confirm" to begin processing

## Understanding the Parameters
- **Canny Upper/Lower**: Controls sensitivity of edge detection. Lower values detect more edges but may introduce noise
- **Max Centroid Distance**: Determines how far an object can move between frames while still being considered the same object
- **Timeout Threshold**: If an object disappears temporarily, tracking will continue if it reappears within this number of frames
- **Expected Cell Radius**: Adds padding to detected contours to better capture cell boundaries. Contours wider or taller than 10× this value are ignored. A cell may also move left by up to this many pixels between frames (outline jitter) and still be matched
- **Min Frames to Count**: Tracks seen in fewer frames than this are treated as noise and not counted. Raise it if noise is being counted; lower it if fast cells crossing the field of view in few frames are being missed

Settings can be saved with **File → Save as default settings** (written to `COUNT/settings.json`) and loaded with **File → Load settings**.

## Output
For each ND2 file, in the chosen CSV folder:
- `<file>_results.csv`: one row per counted cell, with its last position and size, `frames_tracked` (frames it was detected in) and `DEP_response` (True = top outlet)
- `<file>_trajectory_results.csv`: one row per counted cell per frame it was detected in
- Optional overlay images showing tracked objects with unique IDs

## Overlay Visualization
When the "Save Overlay?" option is selected, the program will save visual frames showing:
- Original frame with detected object contours
- Unique ID labels for each tracked cell (once it has been seen in enough frames to count)
- A grey line marking the divide between the top and bottom outlets
- Count information (objects in frame, cells counted so far)

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
It makes three 8-bit AVI videos (12 cells each): `synth_easy_10kHz` (slow, well spaced), `synth_fast_50kHz` (cells crossing the image in 13-15 frames) and `synth_crowded_100kHz` (cells close together, side by side, pausing, and briefly out of focus). COUNT reads ND2 files, so convert each AVI in NIS-Elements (open it, then **File → Save As** `.nd2`), run COUNT on it with the default settings, and compare `<file>_results.csv` with `<file>_answer_key.csv`.

## Running the Tests
From the repository folder:
```
python -m unittest discover COUNT/tests
```
The tests use synthetic data, so they don't need ND2 files or the Nikon SDK.

## Contributing
Contributions are welcome! Feel free to open issues or pull requests.

Contact: bschelske@iastate.edu
