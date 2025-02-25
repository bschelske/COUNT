# Count Objects Until No Tomorrow (C.O.U.N.T.)
🧛‍♂️: _Ah-ah-ah!_ 
## Description
This project is an object tracking system implemented in Python. It provides functionality for detecting and 
tracking objects in video frames using cv2 canny edge detection. The intended application of this project is to 
track cells that flow through a CF-DEP device. Cell response to DEP is recorded within results.csv 

**General workflow:**
1. **Input Files**: Processes Nikon ND2 files directly using the `ND2Reader_SDK` library
2. **Background Subtraction**: MOG2 foreground segmentation isolates moving objects
3. **Edge Detection**: Canny edge detection is used to find contours of objects
4. **Contour Processing**: Overlapping contours are added to a mask and processed to identify distinct objects
5. **Tracking**: Objects are tracked between frames using centroid distance calculations
6. **Classification**: Objects are classified based on their position in the frame (DEP responsive vs. non-responsive)
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
2. Install the required dependencies:
```
pip install -r requirements.txt
```
3. Run the main.py file to start the application

## Required Dependencies
- OpenCV (cv2)
- NumPy
- tkinter (for the UI)
- PIMS with ND2Reader_SDK
- tqdm (for progress bars)

## How to Use
1. Run `main.py` to open the user interface
2. Select an individual ND2 file or a folder containing multiple ND2 files
3. Choose where to save the CSV results (optional)
4. Configure tracking parameters:
   - **Canny Upper/Lower**: Threshold values for edge detection (3:1 or 2:1 ratio recommended)
   - **Max Centroid Distance**: Maximum pixel distance an object can travel between frames
   - **Timeout Threshold**: Number of frames before a lost object is considered gone
   - **Expected Cell Radius**: Used for contour processing to improve detection
5. Optionally check "Save Overlay?" to generate visualization frames with tracking data
6. Preview edge detection to confirm parameter settings
7. Click "Confirm" to begin processing

## Understanding the Parameters
- **Canny Upper/Lower**: Controls sensitivity of edge detection. Lower values detect more edges but may introduce noise
- **Max Centroid Distance**: Determines how far an object can move between frames while still being considered the same object
- **Timeout Threshold**: If an object disappears temporarily, tracking will continue if it reappears within this number of frames
- **Expected Cell Radius**: Adds padding to detected contours to better capture cell boundaries

## Output
- CSV files containing tracking data for each processed file
- Optional overlay images showing tracked objects with unique IDs

## Overlay Visualization
When the "Save Overlay?" option is selected, the program will save visual frames showing:
- Original frame with detected object contours
- Unique ID labels for each tracked object
- Count information (objects in frame, total objects tracked)

These frames are saved to the `overlay/` subdirectory of your chosen CSV output path and can be combined into a video using external tools like ffmpeg.

## Contributing
Contributions are welcome! Feel free to open issues or pull requests.

Contact: bschelske@iastate.edu
