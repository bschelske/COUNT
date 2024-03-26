# Count Objects Until No Tomorrow (C.O.U.N.T.)
_Ah-ah-ah!_ 
## Description
This project is an object tracking system implemented in Python. It provides functionality for detecting and 
tracking objects in video frames using cv2 canny edge detection. The intended application of this project is to 
track cells that flow through a CF-DEP device. Cell response to DEP is recorded within results.csv 

## Features
- Object detection and tracking in video frames or .nd2 files.
- Exporting tracking data to a CSV file.
- User interface 

## How to install
Installation is extremely simple, just

## How to use:
Run main.py

A UI will appear with lots of information.

The region of interest (ROI) is where the tracking will occur.

"Choose .csv path" will change where the results file is saved. By default, the file will save in the "results" folder.

The "Save overlay?" checkbox will allow the user to save an overlay of tracked object labels on top of the inputted 
frames. The overlay frames currently save to the folder named 'nd2_results'. If you are only interested in counts 
of cells, you don't need to save overlays. The overlay is a good way to visualize tracking behavior. 


"Preview ROI" will allow you to define your own ROI on an image. A file/folder must be selected first to open an 
image. To create the ROI, click and drag on the image in the new window. To confirm your ROI, press "enter" or 
"space" and to cancel your selection, press "c"

## Notes to self:
ffmpeg -i 50_kHz.mp4 -vf fps=1 image-%03d.png

convert to images until 1 second into video

ffmpeg -ss 0 -t 1 -i 50_kHz.mp4 image-%03d.png

ffmpeg -framerate 7 -i canny_image-%03d.png canny.mp4

ffmpeg -framerate 10 -i frame_%d.png tracking.mp4


