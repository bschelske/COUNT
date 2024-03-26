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

## Notes to self:
ffmpeg -i 50_kHz.mp4 -vf fps=1 image-%03d.png

convert to images until 1 second into video

ffmpeg -ss 0 -t 1 -i 50_kHz.mp4 image-%03d.png

ffmpeg -framerate 7 -i canny_image-%03d.png canny.mp4

ffmpeg -framerate 10 -i frame_%d.png tracking.mp4


