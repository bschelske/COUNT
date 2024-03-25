"""
===================
Canny edge detector
===================

The Canny filter is a multi-stage edge detector. It uses a filter based on the
derivative of a Gaussian in order to compute the intensity of the gradients.The
Gaussian reduces the effect of noise present in the image. Then, potential
edges are thinned down to 1-pixel curves by removing non-maximum pixels of the
gradient magnitude. Finally, edge pixels are kept or removed using hysteresis
thresholding on the gradient magnitude.

The Canny has three adjustable parameters: the width of the Gaussian (the
noisier the image, the greater the width), and the low and high threshold for
the hysteresis thresholding.

Ben: ffmpeg code
cd  PycharmProjects\scikit\to_image

ffmpeg -i 50_kHz.mp4 -vf fps=1 image-%03d.png
convert to images until 1 second into video
ffmpeg -ss 0 -t 1 -i 50_kHz.mp4 image-%03d.png

ffmpeg -framerate 7 -i canny_image-%03d.png canny.mp4

"""
import os
import tempfile
import shutil
from tracking import tracking, export_to_csv, get_frames
from background_subtraction import nd2_background_subtraction
from ui import create_UI

app = create_UI()
ROI = app.get_roi()
input_file_path = app.file_path
input_folder_path = app.folder_path.get()
print(ROI)
print(input_file_path)
print(input_folder_path)

# # Parameters for tracking function
spots = []  # spot in spots = (x,y,w,h)
output_path = "nd2_results/frame_"  # If overlay = true, save here
canny_lower = 255 // 3
canny_upper = 255

#
# brittanys_path = "Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/"
# brittanys_files = [os.path.join(brittanys_path, f) for f in os.listdir(brittanys_path)]
brittanys_files = ['Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 100kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 105kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 10kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 110kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 115kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 120kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 125kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 130kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 135kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 140kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 145kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 150kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 155kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 15kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 160kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 165kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 170kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 170kHz001.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 175kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 180kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 185kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 190kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 195kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 200kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 20kHz001.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 25kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 30kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 35kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 40kHz001.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 45kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 50kHz..nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 55kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 5kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 60kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 65kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 70kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 75kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 80kHz..nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 85kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 90kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 95kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 100kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 105kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 10kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 110kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 115kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 120kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 125kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 130kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 135kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 140kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 145kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 150kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 155kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 15kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 160kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 165kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 170kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 175kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 180kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 190kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 195kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 200kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 20kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 25kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 30kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 35kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 40kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 45kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 50kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 55kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 5kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 60kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 65kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 70kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 75kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 80kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 85kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 90kHz.nd2', 'Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 RosetteSep 95kHz.nd2']
brittanys_files = ['Z:/Brittany/CF-DEP/CF-DEP Blood 23Feb2024/23Feb2024 Non RosetteSep 5kHz.nd2']

for index, nd2_file in enumerate(brittanys_files):
    print(f"Processing file {index + 1} of {len(brittanys_files)} in brittanys files: {nd2_file}")
    # Get the filename from brittany's files
    file_name = os.path.basename(nd2_file[:-4])

    # Make temp folder for images
    temp_dir = tempfile.mkdtemp()
    print("Temporary directory created:", temp_dir)

    # Convert nd2 file to png files and store into a folder
    nd2_background_subtraction(nd2_file, temp_dir)  # save to temp

    # Load frames from temp
    frame_directory = temp_dir
    print(f'Getting frames from {frame_directory}')
    frames = get_frames(parent_dir=frame_directory)

    # Perform tracking
    print("Tracking")
    overlay_frames, object_final_position, active_id_trajectory = tracking(frames, output_path, ROI, spots, canny_upper,
                                                                           canny_lower, draw_ROI=False,
                                                                           save_overlay=False)
    print("Creating csv files")
    # Create csv file from tracking info
    csv_filename = f"results/{file_name}_results.csv"
    export_to_csv(object_final_position, csv_filename)
    print(f"{csv_filename} saved")

    # # Create csv file from tracking info
    # csv_filename = f"results/active_id_trajectory.csv"
    # export_to_csv(active_id_trajectory, csv_filename)
    # print(f"{csv_filename} saved")

    shutil.rmtree(temp_dir)
    print("Temporary directory deleted:", temp_dir)

# ffmpeg to video code
# ffmpeg -framerate 10 -i frame_%d.png tracking.mp4

# TODO: diagnose overcounting
# TODO: GPU acceleration :)
