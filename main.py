"""
===================
C. O. U. N. T.
    Count objects until no tomorrow...
    By Ben
===================
Check the readme on github for more information. https://github.com/bschelske/C.O.U.N.T.
    "Ah- ah- ah!"
"""
import os
import tempfile
import shutil
import tracking
from background_subtraction import nd2_background_subtraction
import ui

# Create UI and handle errors. Information from the UI is stored within the class-instance "app"
try:
    app = ui.create_UI()
except ValueError as e:
    print("An error occurred in the UI:", e)
    exit(1)

# Utilize inputs from UI (might be able to just pass "app" into tracking()... )
ROI = app.get_roi()
input_file_path = app.file_path
input_folder_path = app.folder_path.get()
canny_lower = app.canny_lower.get()
canny_upper = app.canny_upper.get()
save_overlay = app.save_overlay.get()
timeout = app.timeout.get()
max_centroid_distance = app.max_centroid_distance.get()

# Additional parameters for tracking function
# Spots are rectangular regions to be ignored in tracking. (likely obsolete with inclusion of background subtraction)
spots = []  # spot in spots = (x,y,w,h)
output_path = "nd2_results/frame_"  # If overlay = true, save here

# Iterate through all input files
for index, nd2_file in enumerate(app.files):
    print(f"Processing file {index + 1} of {len(app.files)}: {nd2_file}")
    file_name = os.path.basename(nd2_file[:-4])  # Get the filename from file
    temp_dir = tempfile.mkdtemp()  # Make temp folder for images
    print("Temporary directory created:", temp_dir)

    # Convert nd2 file to png files and store into the temp folder
    nd2_background_subtraction(nd2_file, temp_dir)

    # Load png files from temp folder
    frame_directory = temp_dir
    print(f'Getting frames from {frame_directory}')
    frames = tracking.get_frames(parent_dir=frame_directory)

    # Perform tracking
    print("Tracking...")
    overlay_frames, object_final_position, active_id_trajectory = \
        tracking.tracking(frames, output_path, ROI, spots, canny_upper, canny_lower, max_centroid_distance, timeout,
                          draw_ROI=False, save_overlay=save_overlay)

    # Create csv file from tracking info
    print("Creating csv files")
    csv_filename = f"{app.csv_folder_path.get()}{file_name}_results.csv"
    tracking.export_to_csv(active_id_trajectory, csv_filename)
    print(f"{csv_filename} saved")

    # # Create csv file from tracking info
    # csv_filename = f"results/active_id_trajectory.csv"
    # export_to_csv(active_id_trajectory, csv_filename)
    # print(f"{csv_filename} saved")

    shutil.rmtree(temp_dir)
    print("Temporary directory deleted:", temp_dir)

# TODO: diagnose overcounting
# TODO: GPU acceleration :)
