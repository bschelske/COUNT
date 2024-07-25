"""
===================
C. O. U. N. T.
    Count objects until no tomorrow...
    By Ben
===================
Check the readme on gitHub for more information. https://github.com/bschelske/COUNT
    "Ah- ah- ah!"

This is the file that is meant to be run.
"""
import os
from COUNT import ui, tracking


def main():
    # Create UI and handle errors. Information from the UI is stored within the class-instance "app"
    # This class is initiated in ui.py
    try:
        app = ui.create_ui()
    except ValueError as e:
        print("An error occurred in the UI:", e)
        exit(1)

    # Iterate through all input files (could be one file or many)
    for index, nd2_file in enumerate(app.files):
        print(f"Processing file {index + 1} of {len(app.files)}: {nd2_file}")


        # Get the filename without extension
        file_name = os.path.basename(nd2_file)[:-4]

        # Track the nd2 file using MOG2 background subtraction
        object_final_position, active_id_trajectory = tracking.nd2_mog_contours(nd2_file, app)

        csv_filename = os.path.join(app.csv_folder_path.get(), f"{file_name}_results.csv")
        tracking.export_to_csv(object_final_position, csv_filename)
        print(f"{csv_filename} saved")



if __name__ == '__main__':
    main()
