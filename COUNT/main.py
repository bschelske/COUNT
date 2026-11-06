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
import sys
import time
import traceback

# Allow running this file directly (python COUNT/main.py, or Run in PyCharm) as well as with python -m COUNT.main
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from COUNT import ui, tracking


def main():
    # Create UI. Information from the UI is stored within the class-instance "app"
    # This class is initiated in ui.py
    app = ui.create_ui()
    if not app.confirmed:  # window closed without clicking Confirm
        print("No files processed.")
        return

    # Iterate through all input files (could be one file or many)
    for index, nd2_file in enumerate(app.files):
        print(f"Processing file {index + 1} of {len(app.files)}: {nd2_file}")

        try:
            # Get the filename without extension
            file_name = os.path.splitext(os.path.basename(nd2_file))[0]

            # Track the .nd2 or .avi file using MOG2 background subtraction
            counted_objects = tracking.nd2_mog_contours(nd2_file, app)

            csv_filename = os.path.join(app.csv_folder_path.get(), f"{file_name}_results.csv")
            trajectory_csv_filename = os.path.join(app.csv_folder_path.get(), f"{file_name}_trajectory_results.csv")
            tracking.export_to_csv(counted_objects, csv_filename)
            tracking.export_trajectories_to_csv(counted_objects.values(), trajectory_csv_filename)

        except Exception as e:
            print(f"Error processing file {nd2_file}: {e}")
            traceback.print_exc()


if __name__ == '__main__':
    start_time = time.time()
    main()
    duration = time.time() - start_time
    print(f"Time elapsed: {duration:.2f} s")