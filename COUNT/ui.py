import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from pims import ND2Reader_SDK

"""
This code handles the creation of the user interface (UI).

The UI is a tkinter object. tkinter is a python library for UIs. 
create_UI() is called by main.py to make the UI appear. This function returns an instance of the UI.
The UI is used to gather parameters for the tracking. 
"""


class ROISelectionApp:
    """Contains all information to begin tracking, collected in the user interface"""
    def __init__(self, master):
        """Initiate the class and define default values for tracking parameters"""
        self.master = master
        self.file_path = ""
        self.folder_path = tk.StringVar()  # Variable to store the selected folder path
        self.csv_folder_path = tk.StringVar(value="results/")  # Default csv save path
        self.overlay_path = ""
        self.roi_x = tk.IntVar(value=10)  # Default value for ROI X
        self.roi_y = tk.IntVar(value=0)   # Default value for ROI Y
        self.roi_height = tk.IntVar(value=2048)  # Default value for ROI Height
        self.roi_width = tk.IntVar(value=400)    # Default value for ROI Width
        self.canny_upper = tk.IntVar(value=255)  # Default value for upper canny threshold (3:1 ratio)
        self.canny_lower = tk.IntVar(value=85)    # Default value for lower canny threshold
        self.max_centroid_distance = tk.IntVar(value=70)  # max distance an object will travel between frames (px)
        self.timeout = tk.IntVar(value=7)    # How long before an object is considered lost (frames)
        self.cell_radius = tk.IntVar(value=6)
        self.save_overlay = tk.BooleanVar()
        self.files = []  # Empty list that will accept an individual file, or files from a folder
        self.create_widgets()  # This is the part of the UI you can see

    def create_widgets(self):
        """Makes the visual parts of the UI. Ignore unless changing position of UI elements"""
        # File selection button
        tk.Label(self.master, text="For individual .nd2 file").grid(row=0, column=0, padx=5, pady=5)
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Folder selection button
        tk.Label(self.master, text="For .nd2 files in a folder").grid(row=1, column=0, padx=5, pady=5)
        self.folder_button = tk.Button(self.master, text="Choose Folder", command=self.choose_folder)
        self.folder_button.grid(row=1, column=1, padx=5, pady=5)

        # csv output folder selection button
        tk.Label(self.master, text="Choose .csv save path").grid(row=0, column=2, padx=5, pady=5)
        self.csv_button = tk.Button(self.master, text="Choose Folder", command=self.choose_csv_output)
        self.csv_button.grid(row=0, column=3, padx=5, pady=5)

        # ROI X input field
        tk.Label(self.master, text="ROI X (px):").grid(row=2, column=0, padx=5, pady=5)
        self.roi_x_entry = tk.Entry(self.master, textvariable=self.roi_x)
        self.roi_x_entry.grid(row=2, column=1, padx=5, pady=5)

        # ROI Y input field
        tk.Label(self.master, text="ROI Y (px):").grid(row=3, column=0, padx=5, pady=5)
        self.roi_y_entry = tk.Entry(self.master, textvariable=self.roi_y)
        self.roi_y_entry.grid(row=3, column=1, padx=5, pady=5)

        # ROI Height input field
        tk.Label(self.master, text="ROI Height (px):").grid(row=4, column=0, padx=5, pady=5)
        self.roi_height_entry = tk.Entry(self.master, textvariable=self.roi_height)
        self.roi_height_entry.grid(row=4, column=1, padx=5, pady=5)

        # ROI Width input field
        tk.Label(self.master, text="ROI Width (px):").grid(row=5, column=0, padx=5, pady=5)
        self.roi_width_entry = tk.Entry(self.master, textvariable=self.roi_width)
        self.roi_width_entry.grid(row=5, column=1, padx=5, pady=5)

        # ROI Preview button
        tk.Label(self.master, text="Define ROI on image").grid(row=6, column=0, padx=5, pady=5)
        self.roi_button = tk.Button(self.master, text="Preview ROI", command=self.preview_roi)
        self.roi_button.grid(row=6, column=1, padx=5, pady=5)

        # Preview Edge Dectection button
        tk.Label(self.master, text="Visualize Edge Detection").grid(row=7, column=2, padx=5, pady=5)
        self.edge_preview_button = tk.Button(self.master, text="Preview Detection", command=self.preview_edge_detection)
        self.edge_preview_button.grid(row=7, column=3, padx=5, pady=5)

        # Canny Upper input field
        tk.Label(self.master, text="Canny Upper:").grid(row=2, column=2, padx=5, pady=5)
        self.canny_upper_entry = tk.Entry(self.master, textvariable=self.canny_upper)
        self.canny_upper_entry.grid(row=2, column=3, padx=5, pady=5)

        # Canny Lower input field
        tk.Label(self.master, text="Canny Lower:").grid(row=3, column=2, padx=5, pady=5)
        self.canny_lower_entry = tk.Entry(self.master, textvariable=self.canny_lower)
        self.canny_lower_entry.grid(row=3, column=3, padx=5, pady=5)

        # Max Centroid Distance input field
        tk.Label(self.master, text="Max Centroid Distance (px):").grid(row=4, column=2, padx=5, pady=5)
        self.max_centroid_distance_entry = tk.Entry(self.master, textvariable=self.max_centroid_distance)
        self.max_centroid_distance_entry.grid(row=4, column=3, padx=5, pady=5)

        # Timeout threshold input field
        tk.Label(self.master, text="Timeout Threshold (frames):").grid(row=5, column=2, padx=5, pady=5)
        self.timeout_entry = tk.Entry(self.master, textvariable=self.timeout)
        self.timeout_entry.grid(row=5, column=3, padx=5, pady=5)

        # Expected Cell Radius input field
        tk.Label(self.master, text="Expected Cell Radius (px):").grid(row=6, column=2, padx=5, pady=5)
        self.cell_radius_entry = tk.Entry(self.master, textvariable=self.cell_radius)
        self.cell_radius_entry.grid(row=6, column=3, padx=5, pady=5)

        # Save overlay checkbox
        self.save_overlay_checkbox = tk.Checkbutton(self.master, text="Save Overlay?", variable=self.save_overlay,  onvalue=True, offvalue=False, command=self.on_checkbox_click)
        self.save_overlay_checkbox.grid(row=8, column=1, columnspan=1, padx=5, pady=5)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="      Confirm     ", command=self.confirm_selections)
        self.confirm_button.grid(row=8, column=2, padx=1, pady=10)

        # Quit button
        self.quit_button = tk.Button(self.master, text="        Quit        ", command=self.quit_ui)
        self.quit_button.grid(row=8, column=3, padx=1, pady=10)

    # Some functions of the UI require functions (haha)
    def choose_file(self):
        self.file_path = filedialog.askopenfilename()
        print("Selection:", self.file_path)

    def choose_folder(self):
        self.folder_path.set(filedialog.askdirectory())  # Set the selected folder path
        print("Selection:", self.folder_path.get())

    def choose_csv_output(self):
        self.csv_folder_path.set(filedialog.askdirectory())  # Set the selected folder path
        print("csv save path:", self.csv_folder_path.get())

    def on_checkbox_click(self):
        if self.save_overlay.get() == 1:
            print("An overlay of tracked objects will be saved!")

    def input_handling(self):
        # Handles if the selection was for a folder or a file.
        if self.folder_path.get():
            self.files = [os.path.join(self.folder_path.get(), f) for f in os.listdir(self.folder_path.get())]
        if self.file_path:
            self.files = [self.file_path]

    def confirm_selections(self):
        self.error_handling()
        self.input_handling()

        # Ensure the output directory exists
        os.makedirs(self.csv_folder_path.get(), exist_ok=True)
        os.makedirs(self.csv_folder_path.get()+"overlay/", exist_ok=True)
        self.overlay_path = self.csv_folder_path.get()+"overlay/"
        os.makedirs(self.csv_folder_path.get()+"final_results/", exist_ok=True)

        print(".csv results save path:", self.csv_folder_path.get())
        print("ROI X:", self.roi_x.get())
        print("ROI Y:", self.roi_y.get())
        print("ROI Height:", self.roi_height.get())
        print("ROI Width:", self.roi_width.get())
        print("Canny Upper:", self.canny_upper.get())
        print("Canny Lower:", self.canny_lower.get())
        # Close the Tkinter window
        self.master.destroy()

    def error_handling(self):
        if self.file_path == '' and self.folder_path.get() == '':
            raise ValueError("No file selected!")
        if self.file_path and self.folder_path.get():
            raise ValueError(
                "...Why did you pick a file AND a folder? Choose ONE or the OTHER. What did you expect to happen? :)")

    def preview_roi(self):
        if self.file_path == '' and self.folder_path.get() == '':
            print("Choose a file first!")
        else:
            self.input_handling()

        with ND2Reader_SDK(self.files[0]) as nd2_file:
            frame_data = nd2_file[0]
            image = frame_data
            cv2.namedWindow("Select ROI. Press enter to confirm, 'c' to cancel", cv2.WINDOW_NORMAL)
            ROI = cv2.selectROI("Select ROI. Press enter to confirm, 'c' to cancel", image, cv2.WINDOW_NORMAL)
            cv2.destroyAllWindows()
            self.roi_x.set(ROI[0])
            self.roi_y.set(ROI[1])
            self.roi_width.set(ROI[2])
            self.roi_height.set(ROI[3])

    def get_roi(self):
        ROI = (self.roi_x.get(), self.roi_y.get(), self.roi_height.get(), self.roi_width.get())
        return ROI

    def preview_edge_detection(self):
        """This method is incredibly jank"""

        # Error handling
        if self.file_path == '' and self.folder_path.get() == '':
            print("Choose a file first!")
        else:
            self.input_handling()

        # Get frames
        frames = [self.edge_detection_handling(3), self.edge_detection_handling(4)]
        # Display images
        cv2.namedWindow('Try pressing "4" or "5" "ESC" to change settings/quit', cv2.WINDOW_NORMAL)
        cv2.imshow('Try pressing "4" or "5" "ESC" to change settings/quit', frames[0])

        while True:
            k = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
            if k == 52:  # Key code 4
                cv2.imshow('Try pressing "4" or "5" "ESC" to change settings/quit', frames[1])  # Display the 2nd frame
            elif k == 53:  # Key code 5
                cv2.imshow('Try pressing "4" or "5" "ESC" to change settings/quit', frames[0])  # Display the 1st frame
            elif k == 27:  # ESC key
                break

    def edge_detection_handling(self, frame_index):
        """Edge detection for UI preview"""
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        with ND2Reader_SDK(self.files[0]) as nd2_file:
            # MOG2 Background Method:
            for frame in nd2_file[:frame_index+1]:  # First five frames to calculate background
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                foreground_mask = backSub.apply(frame)

            # Get current frame
            frame = nd2_file[frame_index]
            frame_copy = frame.copy()  # Copy for overlay later...
            frame_copy = cv2.normalize(frame_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)

            # Do background subtraction, and normalize for canny
            normalized_frame = cv2.normalize(foreground_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            canny_img = cv2.Canny(normalized_frame, self.canny_lower.get(), self.canny_upper.get(), 3)
            contours, hierarchy = cv2.findContours(canny_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            # create an empty mask
            mask = np.zeros(frame_copy.shape[:2], dtype=np.uint8)

            # Draw filled contours on a mask using bounding circles
            for cnt in contours:
                (x, y), radius = cv2.minEnclosingCircle(cnt)
                center = (int(x), int(y))
                radius = int(radius + self.cell_radius.get())
                cv2.circle(mask, center, radius, (255), -1)

            # find the contours on the mask (with solid drawn shapes) and draw outline on input image
            contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                cv2.drawContours(frame_copy, [cnt], 0, (0, 0, 255), 2)

            cv2.putText(frame_copy, str(f"Frame {frame_index+1} Objects: {len(contours)}"), (10,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return frame_copy

    def quit_ui(self):
        self.master.destroy()
        quit(2)


def create_ui():
    root = tk.Tk()
    root.iconbitmap(r'count.ico')
    root.title("Count Objects Until No Tomorrow (C.O.U.N.T.)")
    app = ROISelectionApp(root)
    root.mainloop()
    return app
