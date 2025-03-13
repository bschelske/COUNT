import os
import json
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from pims import ND2Reader_SDK
from COUNT import tracking
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
        self.SETTINGS_PATH = "settings.json"
        self.load_settings()
        self.file_path = ""
        self.folder_path = tk.StringVar()  # Variable to store the selected folder path
        self.csv_folder_path = tk.StringVar(value=self.settings.get("csv_save_path"))  # Default csv save path
        self.overlay_path = ""
        self.roi_height = tk.IntVar(value=2048)  # Default value for ROI Height
        self.roi_width = tk.IntVar(value=2048)  # Default value for ROI Width
        self.canny_upper = tk.IntVar(
            value=self.settings.get("canny_upper"))  # Default value for upper canny threshold (3:1 ratio)
        self.canny_lower = tk.IntVar(value=self.settings.get("canny_lower"))  # Default value for lower canny threshold
        self.max_centroid_distance = tk.IntVar(
            value=self.settings.get("max_centroid_distance"))  # max distance an object will travel between frames (px)
        self.timeout = tk.IntVar(
            value=self.settings.get("timeout"))  # How long before an object is considered lost (frames)
        self.cell_radius = tk.IntVar(value=self.settings.get("cell_radius"))
        self.save_overlay = tk.BooleanVar(value=self.settings.get("save_overlay"))
        self.flow_direction = tk.StringVar(value="Towards Right (--->)")
        self.files = []  # Empty list that will accept an individual file, or files from a folder
        self.create_toolbar()
        self.create_widgets()  # This is the part of the UI you can see

    def create_toolbar(self):
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        file_ = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='File', menu=file_)
        file_.add_command(label='Save as default settings', command=self.save_settings)
        file_.add_command(label='Load settings', command=self.choose_settings_file)
        file_.add_command(label='Exit', command=self.master.quit)

        help_ = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='Help', menu=help_)
        help_.add_command(label='Help', command=self.help)

    def create_widgets(self):
        """Makes the visual parts of the UI. Ignore unless changing position of UI elements"""
        justification = "w"
        # File selection button
        tk.Label(self.master, text="For individual .nd2 file", anchor=justification).grid(sticky="w", row=0, column=0,
                                                                                          padx=5, pady=5)
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Folder selection button
        tk.Label(self.master, text="For .nd2 files in a folder", anchor=justification).grid(sticky="w", row=1, column=0,
                                                                                            padx=5, pady=5)
        self.folder_button = tk.Button(self.master, text="Choose Folder", command=self.choose_folder)
        self.folder_button.grid(row=1, column=1, padx=5, pady=5)

        # csv output folder selection button
        tk.Label(self.master, text="Choose .csv save path", anchor=justification).grid(sticky="w", row=0, column=2,
                                                                                       padx=5, pady=5)
        self.csv_button = tk.Button(self.master, text="Choose Folder", command=self.choose_csv_output)
        self.csv_button.grid(row=0, column=3, padx=5, pady=5)

        # Preview Edge Dectection button
        tk.Label(self.master, text="Visualize Edge Detection", anchor=justification).grid(sticky="w", row=8, column=0,
                                                                                          padx=5, pady=5)
        self.edge_preview_button = tk.Button(self.master, text="Preview Detection", command=self.preview_edge_detection)
        self.edge_preview_button.grid(row=8, column=1, padx=5, pady=5)

        # Flow direction
        tk.Label(self.master, text="Flow direction", anchor=justification).grid(sticky="w", row=7, column=0, padx=5,
                                                                                pady=5)
        self.flow_direction_button = tk.OptionMenu(self.master, self.flow_direction,
                                                 *["Towards Right (--->)", "Towards left (<---)"])

        self.flow_direction_button.grid(row=7, column=1, padx=5, pady=5)

        # Canny Upper input field
        tk.Label(self.master, text="Canny Upper:", anchor=justification).grid(sticky="w", row=2, column=0, padx=5,
                                                                              pady=5)
        self.canny_upper_entry = tk.Entry(self.master, textvariable=self.canny_upper)
        self.canny_upper_entry.grid(row=2, column=1, padx=5, pady=5)

        # Canny Lower input field
        tk.Label(self.master, text="Canny Lower:", anchor=justification).grid(sticky="w", row=3, column=0, padx=5,
                                                                              pady=5)
        self.canny_lower_entry = tk.Entry(self.master, textvariable=self.canny_lower)
        self.canny_lower_entry.grid(row=3, column=1, padx=5, pady=5)

        # Max Centroid Distance input field
        tk.Label(self.master, text="Max Centroid Distance (px):", anchor=justification).grid(sticky="w", row=4,
                                                                                             column=0, padx=5, pady=5)
        self.max_centroid_distance_entry = tk.Entry(self.master, textvariable=self.max_centroid_distance)
        self.max_centroid_distance_entry.grid(row=4, column=1, padx=5, pady=5)

        # Timeout threshold input field
        tk.Label(self.master, text="Timeout Threshold (frames):", anchor=justification).grid(sticky="w", row=5,
                                                                                             column=0, padx=5, pady=5)
        self.timeout_entry = tk.Entry(self.master, textvariable=self.timeout)
        self.timeout_entry.grid(row=5, column=1, padx=5, pady=5)

        # Expected Cell Radius input field
        tk.Label(self.master, text="Expected Cell Radius (px):", anchor=justification).grid(sticky="w", row=6, column=0,
                                                                                            padx=5, pady=5)
        self.cell_radius_entry = tk.Entry(self.master, textvariable=self.cell_radius)
        self.cell_radius_entry.grid(row=6, column=1, padx=5, pady=5)

        # Save overlay checkbox
        self.save_overlay_checkbox = tk.Checkbutton(self.master, text="Save Overlay?", variable=self.save_overlay,
                                                    onvalue=True, offvalue=False, command=self.on_checkbox_click)
        self.save_overlay_checkbox.grid(row=9, column=0, columnspan=1, padx=5, pady=5)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="      Confirm     ", command=self.confirm_selections)
        self.confirm_button.grid(sticky="w", row=9, column=2, padx=1, pady=10)

        # Quit button
        self.quit_button = tk.Button(self.master, text="        Quit        ", command=self.quit_ui)
        self.quit_button.grid(sticky="w", row=9, column=3, padx=1, pady=10)

    def choose_settings_file(self):
        self.SETTINGS_PATH = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        self.load_settings()
        tk.messagebox.showinfo("Settings", f"Settings loaded from:\n{self.SETTINGS_PATH}")

    def load_settings(self):
        try:
            with open(self.SETTINGS_PATH, "r") as file:
                self.settings = json.load(file)
        except FileNotFoundError:
            print("no settings.json found... using defaults")
            self.settings = {"canny_upper": 255,
                             "canny_lower": 85,
                             "max_centroid_distance": 70,
                             "timeout": 5,
                             "cell_radius": 6,
                             "save_overlay": False,
                             "csv_save_path": "results/"}

    def save_settings(self):
        settings = {"canny_upper": self.canny_upper.get(),
                    "canny_lower": self.canny_lower.get(),
                    "max_centroid_distance": self.max_centroid_distance.get(),
                    "timeout": self.timeout.get(),
                    "cell_radius": self.cell_radius.get(),
                    "save_overlay": self.save_overlay.get(),
                    "csv_save_path": self.csv_folder_path.get()
                    }
        with open(self.SETTINGS_PATH, "w") as file:
            json.dump(settings, file, indent=4)
        tk.messagebox.showinfo("Settings", f"Settings saved to\n{self.SETTINGS_PATH}")

    def help(self):
        tk.messagebox.showinfo("Help", "no help here")

    def choose_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("ND2 files", "*.nd2")])
        with ND2Reader_SDK(self.file_path) as nd2_file:
            self.roi_width.set(nd2_file.metadata['width'])
            self.roi_height.set(nd2_file.metadata['height'])
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
        os.makedirs(self.csv_folder_path.get() + "overlay/", exist_ok=True)
        self.overlay_path = self.csv_folder_path.get() + "overlay/"
        os.makedirs(self.csv_folder_path.get() + "final_results/", exist_ok=True)

        print(".csv results save path:", self.csv_folder_path.get())
        # Close the Tkinter window
        self.master.destroy()

    def error_handling(self):
        if self.file_path == '' and self.folder_path.get() == '':
            raise ValueError("No file selected!")
        if self.file_path and self.folder_path.get():
            raise ValueError(
                "...Why did you pick a file AND a folder? Choose ONE or the OTHER. What did you expect to happen? :)")


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
            # Check if the window is still open
            if cv2.getWindowProperty('Try pressing "4" or "5" "ESC" to change settings/quit', cv2.WND_PROP_VISIBLE) < 1:
                break  # Exit loop if the window is closed

            k = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
            if k == 53:  # Key code 5
                cv2.imshow('Try pressing "4" or "5" "ESC" to change settings/quit', frames[1])  # Display the 2nd frame
            elif k == 52:  # Key code 4
                cv2.imshow('Try pressing "4" or "5" "ESC" to change settings/quit', frames[0])  # Display the 1st frame
            elif k == 27:  # ESC key
                break

    def edge_detection_handling(self, frame_index):
        """Edge detection for UI preview"""
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        with ND2Reader_SDK(self.files[0]) as nd2_file:
            if 'm' in nd2_file.sizes.keys():  # new nikon weirdness
                nd2_file.iter_axes = 'm'
            # MOG2 Background Method:
            for frame in nd2_file[:frame_index + 1]:  # First five frames to calculate background
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                foreground_mask = backSub.apply(frame)

            # Get current frame
            frame = nd2_file[frame_index]
            frame_copy = frame.copy()  # Copy for overlay later...
            frame_copy = cv2.normalize(frame_copy, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            frame_copy = cv2.cvtColor(frame_copy, cv2.COLOR_GRAY2BGR)

            # Do background subtraction, and normalize for canny
            normalized_frame = cv2.normalize(foreground_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # Morphological operation to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            normalized_frame = cv2.morphologyEx(normalized_frame, cv2.MORPH_OPEN, kernel)
            canny_img = cv2.Canny(normalized_frame, self.canny_lower.get(), self.canny_upper.get(), 3)
            contours, hierarchy = cv2.findContours(canny_img, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)

            _, contours = tracking.remove_overlapped_objects(normalized_frame, contours, self.cell_radius.get())
            total_contours = len(contours)
            for index, cnt in enumerate(contours):
                x, y, w, h = cv2.boundingRect(cnt)
                if (w > self.cell_radius.get() * 10) or (h > self.cell_radius.get() * 10):
                    total_contours -= 1
                else:
                    cv2.drawContours(frame_copy, [cnt], 0, (0, 0, 255), 2)

            cv2.putText(frame_copy, str(f"Frame {frame_index + 1} Objects: {len(contours)}"), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
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
