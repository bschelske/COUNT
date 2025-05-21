import json
import os
import tkinter as tk
from tkinter import filedialog

import cv2
import numpy as np
from pims import ND2Reader_SDK

from COUNT import background_subtraction as bs
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
        self.canny_upper = tk.IntVar(
            value=self.settings.get("canny_upper"))  # Default value for upper canny threshold (3:1 ratio)
        self.canny_lower = tk.IntVar(value=self.settings.get("canny_lower"))  # Default value for lower canny threshold
        self.max_centroid_distance = tk.IntVar(
            value=self.settings.get("max_centroid_distance"))  # max distance an object will travel between frames (px)
        self.timeout = tk.IntVar(
            value=self.settings.get("timeout"))  # How long before an object is considered lost (frames)
        self.cell_max_radius = tk.IntVar(value=self.settings.get("cell_max_radius"))
        self.cell_min_radius = tk.IntVar(value=self.settings.get("cell_min_radius"))
        self.save_overlay = tk.BooleanVar(value=self.settings.get("save_overlay"))
        self.flow_direction = tk.StringVar(value=self.settings.get("flow_direction"))
        self.background_subtraction = tk.BooleanVar(value=self.settings.get("background_subtraction"))
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

        # Background Subtraction Check Box
        self.background_subtraction_checkbox = tk.Checkbutton(self.master, text="Background subtraction",
                                                              variable=self.background_subtraction, onvalue=True,
                                                              offvalue=False)
        self.background_subtraction_checkbox.grid(row=7, column=1, padx=5, pady=5)

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
        tk.Label(self.master, text="Cell size exclusion, radii (min, max)", anchor=justification).grid(sticky="w",
                                                                                                       row=6, column=0,
                                                                                            padx=5, pady=5)
        self.cell_min_radius_entry = tk.Entry(self.master, textvariable=self.cell_min_radius)
        self.cell_min_radius_entry.grid(row=6, column=1, padx=5, pady=5)

        self.cell_max_radius_entry = tk.Entry(self.master, textvariable=self.cell_max_radius)
        self.cell_max_radius_entry.grid(row=6, column=2, padx=5, pady=5)

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
                             "flow_direction": "Towards Right (--->)",
                             "background_subtraction": True,
                             "cell_min_radius": 2,
                             "cell_max_radius": 6,
                             "save_overlay": False,
                             "csv_save_path": "results/"}

    def save_settings(self):
        settings = {"canny_upper": self.canny_upper.get(),
                    "canny_lower": self.canny_lower.get(),
                    "max_centroid_distance": self.max_centroid_distance.get(),
                    "timeout": self.timeout.get(),
                    "flow_direction": self.flow_direction.get(),
                    "background_subtraction": self.background_subtraction.get(),
                    "cell_min_radius": self.cell_min_radius.get(),
                    "cell_max_radius": self.cell_max_radius.get(),
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
            if 'm' in nd2_file.sizes.keys():  # new nikon weirdness
                nd2_file.iter_axes = 'm'
            self.total_frames = len(nd2_file)
        print("Selection:", self.file_path, self.total_frames, "frames")

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
        if self.save_overlay.get():
            print(f"\noverlay saving in {self.overlay_path}")
        print(".csv results save path:", self.csv_folder_path.get())
        # Close the Tkinter window
        self.master.destroy()

    def error_handling(self):
        # if self.file_path == '' and self.folder_path.get() == '':
        #     raise ValueError("No file selected!")
        if self.file_path and self.folder_path.get():
            raise ValueError(
                "...Why did you pick a file AND a folder? Choose ONE or the OTHER. What did you expect to happen? :)")

    def nothing(self, x):
        pass

    def preview_edge_detection(self):
        """This method is incredibly jank"""
        # Error handling
        if not self.file_path and not self.folder_path.get():
            print("Choose a file first!")
        else:
            self.input_handling()

        # Get frames
        self.preview_frame = 0
        self.canny_upper_preview = self.canny_upper.get()
        self.canny_lower_preview = self.canny_lower.get()
        self.cell_min_radius_preview = self.cell_min_radius.get()
        self.cell_max_radius_preview = self.cell_max_radius.get()

        # Display images
        cv2.namedWindow('Preview (ESC to confirm)', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Preview (ESC to confirm)', 600, 600)
        cv2.createTrackbar('Lower', 'Preview (ESC to confirm)', int(self.canny_lower_preview), 255, self.nothing)
        cv2.createTrackbar('Upper', 'Preview (ESC to confirm)', int(self.canny_upper_preview), 255, self.nothing)
        cv2.createTrackbar('rad_min', 'Preview (ESC to confirm)', int(self.cell_min_radius_preview), 20, self.nothing)
        cv2.createTrackbar('rad_max', 'Preview (ESC to confirm)', int(self.cell_max_radius_preview), 30, self.nothing)
        cv2.createTrackbar('Frame', 'Preview (ESC to confirm)', int(self.preview_frame), self.total_frames - 1,
                           self.nothing)

        while True:
            self.canny_upper_preview = cv2.getTrackbarPos('Upper', 'Preview (ESC to confirm)')
            self.canny_lower_preview = cv2.getTrackbarPos('Lower', 'Preview (ESC to confirm)')
            self.cell_min_radius_preview = cv2.getTrackbarPos('rad_min', 'Preview (ESC to confirm)')
            self.cell_max_radius_preview = cv2.getTrackbarPos('rad_max', 'Preview (ESC to confirm)')

            # Lower should not exceed upper
            if self.canny_lower_preview > self.canny_upper_preview:
                self.canny_lower_preview = self.canny_upper_preview
                cv2.setTrackbarPos('Upper', 'Preview (ESC to confirm)', self.canny_upper_preview)
                cv2.setTrackbarPos('Lower', 'Preview (ESC to confirm)', self.canny_lower_preview)

            # Min radius should not exceed max
            if self.cell_min_radius_preview > self.cell_max_radius_preview:
                self.cell_min_radius_preview = self.cell_max_radius_preview
                cv2.setTrackbarPos('rad_min', 'Preview (ESC to confirm)', self.cell_max_radius_preview)
                cv2.setTrackbarPos('rad_max', 'Preview (ESC to confirm)', self.cell_min_radius_preview)

            self.preview_frame = cv2.getTrackbarPos('Frame', 'Preview (ESC to confirm)')
            frame = self.preview_detection()
            cv2.imshow('Preview (ESC to confirm)', frame)

            self.canny_upper.set(self.canny_upper_preview)
            self.canny_lower.set(self.canny_lower_preview)
            self.cell_min_radius.set(self.cell_min_radius_preview)
            self.cell_max_radius.set(self.cell_max_radius_preview)
            k = cv2.waitKey(30) & 0xFF  # Wait indefinitely for a key press
            if k == 27:  # ESC key
                break
        cv2.destroyAllWindows()

    def preview_detection(self):
        """Run contour detection using tracking.process_image() and user input from trackbars"""

        with ND2Reader_SDK(self.files[0]) as nd2_file:
            if 'm' in nd2_file.sizes.keys():  # new nikon weirdness
                nd2_file.iter_axes = 'm'

            frame_data = nd2_file[self.preview_frame]
            # 8-bit normalization required for Canny
            frame_data = cv2.normalize(frame_data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            save_overlay_frame = frame_data.copy()
            save_overlay_frame = cv2.cvtColor(save_overlay_frame, cv2.COLOR_GRAY2BGR)

            # Perform operations and canny edge detection to obtain contours
            if self.background_subtraction.get() is False:
                # No background subtraction
                contours = tracking.process_image(frame_data, self)
            else:
                # Background subtraction
                # Initialize with first few frames
                frame_data = nd2_file[self.preview_frame]
                num_bg_samples = min(20, self.total_frames)  # More samples for better background modeling
                background_sample = np.array([nd2_file[i] for i in range(num_bg_samples)], dtype=np.uint16)
                background = bs.create_background_model(background_sample, method='median')
                # # For tuning and debugging - try different methods
                contours, mask_adaptive = bs.detect_cells(frame_data, background, self, threshold_method='adaptive')

            save_overlay_frame = cv2.drawContours(save_overlay_frame, contours, -1, (0, 0, 255), 2)
            cv2.putText(save_overlay_frame, str(f"Frame {self.preview_frame} Objects: {len(contours)}"), (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return save_overlay_frame

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
