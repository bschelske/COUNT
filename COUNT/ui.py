import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
from COUNT import tracking
"""
This code handles the creation of the user interface (UI).

The UI is a tkinter object. tkinter is a python library for UIs.
create_UI() is called by main.py to make the UI appear. This function returns an instance of the UI.
The UI is used to gather parameters for the tracking.
"""

# settings.json and count.ico live next to this file, wherever COUNT is run from
PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_SETTINGS_PATH = os.path.join(PACKAGE_DIR, "settings.json")
ICON_PATH = os.path.join(PACKAGE_DIR, "count.ico")

DEFAULT_SETTINGS = {"canny_upper": 255,
                    "canny_lower": 85,
                    "max_centroid_distance": 70,
                    "timeout": 20,
                    "cell_radius": 8,
                    "min_frames_tracked": tracking.DEFAULT_MIN_FRAMES_TRACKED,
                    "save_overlay": False,
                    "csv_save_path": "results/"}

# Settings that map directly onto a tkinter variable of the same name
TRACKING_SETTINGS = ["canny_upper", "canny_lower", "max_centroid_distance", "timeout", "cell_radius",
                     "min_frames_tracked"]

PREVIEW_WINDOW = 'Try pressing "4" or "5" "ESC" to change settings/quit'


class ROISelectionApp:
    """Contains all information to begin tracking, collected in the user interface"""

    def __init__(self, master, require_files=True):
        """Initiate the class and define default values for tracking parameters"""
        self.master = master
        self.require_files = require_files  # False when the files come from elsewhere (box.py)
        self.SETTINGS_PATH = DEFAULT_SETTINGS_PATH
        self.load_settings()
        self.file_path = ""
        self.folder_path = tk.StringVar()  # Variable to store the selected folder path
        self.selection_text = tk.StringVar(value="Nothing selected")
        self.csv_folder_path = tk.StringVar(value=self.settings["csv_save_path"])  # Default csv save path
        self.overlay_path = ""
        self.canny_upper = tk.IntVar()  # Upper canny threshold (3:1 ratio)
        self.canny_lower = tk.IntVar()  # Lower canny threshold
        self.max_centroid_distance = tk.IntVar()  # max distance an object will travel between frames (px)
        self.timeout = tk.IntVar()  # How long before a lost object's track is closed (frames)
        self.cell_radius = tk.IntVar()
        self.min_frames_tracked = tk.IntVar()  # A cell must be seen in this many frames to be counted
        self.save_overlay = tk.BooleanVar()
        self.apply_settings()
        self.files = []  # Empty list that will accept an individual file, or files from a folder
        self.confirmed = False  # True only when the user clicks Confirm
        self.create_toolbar()
        self.create_widgets()  # This is the part of the UI you can see

    def create_toolbar(self):
        menu = tk.Menu(self.master)
        self.master.config(menu=menu)

        file_ = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='File', menu=file_)
        file_.add_command(label='Save as default settings', command=self.save_settings)
        file_.add_command(label='Load settings', command=self.choose_settings_file)
        file_.add_command(label='Exit', command=self.quit_ui)

        help_ = tk.Menu(menu, tearoff=0)
        menu.add_cascade(label='Help', menu=help_)
        help_.add_command(label='Help', command=self.help)

    def create_widgets(self):
        """Makes the visual parts of the UI. Ignore unless changing position of UI elements"""
        justification = "w"
        # File selection button
        tk.Label(self.master, text="For individual .nd2 or .avi file", anchor=justification).grid(sticky="w", row=0, column=0,
                                                                                          padx=5, pady=5)
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Folder selection button
        tk.Label(self.master, text="For .nd2/.avi files in a folder", anchor=justification).grid(sticky="w", row=1, column=0,
                                                                                            padx=5, pady=5)
        self.folder_button = tk.Button(self.master, text="Choose Folder", command=self.choose_folder)
        self.folder_button.grid(row=1, column=1, padx=5, pady=5)

        # What will be processed
        tk.Label(self.master, textvariable=self.selection_text, anchor=justification, fg="gray30").grid(
            sticky="w", row=1, column=2, columnspan=2, padx=5, pady=5)

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

        # Minimum frames tracked input field
        tk.Label(self.master, text="Min Frames to Count:", anchor=justification).grid(sticky="w", row=2, column=2,
                                                                                      padx=5, pady=5)
        self.min_frames_tracked_entry = tk.Entry(self.master, textvariable=self.min_frames_tracked)
        self.min_frames_tracked_entry.grid(row=2, column=3, padx=5, pady=5)

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
        path = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        if not path:  # dialog cancelled
            return
        self.SETTINGS_PATH = path
        self.load_settings()
        self.apply_settings()
        messagebox.showinfo("Settings", f"Settings loaded from:\n{self.SETTINGS_PATH}")

    def load_settings(self):
        """Read SETTINGS_PATH into self.settings. Missing keys fall back to DEFAULT_SETTINGS."""
        self.settings = dict(DEFAULT_SETTINGS)
        try:
            with open(self.SETTINGS_PATH, "r") as file:
                self.settings.update(json.load(file))
        except FileNotFoundError:
            print(f"no settings file found at {self.SETTINGS_PATH}... using defaults")
        except json.JSONDecodeError as e:
            print(f"could not read {self.SETTINGS_PATH} ({e})... using defaults")

    def apply_settings(self):
        """Copy self.settings into the fields shown in the window."""
        for name in TRACKING_SETTINGS:
            getattr(self, name).set(self.settings[name])
        self.save_overlay.set(bool(self.settings["save_overlay"]))
        self.csv_folder_path.set(self.settings["csv_save_path"])

    def save_settings(self):
        settings = {name: getattr(self, name).get() for name in TRACKING_SETTINGS}
        settings["save_overlay"] = self.save_overlay.get()
        settings["csv_save_path"] = self.csv_folder_path.get()
        with open(self.SETTINGS_PATH, "w") as file:
            json.dump(settings, file, indent=4)
        messagebox.showinfo("Settings", f"Settings saved to\n{self.SETTINGS_PATH}")

    def help(self):
        messagebox.showinfo("Help", "See the README: https://github.com/bschelske/COUNT")

    def choose_file(self):
        path = filedialog.askopenfilename(filetypes=[("ND2 or AVI files", "*.nd2 *.avi"), ("ND2 files", "*.nd2"),
                                                     ("AVI files", "*.avi")])
        if not path:  # dialog cancelled
            return
        # A file replaces any folder chosen earlier
        self.file_path = path
        self.folder_path.set("")
        self.selection_text.set(f"File: {os.path.basename(path)}")
        print("Selection:", self.file_path)

    def choose_folder(self):
        path = filedialog.askdirectory()
        if not path:  # dialog cancelled
            return
        # A folder replaces any file chosen earlier
        self.folder_path.set(path)
        self.file_path = ""
        count = len(list_video_files(path))
        self.selection_text.set(f"Folder: {os.path.basename(path)} ({count} .nd2/.avi files)")
        print("Selection:", self.folder_path.get())

    def choose_csv_output(self):
        path = filedialog.askdirectory()
        if not path:  # dialog cancelled
            return
        self.csv_folder_path.set(path)
        print("csv save path:", self.csv_folder_path.get())

    def on_checkbox_click(self):
        if self.save_overlay.get():
            print("An overlay of tracked objects will be saved!")

    def input_handling(self):
        # Handles if the selection was for a folder or a file.
        if self.file_path:
            self.files = [self.file_path]
        elif self.folder_path.get():
            self.files = list_video_files(self.folder_path.get())
        else:
            self.files = []

    def confirm_selections(self):
        self.input_handling()
        if self.require_files and not self.files:
            messagebox.showerror("Nothing to process", "Choose an .nd2 or .avi file, or a folder containing them.")
            return
        try:
            for name in TRACKING_SETTINGS:
                getattr(self, name).get()
        except tk.TclError:
            messagebox.showerror("Invalid setting", "Every tracking setting must be a whole number.")
            return

        # Ensure the output directory exists
        os.makedirs(self.csv_folder_path.get(), exist_ok=True)
        self.overlay_path = os.path.join(self.csv_folder_path.get(), "overlay")
        if self.save_overlay.get():
            os.makedirs(self.overlay_path, exist_ok=True)
        os.makedirs(os.path.join(self.csv_folder_path.get(), "final_results"), exist_ok=True)

        print(".csv results save path:", self.csv_folder_path.get())
        self.confirmed = True
        # Close the Tkinter window
        self.master.destroy()

    def preview_edge_detection(self):
        """Show detections on frames 4 and 5 of the selected file (or the first file in the selected folder)."""
        self.input_handling()
        if not self.files:
            messagebox.showerror("Nothing to preview", "Choose an .nd2 or .avi file, or a folder, first.")
            return

        # Get frames
        frames = [self.edge_detection_handling(3), self.edge_detection_handling(4)]
        # Display images
        cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_NORMAL)
        cv2.imshow(PREVIEW_WINDOW, frames[0])

        while True:
            # Check if the window is still open
            if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break  # Exit loop if the window is closed

            k = cv2.waitKey(0) & 0xFF  # Wait indefinitely for a key press
            if k == 53:  # Key code 5
                cv2.imshow(PREVIEW_WINDOW, frames[1])  # Display the 2nd frame
            elif k == 52:  # Key code 4
                cv2.imshow(PREVIEW_WINDOW, frames[0])  # Display the 1st frame
            elif k == 27:  # ESC key
                break
        cv2.destroyWindow(PREVIEW_WINDOW)

    def edge_detection_handling(self, frame_index):
        """Edge detection for UI preview, using the same detection code as the tracker"""
        backSub = cv2.createBackgroundSubtractorMOG2(varThreshold=16, detectShadows=False)
        with tracking.open_video(self.files[0]) as nd2_file:
            # MOG2 Background Method: prime the background exactly as the tracker does, then run up to frame_index
            tracking.prime_background(backSub, nd2_file, len(nd2_file))
            for index in range(frame_index + 1):
                frame = cv2.normalize(nd2_file[index], None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                foreground_mask = backSub.apply(frame)

            frame_copy = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Copy of the current frame for the overlay

        contours = []
        if foreground_mask.any():
            contours = tracking.find_cell_contours(foreground_mask, self.canny_lower.get(), self.canny_upper.get(),
                                                   self.cell_radius.get())
        cv2.drawContours(frame_copy, contours, -1, (0, 0, 255), 2)
        cv2.putText(frame_copy, f"Frame {frame_index + 1} Objects: {len(contours)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        return frame_copy

    def quit_ui(self):
        self.master.destroy()
        quit(2)


def list_video_files(folder):
    """All .nd2 and .avi files directly inside `folder`, sorted by name."""
    return [os.path.join(folder, f) for f in sorted(os.listdir(folder)) if tracking.is_supported_video(f)]


def create_ui(require_files=True):
    root = tk.Tk()
    try:
        root.iconbitmap(ICON_PATH)
    except tk.TclError:
        pass  # .ico icons are only supported on Windows
    root.title("Count Objects Until No Tomorrow (C.O.U.N.T.)")
    app = ROISelectionApp(root, require_files=require_files)
    root.mainloop()
    return app
