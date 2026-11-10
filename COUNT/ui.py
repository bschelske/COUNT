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

DEFAULT_SETTINGS = dict(tracking.DEFAULT_SETTINGS, save_overlay=False, csv_save_path="results/")

# Settings that map directly onto a tkinter variable of the same name
TRACKING_SETTINGS = list(tracking.DEFAULT_SETTINGS)

# (setting, label) for each field in the window, in two columns
LEFT_FIELDS = [("max_centroid_distance", "Max Movement Between Images (px):"),
               ("timeout", "Timeout (images):"),
               ("cell_radius", "Cell Radius (px):"),
               ("min_detections", "Min Detections to Count:"),
               ("detection_threshold", "Detection Threshold (grey levels):"),
               ("background_window", "Background Window (images each side):")]
RIGHT_FIELDS = [("count_line_x", "Counting Line x (px, 0 = auto):"),
                ("outlet_divider_y", "Outlet Divider y (px, 0 = auto):"),
                ("min_area", "Min Speck Area (px):"),
                ("max_area", "Max Speck Area (px):")]

PREVIEW_WINDOW = 'Preview: press "4" or "5" to switch images, "ESC" to close'


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
        # One tkinter variable per tracking setting (see tracking.DEFAULT_SETTINGS for what each one does)
        for name in TRACKING_SETTINGS:
            setattr(self, name, tk.IntVar())
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

        # Preview button
        tk.Label(self.master, text="Check detection and counting line", anchor=justification).grid(
            sticky="w", row=8, column=0, padx=5, pady=5)
        self.edge_preview_button = tk.Button(self.master, text="Preview Detection", command=self.preview_edge_detection)
        self.edge_preview_button.grid(row=8, column=1, padx=5, pady=5)

        # Tracking settings
        for column, fields in ((0, LEFT_FIELDS), (2, RIGHT_FIELDS)):
            for row, (name, label) in enumerate(fields, start=2):
                tk.Label(self.master, text=label, anchor=justification).grid(sticky="w", row=row, column=column,
                                                                             padx=5, pady=5)
                entry = tk.Entry(self.master, textvariable=getattr(self, name))
                entry.grid(row=row, column=column + 1, padx=5, pady=5)
                setattr(self, f"{name}_entry", entry)

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
        """
        Show two consecutive images from the middle of the selected file (or the first file in the selected
        folder) with the moving specks that will be detected, the counting line and the outlet divider.
        """
        self.input_handling()
        if not self.files:
            messagebox.showerror("Nothing to preview", "Choose an .nd2 or .avi file, or a folder, first.")
            return
        try:
            for name in TRACKING_SETTINGS:
                getattr(self, name).get()
        except tk.TclError:
            messagebox.showerror("Invalid setting", "Every tracking setting must be a whole number.")
            return

        # Get frames
        frames = tracking.preview_frames(self.files[0], self, n_images=2)
        if not frames:
            messagebox.showerror("Nothing to preview", f"No images found in {self.files[0]}")
            return
        frames = frames + frames[-1:]  # a one-image file shows the same image for "4" and "5"
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
