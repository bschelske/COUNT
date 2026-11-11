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
               ("background_window", "Background Window (images each side):"),
               ("workers", "Processor Cores (0 = all):")]
RIGHT_FIELDS = [("count_line_x", "Counting Line x (px, 0 = auto):"),
                ("outlet_divider_y", "Outlet Divider y (px, 0 = auto):"),
                ("min_area", "Min Speck Area (px):"),
                ("max_area", "Max Speck Area (px):"),
                ("region_before_line", "Region Before Line (px, 0 = left edge):"),
                ("region_top", "Region Top y (px):"),
                ("region_bottom", "Region Bottom y (px, 0 = bottom):")]

# Help text for every setting, shown by the Help button (a test checks every setting has one)
FIELD_HELP = {
    "max_centroid_distance":
        "How far a cell can move from one image to the next (measured from where it is expected to be) and still "
        "be treated as the same cell. Set it a little above the largest step a cell makes between images. In "
        "recordings with about 3 images per second, cells move 30-100 px between images, so 100 works well. "
        "Too small splits cells into pieces; too large lets nearby cells get mixed up.",
    "timeout":
        "If a cell is not detected for this many images in a row, its track ends. A few images lets a cell that "
        "briefly fades or overlaps another be picked up again.",
    "cell_radius":
        "Cells flow left to right, but a cell may pause or its outline may jitter. A cell can move this many "
        "pixels to the left between images and still be matched.",
    "min_detections":
        "A cell is only counted once it has been detected in at least this many images. This stops two random "
        "specks that happen to fall either side of the counting line from being counted as a cell.",
    "detection_threshold":
        "How much darker (or brighter) than the background a speck must be to be detected, in grey levels "
        "(0-255). Lower finds fainter cells but also more noise. Some cells are only about 9 grey levels from "
        "the background, so keep this low.",
    "background_window":
        "Each image is compared with the images this far before and after it. Anything that stays in the same "
        "place for that long (dust, cells stuck to the device, the electrodes) is treated as background and "
        "ignored, so only moving cells are found. Cells must move further than their own size over this many "
        "images.",
    "workers":
        "How many processor cores to use. Each video is split into chunks processed side by side; the counts "
        "are the same whatever the number. 0 uses all of them; choose fewer to keep the computer responsive.",
    "count_line_x":
        "The pixel column where cells are counted (the blue line in the preview). Each cell is counted once, "
        "when it crosses this line moving right. Put it in the main channel, past the electrodes and before the "
        "channel splits into the two outlets. 0 places it three quarters of the way across the image.",
    "outlet_divider_y":
        "The pixel row that separates the two outlets (the grey line in the preview). Cells crossing the "
        "counting line above it are counted as pDEP (top outlet); below it, as nDEP (bottom outlet). Line it up "
        "with the tip that splits the outlets. 0 uses the middle of the image.",
    "min_area":
        "Specks smaller than this many pixels are ignored (noise).",
    "max_area":
        "Specks larger than this many pixels are ignored (for example a large piece of debris or a change in "
        "lighting).",
    "region_before_line":
        "Only part of each image is analysed (the green box in the preview), which makes COUNT much faster. The "
        "box starts this many pixels left of the counting line. Keep it well before the line (1000 px): each "
        "cell's track needs a few images to learn the cell's speed, and with less, cells in a crowded group can "
        "be mixed up. 0 starts at the left edge of the image.",
    "region_top":
        "The top row of the analysed region (green box). Keep the channel and both outlets inside the box.",
    "region_bottom":
        "The bottom row of the analysed region (green box). 0 uses the bottom of the image.",
}

HELP_INTRODUCTION = (
    "How COUNT counts",
    "COUNT finds the cells that move: each image is compared with the images around it, so dust and stuck "
    "cells are ignored. Moving cells are followed from image to image, and each cell is counted once, when it "
    "crosses the counting line. A cell crossing above the outlet divider is going to the top outlet (pDEP); "
    "below it, to the bottom outlet (nDEP).\n\n"
    "Before running a new set of videos, choose them and click Preview Detection to check that the counting "
    "line, the outlet divider and the analysed region are in the right place.")

HELP_BUTTONS = [
    ("Choose File / Choose Folder (left)",
     "Choose one .nd2 or .avi video, or a folder: every .nd2 and .avi file in it is processed. Choosing one "
     "replaces the other; the grey text shows what will be processed."),
    ("Choose .csv save path",
     "The folder where the results are saved: for each video, <video>_results.csv (one row per counted cell, "
     "with DEP_response True for the top outlet) and <video>_trajectory_results.csv (each cell's path)."),
    ("Preview Detection",
     "Opens two images from the middle of the first video showing, with the current settings: moving cells "
     "found (orange circles), the counting line (blue), the outlet divider (grey) and the analysed region "
     "(green box). Press 4 and 5 to switch images and Esc to close. Use it whenever the camera or device has "
     "moved."),
    ("Save Overlay?",
     "Also save an image of every frame in an 'overlay' folder, showing what was detected and counted, with "
     "running totals. Useful for checking results; processing takes longer."),
    ("Confirm / Quit",
     "Confirm starts counting; progress is shown in the console. Quit closes COUNT without running."),
    ("File menu",
     "Save as default settings: keep the current settings for next time (saved in COUNT/settings.json).\n"
     "Load settings: load a saved settings file, for example one kept for each experimental setup.\n"
     "Settings changed in the window without saving only apply to this run."),
]

PREVIEW_WINDOW = 'Preview: press "4" or "5" to switch images, "ESC" to close'


FIRST_FIELD_ROW = 2
PREVIEW_ROW = FIRST_FIELD_ROW + max(len(LEFT_FIELDS), len(RIGHT_FIELDS))  # first row below the settings
BUTTON_ROW = PREVIEW_ROW + 1


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
        help_.add_command(label='README on GitHub', command=self.open_readme)

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
            sticky="w", row=PREVIEW_ROW, column=0, padx=5, pady=5)
        self.edge_preview_button = tk.Button(self.master, text="Preview Detection", command=self.preview_edge_detection)
        self.edge_preview_button.grid(row=PREVIEW_ROW, column=1, padx=5, pady=5)

        # Tracking settings
        for column, fields in ((0, LEFT_FIELDS), (2, RIGHT_FIELDS)):
            for row, (name, label) in enumerate(fields, start=FIRST_FIELD_ROW):
                tk.Label(self.master, text=label, anchor=justification).grid(sticky="w", row=row, column=column,
                                                                             padx=5, pady=5)
                entry = tk.Entry(self.master, textvariable=getattr(self, name))
                entry.grid(row=row, column=column + 1, padx=5, pady=5)
                setattr(self, f"{name}_entry", entry)

        # Save overlay checkbox
        self.save_overlay_checkbox = tk.Checkbutton(self.master, text="Save Overlay?", variable=self.save_overlay,
                                                    onvalue=True, offvalue=False, command=self.on_checkbox_click)
        self.save_overlay_checkbox.grid(row=BUTTON_ROW, column=0, columnspan=1, padx=5, pady=5)

        # Help button
        self.help_button = tk.Button(self.master, text="      Help      ", command=self.help)
        self.help_button.grid(sticky="w", row=BUTTON_ROW, column=1, padx=1, pady=10)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="      Confirm     ", command=self.confirm_selections)
        self.confirm_button.grid(sticky="w", row=BUTTON_ROW, column=2, padx=1, pady=10)

        # Quit button
        self.quit_button = tk.Button(self.master, text="        Quit        ", command=self.quit_ui)
        self.quit_button.grid(sticky="w", row=BUTTON_ROW, column=3, padx=1, pady=10)

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
        """Open (or bring forward) a window explaining every field, button and menu."""
        if getattr(self, "help_window", None) is not None and self.help_window.winfo_exists():
            self.help_window.lift()
            return
        window = tk.Toplevel(self.master)
        window.title("COUNT Help")
        window.geometry("720x640")
        self.help_window = window

        text = tk.Text(window, wrap="word", padx=12, pady=10, borderwidth=0, font=("TkDefaultFont", 10))
        scrollbar = tk.Scrollbar(window, command=text.yview)
        text.configure(yscrollcommand=scrollbar.set)
        tk.Button(window, text="Close", command=window.destroy, width=12).pack(side="bottom", pady=8)
        scrollbar.pack(side="right", fill="y")
        text.pack(side="left", fill="both", expand=True)

        text.tag_configure("heading", font=("TkDefaultFont", 12, "bold"), spacing1=12, spacing3=4)
        text.tag_configure("name", font=("TkDefaultFont", 10, "bold"), spacing1=8)
        text.tag_configure("body", lmargin1=16, lmargin2=16)
        for heading, sections in help_sections():
            text.insert("end", heading + "\n", "heading")
            for name, body in sections:
                if name:
                    text.insert("end", name + "\n", "name")
                text.insert("end", body + "\n", "body")
        text.configure(state="disabled")  # read only

    def open_readme(self):
        import webbrowser
        webbrowser.open("https://github.com/bschelske/COUNT#readme")

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
        try:
            frames = tracking.preview_frames(self.files[0], self, n_images=2)
        except ValueError as e:  # e.g. a processing region outside the image
            messagebox.showerror("Invalid setting", str(e))
            return
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


def help_sections():
    """The help window's contents: [(heading, [(name, text)])], with every field described under its label."""
    def fields(column):
        return [(label.rstrip(":"), FIELD_HELP[name]) for name, label in column]
    return [
        (HELP_INTRODUCTION[0], [("", HELP_INTRODUCTION[1])]),
        ("Choosing videos and running", HELP_BUTTONS),
        ("Tracking and counting settings (left column)", fields(LEFT_FIELDS)),
        ("Where to count and what to analyse (right column)", fields(RIGHT_FIELDS)),
    ]


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
