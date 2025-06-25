import json
import os
import tkinter as tk
from tkinter import filedialog, messagebox

from pims import ND2Reader_SDK

from COUNT import ui_preview  # Import the new preview module

"""
This code handles the creation of the user interface (UI).

The UI is a tkinter object. tkinter is a python library for UIs. 
create_UI() is called by main.py to make the UI appear. This function returns an instance of the UI.
The UI is used to gather parameters for the tracking. 
"""


class UserInterfaceApp:
    """Contains all information to begin tracking, collected in the user interface"""

    def __init__(self, master):
        """Initiate the class and define default values for tracking parameters"""
        self.master = master
        self.SETTINGS_PATH = "settings.json"
        self.load_settings()
        self.file_path = ""
        self.folder_path = tk.StringVar()
        self.csv_folder_path = tk.StringVar(value=self.settings.get("csv_save_path"))
        self.overlay_path = ""
        self.debug = True

        # Tracking Parameters
        self.canny_upper = tk.IntVar(value=self.settings.get("canny_upper"))
        self.canny_lower = tk.IntVar(value=self.settings.get("canny_lower"))
        self.max_centroid_distance = tk.IntVar(value=self.settings.get("max_centroid_distance"))
        self.timeout = tk.IntVar(value=self.settings.get("timeout"))
        self.cell_max_radius = tk.IntVar(value=self.settings.get("cell_max_radius"))
        self.cell_min_radius = tk.IntVar(value=self.settings.get("cell_min_radius"))
        self.save_overlay = tk.BooleanVar(value=self.settings.get("save_overlay"))
        self.flow_direction = tk.StringVar(value=self.settings.get("flow_direction"))
        self.background_subtraction = tk.BooleanVar(value=self.settings.get("background_subtraction"))

        # Initialize preview manager
        self.preview_manager = ui_preview.PreviewManager(self)

        # Create UI
        self.files = []
        self.create_toolbar()
        self.create_widgets()

    def create_toolbar(self):
        """Create the menu bar"""
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
        """Makes the visual parts of the UI"""
        justification = "w"

        # File selection button
        tk.Label(self.master, text="For individual .nd2 file", anchor=justification).grid(
            sticky="w", row=0, column=0, padx=5, pady=5)
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Folder selection button
        tk.Label(self.master, text="For .nd2 files in a folder", anchor=justification).grid(
            sticky="w", row=1, column=0, padx=5, pady=5)
        self.folder_button = tk.Button(self.master, text="Choose Folder", command=self.choose_folder)
        self.folder_button.grid(row=1, column=1, padx=5, pady=5)

        # CSV output folder selection button
        tk.Label(self.master, text="Choose .csv save path", anchor=justification).grid(
            sticky="w", row=0, column=2, padx=5, pady=5)
        self.csv_button = tk.Button(self.master, text="Choose Folder", command=self.choose_csv_output)
        self.csv_button.grid(row=0, column=3, padx=5, pady=5)

        # Preview Edge Detection button
        tk.Label(self.master, text="Visualize Edge Detection", anchor=justification).grid(
            sticky="w", row=8, column=0, padx=5, pady=5)
        self.edge_preview_button = tk.Button(self.master, text="Preview Detection",
                                             command=self.preview_edge_detection)
        self.edge_preview_button.grid(row=8, column=1, padx=5, pady=5)

        # Background Subtraction Check Box
        self.background_subtraction_checkbox = tk.Checkbutton(
            self.master, text="Background subtraction", variable=self.background_subtraction,
            onvalue=True, offvalue=False)
        self.background_subtraction_checkbox.grid(row=7, column=1, padx=5, pady=5)

        # Parameter input fields
        self._create_parameter_fields()

        # Action buttons
        self._create_action_buttons()

    def _create_parameter_fields(self):
        """Create all the parameter input fields"""
        # Canny Upper input field
        tk.Label(self.master, text="Canny Upper:", anchor="w").grid(
            sticky="w", row=2, column=0, padx=5, pady=5)
        self.canny_upper_entry = tk.Entry(self.master, textvariable=self.canny_upper)
        self.canny_upper_entry.grid(row=2, column=1, padx=5, pady=5)

        # Canny Lower input field
        tk.Label(self.master, text="Canny Lower:", anchor="w").grid(
            sticky="w", row=3, column=0, padx=5, pady=5)
        self.canny_lower_entry = tk.Entry(self.master, textvariable=self.canny_lower)
        self.canny_lower_entry.grid(row=3, column=1, padx=5, pady=5)

        # Max Centroid Distance input field
        tk.Label(self.master, text="Max Centroid Distance (px):", anchor="w").grid(
            sticky="w", row=4, column=0, padx=5, pady=5)
        self.max_centroid_distance_entry = tk.Entry(self.master, textvariable=self.max_centroid_distance)
        self.max_centroid_distance_entry.grid(row=4, column=1, padx=5, pady=5)

        # Timeout threshold input field
        tk.Label(self.master, text="Timeout Threshold (frames):", anchor="w").grid(
            sticky="w", row=5, column=0, padx=5, pady=5)
        self.timeout_entry = tk.Entry(self.master, textvariable=self.timeout)
        self.timeout_entry.grid(row=5, column=1, padx=5, pady=5)

        # Cell radius input fields
        tk.Label(self.master, text="Cell size exclusion, radii (min, max)", anchor="w").grid(
            sticky="w", row=6, column=0, padx=5, pady=5)
        self.cell_min_radius_entry = tk.Entry(self.master, textvariable=self.cell_min_radius)
        self.cell_min_radius_entry.grid(row=6, column=1, padx=5, pady=5)
        self.cell_max_radius_entry = tk.Entry(self.master, textvariable=self.cell_max_radius)
        self.cell_max_radius_entry.grid(row=6, column=2, padx=5, pady=5)

    def _create_action_buttons(self):
        """Create the action buttons at the bottom"""
        # Save overlay checkbox
        self.save_overlay_checkbox = tk.Checkbutton(
            self.master, text="Save Overlay?", variable=self.save_overlay,
            onvalue=True, offvalue=False, command=self.on_checkbox_click)
        self.save_overlay_checkbox.grid(row=9, column=0, columnspan=1, padx=5, pady=5)

        # Confirm button
        self.confirm_button = tk.Button(self.master, text="      Confirm     ",
                                        command=self.confirm_selections)
        self.confirm_button.grid(sticky="w", row=9, column=2, padx=1, pady=10)

        # Quit button
        self.quit_button = tk.Button(self.master, text="        Quit        ",
                                     command=self.quit_ui)
        self.quit_button.grid(sticky="w", row=9, column=3, padx=1, pady=10)

    # Settings management
    def choose_settings_file(self):
        self.SETTINGS_PATH = filedialog.askopenfilename(filetypes=[("JSON files", "*.json")])
        self.load_settings()
        messagebox.showinfo("Settings", f"Settings loaded from:\n{self.SETTINGS_PATH}")

    def load_settings(self):
        try:
            with open(self.SETTINGS_PATH, "r") as file:
                self.settings = json.load(file)
        except FileNotFoundError:
            print("no settings.json found... using defaults")
            self.settings = {
                "canny_upper": 255, "canny_lower": 85, "max_centroid_distance": 70,
                "timeout": 5, "flow_direction": "Towards Right (--->)",
                "background_subtraction": True, "cell_min_radius": 2, "cell_max_radius": 6,
                "save_overlay": False, "csv_save_path": "results/", "file_path": ""
            }

    def save_settings(self):
        settings = {
            "canny_upper": self.canny_upper.get(),
            "canny_lower": self.canny_lower.get(),
            "max_centroid_distance": self.max_centroid_distance.get(),
            "timeout": self.timeout.get(),
            "flow_direction": self.flow_direction.get(),
            "background_subtraction": self.background_subtraction.get(),
            "cell_min_radius": self.cell_min_radius.get(),
            "cell_max_radius": self.cell_max_radius.get(),
            "save_overlay": self.save_overlay.get(),
            "csv_save_path": self.csv_folder_path.get(),
            "file_path": self.file_path
        }
        with open(self.SETTINGS_PATH, "w") as file:
            json.dump(settings, file, indent=4)
        messagebox.showinfo("Settings", f"Settings saved to\n{self.SETTINGS_PATH}")

    # File handling
    def choose_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("ND2 files", "*.nd2")])
        with ND2Reader_SDK(self.file_path) as nd2_file:
            if 'm' in nd2_file.sizes.keys():
                nd2_file.iter_axes = 'm'
            self.total_frames = len(nd2_file)
        print("Selection:", self.file_path, self.total_frames, "frames")

    def choose_folder(self):
        self.folder_path.set(filedialog.askdirectory())
        print("Selection:", self.folder_path.get())

    def choose_csv_output(self):
        self.csv_folder_path.set(filedialog.askdirectory())
        print("csv save path:", self.csv_folder_path.get())

    def input_handling(self):
        """Handle folder vs file selection"""
        if self.folder_path.get():
            self.files = [os.path.join(self.folder_path.get(), f)
                          for f in os.listdir(self.folder_path.get())]
        if self.file_path:
            self.files = [self.file_path]

    # Event handlers
    def on_checkbox_click(self):
        if self.save_overlay.get() == 1:
            print("An overlay of tracked objects will be saved!")

    def preview_edge_detection(self):
        """Delegate to the preview manager"""
        self.preview_manager.preview_edge_detection()

    def confirm_selections(self):
        self.input_handling()

        # Ensure output directories exist
        os.makedirs(self.csv_folder_path.get(), exist_ok=True)
        os.makedirs(self.csv_folder_path.get() + "overlay/", exist_ok=True)
        self.overlay_path = self.csv_folder_path.get() + "overlay/"
        os.makedirs(self.csv_folder_path.get() + "final_results/", exist_ok=True)

        if self.save_overlay.get():
            print(f"\noverlay saving in {self.overlay_path}")
        print(".csv results save path:", self.csv_folder_path.get())

        self.master.destroy()

    def help(self):
        messagebox.showinfo("Help", "no help here")

    def quit_ui(self):
        self.master.destroy()
        quit(2)


def create_ui():
    root = tk.Tk()
    root.iconbitmap(r'count.ico')
    root.title("Count Objects Until No Tomorrow (C.O.U.N.T.)")
    app = UserInterfaceApp(root)
    root.mainloop()
    return app