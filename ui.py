import tkinter as tk
from tkinter import filedialog

class ROISelectionApp:
    def __init__(self, master):
        self.master = master
        self.file_path = ""
        self.roi_x = tk.StringVar(value="10")  # Default value for ROI X
        self.roi_y = tk.StringVar(value="0")   # Default value for ROI Y
        self.roi_height = tk.StringVar(value="2048")  # Default value for ROI Height
        self.roi_width = tk.StringVar(value="400")    # Default value for ROI Width

        self.create_widgets()

    def create_widgets(self):
        # File selection button
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=0, padx=5, pady=5)

        # ROI X input field
        tk.Label(self.master, text="ROI X:").grid(row=1, column=0, padx=5, pady=5)
        self.roi_x_entry = tk.Entry(self.master, textvariable=self.roi_x)
        self.roi_x_entry.grid(row=1, column=1, padx=5, pady=5)

        # ROI Y input field
        tk.Label(self.master, text="ROI Y:").grid(row=2, column=0, padx=5, pady=5)
        self.roi_y_entry = tk.Entry(self.master, textvariable=self.roi_y)
        self.roi_y_entry.grid(row=2, column=1, padx=5, pady=5)

        # ROI Height input field
        tk.Label(self.master, text="ROI Height:").grid(row=3, column=0, padx=5, pady=5)
        self.roi_height_entry = tk.Entry(self.master, textvariable=self.roi_height)
        self.roi_height_entry.grid(row=3, column=1, padx=5, pady=5)

        # ROI Width input field
        tk.Label(self.master, text="ROI Width:").grid(row=4, column=0, padx=5, pady=5)
        self.roi_width_entry = tk.Entry(self.master, textvariable=self.roi_width)
        self.roi_width_entry.grid(row=4, column=1, padx=5, pady=5)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="Confirm", command=self.confirm_selections)
        self.confirm_button.grid(row=5, column=0, columnspan=2, padx=5, pady=5)

    def choose_file(self):
        self.file_path = filedialog.askopenfilename()
        print("File chosen:", self.file_path)

    def confirm_selections(self):
        print("ROI X:", self.roi_x.get())
        print("ROI Y:", self.roi_y.get())
        print("ROI Height:", self.roi_height.get())
        print("ROI Width:", self.roi_width.get())

    def get_roi(self):
        ROI = (self.roi_x.get(), self.roi_y.get(), self.roi_height.get(), self.roi_width.get())
        return ROI

def create_UI():
    root = tk.Tk()
    root.title("Object Detection")
    app = ROISelectionApp(root)
    root.mainloop()
    return app
