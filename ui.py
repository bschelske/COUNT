import tkinter as tk
from tkinter import filedialog

class ROISelectionApp:
    def __init__(self, master):
        self.master = master
        self.file_path = ""
        self.folder_path = tk.StringVar()  # Variable to store the selected folder path
        self.roi_x = tk.StringVar(value="10")  # Default value for ROI X
        self.roi_y = tk.StringVar(value="0")   # Default value for ROI Y
        self.roi_height = tk.StringVar(value="2048")  # Default value for ROI Height
        self.roi_width = tk.StringVar(value="400")    # Default value for ROI Width
        self.canny_upper = tk.StringVar(value="255")  # Default value for ROI Height
        self.canny_lower = tk.StringVar(value="85")    # Default value for ROI Width

        self.create_widgets()

    def create_widgets(self):
        # File selection button
        tk.Label(self.master, text="For individual file").grid(row=0, column=0, padx=5, pady=5)
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Folder selection button
        tk.Label(self.master, text="For files in a folder").grid(row=1, column=0, padx=5, pady=5)
        self.folder_button = tk.Button(self.master, text="Choose Folder", command=self.choose_folder)
        self.folder_button.grid(row=1, column=1, padx=5, pady=5)

        # ROI X input field
        tk.Label(self.master, text="ROI X:").grid(row=2, column=0, padx=5, pady=5)
        self.roi_x_entry = tk.Entry(self.master, textvariable=self.roi_x)
        self.roi_x_entry.grid(row=2, column=1, padx=5, pady=5)

        # ROI Y input field
        tk.Label(self.master, text="ROI Y:").grid(row=3, column=0, padx=5, pady=5)
        self.roi_y_entry = tk.Entry(self.master, textvariable=self.roi_y)
        self.roi_y_entry.grid(row=3, column=1, padx=5, pady=5)

        # ROI Height input field
        tk.Label(self.master, text="ROI Height:").grid(row=4, column=0, padx=5, pady=5)
        self.roi_height_entry = tk.Entry(self.master, textvariable=self.roi_height)
        self.roi_height_entry.grid(row=4, column=1, padx=5, pady=5)

        # ROI Width input field
        tk.Label(self.master, text="ROI Width:").grid(row=5, column=0, padx=5, pady=5)
        self.roi_width_entry = tk.Entry(self.master, textvariable=self.roi_width)
        self.roi_width_entry.grid(row=5, column=1, padx=5, pady=5)

        # Canny Upper
        tk.Label(self.master, text="Canny Upper:").grid(row=2, column=2, padx=5, pady=5)
        self.canny_upper_entry = tk.Entry(self.master, textvariable=self.canny_upper)
        self.canny_upper_entry.grid(row=2, column=3, padx=5, pady=5)

        # Canny Lower
        tk.Label(self.master, text="Canny Lower:").grid(row=3, column=2, padx=5, pady=5)
        self.canny_lower_entry = tk.Entry(self.master, textvariable=self.canny_lower)
        self.canny_lower_entry.grid(row=3, column=3, padx=5, pady=5)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="Confirm", command=self.confirm_selections)
        self.confirm_button.grid(row=6, column=1, columnspan=2, padx=5, pady=5)

    def choose_file(self):
        self.file_path = filedialog.askopenfilename()
        print("Selection:", self.file_path)

    def choose_folder(self):
        self.folder_path.set(filedialog.askdirectory())  # Set the selected folder path
        print("Selection:", self.folder_path.get())

    def confirm_selections(self):
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
            raise ValueError("You didn't pick anything!")
        if self.file_path and self.folder_path.get():
            raise ValueError(
                "...Why did you pick a file AND a folder? Choose ONE or the OTHER. What did you expect to happen? :)")

    def get_roi(self):
        ROI = (self.roi_x.get(), self.roi_y.get(), self.roi_height.get(), self.roi_width.get())
        return ROI

def create_UI():
    root = tk.Tk()
    root.title("Object Detection")
    app = ROISelectionApp(root)
    root.mainloop()
    app.error_handling()
    return app
