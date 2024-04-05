# Plot formatting based off code by BG

import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def get_frequency(string):
    match = re.search(r'\d+kHz', string)

    if match:
        extracted_number = match.group(0)[:-3]
    else:
        match = re.search(r"_(\d+)\.csv$", string)
        if match:
            extracted_number = match.group(1)
        else:
            extracted_number = "fail"
    return extracted_number


class DataSelectionApp:
    def __init__(self, master):
        self.master = master
        self.folder_path = tk.StringVar()  # Variable to store the selected folder path
        self.file_path = ""
        self.files = []  # Empty list that will accept an individual file, or files from a folder

        self.create_widgets()

    def create_widgets(self):
        # Folder selection button
        tk.Label(self.master, text="Consolidate tracking csv files").grid(row=1, column=0, padx=5, pady=5)
        self.folder_button = tk.Button(self.master, text="Choose Folder", command=self.choose_folder)
        self.folder_button.grid(row=1, column=1, padx=5, pady=5)

        # File selection button
        tk.Label(self.master, text="Make plot from file").grid(row=0, column=0, padx=5, pady=5)
        self.file_button = tk.Button(self.master, text="Choose File", command=self.choose_file)
        self.file_button.grid(row=0, column=1, padx=5, pady=5)

        # Button to confirm selections
        self.confirm_button = tk.Button(self.master, text="      Confirm     ", command=self.confirm_selections)
        self.confirm_button.grid(row=8, column=2, padx=1, pady=10)

        # Quit button
        self.quit_button = tk.Button(self.master, text="        Quit        ", command=self.quit_ui)
        self.quit_button.grid(row=8, column=3, padx=1, pady=10)

    def choose_file(self):
        self.file_path = filedialog.askopenfilename()
        print("Selection:", self.file_path)
        self.make_plot()

    def choose_folder(self):
        self.folder_path.set(filedialog.askdirectory())  # Set the selected folder path
        print("Selection:", self.folder_path.get())


    def input_handling(self):
        # Input handling
        if self.folder_path.get():
            self.files = [os.path.join(self.folder_path.get(), f) for f in os.listdir(self.folder_path.get())]
        if self.file_path:
            self.files = [self.file_path]
        print(f"Files:{self.files}")

    def confirm_selections(self):
        self.error_handling()
        self.input_handling()
        self.process_files()

    def process_files(self):
        results = []
        for file in self.files:
            print(f"Processing {file}")
            df = pd.read_csv(file)
            filtered_df = df[df['frames_tracked'] != 1]
            count_DEP_True = (filtered_df['DEP_response'].astype(str).str.strip().str.upper() == 'TRUE').sum()
            count_DEP_False = (filtered_df['DEP_response'].astype(str).str.strip().str.upper() == 'FALSE').sum()
            print(f"count_DEP_True: {count_DEP_True} count_DEP_False: {count_DEP_False}")
            result = {
                'filename': file,
                'frequency': get_frequency(file),
                'DEP_True': count_DEP_True,
                'DEP_False': count_DEP_False,
                'Percent True': (count_DEP_True / (count_DEP_True + count_DEP_False)),
            }
            results.append(result)
        results_df = pd.DataFrame(results)
        # Save processed files
        results_df.to_csv(f"analysis/results.csv")
        print("Done. rename your file :)")

    def make_plot(self):
        df = pd.read_csv(self.file_path)
        x = df['frequency']
        y = df['Percent True'] * 100

        # Fit the Gaussian function to the data
        popt, pcov = curve_fit(gaussian, x, y, p0=[1, np.mean(x), np.std(x)])
        plt.plot(x, gaussian(x, *popt), 'b-', label='Fitted Gaussian')


        # plt.scatter(x,y)
        # plt.show()
        plt.plot(x, y, label='BRAF', marker='o', color='blue', linestyle='none') #linestyle='dotted'

        # derivative = np.gradient(y, x)
        # plt.plot(derivative, label='derivative', marker='s', color='red',
        #          linestyle='dotted')

        # # Define zooming in or fullscale
        y_limit = 100
        increment_size = 20

        # Customize axis labels
        plt.xlabel('Frequency (kHz)', fontsize=16, fontweight='bold')  # Change label as needed
        plt.ylabel('Percentage of pDEP cell (%)', fontsize=16, fontweight='bold')  # Change label as needed

        # Customize main/labelled axis ticks for x-axis
        plt.xticks(np.arange(0, 250, 50), fontsize=12)  # Adjust range and step as needed for x-axis
        plt.xlim(-5, 225)  # Adjust x-axis limits

        # Label every other tick starting from 0 for x-axis
        major_ticks_x = np.arange(0, 250, 50)
        minor_ticks_x = np.arange(25, 250, 50)
        plt.gca().set_xticks(major_ticks_x)
        plt.gca().set_xticks(minor_ticks_x, minor=True)
        plt.tick_params(axis='x', length=4, direction='in')

        # Customize main/labelled axis ticks for y-axis
        plt.yticks(np.arange(0, y_limit + 1, increment_size), fontsize=12)  # Adjust range and step as needed for y-axis
        plt.tick_params(axis='y', length=4, direction='in')

        plt.ylim(-5, y_limit + 1)  # Adjust y-axis limits to start a few pixels higher than the corner

        # Label every other tick starting from 0 for y-axis
        major_ticks_y = np.arange(0, y_limit + 1, increment_size)
        minor_ticks_y = np.arange(10, y_limit + 1, increment_size//2)
        plt.gca().set_yticks(major_ticks_y)
        plt.gca().set_yticks(minor_ticks_y, minor=True)

        # Adjust labelled tick length on the x-axis and y-axis
        plt.tick_params(axis='both', length=4, direction='in')  # Tick marks inside the plot

        # Customize axis thickness and remove top and right borders
        ax = plt.gca()
        ax.spines['bottom'].set_linewidth(2)  # Set x-axis thickness
        ax.spines['left'].set_linewidth(2)  # Set y-axis thickness
        ax.spines['top'].set_visible(False)  # Hide top border
        ax.spines['right'].set_visible(False)  # Hide right border

        # Place legend to the right side of the plot at the top
        # plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), frameon=False,
        #            borderaxespad=0)  # Adjust location as needed
        # plt.legend()  # Adjust location as needed

        plt.show()
        pass


    def error_handling(self):
        if self.file_path == '' and self.folder_path.get() == '':
            raise ValueError("No file selected!")
        if self.file_path and self.folder_path.get():
            raise ValueError(
                "...Why did you pick a file AND a folder? Choose ONE or the OTHER. What did you expect to happen? :)")


    def quit_ui(self):
        self.master.destroy()
        quit(2)

# Define a Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

def create_UI():
    root = tk.Tk()
    root.iconbitmap(r'count.ico')
    root.title("COUNT Plotter")
    app = DataSelectionApp(root)
    root.mainloop()
    return app

app = create_UI()

