import tkinter as tk
from tkinter import filedialog
import pandas as pd
import os
import re
from matplotlib import pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# Define a Gaussian function
def gaussian(x, amplitude, mean, stddev):
    return amplitude * np.exp(-((x - mean) / stddev) ** 2 / 2)

braf = r'C:\Users\bensc\PycharmProjects\scikit\analysis\20240315 CF DEP results BRAF.csv'
a375 = r'C:\Users\bensc\PycharmProjects\scikit\analysis\20240331 CF DEP 4th Trial results updated.csv'


braf_df = pd.read_csv(braf)
x = braf_df['frequency']
y = braf_df['Percent True'] * 100

a375_df = pd.read_csv(a375)
ax = a375_df['frequency']
ay = a375_df['Percent True'] * 100

# Fit the Gaussian function to the data
popt, pcov = curve_fit(gaussian, x, y, p0=[1, np.mean(x), np.std(x)])
plt.plot(x, gaussian(x, *popt), 'b-', label='Fitted Gaussian')

popt, pcov = curve_fit(gaussian, ax, ay, p0=[1, np.mean(x), np.std(x)])
plt.plot(x, gaussian(x, *popt), 'r-', label='Fitted Gaussian')

# plt.scatter(x,y)
# plt.show()
plt.plot(x, y, label='BRAF', marker='o', color='blue', linestyle='none')  # linestyle='dotted'
plt.plot(ax, ay, label='A375', marker='s', color='red', linestyle='none')  # linestyle='dotted'

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
minor_ticks_y = np.arange(10, y_limit + 1, increment_size // 2)
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
