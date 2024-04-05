import pandas as pd
import os
import re

results_path = "results"
results_files = [os.path.join(results_path, f) for f in os.listdir(results_path)]
results_files = ['results\\23Feb2024 Non RosetteSep 100kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 105kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 10kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 110kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 115kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 120kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 125kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 130kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 135kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 140kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 145kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 150kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 155kHz_results.csv', 'results\\23Feb2024 Non RosetteSep 15kHz_results.csv']


for file in results_files:
    df = pd.read_csv(file)

    # Convert 'DEP_response' column values to strings
    df['DEP_response'] = df['DEP_response'].astype(str)

    # Convert 'DEP_response' column values to uppercase for case-insensitive comparison
    df['DEP_response'] = df['DEP_response'].str.upper()

    # Count rows where DEP_response is True
    true_count = df['DEP_response'].value_counts().get('TRUE', 0)

    # Count rows where DEP_response is False
    false_count = df['DEP_response'].value_counts().get('FALSE', 0)

    print(f"{file}\nDEP response: {true_count} No DEP: {false_count}\n")

