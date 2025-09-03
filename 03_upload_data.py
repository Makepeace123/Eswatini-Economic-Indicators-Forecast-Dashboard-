# -*- coding: utf-8 -*-
"""03_upload_data.py"""

from google.colab import files
import pandas as pd

print("Please upload your dataset (CSV file)")
uploaded = files.upload()

# Get the uploaded filename
filename = list(uploaded.keys())[0]
print(f"Uploaded file: {filename}")

# Load the data
df = pd.read_excel(filename)
print(f"Data loaded successfully! Shape: {df.shape}")

# Save raw data
df.to_csv('data/raw_data.csv', index=False)
print("Raw data saved to 'data/raw_data.csv'")


