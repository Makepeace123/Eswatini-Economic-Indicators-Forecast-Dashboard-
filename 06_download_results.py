# -*- coding: utf-8 -*-
"""06_download_results.py"""

from google.colab import files
import os

# Create ZIP archive
!zip -r eswatini_forecast_results.zip model_artifacts/ data/

# Download
files.download('eswatini_forecast_results.zip')
print("Results downloaded successfully!")
