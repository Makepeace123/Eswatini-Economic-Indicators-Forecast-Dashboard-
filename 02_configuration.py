# -*- coding: utf-8 -*-
"""02_configuration.py"""

# Configuration Settings
TARGET_VARIABLES = [
    'Maize meal SZL/1kg',
    'All Items CPI',
    'Inflation rate', 
    'Diesel SZL/1 liter',
    'Cabbage SZL/Head',
    'Tomato (Round) SZL/1kg',
    'Rice SZL/1kg',
    'Beans SZL/1kg',
    'Sugar SZL/1kg',
    'Interest Rate (Prime lending rate)'
]

# Parameters
TEST_SIZE = 0.2  # Use last 20% of data for testing
RANDOM_STATE = 42
FORECAST_HORIZON = 30

# Create directories
import os
os.makedirs('model_artifacts', exist_ok=True)
os.makedirs('data', exist_ok=True)

print("Configuration set up successfully!")
