# -*- coding: utf-8 -*-
"""04_preprocessing.py"""

import pandas as pd
import numpy as np
from scipy import interpolate
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load raw data
df = pd.read_csv('data/raw_data.csv', parse_dates=['Date'], dayfirst=True)
df.set_index('Date', inplace=True)

print("Starting preprocessing...")
print(f"Original data shape: {df.shape}")

# 1. Handle missing values with spline interpolation
print("1. Handling missing values with spline interpolation...")
df_clean = df.copy()

# Identify numeric columns for interpolation
numeric_columns = df_clean.select_dtypes(include=[np.number]).columns.tolist()

# Apply spline interpolation to each numeric column
for col in numeric_columns:
    if df_clean[col].isnull().sum() > 0:
        # Get non-null values and their indices
        valid_mask = df_clean[col].notnull()
        valid_indices = np.where(valid_mask)[0]
        valid_values = df_clean[col].values[valid_mask]
        
        if len(valid_values) > 3:  # Need at least 3 points for spline
            # Create spline function
            spline = interpolate.UnivariateSpline(valid_indices, valid_values, s=0.5)
            
            # Interpolate missing values
            all_indices = np.arange(len(df_clean))
            df_clean[col] = spline(all_indices)
        else:
            # Fallback to linear interpolation if not enough points
            df_clean[col] = df_clean[col].interpolate(method='linear')

print(f"Missing values after interpolation: {df_clean.isnull().sum().sum()}")

# 2. Create date-based features (categorical)
print("2. Creating date features...")
df_clean['day_of_week'] = df_clean.index.dayofweek
df_clean['month'] = df_clean.index.month
df_clean['quarter'] = df_clean.index.quarter

# 3. Drop any remaining NaN rows
df_final = df_clean.dropna()
print(f"Final dataset shape after dropping NaNs: {df_final.shape}")

# 4. Save processed data
df_final.to_csv('data/processed_data.csv')
print("Processed data saved to 'data/processed_data.csv'")

print("Preprocessing completed successfully!")
