# Libraries
import pandas as pd
import numpy as np

# Columns to map
BOOLMAP = {
    'N': False,
    'Y': True,
    'n': False,
    'y': True,
    'No': False,
    'Yes': True,
    'no': False,
    'yes': True,
}

MAPPINGS = {
    'Gender': {
        'M': 0,
        'F': 1
    },
    'Admitted': BOOLMAP,
    'Cough': BOOLMAP,
    'Fever': BOOLMAP,
    'SOB': BOOLMAP,
    'Confusion': BOOLMAP,
    'Current smoker': BOOLMAP,
    'Prev smoker': BOOLMAP,
    'Chronic lung disease': BOOLMAP,
    'Cardiac disease': BOOLMAP,
    'DM': BOOLMAP,
    'HTN': BOOLMAP,
    'ESRD on HD': BOOLMAP,
    'FiO2': {
        '0.1L': 24,
        '8L': 52,
        '4L': 36,
        '15L': 60,
        '?': np.nan
    },
    'Urea': {
        '<1.1': 1.0
    },
    'CRP': {
        '<0.2': 0.1
    },
    'ABX (Yes/No)': BOOLMAP
}

DTYPES = {
    'FiO2': float,
    'Urea': float,
    'CRP': float
}

# ----------------------------
# Data
# ----------------------------
# Comments
# - Date of discharge badly formatted
# - Date og death badly formatted

# Read data ad model
data = pd.read_csv('./raw.csv',
    parse_dates=['Date of discharge', 'Date of death'],
    na_values=['N/A', '-'])

# Lower/uppercase
data['FiO2'] = data['FiO2'].str.upper().str.strip()
data['ABX (Yes/No)'] = data['ABX (Yes/No)'].str.lower()

# Replace values
data = data.replace(MAPPINGS)

# Force some dtypes
data = data.astype(DTYPES)

# Show data
print("\nData:")
print(data)

# Save
data.to_csv('data.csv', index=False)