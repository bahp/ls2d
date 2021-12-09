# Libraries
import pandas as pd
import numpy as np

#
mp1 = {
    'n': False,
    'y': True,
    'probable': True,
    'possible': True,
}

# Cast booleans
columns = [
    'Systemic inflammatory condition',
    'GM>1',
    'Sequence based identification of mould',
    'Mould recovered?',
    'Aspergillus sp. recovered?',
    'Aspergillus sp. on ANY OF culture, PCR, GM>1,microscopy?',
    'Cytology done (y/n)',
    'Microscopic detection of mould',
    'Neutropenia (>10d)',
    'Haematological malignancy or BMT (y/n)',
    'Allograft BMT (y/n)',
    'Soid organ transplantation (y/n)',
    'Steroids (?0.3mg/kg for >3w)',
    'On mould active anti-fungal prior to sample collection',
    'T-cell immune suppression (y/n)',
    'B-cell immune suppression (y/n)',
    'Inherited immune deficiency (y/n)',
    'Refractory GVHD (y/n)',
    'Any host factor',
    'Host factor NOT haem malignancy',
    'Dense well circumscribed lesion (y/n)',
    'Air crescent sign (y/n)',
    'Cavity (y/n)',
    'Wedge/segmental or lobar (y/n)',
    "Reported as possible fungal infection (not just 'atyical infection', y/n)",
    'Any clinical feature',
    'Tracheobronchial sign on bronchoscopy (y/n)',
    'Probable IPA',
    'Possible IPA'
]


DTYPES = {}

# ----------------------------
# Data
# ----------------------------
# Comments
# - Date of discharge badly formatted
# - Date og death badly formatted

# Path to data
path = './fungal-invasive-paul-v18.csv'

# Read data ad model
data = pd.read_csv(path,
    na_values=['N/A', '-'])

for c in columns:
    data[c] = data[c].replace(mp1)

# Fix >6
data['Asp GM Index Value'] = \
    data['Asp GM Index Value'] \
        .replace({'>6': 6}) \
        .astype(float)

# Replace values
#data = data.replace(MAPPINGS)

# Force some dtypes
data = data.astype(DTYPES)

# Show data
print("\nData:")
print(data)

# Save
data.to_csv('data.csv', index=False)