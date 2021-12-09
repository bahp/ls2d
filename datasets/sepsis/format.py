# Libraries
import pandas as pd
import numpy as np

# --------------------------------------
# Constants
# --------------------------------------
# Load dataset
dataset = pd.read_csv('raw.csv',
    #nrows=10000,
    parse_dates=['patient_dob',
                 'date_collected',
                 'date_outcome',
                 'date_sample'])

# Show data
print("\nData:")
print(dataset)

# Show dtypes
print("\nDtypes")
print(dataset.dtypes)

# -------------------
# Filter FBC
# -------------------
# Keep only FBC
dataset = dataset[dataset.examination_code.isin(['FBC'])]

# ---------------------
# Format microorganisms
# ---------------------
# Format organisms
dataset['micro_name'] = dataset['Final ID']
dataset.micro_name = dataset.micro_name.str.strip()
dataset.micro_name = dataset.micro_name.str.lower()

# Uncomment to create initial table for replacements
#dataset.micro_name.value_counts().to_csv("db_organisms-empty.csv"))

# Load database of organisms.
# .. note: It only loads the first 315 because they are
#          the only ones that have been reviewed. Keep
#          reviewing or editing the document as needed.
dborgs = pd.read_csv('organisms.csv',
    nrows=315, encoding="latin")

# Create replace map
replace = dict(zip(dborgs.name, dborgs.micro_code))

# Replace
dataset['micro_code'] = None
dataset.micro_code = dataset.micro_name.map(replace)

# Uncomment to double check the transformations
#dataset.micro_code.value_counts().to_csv("organisms-empty.csv")

# Keep only useful (include resistance information!?)
dataset = dataset[dataset.micro_code.notna()]

# Save
dataset.to_csv('data.csv', index=False)