# Libraries
import pandas as pd
import numpy as np

# Load specific dataset
from sklearn.datasets import load_diabetes

# Load dataset
dataset = load_diabetes(as_frame=True)

# Show keys
print("\nKeys:")
print(dataset.keys())

# Format
data = dataset.data

# Show data
print("\nData:")
print(data)

# Save
data.to_csv('data.csv', index=False)