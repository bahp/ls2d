# Libraries
import pandas as pd
import numpy as np

# Load specific dataset
from sklearn.datasets import load_breast_cancer

# Load dataset
dataset = load_breast_cancer(as_frame=True)

# Show keys
print("\nKeys:")
print(dataset.keys())

# Label conversion
lblmap = dict(enumerate(dataset.target_names))

# Show label conversion
print("\nLabel conversion:")
print(lblmap)

# Format
data = dataset.data
data['target'] = dataset.target
data['label'] = data.target.map(lblmap)

# Show data
print("\nData:")
print(data)

# Save
data.to_csv('data.csv', index=False)