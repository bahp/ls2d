# Libraries
import pandas as pd
import numpy as np

# Load specific dataset
from sklearn.datasets import load_diabetes

# Load dataset
dataset = load_diabetes(as_frame=True)

# This dataset is for regression so it does not have
# a lavel conversion. It is also weird the fact that
# sex is a number.

# Format
data = dataset.data
data['target'] = dataset.target

# Show data
print("\nData:")
print(data)

# Save
data.to_csv('data.csv', index=False)