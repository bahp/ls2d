# Libraries
import pandas as pd

# File path
filepath = './outputs/iris/20220211-142222/results.csv'

# Load data from file
df = pd.read_csv(filepath)

"""
# Filter out those columns starting with param_
df = df.loc[:, ~df.columns.str.startswith('param_')]
# Filter out those columns starting with rank
df = df.loc[:, ~df.columns.str.startswith('rank_')]

# Filter out those columns containing splits
df = df.loc[:, ~df.columns.str.contains('split')]

# Reorder
cols = list(df.columns)
cols.insert(3, cols.pop(cols.index('params')))
cols.insert(4, cols.pop(cols.index('mean_test_pearson')))
cols.insert(5, cols.pop(cols.index('mean_test_spearman')))
df = df[cols]
"""
# -------------------------

# Libraries
from ls2d.utils import format_workbench
from ls2d.utils import format_pipeline

# Format workbench
workbench = format_workbench(df)
pipeline = format_pipeline(df.loc[0, :])

# Show
print("\nWorkbench:")
print(workbench)
print("\n\nPipeline:")
print(pipeline)