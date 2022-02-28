

import os
import glob
import pickle
import numpy as np
import pandas as pd
from pathlib import Path


for p in Path('./outputs/').glob('**'):
    print(p)

print("\n\n\n")
stuff = './outputs/'
depth = 3
for root, dirs, files in os.walk(stuff):
    if root.count(os.sep) == depth:
        print(root.count(os.sep), root)
    #if root.count(os.sep) == 0:
    #    for d in dirs:
    #        print(os.path.join(root, d))


import sys
sys.exit()

PATH = './outputs/iris/20220221-175047/'

# Sample data
X = pd.DataFrame(data=[
    [0.4, 0.5, 0.6, 0.7],
    [0.4, 0.6, 0.9, 1.0]
])

X = X.to_numpy().astype(np.float32)

# Load results
results = pd.read_csv(Path(PATH) / 'results.csv')

# Loop
for path in Path(PATH).rglob('*.p'):

    # Load
    aux = pickle.load(open(str(path.resolve()), "rb"))

    print(path.name)
    #print(path.resolve())
    print(aux.__class__.__name__)
    print(aux.steps[-1][1].__class__.__name__)
    #print(aux)

    print("%s (%s)" % (aux.slug_long, aux.slug_short))
    print("Pipeline: %s" % aux.pipeline)
    print("Split: %s" % aux.split)
    print(aux.uuid)

    if getattr(aux, "predict", None) is not None:
        print("Predict:\n %s" % aux.predict(X))
    if getattr(aux, "transform", None) is not None:
        print("Transform:\n %s" % aux.transform(X))

    print("\n\n")

# Show
print(results)

