# Libraries
import pandas as pd

# -----------------------
# Constants
# -----------------------
# Path to data
path = './ls2d/data/fungal-invasive-paul-v18.csv'

#
mp1 = {
    'n': False,
    'y': True,
    'probable': True,
    'possible': True,
}

# -----------------------
# Load
# -----------------------
# Load data
data = pd.read_csv(path)

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

for c in columns:
    data[c] = data[c].replace(mp1)

# Fix >6
data['Asp GM Index Value'] = \
    data['Asp GM Index Value'] \
        .replace({'>6': 6}) \
        .astype(float)


# Convert dtypes
data = data.convert_dtypes()

# Show types
print("\nData")
print(data)
print("\nDtypes")
print(data.dtypes)


# -----------------------------------------
# Tree
# -----------------------------------------
# Libraries
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.metrics import classification_report

# Features
FEATURES = [
    'Systemic inflammatory condition',
    'Age',
    'Band Intensity',
    'Asp GM Index Value'
]


# --------
# Probable
# --------
# Remove nan
aux = data[FEATURES + ['Probable IPA']]
aux = aux.dropna(how='any')

# Get X and y
X = aux[FEATURES].to_numpy()
y = aux['Probable IPA'].astype(int).to_numpy()

# Show data
print(aux)
print(aux.dtypes)

# Create classifier
clf1 = tree.DecisionTreeClassifier(random_state=0)

# Train
clf1 = clf1.fit(X, y)

# Confusion matrix
rpt = classification_report(y, clf1.predict(X))

# Show
print(rpt)

# Plot tree
tree.plot_tree(clf1)
plt.title("Tree - Probable")



# --------
# Possible
# --------
# Remove nan
aux = data[FEATURES + ['Possible IPA']]
aux = aux.dropna(how='any')

# Get X and y
X = aux[FEATURES].to_numpy()
y = aux['Possible IPA'].astype(int).to_numpy()

# Show data
print(aux)
print(aux.dtypes)

# Create classifier
clf1 = tree.DecisionTreeClassifier(random_state=0)

# Train
clf1 = clf1.fit(X, y)

# Confusion matrix
rpt = classification_report(y, clf1.predict(X))

# Show
print(rpt)

# Plot tree
tree.plot_tree(clf1)
plt.title("Tree - Possible")

plt.show()