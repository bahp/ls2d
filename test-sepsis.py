"""

Import references


- REF1: This reference...

    https://stackoverflow.com/questions/54401560/chunking-dataframe-by-gaps-in-datetime-index

"""


# Libraries
import pandas as pd
import numpy as np


# --------------------------------------------------
# Load data
# --------------------------------------------------
# Load data
data = pd.read_csv('./datasets/sepsis/data.csv',
    nrows=100000,
    parse_dates=['date_collected',
                 'date_sample',
                 'date_outcome'])

# Show
print("\nData:")
print(data)
print("\nDtypes:")
print(data.dtypes)

# ----------------------------
# Filters
# ----------------------------
# These might not be necessary since we just
# want to plot in the 2D space a point which
# represents X days worth of data.

# Keep only those records with information
# n days before the microbiology sample was
# collected.
data = data[(data.day >= -5) & (data.day <= 0)]

# Keep only patients with all the data. Or
# maybe keep patients with at least n days
# worth of data.

# Keep only top 5 organisms.
top = data.micro_code.value_counts()
data = data[data.micro_code.isin(
    top.head(5).index.values)]
a = data.micro_code.value_counts().head(5)

print("\nMicrobiology code counts:")
print(top)


# Select features
FEATURES = [
    'PLT', 'RBC', 'RDW', 'WBC', 'HCT', 'HGB'
]
data = data[data.code.isin(FEATURES)]


# ----------------------------
# Format data structure
# ----------------------------
# Pivot table.
piv = pd.pivot_table(data,
    values=['result'],
    index=['PersonID', 'date_collected'],
    columns=['code'])

# Basic formatting
piv.columns = piv.columns.droplevel(0)
piv = piv.reset_index()
piv = piv.set_index('date_collected')
piv.index = piv.index.normalize()


def reindex_by_date(df):
    """This methods reindex each group.

    Ensures that there is an entry for each day including
    both min and max dates. In addition it forward fills
    and backward fills missing data.

    Would be interesting to use IterativeImputer?
    """
    df = df.groupby(level=0).mean()
    dates = pd.date_range(df.index.min(), df.index.max())
    a = df.reindex(dates).ffill().bfill()
    return a.reset_index()

# Reindex
piv = piv.groupby('PersonID') \
    .apply(reindex_by_date) \
    .reset_index(drop=True)


a = data[['PersonID', 'micro_code']].drop_duplicates()
d = dict(zip(a.PersonID, a.micro_code))
piv['micro_code'] = piv.PersonID.map(d)

# Show pivot
print("\nPivoted:")
print(piv)
print("\nPatients: %s" % piv.PersonID.nunique())
print("\nMissing:")
print(piv.isna().sum().to_frame().T)


from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

piv[FEATURES] = SimpleImputer().fit_transform(piv[FEATURES])
piv[FEATURES] = StandardScaler().fit_transform(piv[FEATURES])

# Create sub-sequences
# --------------------
# Create final 3D matrix (list of lists)
matrix = []

def generate_subsequences(aux, groupby='PersonID', w=5):
    """This method generates subsequences.

    Parameters
    ----------
    aux: pd.DataFrame
        The pandas dataframe
    groupby: str
        The groupby method.
    w: int
        The window length
    """
    # Group
    for i in range(0, (aux.shape[0] - w)):
        matrix.append(aux[i:i+w].to_numpy()[:, 2:])
    # Return
    return None

# Call function (fills matrix)
fmt = piv.groupby('PersonID').apply(generate_subsequences)
# Format matrix
matrix = np.asarray(matrix)

# Show matrix
print("\nMatrix shape: %s\n\n" % str(matrix.shape))

matrix2 = matrix.copy()
matrix = matrix[:,:,:-1].astype('float32')

# -------------------------------------
# Quick Keras Model
# -------------------------------------
# Create
from numpy import array
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from tensorflow.keras import optimizers

import matplotlib.pyplot as plt

"""
# define input sequence
seq_in = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
# reshape input into [samples, timesteps, features]
n_in = len(seq_in)
seq_in = seq_in.reshape((1, n_in, 1))
# prepare output sequence
seq_out = seq_in[:, 1:, :]
n_out = n_in - 1
# define encoder
visible = Input(shape=(n_in,1))
encoder = LSTM(100, activation='relu')(visible)
# define reconstruct decoder
decoder1 = RepeatVector(n_in)(encoder)
decoder1 = LSTM(100, activation='relu', return_sequences=True)(decoder1)
decoder1 = TimeDistributed(Dense(1))(decoder1)
# define predict decoder
decoder2 = RepeatVector(n_out)(encoder)
decoder2 = LSTM(100, activation='relu', return_sequences=True)(decoder2)
decoder2 = TimeDistributed(Dense(1))(decoder2)
# tie it together
model = Model(inputs=visible, outputs=[decoder1, decoder2])
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(seq_in, [seq_in,seq_out], epochs=300, verbose=0)
# demonstrate prediction
yhat = model.predict(seq_in, verbose=0)
print(yhat)

# Show model
print("\nModel:")
model.summary()
print("\nPrediction:")
print(yhat)
print(seq_in)

print(seq_in.reshape(1,-1))

pred = pd.DataFrame()
pred['in'] = seq_in.reshape(-1,1)
#pred['dec1'] = yhat[0].reshape(-1,1)

print(pred)
"""


# -----------------------------------------------------------------
# Example of LSTM autoencoder
# -----------------------------------------------------------------
# REF: https://towardsdatascience.com/using-lstm-autoencoders-on-multidimensional-time-series-data-f5a7a51b29a1

# The model begins with an Encoder: first, the input layer. The input layer is
# an LSTM layer. This is followed by another LSTM layer, of a smaller size. Then,
# I take the sequences returned from layer 2 — then feed them to a repeat vector.
# The repeat vector takes the single vector and reshapes it in a way that allows
# it to be fed to our Decoder network which is symmetrical to our Encoder. Note
# that it doesn’t necessarily have to be symmetrical, but this is standard practice.

from keras import metrics
import keras
import tensorflow as tf


# Define earl stop
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', min_delta=1e-2, patience=5, verbose=0,
    mode='auto', baseline=None, restore_best_weights=True)

# Variables
samples, timesteps, features = matrix.shape

# Construct model
model = Sequential()
model.add(LSTM(64, kernel_initializer='he_uniform',
                   batch_input_shape=(None, timesteps, features),
                   return_sequences=True, name='encoder_1'))
model.add(LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='encoder_2'))
model.add(LSTM(2, kernel_initializer='he_uniform', return_sequences=False, name='encoder_3')) # return false and Repeat Vector
model.add(RepeatVector(timesteps, name='encoder_decoder_bridge'))
model.add(LSTM(2, kernel_initializer='he_uniform', return_sequences=True, name='decoder_1'))
model.add(LSTM(32, kernel_initializer='he_uniform', return_sequences=True, name='decoder_2'))
model.add(LSTM(64, kernel_initializer='he_uniform', return_sequences=True, name='decoder_3'))
model.add(TimeDistributed(Dense(features)))
model.compile(loss="mse", optimizer='adam')
model.build()

# Show model
print(model.summary())

# Fit model
history = model.fit(x=matrix, y=matrix,
    validation_data=(matrix, matrix),
    epochs=200, batch_size=16,
    shuffle=False, callbacks=[early_stop])


aux = model.predict(matrix)

#print(aux)


encoder = Model(inputs=model.inputs, outputs=model.layers[2].output)

encoded = encoder.predict(matrix)

print(encoded.shape)
#print(encoded)

df = pd.DataFrame(data=encoded, columns=['x', 'y'])
df['label'] = matrix2[:,:,-1][:,1] # Get first in row because all ae the same
df['day'] = 'feo'

print(df)

import plotly.express as px

fig = px.scatter(df, x='x', y='y',
                 color='label',
                 hover_data=['day'])
fig.show()

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()
# ------------------------------------------------------------------


"""

matrix = matrix[:,:,1].reshape(-1, 5, 1)

v = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

matrix = np.array(v).reshape((1, 9, 1))




# Create variables
samples, timesteps, features = matrix.shape

# Set params
epochs = 20
batch = 8
lr = 0.0001

# Create model
model = Sequential()
model.add(LSTM(10, input_shape=(timesteps, features), return_sequences=True))
model.add(LSTM(6, activation='relu', return_sequences=True))
model.add(LSTM(1, activation='relu'))
model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(10, kernel_initializer='glorot_normal', activation='relu'))
model.add(Dense(1))
# Show model
print("\n")
model.summary()

# Compile model
model.compile(optimizer=optimizers.Adam(lr), loss='mse')

# Fit model
model.fit(matrix, matrix, epochs=300, verbose=0)

# Predict
yhat = model.predict(matrix[1,:,:].reshape(1, timesteps, 1), verbose=0)

# Show
print(yhat)
"""