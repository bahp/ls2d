# Using Skorch
import torch
import skorch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch import optim
from torch.utils.data import DataLoader

# Basic Functions we use in Skorch
from skorch import NeuralNet
from skorch.callbacks import EpochTimer
from skorch.callbacks import PrintLog
from skorch.callbacks import EpochScoring
from skorch.callbacks import PassthroughScoring
from skorch.dataset import CVSplit
from skorch.utils import get_dim
from skorch.utils import is_dataset
from skorch.utils import to_numpy

# Binary classification model using skorch learn
# Here we are importing the required libraries.

from sklearn.datasets import make_classification
from torch import nn
from skorch import NeuralNetClassifier


class MyModule(nn.Module):
    """Symmetric Autoencoder (SAE)

    It only allows to choose the layers. It would be nice to
    extend this class so that we can choose also the activation
    functions (Sigmoid, ReLu, Softmax), additional dropouts, ...
    """
    def __init__(self, layers):
        super(MyModule, self).__init__()

        # Set layers
        self.layers = layers

        # Create encoder
        enc = []
        for prev, curr in zip(layers, layers[1:]):
            enc.append(nn.Linear(prev, curr))
            enc.append(nn.Sigmoid())
        self.encoder = nn.Sequential(*enc)

        # Reversed layers
        rev = layers[::-1]

        # Create decoder
        dec = []
        for prev, curr in zip(rev, rev[1:]):
            dec.append(nn.Linear(prev, curr))
            dec.append(nn.Sigmoid())
        self.decoder = nn.Sequential(*dec)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    """
    def predict(self, x):
        return np.array([0.6, 0.4])

    def predict_proba(self, x):
        return np.array([0.6, 0.4])
    """

    @torch.no_grad()
    def encode_inputs(self, x):
        z = []
        for e in DataLoader(x, 16, shuffle=False):
            z.append(self.encoder(e))
        return torch.cat(z, dim=0).numpy()



# utility functions
def get_device():
    if torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'
    return device

def train(net, X, optimizer,
          criterion=nn.MSELoss(),
          NUM_EPOCHS=10, BATCH_SIZE=16):
    """Training the neural network manually.

    Parameters
    ----------
    net:
        The neural network.
    X:
    optimizer:
    criterion:
    NUM_EPOCHS:
    BATCH_SIZE:

    Returns
    -------
    train_loss: list
        The loss at each step.
    """
    # Create the loader
    trainloader = DataLoader(
        X, batch_size=BATCH_SIZE,
        shuffle=False
    )

    # Train
    train_loss = []
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for data in trainloader:
            img = data
            img = img.to(device)
            img = img.view(img.size(0), -1)
            optimizer.zero_grad()
            outputs = net(img)
            loss = criterion(outputs, img)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        loss = running_loss / len(trainloader)
        train_loss.append(loss)
        print('Epoch {} of {}, Train Loss: {:.3f}'.format(
            epoch + 1, NUM_EPOCHS, loss))

    return train_loss


# ------------------------------------
# Config
# ------------------------------------
BATCH_SIZE = 128
MAX_EPOCHS = 100
LR = 0.1


# ------------------------------------
# Load Data
# ------------------------------------

# Load data
X, y = make_classification(1000, 20,
    n_informative=10, random_state=0)
X = X.astype(np.float32)
y = y.astype(np.int64)

# ------------------------------------
# Manual example
# ------------------------------------
# Autoncoder
net = MyModule(layers=[20, 10, 5, 2])

# Criterion and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=LR)
optimizer = optim.SGD(net.parameters(), lr=LR)

# get the computation device
device = get_device()
print("\nUsing device: %s\n" % device)

# load the neural network onto the device
net.to(device)

# Train the network
train_loss = train(net, X,
    criterion=criterion,
    optimizer=optimizer,
    NUM_EPOCHS=MAX_EPOCHS,
    BATCH_SIZE=BATCH_SIZE)

# Show training loss
plt.figure()
plt.plot(train_loss)
plt.title('Train Loss (Manual)')
plt.xlabel('Epochs')
plt.ylabel('Loss')


# ------------------------------------
# Skorch example
# ------------------------------------
# By default it uses SGD optimizer
# Create model
model = NeuralNet(
    MyModule,
    module__layers=[20, 10, 5, 2],
    max_epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    optimizer__lr=LR,
    criterion=torch.nn.MSELoss,
    iterator_train__shuffle=False,
    #train_split=None
)

# Fit
model.fit(X, y=X)

# Get history
history = pd.DataFrame(model.history)

# Show info
print("\nModel: %s\n" % model.module_)
print("\nHistory: %s\n" % history)

# Show training loss
plt.figure()
plt.plot(history['train_loss'])
if 'valid_loss' in history:
    plt.plot(history['valid_loss'])
plt.title('Train Loss (Skorch)')
plt.xlabel('Epochs')
plt.ylabel('Loss')

# Show
plt.show()