"""


https://stackoverflow.com/questions/67456368/pytorch-getting-runtimeerror-found-dtype-double-but-expected-float

"""

import pandas as pd
import numpy as np

# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.base import BaseEstimator

from torch.utils.data import DataLoader
from skorch import NeuralNet

class SkorchAE(NeuralNet):
    def transform(self, X, *args, **kwargs):
        if isinstance(X, pd.DataFrame) or isinstance(X, pd.Series):
            X = X.to_numpy().astype(np.float32)
        if isinstance(X, np.ndarray):
            X = X.astype(np.float32)
        return self.module_.encode_inputs(X, *args, **kwargs)

    def fit(self, X, y, *args, **kwargs):
        X_ = torch.from_numpy(X).float()
        super().fit(X_, X_)

    def get_params(self, deep=True, **kwargs):
        """Overwritten.

        In order to get unique signatures when creating the pipelines,
        it is necessary to remove those parameters from NeuralNet that
        are created dynamically. Thus, in addition to avoid including
        the callbacks, we have to remove also the valid split as these
        are not part of the model hyperparameter configuration.

        'train_split':
            <skorch.dataset.ValidSplit object at 0x1544ffcd0>
        """
        params = BaseEstimator.get_params(self, deep=deep, **kwargs)
        # Callback parameters are not returned by .get_params, needs
        # special treatment.
        #params_cb = self._get_params_callbacks(deep=deep)
        #params.update(params_cb)
        # don't include the following attributes
        to_exclude = {'_modules', '_criteria', '_optimizers', 'train_split'}
        return {key: val for key, val in params.items()
            if key not in to_exclude}



class AE(nn.Module):
    """Symmetric Autoencoder (SAE)

    It only allows to choose the layers. It would be nice to
    extend this class so that we can choose also the activation
    functions (Sigmoid, ReLu, Softmax), additional dropouts, ...
    """
    def __init__(self, layers):
        super(AE, self).__init__()

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
        encoded = self.encoder(x.float())
        decoded = self.decoder(encoded)
        return decoded

    @torch.no_grad()
    def encode_inputs(self, x):
        #print(self.encoder(torch.tensor(x)))
        z = []
        for e in DataLoader(x, 16, shuffle=False):
            z.append(self.encoder(e))
        return torch.cat(z, dim=0).numpy()






if __name__ == '__main__':

    # Libraries
    import numpy as np
    import pandas as pd

    from sklearn import datasets
    from sklearn.pipeline import Pipeline
    from sklearn.metrics import mean_squared_error
    from sklearn.datasets import make_classification
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import GridSearchCV

    from skorch import NeuralNet
    from skorch import NeuralNetClassifier
    from skorch import NeuralNetRegressor

    # -----------
    # Create data
    # -----------
    # Load data
    X, y = iris = datasets.load_iris(return_X_y=True)
    #X, y = make_classification(1000, 20, n_informative=10, random_state=0)
    X = X.astype(np.float32)
    y = y.astype(np.int64)

    # Sample
    top = X[:2, :]

    # Show
    print("\nData: %s\n" % str(X.shape))

    # --------------------------------------
    # Create Autoencoder
    # --------------------------------------
    ae1 = NeuralNet(AE,
          max_epochs=10, lr=0.1,
          train_split=None,
          #criterion=torch.nn.NLLLoss,
          criterion=torch.nn.MSELoss,
          iterator_train__shuffle=False,
          module__layers=[4, 3, 2],
          #optimizer=torch.optim.Adam,
          #optimizer=torch.optim.RMSprop
    )

    # Show NeuralNet
    print("\nNeuralNet: \n%s" % ae1)

    # Fit skorch
    print("\nFit:")
    aux = ae1.fit(X, y=X)

    print("\nAutoencoder: \n%s" % aux.module_)



    # Prediction and encoding
    p0 = aux.predict(top)
    p1 = aux.predict_proba(top)
    e1 = aux.module_.encode_inputs(top)

    # Errors
    l1 = F.mse_loss( \
            torch.from_numpy(p1),
            torch.from_numpy(top),
            reduction='mean')

    l2 = torch.nn.MSELoss(reduction='mean')( \
            torch.from_numpy(p1),
            torch.from_numpy(top))

    l3 = mean_squared_error(p1, top)

    # Show
    print("\n")
    print("Top: %s" % top)
    print("Pred 0:  %s" % p0)
    print("Pred 1:  %s" % p1)
    print("Encode:  %s" % e1)
    print("\n")
    print("Loss 1: %s" % l1)
    print("Loss 2: %s" % l2)
    print("Loss 2: %s" % l3)



    # -------------------------------------------------
    # Create pipeline
    # -------------------------------------------------
    # Create pipeline
    pipe = Pipeline([
        ('net', aux),
    ])

    # Fit pipeline
    pipe = pipe.fit(X, y)

    # Prediction and encoding
    p0 = pipe.predict(top)
    p1 = pipe.predict_proba(top)
    e1 = pipe.module_.encode_inputs(top)

    # Show
    print("\n")
    print("Top: %s" % top)
    print("Pred 0:  %s" % p0)
    print("Pred 1:  %s" % p1)
    print("Encode:  %s" % e1)
    print("\n")

    import sys
    sys.exit()


    # -------------------------------------------------
    # Grid Search
    # -------------------------------------------------

    # Create pipeline
    pipe = Pipeline([
        ('scale', StandardScaler()),
        ('net', ae1),
    ])

    # Parameters
    params = {
        'net__lr': [0.01],
        'net__max_epochs': [20],
        'net__module__layers': [[4, 2]]
    }

    def loss_fn(recon_x, x):
        return F.mse_loss(recon_x, x, reduction='sum')


    def custom_metrics(est, X, y):
        """This method computes the metrics."""
        # Transform
        y = est.predict(X)
        # Return
        return mean_squared_error(y, X)


    # Create grid search
    # Since we are not really classification accuracy might be a
    # bad metric, since we are using the encoder/decoder the loss
    # would be the best? But not for usability.
    gs = GridSearchCV(pipe, params, refit=True,
        cv=2, scoring='neg_mean_squared_error')

    # Fit
    gs.fit(X, y=X)

    # Predictions
    p3 = gs.best_estimator_.predict_proba(top)

    # Show
    print("\n")
    print("Best score: %s" % gs.best_score_)
    print("Best param: %s" % gs.best_params_)
    print("Best model: \n%s" % gs.best_estimator_)
    print("\n")
    print("DataFrame:")
    print(pd.DataFrame(gs.cv_results_))
    print("Predictions:")
    print(p3)



    import sys
    sys.exit()

    # --------------------------
    # Create Autoencoder (image)
    # --------------------------
    # Create model
    model = AE()

    # Validation using MSE Loss function
    loss_function = torch.nn.MSELoss()

    # Using an Adam Optimizer with lr = 0.1
    optimizer = torch.optim.Adam(model.parameters(),
        lr=1e-1, weight_decay=1e-8) # optim.RMSprop




