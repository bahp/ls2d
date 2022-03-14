###############################################################################
# Author: Bernard Hernandez
# Filename:
# Date:
# Description:
#
###############################################################################


# Import scikits preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import Normalizer
from sklearn.preprocessing import QuantileTransformer
from sklearn.preprocessing import KBinsDiscretizer

# Import outlier detection
from sklearn.ensemble import IsolationForest

# Import imputers
from sklearn.impute import SimpleImputer

# Import default samplers from imblearn
#from imblearn.over_sampling import SMOTE
#from imblearn.under_sampling import RandomUnderSampler
#from imblearn.over_sampling import RandomOverSampler

# Import model selection
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# Import cross validation method
from sklearn.calibration import CalibratedClassifierCV

# Import classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier

# Import dimensionality reduction algorithms
from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import FastICA
from sklearn.decomposition import NMF
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding

# Import scikits metrics
from sklearn.metrics import make_scorer

# XGboost
#import xgboost as xgb

# Lightbm
#import lightgbm as lgbm

# Torch
import torch

# Skorch
import skorch

# Keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.constraints import maxnorm
# from keras.optimizers import SGD
# from keras.wrappers.scikit_learn import KerasClassifier

# Own libraries
#from pySML2 import metrics
#from pySML2.preprocessing import outliers

from ls2d.autoencoder import AE
from ls2d.autoencoder import SkorchAE

# ----------------------------------------------------------------------------
#                          Deault Keras NN
# ----------------------------------------------------------------------------
def build_keras_cnn(activation='relu',
                    dropout_rate=0.2,
                    optimizer='Adam'):
    """This method....

    .. note: The first layer should be same number as input features.
    .. note: The last layer should be same number as output features.
    """

    # Create model
    # model = Sequential()
    # model.add(Dense(21, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(30, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    # model.add(Dense(1, activation='sigmoid'))

    # Compile
    # model.compile(
    #    loss='binary_crossentropy',
    #    optimizer=optimizer,
    #    metrics=['accuracy']
    # )

    model = None

    # Return
    return model


# ----------------------------------------------------------------------------
#                     Pipeline Acronyms Definitions
# ----------------------------------------------------------------------------
# This are some acronyms that can be used by default. The acronym will be
# directly translated to the corresponding instance during the construction
# of the Pipeline object. Note that there is no need to use the acronyms
# since object instances can be passed directly.

# Default list of filters.
_DEFAULT_FILTERS = {
    'iso15': IsolationForest(contamination=0.15, random_state=42),
    #'iqr15': outliers.IQRTransformer()
}

_DEFAULT_TRANSFORMERS = {
    'kbins10': KBinsDiscretizer(n_bins=10, encode='ordinal')
}

# Default list of imputers.
_DEFAULT_IMPUTERS = {
    'median': SimpleImputer(strategy='median'),
    'mean': SimpleImputer(strategy='mean')
}

# Default list of samplers.
_DEFAULT_SAMPLERS = {
    #'randomu': RandomUnderSampler(random_state=42),
    #'randomo': RandomOverSampler(random_state=42),
    #'smote': SMOTE(k_neighbors=3),
}

# Default list of scalers.
_DEFAULT_SCALERS = {
    'minmax': MinMaxScaler(),
    'norm': Normalizer(),
    'robust': RobustScaler(),
    'std': StandardScaler(),
    'quant': QuantileTransformer(),
}

# Default list of splitters.
_DEFAULT_SPLITTERS = {
    'skfold': StratifiedKFold(n_splits=10, shuffle=False),
    'skfold5': StratifiedKFold(n_splits=5, shuffle=False),
    'skfold2': StratifiedKFold(n_splits=2, shuffle=False),
    'kfold5': KFold(n_splits=5),
    'kfold2': KFold(n_splits=2)
}

# Default list of estimators.
_DEFAULT_ESTIMATORS = {
    'gnb': GaussianNB(),
    'dtc': DecisionTreeClassifier(),
    'rfc': RandomForestClassifier(),
    'svm': SVC(),
    'ann': MLPClassifier(),
    'llr': LogisticRegression(),
    'etc': ExtraTreesClassifier(),
    #'xgb': xgb.XGBClassifier(),
    #'lgbm': lgbm.LGBMClassifier(),
    #'keras': KerasClassifier(build_fn=build_keras_cnn)

    'pca': PCA(),
    'nmf': NMF(),
    'lda': LatentDirichletAllocation(),
    'pcak': KernelPCA(),
    'pcai': IncrementalPCA(),
    'icaf': FastICA(n_components=2),
    'iso': Isomap(n_components=2),
    'lle': LocallyLinearEmbedding(n_components=2),
    'sae': SkorchAE(AE, criterion=torch.nn.MSELoss)
}

# Add calibrated estimators
def _calibrated_estimator(e):
    return CalibratedClassifierCV(base_estimator=e)

_DEFAULT_ESTIMATORS.update(
    {'c{0}'.format(k):_calibrated_estimator(v)
        for k,v in _DEFAULT_ESTIMATORS.items()})

# Create the final list of acronyms.
_DEFAULT_ACRONYMS = {'none': None}
_DEFAULT_ACRONYMS.update(_DEFAULT_FILTERS)
_DEFAULT_ACRONYMS.update(_DEFAULT_SAMPLERS)
_DEFAULT_ACRONYMS.update(_DEFAULT_IMPUTERS)
_DEFAULT_ACRONYMS.update(_DEFAULT_SCALERS)
_DEFAULT_ACRONYMS.update(_DEFAULT_SPLITTERS)
_DEFAULT_ACRONYMS.update(_DEFAULT_ESTIMATORS)

# ----------------------------------------------------------------------------
#                Pipeline Estimators Grid Search
# ----------------------------------------------------------------------------
param_grid_gnb = {
    'gnb__var_smoothing': [1e-2, 1e-9],
    # 'gnb__priors': [None]
}

param_grid_cgnb = {
    'gnb__base_estimator__var_smoothing': [1e-1, 1e-3]
}

param_grid_llr = {
    'llr__penalty': ['l2'],
    'llr__C': [0.01, 0.1, 1.0],
    'llr__fit_intercept': [True, False],
    'llr__class_weight': [None, 'balanced'],
    'llr__max_iter': [500]
}

param_grid_dtc = {
    'dtc__criterion': ['gini', 'entropy'],
    'dtc__max_features': [None],
    'dtc__class_weight': [None],
    'dtc__min_samples_leaf': [5, 50, 100],
    'dtc__min_samples_split': [10, 100, 200]
}

param_grid_rfc = {
    'rfc__n_estimators': [3, 5, 10, 100],
    'rfc__criterion': ['gini'],
    'rfc__max_features': [None],
    'rfc__class_weight': [None],
    'rfc__min_samples_leaf': [5, 50, 100],
    'rfc__min_samples_split': [5, 50]
}

param_grid_svm = {
    'svm__C': [1, 0.1, 0.01],
    'svm__gamma': [1, 0.1, 0.01],
    'svm__probability': [True],
    'svm__max_iter': [500]
}

param_grid_ann = {
    'ann__hidden_layer_sizes': [(20, 20),
                                (100, 100),
                                (100, 100, 100)],
    'ann__activation': ['relu'],
    'ann__solver': ['adam'],
    'ann__alpha': [1., 0.01],
    'ann__batch_size': ['auto'],
    'ann__learning_rate': ['constant'],
    'ann__learning_rate_init': [0.001],
    'ann__power_t': [0.5],
    'ann__max_iter': [1000],
    'ann__tol': [1e-4],
    'ann__warm_start': [False],
    'ann__momentum': [0.9],
}

param_grid_xgb = {
    "xgb__use_label_encoder": [False],
    "xgb__colsample_bytree": [0.1, 0.8],
    "xgb__gamma": [0.001, 1],
    "xgb__learning_rate": [0.01],    # default 0.1
    "xgb__max_depth": [2, 10],       # default 3
    "xgb__n_estimators": [10, 100],  # default 100
    "xgb__subsample": [0.2, 0.8]
}

param_grid_lgbm = {
    'lgbm__application': ['binary'],
    'lgbm__objective': ['binary'],
    'lgbm__metric': ['auc'],
    'lgbm__is_unbalance': ['true'],
    'lgbm__boosting': ['gbdt'],
    'lgbm__num_leaves': [31],
    'lgbm__feature_fraction': [0.5],
    'lgbm__bagging_fraction': [0.5],
    'lgbm__bagging_freq': [20],
    'lgbm__learning_rate': [0.05],
    'lgbm__verbose': [0],
    # 'lgbm__n_estimators': [10, 50, 100] ??
}

param_grid_keras = {
    'keras__epochs': [1, 2, 3],
    'keras__batch_size': [128]
    # 'keras__epochs' :              [100,150,200],
    # 'keras__batch_size' :          [32, 128],
    # 'keras__optimizer' :           ['Adam', 'Nadam'],
    # 'keras__dropout_rate' :        [0.2, 0.3],
    # 'keras__activation' :          ['relu', 'elu']
}

_DEFAULT_PARAM_GRIDS = {
    'gnb': param_grid_gnb,
    'llr': param_grid_llr,
    'dtc': param_grid_dtc,
    'rfc': param_grid_rfc,
    'svm': param_grid_svm,
    'ann': param_grid_ann,
    'xgb': param_grid_xgb,
    'lgbm': param_grid_lgbm,
    'keras': param_grid_keras,
}

# Add calibrated param grids
# .. note: It is also possible to select the calibration
#          method by including the following attribute in
#          the dictionary:
#          {{0}__method: ['sigmoid', 'isotonic']}
# .. example: csvm__base_estimator__C
def _calibrated_param_grids(name, grid):
    return {'{0}__base_estimator__{1}'.format(name, k.split('__')[1]):v
        for k,v in grid.items()
    }

_DEFAULT_PARAM_GRIDS.update(
    {'c{0}'.format(k):_calibrated_param_grids('c{0}'.format(k), v)
        for k,v in _DEFAULT_PARAM_GRIDS.items()})



# ----------------------------------------------------------------------------
#                           Performance Scores
# ----------------------------------------------------------------------------

"""
_DEFAULT_METRICS = {
    'roc_auc': 'roc_auc',  # Usually larger
    'aucroc': make_scorer(metrics._auc),  # Usually lower
    'aucpr': make_scorer(metrics._auc_pr),
    'sens': make_scorer(metrics._sens),
    'spec': make_scorer(metrics._spec),
    'gmean': make_scorer(metrics._gmean),
    'tp': make_scorer(metrics._confusion_matrix, group='tp'),
    'fp': make_scorer(metrics._confusion_matrix, group='fp'),
    'tn': make_scorer(metrics._confusion_matrix, group='tn'),
    'fn': make_scorer(metrics._confusion_matrix, group='fn'),
    #'mcc': make_scorer(metrics._mcc),
    #'plike': make_scorer(metrics._plike),
    #'nlike': make_scorer(metrics._nlike),
    #'bcr': make_scorer(metrics._bcr),
    #'ber': make_scorer(metrics._ber),
    # 'f1_micro': 'f1_micro',
    # 'f1-score': 'f1',
    # 'accuracy': 'accuracy',
    # 'precision': make_scorer(metrics.precision),
    # 'recall': make_scorer(metrics.recall),
    # 'f-score': make_scorer(metrics.fscore),
    # 'samples': make_scorer(metrics.samples)
}
"""
