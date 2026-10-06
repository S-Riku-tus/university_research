"""3 kHz acoustic-summary regressors; raw power -> fixed 34 features -> model.

The pipeline accepts the same time/frequency/channel arrays as the neural
models. It learns no feature transform from recording identities or labels.
Targets are scaled by ModelTrainer, just as for the original RF.
"""
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline

from utils.dataloading.acoustic_summary_features import AcousticFrequency34


def hgb_frequency34(**params):
    defaults = dict(max_iter=150, learning_rate=.05, max_leaf_nodes=31,
                    min_samples_leaf=20, l2_regularization=1., early_stopping=False)
    return Pipeline([("features", AcousticFrequency34()),
                     ("regressor", HistGradientBoostingRegressor(**{**defaults, **params}))])


def extra_trees_frequency34(**params):
    defaults = dict(n_estimators=256, min_samples_leaf=2, max_features=1., n_jobs=2)
    return Pipeline([("features", AcousticFrequency34()),
                     ("regressor", ExtraTreesRegressor(**{**defaults, **params}))])
