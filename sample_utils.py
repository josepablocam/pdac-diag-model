import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def sample_imbalanced(y, ix, size, seed):
    """
    Sample dataset maintaining imbalance
    """
    # down sample entire dataset to size, maintaing imbalance
    splitter = StratifiedShuffleSplit(2, test_size=size, random_state=seed)
    y_at_ix = y[ix]
    split = splitter.split(y_at_ix, y_at_ix)
    _, test_ix = next(split)
    return ix[test_ix]
