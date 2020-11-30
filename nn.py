# simple NN with skorch
# that does balanced dataset at training time
#
from collections import OrderedDict

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torch.nn.functional as F
import skorch
from sklearn.model_selection import StratifiedKFold


def make_ff_nn(
        input_size,
        num_hidden_layers,
        hidden_size,
        activation,
        dropout,
):
    """
    Wrapper function to assemble feed forward NN using pytorch sequential
    """
    modules = [
        ("input_layer", nn.Linear(input_size, hidden_size)),
        ("input_drop_out", nn.Dropout(p=dropout)),
    ]
    for ix in range(num_hidden_layers - 1):
        l = nn.Linear(hidden_size, hidden_size)
        modules.append(("hidden_layer_{}".format(ix), l))
        if activation == "logistic":
            a = nn.Sigmoid()
        elif activation == "tanh":
            a = nn.Tanh()
        elif activation == "relu":
            a = nn.ReLU()
        else:
            raise ValueError("No such activation function defined")
        modules.append(("activation_{}".format(ix), a))
        if dropout > 0:
            modules.append(("drop_out_{}".format(ix), nn.Dropout(p=dropout)))
    # output layer
    modules.append(("output_layer", nn.Linear(hidden_size, 1)))
    modules.append(("output_fun", nn.Sigmoid()))
    return nn.Sequential(OrderedDict(modules))


class FeedForwardNN(nn.Module):
    """
    Simple feed forward network with dropout
    """
    def __init__(
            self,
            input_size,
            num_hidden_layers=3,
            hidden_size=100,
            activation="logistic",
            dropout=0.5,
    ):
        super().__init__()
        self.model = make_ff_nn(
            input_size,
            num_hidden_layers,
            hidden_size,
            activation,
            dropout,
        )

    def forward(self, X):
        return self.model(X)


def to_float32(elem):
    """
    Need to cast X or y to float32
    """
    return elem.astype(np.float32)


class PatchedNeuralNetClassifier(skorch.NeuralNetClassifier):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def fit(self, X, y):
        # conveniences
        if X.dtype != np.float32:
            X = to_float32(X)

        if y.dtype != np.float32:
            y = to_float32(y)

        # to align with sklearn expectations
        self.classes_ = np.array([False, True])
        super().fit(X, y)

    def predict_proba(self, X):
        prob_true = super().predict_proba(X)
        prob_false = 1.0 - prob_true
        return np.hstack((prob_false, prob_true))

    def predict(self, X):
        prob_true = super().predict_proba(X)
        return prob_true.flatten() > 0.5


class WeightedBCELoss(nn.Module):
    def __init__(self, weight=None):
        super().__init__()
        self.weight = None

    def forward(self, _input, target):
        # compute weights automatically
        if self.weight is None:
            num_pos = target.sum()
            num_neg = (1 - target).sum()
            # need to match dimensions
            weight = torch.tensor([1.0, num_neg / num_pos],
                                  dtype=torch.float32)
            weights = weight[target.to(torch.long)].reshape(-1, 1)
            weights = weights.to(_input.device)
        return F.binary_cross_entropy(
            _input,
            target.reshape(-1, 1),
            weight=weights,
        )


def get_nn():
    """
    Wrapper to produce model
    """
    net = PatchedNeuralNetClassifier(
        FeedForwardNN,
        optimizer=optim.Adam,
        criterion=WeightedBCELoss,
        # use all data in each batch, so that
        # the weighted loss makes sense...
        batch_size=-1,
        max_epochs=5,
        # early stopping
        callbacks=[
            ('early-stopping',
             skorch.callbacks.EarlyStopping(
                 monitor='valid_loss',
                 patience=3,
                 threshold=0.0001,
                 threshold_mode='rel',
                 lower_is_better=True,
             )),
        ],
        device="cuda",
    )
    return net
