import torch
import torch.nn as nn
import numpy as np

from math import pi
from scipy.special import logsumexp

# Placeholder: add an MLP autoencoder

# Otherwise follow https://tomekkorbak.com/2021/02/09/em-for-gaussian-mixtures-using-einsum/

class PCAutoencoder(nn.Module):
    def __init__(self, n_hidden):
        super(PCAutoencoder, self).__init__()

    def fit(self, x, delta=1e-3, n_iter=100, warm_start=False):
        pass

    def predict(self, x, probs=False):
        pass

    def predict_proba(self, x):
        pass

    def sample(self, n):
        pass