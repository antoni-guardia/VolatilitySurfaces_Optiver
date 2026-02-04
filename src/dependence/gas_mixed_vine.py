import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class GAS_Pair_Copula(nn.Module):
    def __init__(self, family='gaussian'):
        super().__init__()
        self.family = family