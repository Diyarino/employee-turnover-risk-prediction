# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 10:49:56 2025

@author: Altinses
"""

# %% imports

import torch

# %% model

class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = torch.nn.Linear(input_dim, 1)
        
    def forward(self, x):
        return torch.sigmoid(self.linear(x))