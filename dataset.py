# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 10:47:04 2025

@author: Altinses, M.Sc.
"""

# %% imports

import torch
import numpy as np

# %% dataset

class EmployeeDataset(torch.utils.data.Dataset):
    def __init__(self, num_samples=10000):
        self.num_samples = num_samples
        self.features, self.labels = self.generate_data()
        
    def generate_data(self):
        satisfaction = np.random.normal(0.5, 0.2, self.num_samples)
        salary = np.random.normal(0.7, 0.15, self.num_samples)
        tenure = np.random.exponential(5, self.num_samples) / 10  # Durchschnittlich 5 Jahre
        overtime = np.random.poisson(5, self.num_samples) / 10
        age = np.random.normal(35, 10, self.num_samples) / 50
        
        features = np.column_stack([satisfaction, salary, tenure, overtime, age])
        
        logit = (-2.5 + 4*satisfaction + 6*salary - 1.5*tenure + 2*overtime - 1*age)
        probabilities = 1 / (1 + np.exp(-logit))
        
        # Labels generieren (0 = bleibt, 1 = kündigt)
        labels = np.random.binomial(1, probabilities)
        
        return features, labels
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.features[idx]), torch.FloatTensor([self.labels[idx]])
