# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 10:52:51 2025

@author: Altinses
"""

# %% imports

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# %%

def evaluate_model(y_true, y_pred, y_probs, model_name):
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)
    
    print(f"\n{model_name} Evaluation:")
    print(f"Accuracy: {acc:.4f}")
    print(f"ROC AUC: {roc_auc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    return acc, roc_auc, cm