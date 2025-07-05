# -*- coding: utf-8 -*-
"""
Created on Sat Jul  5 09:54:54 2025

@author: Diyar Altinses, M.Sc.
"""

# %% imports

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from tqdm import tqdm
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


from dataset import EmployeeDataset
from model import LogisticRegressionModel
from evaluate import evaluate_model
from config_plots import configure_plt

# %% setup

# Set random seeds for reproducibility
# torch.manual_seed(42)
# np.random.seed(42)
configure_plt()

# %% dataset

dataset = EmployeeDataset(10000)
features, labels = dataset.features, dataset.labels

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
input_dim = X_train.shape[1]
batch_size = 16
train_dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# %% model 

learning_rate = 0.00001
num_epochs = 200
log_reg_model = LogisticRegressionModel(input_dim)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(log_reg_model.parameters(), lr=learning_rate)

# %% training

losses = []
for epoch in range(num_epochs):
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = log_reg_model(inputs)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        optimizer.step()
    losses.append(loss.item())
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# %% Training des Random Forest

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# %% evaluation

with torch.no_grad():
    log_reg_probs = log_reg_model(torch.FloatTensor(X_test)).numpy()
    log_reg_preds = (log_reg_probs > 0.5).astype(int)

rf_preds = rf_model.predict(X_test)
rf_probs = rf_model.predict_proba(X_test)[:, 1]

log_reg_acc, log_reg_auc, log_reg_cm = evaluate_model(y_test, log_reg_preds, log_reg_probs, "Logistic Regression")
rf_acc, rf_auc, rf_cm = evaluate_model(y_test, rf_preds, rf_probs, "Random Forest")

# %% plots

results = pd.DataFrame({
    'Model': ['Logistic Regression', 'Random Forest'],
    'Accuracy': [log_reg_acc*100, rf_acc*100],
    'ROC AUC': [log_reg_auc, rf_auc]
})

print("\nComparison:")
print(results)

# %% Confusion Matrix 

plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
sns.heatmap(log_reg_cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression')
plt.subplot(1, 2, 2)
sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Greens')
plt.title('Random Forest')
plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi = 300)
plt.show()

# Feature Importance for Random Forest
feature_importance = rf_model.feature_importances_
features = ['Satisfaction', 'Salary', 'Servicelength', 'Overtime', 'Age']
plt.figure(figsize=(6, 3))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance (Random Forest)')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi = 300)
plt.show()

# %% GIF

os.makedirs('training_frames', exist_ok=True)

X_train_2d = X_train[:, :2]
X_test_2d = X_test[:, :2]

vis_model = LogisticRegressionModel(2)
vis_optimizer = torch.optim.Adam(vis_model.parameters(), lr=learning_rate)

x_min, x_max = X_train_2d[:, 0].min() - 0.1, X_train_2d[:, 0].max() + 0.1
y_min, y_max = X_train_2d[:, 1].min() - 0.1, X_train_2d[:, 1].max() + 0.1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                     np.linspace(y_min, y_max, 100))

for epoch in tqdm(range(num_epochs)):
    for inputs, targets in train_loader:
        inputs_2d = inputs[:, :2]  # Nur die ersten beiden Features
        vis_optimizer.zero_grad()
        outputs = vis_model(inputs_2d)
        loss = criterion(outputs, targets.unsqueeze(1))
        loss.backward()
        vis_optimizer.step()
    
    if epoch % 5 == 0:
        plt.figure(figsize=(6, 3))
        
        with torch.no_grad():
            Z = vis_model(torch.FloatTensor(np.c_[xx.ravel(), yy.ravel()])).numpy()
            Z = Z.reshape(xx.shape)
        
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', cmap='coolwarm')
        plt.xlabel('Job satisfaction')
        plt.ylabel('Salary (normalized)')
        plt.title(f'Decision limit - Epoch {epoch+1}')
        plt.colorbar(label='Probability of termination')
        
        # Frame speichern
        plt.tight_layout()
        plt.savefig(f'training_frames/frame_{epoch:03d}.png', dpi = 150)
        plt.close()


example_employee = np.array([[0.6, 0.8, 0.4, 0.3, 0.6]])  # Zufriedenheit 0.6, Gehalt 0.8, etc.

with torch.no_grad():
    log_reg_prob = log_reg_model(torch.FloatTensor(example_employee)).item()
rf_prob = rf_model.predict_proba(example_employee)[0, 1]

# %%

frames = []
path = os.path.join(os.getcwd(), 'training_frames')

for frame in os.listdir(path):
    img = Image.open(os.path.join(path, frame))
    frames.append(img)

frames[0].save("animation.gif",
               format="GIF",
               append_images=frames[1:],
               save_all=True,
               duration=500,
               loop=0)

