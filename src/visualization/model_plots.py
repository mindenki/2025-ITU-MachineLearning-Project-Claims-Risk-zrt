import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import LearningCurveDisplay



def residual_plot(y_true,y_pred):
    """ Plots the residuals of a regression model."""
    
    residuals = y_true - y_pred
    plt.figure(figsize=(10,6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted Values')
    plt.show()
    
def parity_plot(y_true, y_pred):
    """ Plots the parity plot of true vs predicted values."""
    
    plt.figure(figsize=(10,6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    max_val = max(max(y_true), max(y_pred))
    min_val = min(min(y_true), min(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--')
    plt.xlabel('True Values')
    plt.ylabel('Predicted Values')
    plt.title('Parity Plot: True vs Predicted Values')
    plt.show()

def prediction_distribution(y_true, y_pred):
    """ Plots the distribution of true vs predicted values."""
    
    plt.figure(figsize=(10,6))
    sns.kdeplot(y_true, label='True Values', color='blue', fill=True, alpha=0.5)
    sns.kdeplot(y_pred, label='Predicted Values', color='orange', fill=True, alpha=0.5)
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.title('Distribution of True vs Predicted Values')
    plt.legend()
    plt.show()
    
    
def plot_learning_curve(model, X, y, cv=None, scoring="neg_mean_absolute_error"):
    lc = LearningCurveDisplay.from_estimator(
        model,
        X,
        y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        figsize=(10, 6),
    )
    lc.ax_.set_title("Learning Curve")
    lc.ax_.set_xlabel("Training Examples")
    lc.ax_.set_ylabel(scoring)
    plt.grid()
    plt.show()
    