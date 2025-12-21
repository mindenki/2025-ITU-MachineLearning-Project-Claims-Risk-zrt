import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import learning_curve
from sklearn.model_selection import LearningCurveDisplay



def residual_plot(y_true, y_pred, gridsize=50, name=""):
    sns.set(style="whitegrid")
    residuals = y_true - y_pred
    
    plt.figure(figsize=(10,6))
    
    # Hexbin density plot
    hb = plt.hexbin(
        y_pred, residuals,
        gridsize=gridsize,
        cmap="viridis",
        mincnt=1,          
        linewidths=0.5,
        edgecolors='grey'
    )
    
    # Color bar for count
    cb = plt.colorbar(hb)
    cb.set_label('Number of points')
    
    # Zero residual reference line
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5)
    
    # Labels and title
    plt.xlabel('Predicted Values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title(f'Residuals vs Predicted Values (Density): {name}', fontsize=14)
    
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()
    
    


def parity_plot(y_true, y_pred, jitter=0.0, circle_size=80, name=""):

    
    sns.set(style="whitegrid")
    
    y_true_plot = y_true
    y_pred_plot = y_pred

    plt.figure(figsize=(10,6))

    # Scatter plot
    plt.scatter(
        y_true_plot, y_pred_plot,
        alpha=0.6,
        s=circle_size,
        edgecolors='black',
        linewidth=0.4,
        c=sns.color_palette("viridis", as_cmap=True)(0.6)
    )

    # Identity line
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', linewidth=1.5)

    plt.xlabel('True Values', fontsize=12)
    plt.ylabel('Predicted Values', fontsize=12)
    plt.title(f'Parity Plot: True vs Predicted Values: {name}', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()
    
def prediction_distribution(y_true, y_pred, bins=50, name=""):
    sns.set(style="whitegrid")  

    plt.figure(figsize=(10,6))
    
    # Plot histograms
    sns.histplot(y_true, bins=bins, color='blue', alpha=0.5, stat='density', label='True Values', kde=False)
    # lets make the orange thicker for predicted values
    sns.histplot(y_pred, bins=bins, color='orange', alpha=1, stat='density', label='Predicted Values', kde=False)


    plt.yscale('log')

    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Density (log scale)', fontsize=12)
    plt.title(f'Distribution of True vs Predicted Values: {name}', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.show()
    
def prediction_distribution_violin(y_true, y_pred, name=""):
    import pandas as pd

    df = pd.DataFrame({
        'True Values': y_true,
        'Predicted Values': y_pred
    })

    df_melt = df.melt(var_name='Type', value_name='Value')

    plt.figure(figsize=(8,6))
    sns.violinplot(x='Type', y='Value', data=df_melt, palette=['blue', 'orange'], inner='quartile')
    plt.title(f'Distribution of True vs Predicted Values (Violin Plot): {name}')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.show()

    
def plot_learning_curve(model, X, y, cv=None, scoring="neg_mean_absolute_error", train_sizes=np.linspace(0.1,1.0,10), name=""):
    # Compute learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        train_sizes=train_sizes
    )
    
    train_scores = -train_scores
    val_scores = -val_scores

    # clculate mean and std
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)

    plt.figure(figsize=(10,6))


    color_train = "blue"   
    color_val = "orange"     

    # Plot train score with shadedd std
    plt.plot(train_sizes, train_mean, color=color_train, label="Train Score", marker='o')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color=color_train, alpha=0.2)

    # Plot validation score with shaded sdt
    plt.plot(train_sizes, val_mean, color=color_val, label="Validation Score", marker='o')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, color=color_val, alpha=0.2)

    plt.title(f"Learning Curve: {name}", fontsize=14)
    plt.xlabel("Training Examples", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.show()
    
    
def tail_residuals_plot(y_true, y_pred, name):
        
        residuals = y_true - y_pred
        threshold = np.percentile(y_true, 99)
        top_idx = y_true >= threshold
        top_residuals = residuals[top_idx]
        plt.figure(figsize=(8,5))
        sns.histplot(top_residuals, color='purple', kde=True, bins=20)
        plt.axvline(0, color='red', linestyle='--')
        plt.title(f'Tail Residuals (Top 1% Claims): {name}')
        plt.xlabel('Residuals')
        plt.show()
        