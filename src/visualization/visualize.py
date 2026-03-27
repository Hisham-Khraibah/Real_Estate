import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_correlation_heatmap(data):
    """
    Plot a correlation heatmap for the given data.
    
    Args:
        data (pandas.DataFrame): The input data.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap', fontsize=16)
    plt.show()

def plot_feature_importance(model, x):
    """
    Plot a bar chart showing the feature importances.
    
    Args:
        model: Trained model with feature_importances_ attribute.
        x (pandas.DataFrame): Feature dataset.
    """
    fig, ax = plt.subplots()
    ax = sns.barplot(x=model.feature_importances_, y=x.columns)
    plt.title("Feature importance chart")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    fig.savefig("feature_importance.png")