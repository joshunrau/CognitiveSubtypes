import math

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_distributions(df: pd.DataFrame, variables: list, filepath=None):
    
    n_rows = math.ceil(len(variables) / 2)

    fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=100)

    for i, ax in enumerate(axes.flat):
        try:
            sns.histplot(data=df, x=variables[i], ax=ax)
        except IndexError:
            break
        ax.set_title(variables[i])
    
    fig.set_figwidth(16)
    fig.set_figheight(8 * n_rows)
    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath)


def plot_transforms(df1: pd.DataFrame, df2: pd.DataFrame, variables: list, filepath=None):
    
    n_rows = len(variables)
    fig, axes = plt.subplots(nrows=n_rows, ncols=2, dpi=100)
    
    for i in range(n_rows):
        sns.histplot(data=df1, x=variables[i], ax=axes[i][0])
        sns.histplot(data=df2, x=variables[i], ax=axes[i][1])
    
    fig.set_figwidth(16)
    fig.set_figheight(8 * n_rows)
    fig.tight_layout()
    
    if filepath is not None:
        plt.savefig(filepath)


def plot_missing_data(df: pd.DataFrame, variables: list, filepath=None):
    plt.figure(figsize=(10, len(variables)))
    sns.heatmap(df[variables].isna().transpose(), cmap="YlGnBu", cbar_kws={'label': 'Missing Data'})

