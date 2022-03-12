import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_comparison_histogram(
    metric: pd.Series,
    categories: pd.Series,
    *args,
    title: str = None,
    dropna: bool = True,
    vlines: list = None, 
    **kwargs
):
    for cat in categories.unique():

        if dropna:
            if (cat is None) | (cat is np.nan):
                continue

        plt.hist(
            metric[categories == cat],
            *args,
            label=cat,
            density=True,
            alpha=.5,
            **kwargs,
        )

    if title is not None:
        plt.title(title)

    if vlines is not None:
        for line in vlines:
            plt.axvline(line, linestyle='--', color='grey')
        

    plt.legend()
    plt.show()

