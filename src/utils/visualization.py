import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_comparison_histogram(
    metric: pd.Series,
    categories: pd.Series,
    *args,
    title: str = None,
    dropna=True,
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
            **kwargs
        )

    if title is not None:
        plt.title(title)

    plt.legend()
    plt.show()

