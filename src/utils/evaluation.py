import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.metrics

from dataclasses import dataclass
from typing import Optional, Iterable


class ModelEvaluator:
    """Helper class to plot evaluation statistics and compare models.
    """
 
    def __init__(self):
        self._class_results_container = {}
        self._macro_avgs_container = {}
        self._weighted_avgs_container = {}
        self._accuracies_container = {}
        self.classes = None

    def store(self, y_true: np.array, y_pred: np.array, *, name: str) -> None:
        """Stores model performance statistics.

        Args:
            y_true (np.array): The true labels
            y_pred (np.array): The predictions
            name (str): A name for the current model
        """

        classification_report = sklearn.metrics.classification_report(y_true, y_pred, output_dict=True)

        # Pop avg results from classification report
        # so that only results per class remain.
        macro_avg = classification_report.pop('macro avg')
        accuracy = classification_report.pop('accuracy')
        weighted_avg = classification_report.pop('weighted avg')

        if not self.classes:
            self.classes = [key for key, _ in classification_report.items()]

        results_per_class = pd.DataFrame.from_dict(classification_report, orient='index')
        self._class_results_container[name] = results_per_class

        self._macro_avgs_container[name] = macro_avg
        self._weighted_avgs_container[name] = weighted_avg
        self._accuracies_container[name] = accuracy


    def get_summary(
        self, y_true: np.array, y_pred: np.array, *, 
        name: str,
        classification_report: bool = True,
        store: bool = True
    ) -> None:

        sklearn.metrics.ConfusionMatrixDisplay.from_predictions(y_true, y_pred)

        if name is not None:
            plt.title(name)

        plt.show()

        if classification_report:
            print(sklearn.metrics.classification_report(y_true, y_pred))

        if store:
            self.store(y_true, y_pred, name=name)


    @property
    def class_results(self) -> pd.DataFrame:
        """Results per class for each of the models

        Returns:
            pd.DataFrame: A multiindexed dataframe (name, class)
        """
        return pd.concat(self._class_results_container)


    @property
    def macro_results(self) -> pd.DataFrame:
        """Results per class for each of the models

        Returns:
            pd.DataFrame: A multiindexed dataframe (name, class)
        """
        return pd.DataFrame(self._macro_avgs_container).T


    @staticmethod
    def _plot_abstract(
        df, *,
        metric,
    ) -> None:
 
        df.plot(kind='bar')
        plt.title(metric)
        plt.xticks(rotation=45)
        #plt.legend(title='class')
        plt.show()


    def _get_plot_df(
        self, *,
        type: str,
        metric: str
    ) -> pd.DataFrame:
        

        transformations = {
            'class_results': self.class_results[metric].unstack()
        }

        if type not in transformations.keys():
            allowed_params: str = str(list(transformations.keys()))
            raise ValueError(f'Parameter type must be in {allowed_params}')

        return transformations[type]

    
    def plot(
        self,
        type='class_results',
        metric='f1-score'
    ) -> None:

        self._plot_abstract(
            df=self._get_plot_df(type=type, metric=metric),
            metric=metric
        )