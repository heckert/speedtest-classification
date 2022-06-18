"""Sklearn custom transformers for pipeline"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
from typing import Union


def _slice(a: np.ndarray, start: int,end: int):
    """https://stackoverflow.com/questions/
       39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
    """
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.fromstring(b.tostring(),dtype=(str,end-start))

def _day_of_week(dts: np.ndarray):
    """https://stackoverflow.com/questions/
       52398383/finding-day-of-the-week-for-a-datetime64
    """
    return (dts.astype('datetime64[D]').view('int64') - 4) % 7


class FringeCategoryBucketer(BaseEstimator, TransformerMixin):
    """Group small categories into a common bucket.

    Args:
        keep_top_n: How many categories to keep
        bucket_name: What to return for the new category name
    """

    def __init__(self,
                 keep_top_n: int = 10,
                 bucket_name: str = 'other'):

        self.keep_top_n = keep_top_n
        self.bucket_name = bucket_name
        
    def fit(self, X, y = None):
        X = X.copy().astype(str)

        unique, counts = np.unique(X, return_counts=True)
        unique_counts = np.column_stack([unique, counts])

        sorter = unique_counts[:,1].astype(int).argsort()
        sorted_counts = unique_counts[sorter][::-1]

        self.selected_cats = sorted_counts[:self.keep_top_n, 0]

        return self

    def transform(self, X, y = None):
        X = X.copy().astype(str)

        cleaner_func = np.vectorize(lambda x: x if x in self.selected_cats else self.bucket_name)

        return cleaner_func(X)


class WeekendExtractor(BaseEstimator, TransformerMixin):
    """Transforms date column into 1s and 0s 
        if date was on a weekend or not.
    
    """
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        datetimes = X.astype('datetime64[D]')
        weekdays = _day_of_week(datetimes)
        weekend_bin = np.ones(shape=weekdays.shape)
        weekend_bin[weekdays<5] = 0

        return weekend_bin.reshape(-1,1)


class HourExtractor(BaseEstimator, TransformerMixin):
    """Extract hour from date field.
    
    """
    def __init__(self):
        pass
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        hours = _slice(X.astype(str), 11, 13).reshape(-1,1)

        return hours.astype(int)


class FeatureCrosser(BaseEstimator, TransformerMixin):
    """Append two columns into one.

    """
    def __init__(self, sep='-'):
        self.sep = sep
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()
        
        if X.shape[1] != 2:
            raise Exception('Input has to have two columns')

        array_list = []

        for i in range(X.shape[1]):
            arr = X[:, i]
            # If numeric, convert to int
            if np.issubdtype(arr.dtype, np.number):
                arr = arr.astype(int)
            str_array = np.array(arr, dtype=str)
            array_list.append(str_array)

        result = np.apply_along_axis(self.sep.join, 0, array_list)

        return result.reshape(-1,1)


class OutlierRemover(BaseEstimator, TransformerMixin):
    """Cap values lying a certain number of
        standard deviations above the median.

    Args:
        upper_threshold_factor: How many standarddeviations above
            the median to set the threshold.
        lower_threshold_factor: Not yet implemented
    """

    def __init__(self,
                 upper_threshold_factor: Union[int, None] = 3,
                 lower_threshold_factor: Union[int, None] = None):

        self.upper_threshold_factor = upper_threshold_factor
        self.lower_threshold_factor = lower_threshold_factor

    @staticmethod
    def _replace_values_above_thresholds(X: np.ndarray, thresholds: np.ndarray):
        X = X.copy()

        # HACK
        # If input is pandas, transform to numpy array
        if isinstance(X, pd.DataFrame):
            X = X.values

        elif isinstance(X, pd.Series):
            X = X.values.reshape(-1, 1)

        mask = X > thresholds

        tiled_thresholds = np.tile(thresholds, (X.shape[0],1))

        X[mask] = tiled_thresholds[mask]

        return X
        
    def fit(self, X, y = None):

        self.medians = np.nanmedian(X, axis=0)
        self.stds = np.nanstd(X, axis=0)

        if self.upper_threshold_factor is not None:
            self.upper_thresholds = self.medians + self.upper_threshold_factor * self.stds

        if self.lower_threshold_factor is not None:
            self.lower_thresholds = self.medians - self.lower_threshold_factor * self.stds

        return self

    def transform(self, X, y = None):
        X = X.copy()

        if self.upper_threshold_factor is not None:
            X = self._replace_values_above_thresholds(X, self.upper_thresholds)

        if self.lower_threshold_factor is not None:
            raise NotImplementedError('_replace_values_below_thresholds not yet implemented')

        return X
