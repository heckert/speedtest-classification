import numpy as np
import operator
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from typing import Union, Callable


def _slice(a: np.ndarray, start: int,end: int):
    """https://stackoverflow.com/questions/
       39042214/how-can-i-slice-each-element-of-a-numpy-array-of-strings
    """
    b = a.view((str,1)).reshape(len(a),-1)[:,start:end]
    return np.fromstring(b.tostring(),dtype=(str,end-start))

def _day_of_week(dates: np.ndarray) -> np.ndarray:
    """https://stackoverflow.com/questions/
       52398383/finding-day-of-the-week-for-a-datetime64
    """
    return (dates.astype('datetime64[D]').view('int64') - 4) % 7
    
def _convert_pd_to_np(X: Union[pd.DataFrame, pd.Series]) -> np.ndarray:
    X = X.values

    # Series are one-dimensional. Reshape to 2 dims.
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    return X


class HourExtractor(BaseEstimator, TransformerMixin):
    """Extract hour from date field.
    
    """
    def __init__(self, utc_to_cet: bool = True):
        self.utc_to_cet = utc_to_cet
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        # If input is pandas, transform to numpy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = _convert_pd_to_np(X)

        hours = _slice(X.astype(str), 11, 13).reshape(-1,1)

        return hours.astype(int)


class WeekdayExtractor(BaseEstimator, TransformerMixin):
    """Transforms date column into 1s and 0s 
        if date falls on a weekend or not.
    
    """
        
    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        # If input is pandas, transform to numpy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = _convert_pd_to_np(X)

        datetimes = X.astype('datetime64[D]')
        weekdays = _day_of_week(datetimes)

        return weekdays.reshape(-1,1)
    

class WeekendExtractor(BaseEstimator, TransformerMixin):
    """Transforms date column into 1s and 0s 
        if date falls on a weekend or not.
    
    """

    def fit(self, X, y = None):
        return self

    def transform(self, X, y = None):
        X = X.copy()

        datetimes = X.astype('datetime64[D]')
        weekdays = _day_of_week(datetimes)
        weekend_bin = np.ones(shape=weekdays.shape)
        weekend_bin[weekdays<5] = 0

        return weekend_bin.reshape(-1,1)


class OutlierClipper(BaseEstimator, TransformerMixin):
    """Cap values according to quantiles.

    Args:
        upper_limit: Percentile beyond which to clip the values. Between 0 and 1
        lower_limit: Not yet implemented
    """

    def __init__(self,
                 lower_limit: float = .01,
                 upper_limit: float = .99):

        self.lower_limit = lower_limit
        self.upper_limit = upper_limit

    @staticmethod
    def _replace_values_beyond_thresholds(
        X: np.ndarray,
        thresholds: np.ndarray,
        compare: Callable
    ) -> np.ndarray:

        X = X.copy()

        # If input is pandas, transform to numpy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = _convert_pd_to_np(X)

        mask = compare(X, thresholds)

        tiled_thresholds = np.tile(thresholds, (X.shape[0], 1))

        X[mask] = tiled_thresholds[mask]

        return X

        
    def fit(self, X, y = None):

        self.upper_quantiles = np.nanquantile(X, self.upper_limit, axis=0)
        self.lower_quantiles = np.nanquantile(X, self.lower_limit, axis=0)

        return self

    def transform(self, X, y = None):
        X = X.copy()

        X = self._replace_values_beyond_thresholds(X, self.upper_quantiles, operator.gt)
        X = self._replace_values_beyond_thresholds(X, self.lower_quantiles, operator.lt)

        return X


class RatioCalculator(BaseEstimator, TransformerMixin):
    def fit(self, X, y = None):

        return self

    def transform(self, X, y = None):
        X = X.copy()

        # If input is pandas, transform to numpy array
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = _convert_pd_to_np(X)

        if X.shape[1] != 2:
            raise Exception('Input has to have two columns')
        
        # Replace 0 in denominator before division
        denom = X[:, 1]
        denom[denom==0] = 1e-6

        return (X[:, 0] / denom).reshape(-1,1)




class FringeCategoryBucketer(BaseEstimator, TransformerMixin):
    """Group small categories into a common bucket.

    Useful to apply before One-Hot-Encoding to avoid exploding dimensionalities.
    No longer needed since sklearn's OHE introduced kwarg `max_categories` 
    in version 1.1.

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
