import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels


# See link for how to use sklearn BaseEstimator
# https://github.com/scikit-learn-contrib/project-template/blob/master/skltemplate/_template.py

class BaselineClassifier(BaseEstimator, ClassifierMixin):
    """Assigns labels based on upper limit threshold values.

    Args:
        upper_limits (dict): Keys are classes, values represent
            numeric threshold up until which the correspoding
            class is assigned.
            For the highest class no threshold must be defined.
            Highest class is assigned for values greater than the
            highest threshold.

    """

    def __init__(self, col_index: int = 0, upper_limits: dict = None):
        self.col_index = col_index
        self.upper_limits = upper_limits

    def fit(self, X, y):
        """Fetches open category (higher than highest threshold)
        and prepares descendingly ordered array of threshold values.

        X is not actually used during fit call, but is passed regardless
        to comply with scikit-learn's API.
        """

        # Store the classes seen during fit
        self.classes_ = unique_labels(y)
        # Store indices in dictionary
        self.class_indices_ = {
            class_: index for index, class_ in enumerate(self.classes_)
        }

        # Upper limits contains 3G and 4G.
        # Values above 4G upper limit should be 5G.
        # Extract open_category (=5G) by comparing
        # keys in upper_limits and all available categories
        # in y.
        covered_categories = set(self.upper_limits.keys())
        open_category = set(self.classes_) - covered_categories
        self.open_category_ = list(open_category)[0]

        # Create series from upper_limits and order
        # descending (highest on top).
        class_keys = [
            self.class_indices_[class_] for class_ in self.upper_limits.keys()
        ]
        class_thresholds = list(self.upper_limits.values())
        limits = np.column_stack([class_keys, class_thresholds])
        # Order array descending by column w index 1
        self.ordered_limits_ = limits[limits[:, 1].argsort()][::-1]

        # Return the classifier
        return self

    def predict(self, X):
        # Check if fit had been called
        check_is_fitted(self)

        # Transform DataFrame to np ndarray
        if isinstance(X, pd.DataFrame):
            X = X.values

        # Only one column is used to compare
        # against upper_limits
        X = X[:, self.col_index].reshape(-1, 1)

        # Input validation
        X = check_array(X)

        # Prepare output array in length of X
        y_pred = np.zeros(shape=(X.shape[0], 1))

        # Values above highest limit are assigned the open
        # category (-> 5G)
        open_cat_mask = X >= self.ordered_limits_[0, 1]
        y_pred[open_cat_mask] = self.class_indices_[self.open_category_]

        # Then, iterate through the ordered limits and
        # assign the label if X is smaller than the upper_limit
        for label, limit in self.ordered_limits_:
            y_pred[X < limit] = label

        # Transform indices back to labels
        inv_map = {v: k for k, v in self.class_indices_.items()}
        y_pred = np.array(list(map(lambda x: inv_map[x], y_pred[:, 0])))

        return y_pred
