import numpy as np

from .abstract_methods import SingleClassifier
from .utils import return_labels_above_threshold, return_plurality_label


class PluralityWeightsClassifier(SingleClassifier):
    """A wrapper method that takes a classifier and soft labeled data.
    It selects the most common label as the final label and uses its probability
    as a weight."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # steps to execute before fitting
    def _preprocessing(self):
        self.labels_ = return_plurality_label(self.prob_dist_, self.classes_)
        self.max_probability_ = np.max(self.prob_dist_, axis=1)
        self._reweight_data()


class ThresholdWeightsClassifier(SingleClassifier):
    """A wrapper method that takes a classifier and soft labeled data.
    It applies plurality voting to the soft labels and then trains a
    classifier on the resulting labels, including only the percentage of
    most confident labels indicated by the threshold parameter.

    Parameters
    ----------
    threshold : float, default=0.5
        The percentage of most confident labels to include in the final
        training set. The threshold must be between 0 and 1.
    """

    def __init__(self, **kwargs):
        self.threshold = kwargs.pop("threshold", 0.5)
        super().__init__(**kwargs)

    # steps to execute before fitting
    def _preprocessing(self):
        self.labels_ = return_plurality_label(self.prob_dist_, self.classes_)
        self.eligible_label_indices_ = return_labels_above_threshold(
            self.prob_dist_, self.threshold
        )
        self.X_ = self.X_[self.eligible_label_indices_]
        self.labels_ = self.labels_[self.eligible_label_indices_]
        self.prob_dist_ = self.prob_dist_[self.eligible_label_indices_]
        self.max_probability_ = np.max(self.prob_dist_, axis=1)
        self._reweight_data()


class DuplicateWeightsClassifier(SingleClassifier):
    """A wrapper method that takes a classifier and soft labeled data.
    It duplicates the instances such that the duplicated set contains an instance
    for every label, with a weight corresponding to the original probability."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    # steps to execute before fitting
    def _preprocessing(self):
        self._duplicate_data()
        self._reweight_data()
