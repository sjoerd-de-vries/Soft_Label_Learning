import numpy as np
from numpy.random import default_rng

from .abstract_methods import EnsembleClassifier
from .utils import (
    duplicate_labels,
    return_labels_above_threshold,
    return_plurality_label,
    transform_weights,
    transform_weights_per_instance,
)

rng = default_rng()


class PluralityEnsembleClassifier(EnsembleClassifier):
    """A wrapper method that takes a classifier and soft labelled data.
    It selects the most likely label for each instance, constructs an
    ensemble by sampling, possibly weighted, and trains its members on the
    plurality labels, possibly weighted.

    Parameters
    ----------
    weighted_first_sampling : bool, default=True
        Whether to sample the instances for the ensemble members
        with replacement, weighted by the maximum probability of the
        soft labels.
    weighted_fit : bool, default=False
        Whether to fit the ensemble members with weights, based on the
        maximum probability of the soft labels.
    """

    def __init__(self, **kwargs):
        self.weighted_first_sampling = kwargs.pop("weighted_first_sampling", True)
        self.weighted_fit = kwargs.pop("weighted_fit", False)

        super().__init__(**kwargs)

    # steps to execute before fitting
    def _preprocessing(self):
        self.labels_ = return_plurality_label(self.prob_dist_, self.classes_)
        self.max_probability_ = np.max(self.prob_dist_, axis=1)
        self._reweight_data()


class ThresholdEnsembleClassifier(EnsembleClassifier):
    """A wrapper method that takes a classifier and soft labeled data.
    It selects the most likely label for each instance, and check whether it is in the
    fraction of most confident instances, specified by the threshold parameter.
    If so, it constructs an ensemble by sampling, possibly weighted,
    and trains its members on the plurality labels, possibly weighted.

    Parameters
    ----------
    threshold : float, default=0.5
        The threshold that corresponds to the fraction of the most confident instances
        to consider during training.
    weighted_first_sampling : bool, default=True
        Whether to sample the instances for the ensemble members
        with replacement, weighted by the maximum probability of the
        soft labels.
    weighted_fit : bool, default=False
        Whether to fit the ensemble members with weights, based on the
        maximum probability of the soft labels.

    """

    def __init__(self, **kwargs):
        self.threshold = kwargs.pop("threshold", 0.5)
        self.weighted_first_sampling = kwargs.pop("weighted_first_sampling", True)
        self.weighted_fit = kwargs.pop("weighted_fit", False)

        super().__init__(**kwargs)

    # steps to execute before sampling
    def _preprocessing(self):
        self.labels_ = return_plurality_label(self.prob_dist_, self.classes_)
        self.eligible_label_indices_ = return_labels_above_threshold(
            self.prob_dist_, self.threshold
        )
        self.X_ = self.X_[self.eligible_label_indices_]
        self.labels_ = self.labels_[self.eligible_label_indices_]
        self.prob_dist_ = self.prob_dist_[self.eligible_label_indices_]
        self.labeled_size_ = self.X_.shape[0]
        self.max_probability_ = np.max(self.prob_dist_, axis=1)
        self._reweight_data()


class DuplicateEnsembleClassifier(EnsembleClassifier):
    """A wrapper method that takes a classifier and data with soft labels.
    It first duplicates all of the data, and then generates an ensemble via
    sampling, using the probability of each label as a weight.

    Parameters
    ----------
    weighted_first_sampling : bool, default=True
        Whether to sample the instances for the ensemble members
        with replacement, weighted by the maximum probability of the
        soft labels.
    weighted_fit : bool, default=False
        Whether to fit the ensemble members with weights, based on the
        maximum probability of the soft labels.
    """

    def __init__(self, **kwargs):
        self.weighted_first_sampling = kwargs.pop("weighted_first_sampling", True)

        self.weighted_fit = False
        super().__init__(**kwargs)

    # steps to execute before fitting
    def _preprocessing(self):
        self._duplicate_data()
        self._reweight_data()


# Abstract class
class EnsembleDuplicateClassifier(EnsembleClassifier):
    """An abstract class, to be used by EnsembleDuplicateWeightsClassifier and
    EnsembleDuplicateSamplingClassifier."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        if self.alpha < 0:
            if self.reweighted_second_step:
                raise ValueError("alpha < 0 can cause division by 0 errors.")

    # Override as first we sample on the original data
    def _duplicate_data(self):
        (
            self.X_duplicated_,
            self.labels_duplicated_,
            self.max_probability_duplicated_,
        ) = duplicate_labels(self.X_, self.prob_dist_, self.classes_)

    def _preprocessing(self):
        self._duplicate_data()
        self.labels_ = return_plurality_label(self.prob_dist_, self.classes_)
        self.max_probability_ = np.max(self.prob_dist_, axis=1)
        self._reweight_data()

    # For duplicated data, need to multiply the labeled_indices_used
    # by the number of classes
    def _second_sample(self, labeled_indices_used):
        multiplied_labeled_indices_used = len(self.classes_) * np.array(
            labeled_indices_used
        )

        duplicated_labeled_indices_used = np.repeat(
            multiplied_labeled_indices_used, len(self.classes_)
        )

        for i in range(1, len(self.classes_)):
            duplicated_labeled_indices_used[i :: len(self.classes_)] += i

        X_used = self.X_duplicated_[duplicated_labeled_indices_used, :]
        y_used = self.labels_duplicated_[duplicated_labeled_indices_used]
        weights_used_temp = self.max_probability_duplicated_[
            duplicated_labeled_indices_used
        ]

        if self.reweighted_second_step:
            weights_used = transform_weights(weights_used_temp, self.alpha)
        else:
            weights_used = weights_used_temp

        return X_used, y_used, weights_used


class EnsembleDuplicateWeightsClassifier(EnsembleDuplicateClassifier):
    """A wrapper method that takes a classifier and data with soft labels.
    It generated an ensemble via sampling, after which all selected instances
    are duplicated. The duplicated instances are then weighted during fitting using
    their soft label as a weight.

    Parameters
    ----------
    weighted_first_sampling : bool, default=True
        Whether to sample the instances for the ensemble members
        with replacement, weighted by the maximum probability of the
        soft labels.
    reweighted_second_step : bool, default=True
        Whether to reweight the samples in the fitting step.
    """

    def __init__(self, **kwargs):
        self.weighted_first_sampling = kwargs.pop("weighted_first_sampling", True)
        self.reweighted_second_step = kwargs.pop("reweighted_second_step", True)

        self.weighted_fit = True
        super().__init__(**kwargs)


class EnsembleDuplicateSamplingClassifier(EnsembleDuplicateClassifier):
    """A wrapper method that takes a classifier and data with soft labels.
    It generated an ensemble via sampling, after which all selected instances
    are duplicated. The duplicated instances are then sampled again using
    their soft label as a weight.

    Parameters
    ----------
    weighted_first_sampling : bool, default=True
        Whether to sample the instances for the ensemble members
        with replacement, weighted by the maximum probability of the
        soft labels.
    reweighted_second_step : bool, default=True
        Whether to reweight the samples in the second sampling step.
    """

    def __init__(self, **kwargs):
        self.weighted_first_sampling = kwargs.pop("weighted_first_sampling", True)
        self.reweighted_second_step = kwargs.pop("reweighted_second_step", True)

        self.weighted_fit = False
        super().__init__(**kwargs)

    # For duplicated data, need to multiply the labeled_indices_used
    # by the number of classes
    def _second_sample(self, labeled_indices_used):
        X_temp, y_temp, weights_temp = super()._second_sample(labeled_indices_used)

        indices_used = self._sample_data(X_temp, self.labeled_size_, weights_temp, True)
        X_used, y_used, weights_used = (
            X_temp[indices_used],
            y_temp[indices_used],
            weights_temp[indices_used],
        )

        return X_used, y_used, weights_used


class EnsembleSamplingClassifier(EnsembleClassifier):
    """A wrapper method that takes a classifier and data with soft labels.
    It first samples the instances to train each ensemble member on,
    after which the label that is assigned to said instance is sampled from
    the original soft label.

    Effectively the same as a data duplication strategy.

    Parameters
    ----------
    weighted_first_sampling : bool, default=True
        Whether to sample the instances for the ensemble members
        with replacement, weighted by the maximum probability of the
        soft labels.
    reweighted_second_step : bool, default=True
        Whether to reweight the samples in the second sampling step.
    weighted_fit : bool, default=False
        Whether to fit the ensemble members with weights, based on the
        maximum probability of the soft labels.
    """

    def __init__(self, **kwargs):
        self.weighted_first_sampling = kwargs.pop("weighted_first_sampling", True)
        self.reweighted_second_step = kwargs.pop("reweighted_second_step", True)

        self.weighted_fit = False
        super().__init__(**kwargs)

        if self.alpha < 0:
            if self.reweighted_second_step:
                raise ValueError("alpha < 0 can cause division by 0 errors.")

    # steps to execute before fitting
    def _preprocessing(self):
        self.max_probability_ = np.max(self.prob_dist_, axis=1)
        self._reweight_data()

    def _second_sample(self, labeled_indices_used):
        label_distributions = self.prob_dist_[labeled_indices_used, :]
        labels_used = np.full(label_distributions.shape[0], -1)

        if self.reweighted_second_step:
            weights_used = transform_weights_per_instance(
                label_distributions, self.alpha
            )
        else:
            weights_used = label_distributions

        cumulative_prob = weights_used.cumsum(axis=1)
        random_numbers = np.random.rand(len(weights_used), 1)
        choices = (random_numbers < cumulative_prob).argmax(axis=1)
        labels_used = self.classes_[choices]

        X_used = self.X_[labeled_indices_used, :]
        y_used = labels_used

        return X_used, y_used, weights_used
