import numpy as np
from joblib import Parallel, delayed
from numpy.random import default_rng
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import SGDClassifier
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

from .utils import (
    adjust_predictions,
    average_prob_predictions,
    check_empty_kwargs,
    duplicate_labels,
    get_remaining_indices,
    majority_vote,
    propagate_prob_predictions,
    transform_weights,
)

rng = default_rng()


class SoftLabelClassifier(ClassifierMixin, BaseEstimator):
    """An abstract class for a wrapper method that trains a single classifier
    on soft-labelled data.

    Parameters
    ----------
    alpha : float, default = 1
        The alpha value used for weighting soft labels.
        This is not currently used in experiments but is included for future use.
        alpha = 0 -> No weighting
        alpha = 1 -> Weights are proportional to the probabilities.
        alpha > 1 -> Higher probabilities receive larger relative weights.
        alpha < 0 -> Smaller probabilities receive larger relative weights.
        Not applicable to all methods.
    base_estimator : classifier object,
        default = sklearn.linear_model.SGDClassifier(loss="modified_huber", n_jobs=1)
        Any classifier, as long as the fit() and predict() functions.
    smoothing : float, default = 0
        The amount of smoothing applied to the soft labels.
        This is not currently used in experiments but is included for future use.
    verbose : boolean, default = True
        Whether or not to print detailed information about the training process.
    estimator_params : dict, parameters that are passed to the base estimator
        The parameters values to set for the base estimator.

    Attributes
    ----------
    fit_estimator_ : classifier
        The classifier trained by the method.
    is_fitted_ : boolean
        Whether the fit method has been called.
    max_probability_ : ndarray, shape (n_samples,)
        The maximum probability for each sample.
    sample_weights_ : ndarray, shape (n_samples,)
        The weights used for each sample.
    X_ : ndarray, shape (n_samples, n_features)
        The data used for training.
    classes_ : ndarray, shape (n_classes,)
        The classes seen at the fit method.
    prob_dist_ : ndarray, shape (n_samples, n_classes)
        The probability distribution for each sample.
    labels_ : ndarray, shape (n_samples,)
        The labels used for training.
    """

    def __init__(self, **kwargs):
        self.alpha = kwargs.pop("alpha", 1)
        self.smoothing = kwargs.pop("smoothing", 0)
        self.base_estimator = kwargs.pop(
            "base_estimator", SGDClassifier(loss="modified_huber", n_jobs=1)
        )
        self.verbose = kwargs.pop("verbose", True)
        self.estimator_params = kwargs.pop("estimator_params", {})
        check_empty_kwargs(kwargs)

    def _duplicate_data(self):
        self.X_, self.labels_, self.max_probability_ = duplicate_labels(
            self.X_, self.prob_dist_, self.classes_
        )

    def _smooth_labels(self):
        if self.smoothing != 0:
            self.prob_dist_ = self.prob_dist_ + self.smoothing
            self.prob_dist_ = self.prob_dist_ / np.sum(self.prob_dist_, axis=1)[:, None]

    # Steps to execute before fitting
    # Threshold, Plurality Vote, Label Adjustment and Duplication
    def _preprocessing(self):
        self.max_probability_ = np.max(self.prob_dist_, axis=1)

    def _prepare_estimator(self):
        estimator = clone(self.base_estimator)
        estimator.set_params(**self.estimator_params)

        return estimator

    def _reweight_data(self):
        self.sample_weights_ = transform_weights(self.max_probability_, self.alpha)

    def fit(self, X, classes, prob_dist):
        """A function for fitting the classifier.

        Parameters
        ----------
        X : array-like
            2-dimensional array containing the independent data used for training
        classes : array-like
            1-dimensional array containing the classes used for training
        prob_dist : array-like
            2-dimensional array containing probabilities corresponding to each of
            the classes from the classes parameter for each instance in X.

        Returns
        -------
        self : object
            Returns fitted ensemble object.
        """

        # Check that X and y have correct shape
        X, _ = check_X_y(X, prob_dist[:, 0])

        # Store the data seen during fit
        self.X_ = X
        self.classes_ = unique_labels(classes)
        self.prob_dist_ = prob_dist

        # Apply smoothing if applicable
        self._smooth_labels()

        # Apply the necessary preprocessing steps
        self._preprocessing()

    def predict(self, X):
        """A function for predicting using the trained classifier.

        Parameters
        ----------
        X : array-like
            2-dimensional array containing the independent data used for training

        Returns
        -------
        y : array-like (n_samples,)
            The estimated label for each sample
        """

        # Check is fit had been called
        check_is_fitted(self, "is_fitted_")

        # Input validation
        X = check_array(X)

        return X

    def predict_proba(self, X):
        pass


class SingleClassifier(SoftLabelClassifier):
    """A base class for a wrapper method that trains a single classifier
    on soft label data.

    Parameters
    ----------
    weighted_fit : boolean, default = True
        Whether or not to use the sample weights during the fit.

    As well as all the parameters from the SoftLabelClassifier class.

    Attributes
    ----------
    All of the attributes from the SoftLabelClassifier class.
    """

    def __init__(self, **kwargs):
        self.weighted_fit = kwargs.pop("weighted_fit", True)
        super().__init__(**kwargs)

    def _fit_estimator(self):
        estimator = self._prepare_estimator()

        if self.weighted_fit:
            estimator.fit(self.X_, self.labels_, sample_weight=self.sample_weights_)
        else:
            estimator.fit(self.X_, self.labels_)

        self.fit_estimator_ = estimator

    def fit(self, X, classes, prob_dist):
        super().fit(X, classes, prob_dist)

        # Starting the fit
        self._fit_estimator()
        self.is_fitted_ = True

        # Return the classifier
        return self

    def predict(self, X):
        X = super().predict(X)
        y = self.fit_estimator_.predict(X)

        return y

    def predict_proba(self, X):
        X = super().predict(X)

        y = adjust_predictions(
            self.fit_estimator_.predict_proba(X),
            self.fit_estimator_.classes_,
            self.classes_,
        )

        return y


class EnsembleClassifier(SoftLabelClassifier):
    """A base class for ensemble wrapper methods that are trained directly
    from datasets with soft labels.

    Parameters
    ----------
    ensemble_size : int, default = 25
        The number of members the ensemble consists of.
    n_jobs : int, default = 1
        The number of parallel jobs used to train the different ensemble members.
    replace: boolean, default = True
        Whether or not to sample with replacement.

    As well as all the parameters from the SoftLabelClassifier class.

    Attributes
    ----------
    fit_ensemble_ : list (classifiers)
        A list containing the fit ensemble members.

    as well as all the attributes from the SoftLabelClassifier class.
    """

    def __init__(self, **kwargs):
        self.ensemble_size = kwargs.pop("ensemble_size", 25)
        self.n_jobs = kwargs.pop("n_jobs", 1)
        self.replace = kwargs.pop("replace,", True)
        super().__init__(**kwargs)

    # Private method for sampling the data,
    def _sample_data(self, X, n_samples, sample_weights, weighted_sampling):
        if weighted_sampling:
            # Normalize the weights
            probabilities = sample_weights / np.sum(sample_weights)
        else:
            probabilities = None

        # Sample based on the maximum confidence
        indices = rng.choice(
            range(X.shape[0]),
            n_samples,
            replace=self.replace,
            p=probabilities,
        ).tolist()

        return indices

    def _first_sample(self):
        labeled_indices_used = self._sample_data(
            self.X_,
            self.labeled_size_,
            self.sample_weights_,
            self.weighted_first_sampling,
        )
        return labeled_indices_used

    def _second_sample(self, labeled_indices_used):
        X_used = self.X_[labeled_indices_used, :]
        y_used = self.labels_[labeled_indices_used]
        weights_used = self.sample_weights_[labeled_indices_used]
        return X_used, y_used, weights_used

    # Private method to train the ensemble classifier in parallel,
    # using the confidence sampling method
    def _train_ensemble_parallel(self):
        """Private method for training a base classifier using confidence sampling

        Parameters
        ----------
        self : ConfidenceSamplingClassifier object
            Contains all of the relevant settings used in the training procedure
        X : array-like
            2-dimensional array containing the independent data used for training
        prob_dist : array-like
            2-dimensional array containing probabilities corresponding to each of
            the classes from the classes parameter for each instance in X.

        Returns
        -------
        ConfidenceSamplingClassifier object
            A classifier trained by using confidence sampling
        """

        estimator = clone(self.base_estimator)
        estimator.set_params(**self.estimator_params)

        # Selecting the labeled data used for training
        labeled_indices_used = self._first_sample()

        # Calculating the indices of the out-of-bag data for used for error-checking
        labeled_indices_unused = get_remaining_indices(
            self.X_.shape[0], labeled_indices_used
        )

        # Obtaining the labels used by sampling
        X_used, y_used, weights_used = self._second_sample(labeled_indices_used)

        # Applying the selected base classifier to the sampled data
        if self.weighted_fit:
            fitted_estimator = estimator.fit(X_used, y_used, sample_weight=weights_used)
        else:
            fitted_estimator = estimator.fit(X_used, y_used)

        return fitted_estimator

    def fit(self, X, classes, prob_dist):
        super().fit(X, classes, prob_dist)
        self.labeled_size_ = X.shape[0]

        # Starting the fit
        results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._train_ensemble_parallel)() for _ in range(self.ensemble_size)
        )

        self.fit_ensemble_ = results
        self.is_fitted_ = True

        # Return the ensemble classifier
        return self

    def predict(self, X):
        """A function for making a prediction using the ensemble.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        result : array-like, shape (n_samples,)
            The estimated label for each sample
        """
        X = super().predict(X)

        # Actual prediction
        result = majority_vote(self.fit_ensemble_, X)

        return result

    def predict_proba(self, X, method="average"):
        # Method is either average or prob_propagation
        X = super().predict(X)

        # Actual prediction
        if method == "average":
            result = average_prob_predictions(self.fit_ensemble_, X, self.classes_)
        elif method == "prob_propagation":
            result = propagate_prob_predictions(self.fit_ensemble_, X, self.classes_)
        else:
            raise ValueError("Invalid method parameter")

        return result
