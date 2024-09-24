import inspect

import numpy as np


def get_remaining_indices(size, indices_used):
    """Returns the indices within range(size) which are not contained in indices_used

    Parameters
    ----------
    size : int
        Number of total indices
    indices_used : array-like
        The indices already used

    Returns
    -------
    ndarray,
        Array including all indices not contained in indices_used
    """
    remaining_indices = np.setdiff1d(np.arange(size), indices_used)

    return remaining_indices


def majority_vote(ensemble, X_test):
    """Performs a majority vote on the predictions of an ensemble

    Parameters
    ----------
    ensemble : list
        a list containing the ensemble classifiers
    X_test : ndarray
        2-dimensional array containing the independent test data

    Returns
    -------
    majority_prediction : ndarray
        1-dimensional array containing the majority predictions of the ensemble
    """

    predictions = np.empty((len(ensemble), len(X_test)), dtype=int)

    for i in range(len(ensemble)):
        predictions[i, :] = ensemble[i].predict(X_test)

    majority_prediction = np.apply_along_axis(
        lambda x: np.bincount(x).argmax(), axis=0, arr=predictions
    )

    return majority_prediction


def adjust_predictions(predictions, prediction_classes, actual_classes):
    if len(prediction_classes) != len(actual_classes):
        adjusted_predictions = np.zeros((len(predictions), len(actual_classes)))
        for pred_index, pred_value in enumerate(prediction_classes):
            original_index = actual_classes.tolist().index(pred_value)
            adjusted_predictions[:, original_index] = predictions[:, pred_index]

        predictions = adjusted_predictions

    return predictions


def average_prob_predictions(ensemble, X_test, classes):
    predictions = np.empty((len(ensemble), len(X_test)), dtype=int)

    for i in range(len(ensemble)):
        predictions[i, :] = ensemble[i].predict(X_test)

    bins = max(classes) + 1

    counts_per_class = np.apply_along_axis(
        lambda x: np.bincount(x, minlength=bins), axis=0, arr=predictions
    )
    soft_labels = counts_per_class[classes, :].T / len(ensemble)

    return soft_labels


def propagate_prob_predictions(ensemble, X_test, classes):
    predictions = np.zeros((len(ensemble), len(X_test), len(classes)))
    for i in range(len(ensemble)):
        clf_predictions = ensemble[i].predict_proba(X_test)
        if len(ensemble[i].classes_) != len(classes):
            for pred_index, pred_value in enumerate(ensemble[i].classes_):
                original_index = classes.tolist().index(pred_value)
                predictions[i, :, original_index] = clf_predictions[:, pred_index]
        else:
            predictions[i, :, :] = clf_predictions
    result = np.mean(predictions, axis=0)
    return result


def return_plurality_label(prob_dist, classes=None):
    if classes is None:
        classes = np.arange(prob_dist.shape[1])
    if not isinstance(classes, np.ndarray):
        classes = np.array(classes)

    max_indices = np.argmax(prob_dist, axis=1)
    labels = classes[max_indices]

    return labels


def return_labels_above_threshold(prob_dist, threshold):
    max_values = np.max(prob_dist, axis=1)
    cutoff = np.quantile(max_values, threshold)
    above_threshold = max_values >= cutoff
    indices_above_threshold = np.where(above_threshold)[0]

    return indices_above_threshold


def duplicate_labels(X, prob_dist, classes):
    X_new = np.repeat(X, prob_dist.shape[1], axis=0)
    labels = np.tile(classes, prob_dist.shape[0])
    max_probability = np.reshape(prob_dist, (-1,))

    return X_new, labels, max_probability


def transform_weights(weights, alpha):
    assert weights.ndim == 1

    # Weight samples for selection based on the
    # maximum probability among their soft labels
    new_weights = np.power(weights, alpha)

    # The absolute values of weights affect some algorithms,
    # Therefore scale such that the average weight is 1
    rescaled_weights = new_weights * (new_weights.shape[0] / np.sum(new_weights))

    return rescaled_weights


def transform_weights_per_instance(weights, alpha):
    assert weights.ndim == 2

    # Weight samples for selection
    new_weights = np.power(weights, alpha)
    rescaled_weights = new_weights / np.sum(new_weights, axis=1)[:, None]

    return rescaled_weights


def check_empty_kwargs(kwargs):
    try:
        keys = kwargs.keys()
        assert len(keys) == 0
    except AssertionError:
        msg = "{0}() got an unexpected keyword argument{1} {2}".format(
            inspect.stack()[1][3],  # caller name
            "s" if len(keys) > 1 else "",
            ", ".join(["'{0}'".format(k) for k in keys]),
        )
        raise TypeError(msg)
