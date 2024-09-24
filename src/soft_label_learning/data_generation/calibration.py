import numpy as np


def calibration_func(x, a, extremity):
    scale = 2 if extremity else 1
    return x + a / scale * np.sin(x * scale * np.pi)


def max_normalize_soft_label(
    old_labels, new_max_labels, old_max_labels, max_label_indices
):
    # Normalize the soft labels such that the new value for the maximum label
    # is preserved

    # Calculate the sum of the values that were not adjusted
    normalization_const = old_labels.sum(axis=1)
    delta_max_labels = old_max_labels - new_max_labels

    zero_norm = normalization_const == 0

    # pre-normalize soft labels for which the nonmax labels are nonzero
    new_labels = old_labels.copy()
    multiplier = (
        normalization_const[~zero_norm] + delta_max_labels[~zero_norm]
    ) / normalization_const[~zero_norm]
    new_labels[~zero_norm] = new_labels[~zero_norm] * multiplier[:, np.newaxis]

    # pre-normalize soft labels for which the nonmax labels are zero
    new_values = delta_max_labels[zero_norm] / (old_labels.shape[1] - 1)
    new_values_2d = np.repeat(new_values[:, np.newaxis], old_labels.shape[1], axis=1)
    new_labels[zero_norm] = new_values_2d.round(10)

    # Insert the new maximum labels into the pre-normalized soft labels
    indices = np.arange(old_labels.shape[0])
    new_labels[indices, max_label_indices] = new_max_labels

    return new_labels


def max_transform_soft_labels(soft_labels, calibration_func):
    old_labels = np.array(soft_labels)
    max_label_indices = np.argmax(old_labels, axis=1)
    indices = np.arange(old_labels.shape[0])
    max_labels = old_labels[indices, max_label_indices]

    # access all maximum values in the matrix old labels
    old_labels[indices, max_label_indices] = 0

    # apply the calibration function to all max labels
    new_max_labels = calibration_func(max_labels)

    # normalize the new labels, so that the sum of all labels is 1
    new_labels = max_normalize_soft_label(
        old_labels, new_max_labels, max_labels, max_label_indices
    )

    return new_labels


def apply_noise_model_stochastically(x, noise_model, sigma=0.5):
    """This function first draws a scaling parameter from a normal distribution
    centred around 1, and then uses it to determining the strength of the noise
    model application for a single soft label to simulate the randomness caused by
    sampling when constructing a single dataset.

    Should be instantiated as a partial function to be used with SYNLABEL

    Parameters
    ----------
    x : array-like
        the soft labels
    noise_model : function
        function that can be applied directly to the soft labels
    sigma : float, optional
        the standard deviation of the noise, by default 0.5

    Returns
    -------
    soft_labels: array-like
        The soft labels with all noise applied to them
    """
    scaling_parameter = np.random.normal(1, sigma, x.shape[0])
    inverted_scaling_parameter = 1 - scaling_parameter
    original_soft_labels = x
    noisy_soft_labels = noise_model(x)

    transformed_soft_labels = np.multiply(
        original_soft_labels, inverted_scaling_parameter[:, np.newaxis]
    ) + np.multiply(noisy_soft_labels, scaling_parameter[:, np.newaxis])

    # Clip values <0, >1, and renormalize so they sum to 1
    clipped_soft_labels = np.clip(transformed_soft_labels, 0, 1)
    normalized_soft_labels = (
        clipped_soft_labels
        / np.linalg.norm(clipped_soft_labels, 1, axis=1)[:, np.newaxis]
    )

    return normalized_soft_labels
