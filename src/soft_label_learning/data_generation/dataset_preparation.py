from functools import partial

import numpy as np
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestClassifier
from synlabel import DiscreteObservedDataset
from synlabel.utils.helper_functions import (
    apply_transition_matrix,
    generate_random_transition_matrix,
    generate_uniform_noise_matrix,
    rescale_transition_matrix,
)

from ..config import path_uci_keel_data
from .calibration import (
    apply_noise_model_stochastically,
    calibration_func,
    max_transform_soft_labels,
)
from .data_extraction import read_data


def get_feature_ranking(coefficients):
    return np.argsort(np.abs(coefficients))[::-1]


def construct_ground_truth(data, clf):
    data = data.copy()
    X = data[:, :-1]
    y = data[:, -1].astype(int)

    # Shuffle
    shuffled_indices = np.random.permutation(len(y))
    X = X[shuffled_indices].copy()
    y = y[shuffled_indices].copy()

    # Initialize a Discrete Observed Dataset
    D_OH = DiscreteObservedDataset(X=X, y=y)

    # Fit a deterministic function
    clf.fit(X, y)

    # Construct the Ground Truth dataset
    D_G = D_OH.to_ground_truth("function_based", function=clf)

    return D_G, X, y


def generate_test_data(
    dataset="keel_contraceptive.csv",
    clf=RandomForestClassifier(),
    n_features_to_hide=1,
    n_samples=100,
    method="multivariate_kde_scipy",
    reverse_feature_order=False,
    model_importance=True,
    verbose=False,
):
    """Generates a D_G and D_PG dataset for testing purposes based on
    another, provided dataset.

    Returns
    -------
    D_G, D_PG : GroundTruthDataset, PartialGroundTruthDataset
    """

    if isinstance(dataset, str):
        data = read_data(dataset, read_path=path_uci_keel_data)
    else:
        data = dataset

    D_G, X, y = construct_ground_truth(data, clf)

    if verbose:
        # Measure disagreement between new and old labels
        difference = (y != D_G.y).sum()
        print(f"Difference GT - original: {difference}")
        print(f"% difference: {(100*difference / len(y)).round(2)}")

    if model_importance:
        if hasattr(D_G.func, "feature_importances_"):
            # in case of a tree based method
            importances = D_G.func.feature_importances_
        elif hasattr(D_G.func, "coef_"):
            importances = D_G.func.coef_[0]
        else:
            raise ValueError("Model does not have feature importances or coefficients")
    else:
        # Calculate the correlation between the features in X and the outcome y
        # Should only be used for continuous features / outcomes
        importances = [pearsonr(X[:, i], y)[0] for i in range(X.shape[1])]

    if verbose:
        print(importances)

    # From strongest to weakest correlation
    feature_ranking = get_feature_ranking(importances)

    if verbose:
        print(feature_ranking)

    if reverse_feature_order:
        features_to_hide = feature_ranking[-n_features_to_hide:].astype(int).tolist()
    else:
        features_to_hide = feature_ranking[:n_features_to_hide].astype(int).tolist()

    if verbose:
        print(features_to_hide)

    D_PG = D_G.to_partial_ground_truth(
        "feature_hiding",
        features_to_hide=features_to_hide,
        samples_per_instance=n_samples,
        sampling_method=method,
    )

    return D_G, D_PG


def get_ncar_function(noise_level, n_classes):
    uniform_base = generate_uniform_noise_matrix(n_classes, 0.01)
    uniform = rescale_transition_matrix(uniform_base, noise_level)
    atm_uni = partial(apply_transition_matrix, matrix=uniform)
    transform_uni = partial(apply_noise_model_stochastically, noise_model=atm_uni)

    return transform_uni


def get_nar_function(noise_level, n_classes):
    random_base = generate_random_transition_matrix(n_classes, 0.01)
    random = rescale_transition_matrix(random_base, noise_level)
    atm_random = partial(apply_transition_matrix, matrix=random)
    transform_random = partial(apply_noise_model_stochastically, noise_model=atm_random)

    return transform_random


def get_calibration_noise_function(noise_level, extremity):
    calibration_func_partial = partial(
        calibration_func, a=noise_level, extremity=extremity
    )
    tsl = partial(max_transform_soft_labels, calibration_func=calibration_func_partial)
    transform_miscalibrated = partial(apply_noise_model_stochastically, noise_model=tsl)

    return transform_miscalibrated
