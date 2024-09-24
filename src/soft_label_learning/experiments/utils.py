from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from ..methods.ensemble_classifiers import (
    DuplicateEnsembleClassifier,
    EnsembleDuplicateSamplingClassifier,
    EnsembleDuplicateWeightsClassifier,
    EnsembleSamplingClassifier,
    PluralityEnsembleClassifier,
    ThresholdEnsembleClassifier,
)
from ..methods.single_classifiers import (
    DuplicateWeightsClassifier,
    PluralityWeightsClassifier,
    ThresholdWeightsClassifier,
)


def get_classifier(clf_name, dataset):
    large_sets = [
        "uci_mice.csv",
        "uci_nursery.csv",
        "keel_spambase.csv",
        "uci_abalone.csv",
        "uci_madelon.csv",
        "uci_red.csv",
        "uci_white.csv",
    ]

    if clf_name == "SGD":
        if dataset in large_sets:
            # set early stopping to true, or fitting will take very long
            clf = SGDClassifier(loss="modified_huber", n_jobs=1, early_stopping=True)
        else:
            clf = SGDClassifier(loss="modified_huber", n_jobs=1)
    elif clf_name == "LR":
        clf = LogisticRegression(n_jobs=1)
    elif clf_name == "GNB":
        clf = GaussianNB()
    elif clf_name == "DT":
        clf = DecisionTreeClassifier(max_features="sqrt")
    else:
        raise ValueError("Classifier not recognized")

    return clf


def get_method(method_name):
    method_map = {
        "PluralityClf": (PluralityWeightsClassifier, {"weighted_fit": False}),
        "PluralityWeightsClf": (PluralityWeightsClassifier, {}),
        "DuplicateWeightsClf": (DuplicateWeightsClassifier, {}),
        "PluralityBootstrapClf": (
            PluralityEnsembleClassifier,
            {"weighted_first_sampling": False},
        ),
        "PluralityBootstrapWFClf": (
            PluralityEnsembleClassifier,
            {"weighted_first_sampling": False, "weighted_fit": True},
        ),
        "PluralityEnsClf": (PluralityEnsembleClassifier, {}),
        "BootstrapSamplingClf": (
            EnsembleSamplingClassifier,
            {"weighted_first_sampling": False},
        ),
        "EnsSamplingClf": (EnsembleSamplingClassifier, {}),
        "EnsSamplingNRClf": (
            EnsembleSamplingClassifier,
            {"reweighted_second_step": False},
        ),
        "DupEnsClf": (DuplicateEnsembleClassifier, {}),
        "BootstrapDupWeightsClf": (
            EnsembleDuplicateWeightsClassifier,
            {"weighted_first_sampling": False},
        ),
        "EnsDupWeightsClf": (EnsembleDuplicateWeightsClassifier, {}),
        "EnsDupWeightsNRClf": (
            EnsembleDuplicateWeightsClassifier,
            {"reweighted_second_step": False},
        ),
        "BootstrapDupSamplingClf": (
            EnsembleDuplicateSamplingClassifier,
            {"weighted_first_sampling": False},
        ),
        "EnsDupSamplingClf": (EnsembleDuplicateSamplingClassifier, {}),
        "EnsDupSamplingNRClf": (
            EnsembleDuplicateSamplingClassifier,
            {"reweighted_second_step": False},
        ),
        # "base_clf_pv": ("base_clf_pv", {}),
        "SampleClf": ("SampleClf", {}),
    }

    if "Threshold" in method_name:
        return get_threshold_method(method_name)

    if method_name not in method_map:
        raise ValueError(f"Invalid method name: {method_name}")

    return method_map[method_name]


def get_threshold_method(method_name):
    method, threshold_str = method_name.split("_")
    threshold = float(threshold_str) / 100

    method_map = {
        "ThresholdClf": (
            ThresholdWeightsClassifier,
            {"threshold": threshold, "weighted_fit": False},
        ),
        "ThresholdWeightsClf": (ThresholdWeightsClassifier, {"threshold": threshold}),
        "ThresholdBootstrapClf": (
            ThresholdEnsembleClassifier,
            {"threshold": threshold, "weighted_first_sampling": False},
        ),
        "ThresholdBootstrapWFClf": (
            ThresholdEnsembleClassifier,
            {
                "threshold": threshold,
                "weighted_first_sampling": False,
                "weighted_fit": True,
            },
        ),
        "ThresholdEnsClf": (ThresholdEnsembleClassifier, {"threshold": threshold}),
    }

    if method not in method_map:
        raise ValueError(f"Invalid method name: {method_name}")

    return method_map[method]
