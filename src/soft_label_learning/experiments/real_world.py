import datetime
import json
import logging

import numpy as np
from joblib import Parallel, delayed
from sklearn import preprocessing
from sklearn.base import clone
from sklearn.metrics import accuracy_score, roc_auc_score

from ..config import path_output
from ..methods.ensemble_classifiers import EnsembleClassifier
from ..methods.utils import adjust_predictions, return_plurality_label
from ..methods.utils import return_plurality_label as pv
from .metrics import mean_total_variation_distance as tvd
from .utils import get_classifier, get_method


# Evaluate the predictions of the model
def evaluate_predictions(
    y_pred_train_hard,
    y_pred_test_hard,
    y_pred_train_soft,
    y_pred_test_soft,
    y_pred_train_soft_prop,
    y_pred_test_soft_prop,
    y_OH_pv_train,
    y_OH_pv_test,
    y_OH_s_train,
    y_OH_s_test,
    y_OS_train,
    y_OS_test,
    classes,
):
    metric_dict = {}

    ## Accuracy: tested on = [pv, sampled]
    # tested on pv
    test_set = y_OH_pv_test
    metric_dict["accuracy-test-pv-pv"] = accuracy_score(test_set, y_pred_test_hard)
    metric_dict["accuracy-test-soft-pv"] = accuracy_score(
        test_set, pv(y_pred_test_soft, classes)
    )

    if y_pred_train_soft_prop is None:
        metric_dict["accuracy-test-soft_prop-pv"] = metric_dict["accuracy-test-soft-pv"]
    else:
        metric_dict["accuracy-test-soft_prop-pv"] = accuracy_score(
            test_set, pv(y_pred_test_soft_prop, classes)
        )

    # tested on sampled
    test_set = y_OH_s_test
    metric_dict["accuracy-test-pv-samp"] = accuracy_score(test_set, y_pred_test_hard)
    metric_dict["accuracy-test-soft-samp"] = accuracy_score(
        test_set, pv(y_pred_test_soft, classes)
    )

    if y_pred_train_soft_prop is None:
        metric_dict["accuracy-test-soft_prop-samp"] = metric_dict[
            "accuracy-test-soft-samp"
        ]
    else:
        metric_dict["accuracy-test-soft_prop-samp"] = accuracy_score(
            test_set, pv(y_pred_test_soft_prop, classes)
        )

    ## AUC: pred type = [soft, soft_prop], tested on = [pv, samp]
    # tested on pv
    test_set = y_OH_pv_test
    metric_dict["auc-test-soft-pv"] = roc_auc_score(test_set, y_pred_test_soft[:, 1])

    if y_pred_test_soft_prop is None:
        metric_dict["auc-test-soft_prop-pv"] = metric_dict["auc-test-soft-pv"]
    else:
        metric_dict["auc-test-soft_prop-pv"] = roc_auc_score(
            test_set, y_pred_test_soft_prop[:, 1]
        )

    # tested on sampled
    test_set = y_OH_s_test
    metric_dict["auc-test-soft-samp"] = roc_auc_score(test_set, y_pred_test_soft[:, 1])

    if y_pred_test_soft_prop is None:
        metric_dict["auc-test-soft_prop-samp"] = metric_dict["auc-test-soft-samp"]
    else:
        metric_dict["auc-test-soft_prop-samp"] = roc_auc_score(
            test_set, y_pred_test_soft_prop[:, 1]
        )

    ## TVD: pred type = [soft, soft_prop], tested on = [soft]
    test_set = y_OS_test
    metric_dict["tvd-test-soft-soft"] = tvd(test_set, y_pred_test_soft)

    if y_pred_test_soft_prop is None:
        metric_dict["tvd-test-soft_prop-soft"] = metric_dict["tvd-test-soft-soft"]
    else:
        metric_dict["tvd-test-soft_prop-soft"] = tvd(test_set, y_pred_test_soft_prop)

    return metric_dict


# Main experiment loop
# This method is called in parallel for each repetition
# of the experiment
def inner_experiment(
    X,
    y_OS,
    base_clf,
    learning_method,
    kwargs,
    alpha,
    rep_counter,
    train_percentage=0.8,
):
    rng = np.random.default_rng(seed=rep_counter)
    classes = np.array([0, 1])

    # Transforming the soft labels to hard labels
    y_OH_pv = return_plurality_label(y_OS)
    cumulative_prob = y_OS.cumsum(axis=1)
    random_numbers = np.random.rand(len(y_OS), 1)
    choices = (random_numbers < cumulative_prob).argmax(axis=1)
    y_OH_s = classes[choices]

    # Preparing the train-test split
    n_samples = X.shape[0]
    n_train = int(n_samples * train_percentage)

    # Obtaining the indices
    two_classes = False
    minval = 9

    # Ensure the samples contains enough data for both classes
    while not two_classes:
        train_indices = rng.choice(n_samples, n_train, replace=False)
        test_indices = np.setdiff1d(np.arange(n_samples), train_indices)
        if (
            np.unique(y_OH_pv[train_indices], return_counts=True)[1].min() >= minval
        ) and (np.unique(y_OH_s[train_indices], return_counts=True)[1].min() >= minval):
            two_classes = True

    scaler = preprocessing.StandardScaler()
    scaler.fit(X.iloc[train_indices])

    X_train_transformed = scaler.transform(X.iloc[train_indices])
    X_test_transformed = scaler.transform(X.iloc[test_indices])

    # Fit the methods
    if isinstance(learning_method, str):
        clf = clone(base_clf)
        if learning_method == "SampleClf":
            clf.fit(X_train_transformed, y_OH_s[train_indices])
        else:
            raise ValueError(f"Invalid learning method: {learning_method}")

    else:
        smoothing = 0.0

        clf = learning_method(
            base_estimator=clone(base_clf),
            alpha=alpha,
            smoothing=smoothing,
            **kwargs,
        )
        clf.fit(X_train_transformed, classes, y_OS[train_indices])

    # Predict
    y_pred_train_hard = clf.predict(X_train_transformed)
    y_pred_test_hard = clf.predict(X_test_transformed)

    y_pred_train_soft = adjust_predictions(
        clf.predict_proba(X_train_transformed),
        clf.classes_,
        classes,
    )
    y_pred_test_soft = adjust_predictions(
        clf.predict_proba(X_test_transformed),
        clf.classes_,
        classes,
    )

    y_pred_train_soft_prop = None
    y_pred_test_soft_prop = None

    if not isinstance(learning_method, str) and issubclass(
        learning_method, EnsembleClassifier
    ):
        y_pred_train_soft_prop = adjust_predictions(
            clf.predict_proba(X_train_transformed, method="prob_propagation"),
            clf.classes_,
            classes,
        )
        y_pred_test_soft_prop = adjust_predictions(
            clf.predict_proba(X_test_transformed, method="prob_propagation"),
            clf.classes_,
            classes,
        )

    metric_dict = evaluate_predictions(
        y_pred_train_hard,
        y_pred_test_hard,
        y_pred_train_soft,
        y_pred_test_soft,
        y_pred_train_soft_prop,
        y_pred_test_soft_prop,
        y_OH_pv[train_indices],
        y_OH_pv[test_indices],
        y_OH_s[train_indices],
        y_OH_s[test_indices],
        y_OS[train_indices],
        y_OS[test_indices],
        classes,
    )

    return metric_dict


# Experiment loop: iterate over classifiers and methods
# and run the inner experiment method where model training
# happens in parallel
def run_experiment(
    exp_parameters,
    X,
    y_soft,
    repeats=100,
    n_jobs=50,
    save=False,
    alpha=1.0,
    train_percentage=0.8,
):
    # Set up logging
    # get current time
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d_%H_%M_%S")

    logging.info("Conducting experiment for Real-World data")

    temp_result_dict = {}
    result_dict = {}

    for clf_name in exp_parameters["clf"]:
        clf = get_classifier(clf_name, "real-world")
        logging.info(f"Classifier: {clf_name}")

        temp_result_dict[clf_name] = {}
        result_dict[clf_name] = {}

        for method_name in exp_parameters["method"]:
            method, kwargs = get_method(method_name)
            logging.info(f"Learning method: {method_name}")

            temp_result_dict[clf_name][method_name] = {}
            result_dict[clf_name][method_name] = {}

            repeated_results = Parallel(n_jobs=n_jobs)(
                delayed(inner_experiment)(
                    X,
                    y_soft,
                    clf,
                    method,
                    kwargs,
                    alpha,
                    rep_counter=i,
                    train_percentage=train_percentage,
                )
                for i in range(repeats)
            )

            for repeat in range(repeats):
                # Transform result to dict
                temp_result = repeated_results[repeat]
                for key, value in temp_result.items():
                    if key not in temp_result_dict[clf_name][method_name]:
                        temp_result_dict[clf_name][method_name][key] = []
                    temp_result_dict[clf_name][method_name][key].append(value)

            for key, _ in temp_result_dict[clf_name][method_name].items():
                mean = np.mean(temp_result_dict[clf_name][method_name][key])
                std = np.std(temp_result_dict[clf_name][method_name][key])
                result_dict[clf_name][method_name][key] = {"mean": mean, "std": std}

            # Check if there was an unsuccessful run
            if None in repeated_results:
                logging.warning("Unsuccessful run of conduct experiment")
                logging.warning(f"clf_name: {clf_name}, method_name: {method_name}")

    # Save the results
    if save:
        complete_path = path_output / "real_data"
        complete_path.mkdir(parents=True, exist_ok=True)

        path_and_file = complete_path / f"{time_string}.json"

        # Convert and write JSON object to file
        with open(path_and_file, "w") as outfile:
            json.dump(result_dict, outfile)

    return result_dict
