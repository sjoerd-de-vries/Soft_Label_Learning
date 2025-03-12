import datetime
import json
import logging
import math
from collections import defaultdict
from copy import deepcopy
from functools import partial

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score

# Synlabel import
from synlabel.utils.helper_functions import one_hot_encoding

from ..config import path_repository, path_uci_keel_data
from ..data_generation.data_extraction import datasets, read_data
from ..data_generation.dataset_preparation import (
    generate_test_data,
    get_calibration_noise_function,
    get_nar_function,
    get_ncar_function,
)
from ..methods.ensemble_classifiers import EnsembleClassifier
from ..methods.utils import adjust_predictions, return_plurality_label
from .metrics import mean_log_loss as ll
from .metrics import mean_squared_error as mse
from .metrics import mean_total_variation_distance as tvd
from .utils import get_classifier, get_method


def conduct_experiment(
    X,
    y_G,
    y_PG,
    y_OS,
    classes,
    learning_method,
    kwargs,
    base_clf,
    alpha_values,
    ens_propagation,
    smoothing,
    train_percentage=0.7,
    rep_counter=1,
):
    """
    Conducts a single experiment with the given data and learning method.

    """

    try:
        # Transforming soft labels to obtain hard labels
        y_PG_pv = return_plurality_label(y_PG)
        cumulative_prob = y_PG.cumsum(axis=1)
        random_numbers = np.random.rand(len(y_PG), 1)
        choices = (random_numbers < cumulative_prob).argmax(axis=1)
        y_PG_s = classes[choices]
        y_OH_pv = return_plurality_label(y_OS)
        cumulative_prob = y_OS.cumsum(axis=1)
        random_numbers = np.random.rand(len(y_OS), 1)
        choices = (random_numbers < cumulative_prob).argmax(axis=1)
        y_OH_s = classes[choices]

        # Soft label transform of hard labels
        y_G_dist = one_hot_encoding(y_G, classes)
        y_OH_pv_dist = one_hot_encoding(y_OH_pv, classes)
        y_OH_s_dist = one_hot_encoding(y_OH_s, classes)

        # Preparing the train-test split
        n_samples = X.shape[0]
        n_train = int(n_samples * train_percentage)

        # Obtaining the indices
        rng = np.random.default_rng(seed=rep_counter)
        training_indices = rng.choice(n_samples, n_train, replace=False)
        test_indices = np.setdiff1d(np.arange(n_samples), training_indices)

        # Obtaining the actual data to train on
        X_train = X[training_indices]
        y_OS_train = y_OS[training_indices]
        y_OH_pv_train = y_OH_pv[training_indices]
        y_OH_s_train = y_OH_s[training_indices]

        # The outcome labels
        hard_label_keys = ["G", "PG_pv", "PG_s", "OH_pv", "OH_s"]
        values = [y_G, y_PG_pv, y_PG_s, y_OH_pv, y_OH_s]
        hard_label_dict = dict(zip(hard_label_keys, values, strict=False))
        soft_label_keys = ["G_dist", "PG", "OS", "OH_pv_dist", "OH_s_dist"]
        values = [y_G_dist, y_PG, y_OS, y_OH_pv_dist, y_OH_s_dist]
        soft_label_dict = dict(zip(soft_label_keys, values, strict=False))

        # One hot encoded hard labels for AUC and log loss
        hard_soft_label_keys = ["G_dist", "OH_pv_dist", "OH_s_dist"]
        values = [y_G_dist, y_OH_pv_dist, y_OH_s_dist]
        hard_soft_label_dict = dict(zip(hard_soft_label_keys, values, strict=False))
        outcome_label_dict = {
            "hard": hard_label_dict,
            "soft": soft_label_dict,
            "hard-soft": hard_soft_label_dict,
        }

        # Fitting the methods
        model_dict = {}

        # Adding different values of alpha
        for alpha in alpha_values:
            if isinstance(learning_method, str):
                # Values of alpha other than 1 don't make sense when learning from
                # hard labels
                if alpha == 1 and smoothing == 0:
                    clf = clone(base_clf)

                    if learning_method == "SampleClf":
                        clf.fit(X_train, y_OH_s_train)
                    else:
                        raise ValueError(f"Invalid learning method: {learning_method}")
                    model_dict[f"clf_{alpha}_{smoothing}"] = deepcopy(clf)
                else:
                    model_dict[f"clf_{alpha}_{smoothing}"] = None

                # No ensemble probability propagation
                model_dict[f"clf_prop_{alpha}_{smoothing}"] = None
            else:
                clf = learning_method(
                    base_estimator=clone(base_clf),
                    alpha=alpha,
                    smoothing=smoothing,
                    **kwargs,
                )
                clf.fit(X_train, classes, y_OS_train)
                model_dict[f"clf_{alpha}_{smoothing}"] = deepcopy(clf)

                if issubclass(learning_method, EnsembleClassifier) and ens_propagation:
                    model_dict[f"clf_prop_{alpha}_{smoothing}"] = deepcopy(clf)
                else:
                    model_dict[f"clf_prop_{alpha}_{smoothing}"] = None

        # Predicting
        train_test_keys = ["train", "test"]
        values = [training_indices, test_indices]
        indices_dict = dict(zip(train_test_keys, values, strict=False))

        prediction_dict = {
            "hard": {"train": {}, "test": {}},
            "soft": {"train": {}, "test": {}},
            "hard-soft": {"train": {}, "test": {}},
        }

        for index_key, indices in indices_dict.items():
            for model_key, model_value in model_dict.items():
                if model_value is None:
                    prediction_dict["hard"][index_key][model_key] = None
                    prediction_dict["soft"][index_key][model_key] = None
                else:
                    prediction_dict["hard"][index_key][model_key] = model_value.predict(
                        X[indices]
                    )
                    if "clf_prop" in model_key:
                        predictions = adjust_predictions(
                            model_value.predict_proba(
                                X[indices], method="prob_propagation"
                            ),
                            model_value.classes_,
                            classes,
                        )
                    else:
                        predictions = adjust_predictions(
                            model_value.predict_proba(X[indices]),
                            model_value.classes_,
                            classes,
                        )

                    prediction_dict["soft"][index_key][model_key] = predictions

        prediction_dict["hard-soft"]["train"] = prediction_dict["soft"]["train"]
        prediction_dict["hard-soft"]["test"] = prediction_dict["soft"]["test"]

        # Prediction metrics
        ## Hard labels
        hard_metric_keys = ["accuracy"]
        values = [accuracy_score]
        hard_metric_dict = dict(zip(hard_metric_keys, values, strict=False))

        ## Soft labels
        soft_metric_keys = ["LL", "TVD", "MSE"]
        values = [ll, tvd, mse]
        soft_metric_dict = dict(zip(soft_metric_keys, values, strict=False))

        # Hard labels, Soft label predictions
        hard_soft_metric_keys = ["hard_soft_LL", "hard_soft_AUC"]
        values = [log_loss, partial(roc_auc_score, average="weighted")]
        hard_soft_metric_dict = dict(zip(hard_soft_metric_keys, values, strict=False))
        metric_dict = {
            "hard": hard_metric_dict,
            "soft": soft_metric_dict,
            "hard-soft": hard_soft_metric_dict,
        }

        # The experiment loop
        nested_dict = lambda: defaultdict(nested_dict)
        result_dict = nested_dict()

        # Obtaining the outcome labels
        for label_type in outcome_label_dict.keys():
            for val_set_key, val_set_value in outcome_label_dict[label_type].items():
                # Obtaining the predicted values
                for prediction_set in train_test_keys:  # ["train", "test"]
                    # hard-hard & soft-soft validation and prediction metrics
                    for (
                        prediction_result_key,
                        prediction_result_value,
                    ) in prediction_dict[label_type][prediction_set].items():
                        pred = prediction_result_value

                        # Settings that were previously identified as not useable
                        if pred is None:
                            result_dict[label_type][prediction_result_key] = None
                        else:
                            val = val_set_value[indices_dict[prediction_set]]

                            for metric_key, metric in metric_dict[label_type].items():
                                # Since this does not work if a class never occurs
                                if metric_key == "hard_soft_AUC":
                                    temp_val = np.append(
                                        val, np.eye(len(classes)), axis=0
                                    )
                                    temp_pred = np.append(
                                        pred, np.eye(len(classes)), axis=0
                                    )
                                    score = metric(temp_val, temp_pred)
                                else:
                                    score = metric(val, pred)
                                result_dict[label_type][prediction_result_key][
                                    val_set_key
                                ][metric_key][prediction_set] = round(score, 6)

    except Exception as e:
        result_dict = None
        logging.warning("Unsuccessful run of conduct experiment")
        logging.warning(f"clf_name: {base_clf}, method_name: {learning_method}")
        logging.error(repr(e))

    return result_dict


def create_and_return_save_path(name_specifier, time_string=None):
    # get current time
    now = datetime.datetime.now()
    temp_time_string = now.strftime("%Y%m%d_%H_%M_%S")

    if time_string is None:
        time_string = temp_time_string

    # get path to parent folder
    save_path = path_repository / "experiments" / "synthetic_data_results" / time_string

    # makes the new folder, if it does not yet exist
    save_path.mkdir(parents=False, exist_ok=True)
    complete_path = save_path / name_specifier

    return complete_path


def complete_experiment(
    exp_parameters,
    lr_feature_hiding_path,
    rf_feature_hiding_path,
    fh_samples=1000,
    repeats=100,
    n_jobs=50,
    save=True,
    test_run=False,
    gt_save_and_load=False,
    generate_gt_sets=False,
):
    """Main method to conduct the experiments
    This methods iterates over the different settings,
    prepares the different datasets and adds noise to them
    and then calls the conduct_experiment method to run the
    actual experiment.

    Parameters
    ----------
    exp_parameters : dict
        dictionary containing the parameters for the experiment
    lr_feature_hiding_path : Path
        path to the feature hiding cut-off values for the logistic regression model
    rf_feature_hiding_path : Path
        path to the feature hiding cut-off values for the random forest model
    fh_samples : int, optional
        the number of iterations of the feature hiding method, by default 1000
    repeats : int, optional
        the number of repeated executions of the experiment, by default 100
    n_jobs : int, optional
        number of parallel jobs, by default 50
    save : bool, optional
        whether to save the results of the experiment, by default True
    test_run : bool, optional
        whether to execute a dummy run for testing purposes, by default False
    gt_save_and_load : bool, optional
        whether to check for existing versions of the different ground truth
        datasets, that ensures the same sets are used for evaluation if the
        experiments are split up over multiple method calls, by default False
    generate_gt_sets : bool, optional
        whether this is a run with the purpose of preparing the different versions
        of the ground truth data, by default False

    """
    allowed_keys = [
        "dataset",
        "gt",
        "mtvd",
        "noise",
        "noise_type",
        "clf",
        "method",
        "label_eval",
        "alpha",
        "eval_set",
        "metric",
        "train_test",
        "ens_propagation",
        "smoothing",
    ]

    # Validate the experiment parameters
    for key in exp_parameters.keys():
        if key not in allowed_keys:
            raise ValueError(f"Invalid key: {key}")

    # Get current time
    now = datetime.datetime.now()
    time_string = now.strftime("%Y%m%d_%H_%M_%S")

    for dataset in exp_parameters["dataset"]:
        logging.info(f"Conducting experiment for {dataset}")

        # Check if paths are correct
        if save:
            dataset_path = create_and_return_save_path(
                dataset[:-4], time_string=time_string
            )

        for gt_model in exp_parameters["gt"]:
            logging.info(f"Current GT model: {gt_model}")

            fh_result_dict = read_fh_results(
                gt_model, lr_feature_hiding_path, rf_feature_hiding_path
            )[dataset]

            for mtvd_setting in exp_parameters["mtvd"]:
                logging.info(f"Current mtvd: {mtvd_setting}")
                n_features_to_hide = fh_result_dict[f"cutoff_{mtvd_setting}"]

                if gt_model == "lr":
                    clf = LogisticRegression(
                        penalty="elasticnet", solver="saga", l1_ratio=0.5
                    )
                elif gt_model == "rf":
                    clf = RandomForestClassifier()

                path_gt = path_repository / "experiments" / "processed_G_PG_sets"

                # Check if a file named "dataset_gt_model_mtvd.joblib" exists at path_gt
                if gt_save_and_load:
                    # Check if file exists
                    if (
                        path_gt
                        / f"{dataset}_{gt_model}_{mtvd_setting}_{fh_samples}_G.joblib"
                    ).exists():
                        logging.info("Loading GT sets")
                        D_G = joblib.load(
                            path_gt
                            / (
                                f"{dataset}_{gt_model}_{mtvd_setting}_{fh_samples}_G"
                                + ".joblib"
                            )
                        )
                        D_PG = joblib.load(
                            path_gt
                            / (
                                f"{dataset}_{gt_model}_{mtvd_setting}_{fh_samples}_PG"
                                + ".joblib"
                            )
                        )
                    else:
                        logging.info("Generating and saving GT sets")
                        D_G, D_PG = generate_test_data(
                            dataset=dataset,
                            clf=clf,
                            n_features_to_hide=n_features_to_hide,
                            n_samples=fh_samples,
                            method="multivariate_kde_sklearn",
                            reverse_feature_order=True,
                        )
                        joblib.dump(
                            D_G,
                            path_gt
                            / (
                                f"{dataset}_{gt_model}_{mtvd_setting}_{fh_samples}_G"
                                + ".joblib"
                            ),
                        )
                        joblib.dump(
                            D_PG,
                            path_gt
                            / (
                                f"{dataset}_{gt_model}_{mtvd_setting}_{fh_samples}_PG"
                                + ".joblib"
                            ),
                        )
                else:
                    # Generate the ground truth data
                    D_G, D_PG = generate_test_data(
                        dataset=dataset,
                        clf=clf,
                        n_features_to_hide=n_features_to_hide,
                        n_samples=fh_samples,
                        method="multivariate_kde_sklearn",
                        reverse_feature_order=True,
                    )

                classes = D_G.classes
                n_classes = len(classes)

                # Generate the observed soft label data
                D_OS_noiseless = D_PG.to_distributed_observed("identity")

                # Generate the observed soft label data, to ensure the same sets
                # are used when experiments are run subsequently
                if generate_gt_sets:
                    continue

                for noise_level in exp_parameters["noise"]:
                    logging.info(f"Current noise level: {noise_level}")

                    noise_float = float(noise_level) / 100.0

                    transform_uni = get_ncar_function(noise_float, n_classes)
                    D_OS_NCAR = D_PG.to_distributed_observed(
                        "transform_y", transformation=transform_uni
                    )

                    transform_random = get_nar_function(noise_float, n_classes)
                    D_OS_NAR = D_PG.to_distributed_observed(
                        "transform_y", transformation=transform_random
                    )

                    # Miscalibration
                    transform_miscalibrated = get_calibration_noise_function(
                        noise_float, extremity=False
                    )
                    D_OS_miscalibrated_pos_false = D_PG.to_distributed_observed(
                        "transform_y", transformation=transform_miscalibrated
                    )

                    transform_miscalibrated = get_calibration_noise_function(
                        -noise_float, extremity=False
                    )
                    D_OS_miscalibrated_neg_false = D_PG.to_distributed_observed(
                        "transform_y", transformation=transform_miscalibrated
                    )

                    transform_miscalibrated = get_calibration_noise_function(
                        noise_float, extremity=True
                    )
                    D_OS_miscalibrated_pos_true = D_PG.to_distributed_observed(
                        "transform_y", transformation=transform_miscalibrated
                    )

                    transform_miscalibrated = get_calibration_noise_function(
                        -noise_float, extremity=True
                    )
                    D_OS_miscalibrated_neg_true = D_PG.to_distributed_observed(
                        "transform_y", transformation=transform_miscalibrated
                    )

                    set_dict = {
                        "noiseless": D_OS_noiseless,
                        "NCAR": D_OS_NCAR,
                        "NAR": D_OS_NAR,
                        "miscalibrated_pos_false": D_OS_miscalibrated_pos_false,
                        "miscalibrated_neg_false": D_OS_miscalibrated_neg_false,
                        "miscalibrated_pos_true": D_OS_miscalibrated_pos_true,
                        "miscalibrated_neg_true": D_OS_miscalibrated_neg_true,
                    }

                    # Conducting the experiment
                    for os_name, D_OS in set_dict.items():
                        # Skip if the noise type is not in the experiment parameters
                        if os_name not in exp_parameters["noise_type"]:
                            continue

                        logging.info(f"Current observed setting: {os_name}")

                        for clf_name in exp_parameters["clf"]:
                            clf = get_classifier(clf_name, dataset)
                            logging.info(f"Classifier: {clf_name}")

                            result_dict = {}

                            # Skip if the noise type is noiseless and
                            # has already been done
                            if (os_name == "noiseless") and (
                                noise_level != exp_parameters["noise"][0]
                            ):
                                continue

                            for method_name in exp_parameters["method"]:
                                method, kwargs = get_method(method_name)
                                logging.info(f"Learning method: {method_name}")

                                result_dict[method_name] = {}

                                if test_run:
                                    repeated_results = [None for i in range(repeats)]
                                else:
                                    repeated_results = Parallel(n_jobs=n_jobs)(
                                        delayed(conduct_experiment)(
                                            D_PG.X,
                                            D_G.y,
                                            D_PG.y,
                                            D_OS.y,
                                            classes,
                                            method,
                                            kwargs,
                                            clf,
                                            exp_parameters["alpha"],
                                            exp_parameters["ens_propagation"],
                                            exp_parameters["smoothing"],
                                            rep_counter=i,
                                        )
                                        for i in range(repeats)
                                    )

                                for repeat in range(repeats):
                                    # Transform result to dict
                                    temp_result = repeated_results[repeat]
                                    result_dict[method_name][repeat] = temp_result

                                # Check if there was an unsuccessful run
                                if None in repeated_results:
                                    logging.warning(
                                        "Unsuccessful run of conduct experiment"
                                    )
                                    logging.warning(
                                        f"clf_name: {clf_name}, \
                                        method_name: {method_name}"
                                    )

                            # Save the results
                            if save:
                                complete_path = (
                                    dataset_path
                                    / gt_model
                                    / mtvd_setting
                                    / noise_level
                                    / os_name
                                )
                                complete_path.mkdir(parents=True, exist_ok=True)

                                path_and_file = complete_path / f"{clf_name}.json"

                                # Convert and write JSON object to file
                                with open(path_and_file, "w") as outfile:
                                    json.dump(result_dict, outfile)


def determine_mtvd_per_dataset(
    model_to_use="rf", fh_samples=500, save_data=True, dataset=None, reverse_order=True
):
    """
    Used to determine the number of features to hide to achieve a certain MTVD
    value for a given dataset corresponding to either the low or high uncertainty
    setting
    """

    mtvd_dict = {}

    # Check if paths are correct
    if save_data:
        # Get current time
        now = datetime.datetime.now()
        time_string = now.strftime("%Y%m%d_%H_%M_%S")
        result_string = time_string + f"_dataset_mtvd_{model_to_use}.json"

        # Get save path
        save_path = path_repository / "experiments" / "feature_hiding_mtvd"

        if save_path.exists():
            complete_path = save_path / result_string
        else:
            raise FileNotFoundError(f"Path {save_path} does not exist")

    if dataset is None:
        datasets_to_use = datasets
    else:
        datasets_to_use = [dataset]

    for dataset in datasets_to_use:
        logging.info(dataset)
        mtvd_dict[dataset] = {}
        data = read_data(dataset, read_path=path_uci_keel_data)
        logging.info(data.shape)

        # Construct range such that the last 20 features are always included
        initial_range = list(
            range(1, data.shape[1] - 1, math.ceil(data.shape[1] / 100))
        )
        last_20_features = list(range(max(data.shape[1] - 21, 1), data.shape[1] - 1))
        complete_range = list(set(initial_range + last_20_features))
        complete_range.sort()

        for features_to_exclude in complete_range:
            if model_to_use == "lr":
                model = LogisticRegression(
                    penalty="elasticnet", solver="saga", l1_ratio=0.5
                )
            elif model_to_use == "rf":
                model = RandomForestClassifier()

            D_G, D_PG = generate_test_data(
                dataset=data,
                clf=model,
                n_samples=fh_samples,
                method="multivariate_kde_sklearn",
                n_features_to_hide=features_to_exclude,
                reverse_feature_order=reverse_order,
            )

            y_G_dist = D_G.to_partial_ground_truth("identity").y
            mtvd = round(tvd(y_G_dist, D_PG.y), 5)
            logging.info(f"features to exclude {features_to_exclude}, MTVD: {mtvd}")

            mtvd_dict[dataset][features_to_exclude] = mtvd
        logging.info("   ")

        # Re-save the results after every dataset has been finished
        if save_data:
            # Convert and write JSON object to file
            with open(complete_path, "w") as outfile:
                json.dump(mtvd_dict, outfile)

    return mtvd_dict


def read_fh_results(gt_model, lr_feature_hiding_path, rf_feature_hiding_path):
    if gt_model == "rf":
        final_path = rf_feature_hiding_path
    elif gt_model == "lr":
        final_path = lr_feature_hiding_path

    with open(final_path, "r") as file:
        loaded_dict = json.load(file)

    return loaded_dict
