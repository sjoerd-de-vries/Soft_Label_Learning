import json
from collections import defaultdict

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import ScalarFormatter

from ..config import path_repository


def get_result_dict(folder_path, fixed_settings):
    """Load in the results of an experiment.

    Parameters
    ----------
    folder_path : string
        the name of the folder containing the experiment results. Typically a datetime
    fixed_settings : dict
        dictionary containing the setting of the experiment we want to get the results
        for. The values that are used specifically are:

        - dataset
        - gt
        - mtvd
        - noise
        - noise_type
        - clf

    Returns
    -------
    dict
        results of the experiment
    """

    experiment_path = (
        path_repository / "experiments" / "synthetic_data_results" / folder_path
    )

    settings_path = (
        experiment_path
        / fixed_settings["dataset"]
        / fixed_settings["gt"]
        / fixed_settings["mtvd"]
        / fixed_settings["noise"]
        / fixed_settings["noise_type"]
    )
    complete_path = settings_path / f"{fixed_settings['clf']}.json"

    with open(complete_path, "r") as file:
        exp_results = json.load(file)

    return exp_results


def get_alpha_prop_string(fixed_settings):
    if fixed_settings["ens_propagation"]:
        alpha_prop_string = (
            f"clf_prop_{fixed_settings['alpha']}_{fixed_settings['smoothing']}"
        )
    else:
        alpha_prop_string = (
            f"clf_{fixed_settings['alpha']}_{fixed_settings['smoothing']}"
        )

    return alpha_prop_string


def get_metric_settings(fixed_settings, eval_metric):
    if eval_metric == "TVD":
        fixed_settings["label_eval"] = "soft"
        fixed_settings["eval_set"] = "PG"
        fixed_settings["metric"] = "TVD"
        fixed_settings["train_test"] = "test"
    elif eval_metric == "hard_soft_AUC":
        fixed_settings["label_eval"] = "hard-soft"
        fixed_settings["eval_set"] = "G_dist"
        fixed_settings["metric"] = "hard_soft_AUC"
        fixed_settings["train_test"] = "test"
    elif eval_metric == "accuracy":
        fixed_settings["label_eval"] = "hard"
        fixed_settings["eval_set"] = "G"
        fixed_settings["metric"] = "accuracy"
        fixed_settings["train_test"] = "test"

    return fixed_settings


def results_by_noise_level(
    result_dict, metric, clf, noise_type, dataset, method, base_norm=True
):
    base = result_dict[metric][clf]["noiseless"]["5"][dataset][method]
    result_list = [base]
    if base_norm:
        result_list = [0]

    for result in result_dict[metric][clf][noise_type].values():
        if base_norm:
            result_list.append(result[dataset][method] - base)
        else:
            result_list.append(result[dataset][method])

    return result_list


def get_q2_result_dict(
    settings,
    methods,
    metrics,
    classifiers,
    datasets,
    non_ens_methods,
    result_string,
):
    """Method for creating a dictionary of the results of the Q2 experiment,
    converting them to a more usable format.
    """
    # initialize result dict
    nested_dict = lambda: defaultdict(nested_dict)
    result_dict = nested_dict()

    fixed_settings = settings.copy()

    for gt in settings["gt"]:
        fixed_settings["gt"] = gt
        print(f"gt: {gt}")
        for mtvd in fixed_settings["mtvd"]:
            fixed_settings["mtvd"] = mtvd
            print(f"mtvd: {mtvd}")
            for noise_type in settings["noise_type"]:
                fixed_settings["noise_type"] = noise_type
                print(f"noise_type: {noise_type}")
                for noise_level in settings["noise"]:
                    fixed_settings["noise"] = noise_level

                    if noise_type == "noiseless" and noise_level != "5":
                        continue

                    for clf in classifiers:
                        fixed_settings["clf"] = clf

                        for temp_set in datasets:
                            fixed_settings["dataset"] = temp_set

                            repeated_result_dict, nan_set = get_method_metric_results(
                                result_string,
                                fixed_settings,
                                methods,
                                metrics,
                                non_ens_methods,
                            )

                            for method in methods:
                                for eval_metric in metrics:
                                    repeated_result = repeated_result_dict[method][
                                        eval_metric
                                    ]

                                    if len(nan_set) == 0:
                                        mean_result = np.mean(repeated_result)
                                    else:
                                        usable_iterations = np.setdiff1d(
                                            np.arange(len(repeated_result)),
                                            list(nan_set),
                                            assume_unique=True,
                                        )

                                        mean_result = np.mean(
                                            np.array(repeated_result)[
                                                np.array(usable_iterations)
                                            ]
                                        )

                                    result_dict[gt][mtvd][eval_metric][clf][noise_type][
                                        noise_level
                                    ][temp_set][method] = mean_result

    return result_dict


def get_q1_result_dict(
    settings, methods, metrics, classifiers, datasets, non_ens_methods, result_string
):
    """Method for creating a dictionary of the results the Q1 experiment,
    converting them to a more usable format.
    """
    fixed_settings = settings.copy()
    result_dict = {}

    for eval_metric in metrics:
        result_dict[eval_metric] = {}
        for clf in classifiers:
            result_dict[eval_metric][clf] = np.zeros((len(methods), len(datasets)))

    for clf in classifiers:
        fixed_settings["clf"] = clf

        for set_idx, temp_set in enumerate(datasets):
            fixed_settings["dataset"] = temp_set

            repeated_result_dict, nan_set = get_method_metric_results(
                result_string, fixed_settings, methods, metrics, non_ens_methods
            )

            for method_idx, method in enumerate(methods):
                for eval_metric in metrics:
                    repeated_result = repeated_result_dict[method][eval_metric]

                    if len(nan_set) == 0:
                        mean_result = np.mean(repeated_result)
                    else:
                        usable_iterations = np.setdiff1d(
                            np.arange(len(repeated_result)),
                            list(nan_set),
                            assume_unique=True,
                        )

                        mean_result = np.mean(
                            np.array(repeated_result)[np.array(usable_iterations)]
                        )

                    result_dict[eval_metric][clf][method_idx, set_idx] = mean_result

    return result_dict


def q1_statistics_result_dict(
    specific_settings,
    settings_dict,
    result_path,
    methods,
    metrics,
    non_ens_methods,
    classifiers,
    datasets,
):
    """Method for creating a dictionary of the results of the Q1 experiment,
    converting them to a more usable format for generating results averaged over
    multiple datasets.
    """
    # initialize result dict
    nested_dict = lambda: defaultdict(nested_dict)
    result_dict = nested_dict()

    for gt_model in settings_dict["gt"]:
        specific_settings["gt"] = gt_model

        for mtvd in settings_dict["mtvd"]:
            specific_settings["mtvd"] = mtvd

            for dataset in datasets:
                specific_settings["dataset"] = dataset

                for clf in classifiers:
                    specific_settings["clf"] = clf

                    repeated_result_dict, nan_set = get_method_metric_results(
                        result_path,
                        specific_settings,
                        methods,
                        metrics,
                        non_ens_methods,
                    )

                    for _, method in enumerate(methods):
                        for eval_metric in metrics:
                            repeated_result = repeated_result_dict[method][eval_metric]

                            if len(nan_set) == 0:
                                mean_result = np.mean(repeated_result)
                            else:
                                usable_iterations = np.setdiff1d(
                                    np.arange(len(repeated_result)),
                                    list(nan_set),
                                    assume_unique=True,
                                )

                                mean_result = np.mean(
                                    np.array(repeated_result)[
                                        np.array(usable_iterations)
                                    ]
                                )

                            result_dict[gt_model][mtvd][clf][eval_metric][method][
                                dataset
                            ] = mean_result

    return result_dict


def q1_statistics_table(
    result_dict, settings_dict, methods, metrics, classifiers, datasets, threshold=False
):
    """Method for creating a table of the results of an experiment,
    averaged over all datasets."""

    # Initialize table array
    table_array = np.zeros((len(methods), len(classifiers) * len(metrics)))

    clf_metric = []

    # Iterate over rows and columns
    for metric_idx, metric in enumerate(metrics):
        for clf_idx, clf in enumerate(classifiers):
            if metric == "hard_soft_AUC":
                clf_metric.append(f"{clf} - AUC")
            else:
                clf_metric.append(f"{clf} - " + r"$\overline{TVD}$")

            for method_idx, method in enumerate(methods):
                n_metrics = 0

                # Obtain the results to combine
                for gt_model in settings_dict["gt"]:
                    for mtvd in settings_dict["mtvd"]:
                        for dataset in datasets:
                            result = result_dict[gt_model][mtvd][clf][metric][method][
                                dataset
                            ]

                            if metric == "TVD":
                                result = -result

                            # in case of threshold clf
                            if threshold:
                                if not np.isnan(result):
                                    n_metrics += 1

                                    table_array[
                                        method_idx,
                                        metric_idx * len(classifiers) + clf_idx,
                                    ] += result
                            else:
                                n_metrics += 1

                                table_array[
                                    method_idx, metric_idx * len(classifiers) + clf_idx
                                ] += result

                table_array[method_idx, metric_idx * len(classifiers) + clf_idx] /= (
                    n_metrics
                )
    return table_array, clf_metric


def get_method_metric_results(folder_path, settings, methods, metrics, non_ens_methods):
    """Method for converting the results from an experiment to a more usable format."""

    exp_results = get_result_dict(folder_path, settings)

    result_dict = {}

    # Keeping track of iterations where a methods result is None
    nan_dict = {}
    nan_set = set()
    # To vary: method, metric

    for method in methods:
        # As non ensemble methods do not have ens propagation,
        # we need to set it to False
        fixed_settings = settings.copy()

        if method in non_ens_methods:
            fixed_settings["ens_propagation"] = False

        alpha_prop_string = get_alpha_prop_string(fixed_settings)

        method_results = exp_results[method]

        result_dict[method] = {}
        nan_dict[method] = []

        for metric in metrics:
            result_list = []

            fixed_settings = get_metric_settings(fixed_settings, metric)

            for key, iteration_result in method_results.items():
                if iteration_result is None:
                    result_list.append(np.nan)
                    nan_set.add(key)

                else:
                    intermediate_result = iteration_result[
                        fixed_settings["label_eval"]
                    ][alpha_prop_string]
                    if intermediate_result is None:
                        raise ValueError("None result for intermediate result")

                    result_list.append(
                        intermediate_result[fixed_settings["eval_set"]][
                            fixed_settings["metric"]
                        ][fixed_settings["train_test"]]
                    )

            result_dict[method][metric] = result_list

    return result_dict, nan_set


def plot_heatmap(
    result_array,
    xlabels,
    ylabels,
    base_row,
    add_mean=True,
    figsize=(12, 8),
):
    """Plot a heatmap of the results in result_array."""
    n_rows = result_array.shape[0]

    result_array = result_array * 100

    # add the mean of each row
    if add_mean:
        result_array = np.concatenate(
            (result_array, np.mean(result_array, axis=1).reshape(-1, 1)),
            axis=1,
        )

    # changes order, 0 = sampling, 4 is bootstrap
    result_array_norm = result_array - np.tile(result_array[base_row, :], (n_rows, 1))

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(result_array_norm)

    min_max = np.max(abs(result_array_norm))

    plt.imshow(result_array_norm, cmap="seismic", vmin=-min_max, vmax=min_max)
    plt.colorbar()

    # Show all ticks and label them with the respective list entries
    xlabels_new = xlabels.copy()
    if add_mean:
        xlabels_new.append("mean")
    ax.set_xticks(np.arange(len(xlabels_new)), labels=xlabels_new)
    ax.set_yticks(np.arange(n_rows), labels=ylabels)

    # Plot the values
    for set_idx in range(len(xlabels_new)):
        for method_idx in range(n_rows):
            temp_value = round(result_array[method_idx, set_idx], ndigits=1)
            text = ax.text(
                set_idx,
                method_idx,
                temp_value,
                ha="center",
                va="center",
                color="black",
                path_effects=[pe.withStroke(linewidth=2.1, foreground="white")],
                fontsize=9,
            )

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    return fig


def plot_variation_over_sets_on_axis(
    ax,
    result_dict,
    datasets,
    title,
    linestyle_dict,
    base_norm=True,
    **kwargs,
):
    """
    Plot the results of different methods over different noise levels.
    if base_norm is True, the results are normalized by the noiseless result.
    """
    plot_kwargs = {}

    # If the value is a list, we vary over the list items
    for key, value in kwargs.items():
        if isinstance(value, str):
            plot_kwargs[key] = value
        else:
            vary_key = key

    for value in kwargs[vary_key]:
        plot_kwargs[vary_key] = value
        result_list = []

        for temp_set in datasets:
            plot_kwargs["dataset"] = temp_set

            result = results_by_noise_level(
                result_dict, base_norm=base_norm, **plot_kwargs
            )
            result_list.append(result)

        avg_result = np.mean(np.array(result_list), axis=0)
        value_label = value

        ax.plot(avg_result, linestyle=linestyle_dict[value], label=value_label)

    # plot a horizontal line
    if base_norm:
        ax.axhline(y=0.0, color="gray", linestyle=(0, (8, 16)))

    ax.set_title(title)

    # Apply the custom formatter to the x-axis
    if base_norm:
        formatter = ScalarFormatter(useMathText=True)
        formatter.set_scientific(True)
        formatter.set_powerlimits((-1, 1))
        ax.yaxis.set_major_formatter(formatter)


def get_noise_replace_dict():
    noise_replace_dict = {
        "noiseless": "Noiseless",
        "NCAR": "NCAR",
        "NAR": "NAR",
        "miscalibrated_pos_false": "Overprediction",
        "miscalibrated_neg_false": "Underprediction",
        "miscalibrated_pos_true": "Underextremity",
        "miscalibrated_neg_true": "Overextremity",
    }

    return noise_replace_dict
