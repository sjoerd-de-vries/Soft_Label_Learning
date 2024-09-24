import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal


def fit_gaussian(data, weights):
    # Calculate the weighted mean
    weighted_mean = np.average(data, axis=0, weights=weights)

    # Calculate the weighted covariance matrix
    weighted_cov = np.cov(data.T, aweights=weights)

    return (weighted_mean, weighted_cov)


def create_multivariate_normal_gaussian(mean, cov):
    # Create a multivariate normal distribution object
    gaussian = multivariate_normal(mean=mean, cov=cov)

    return gaussian


def generate_distributions(dist=0.5, dimension=2):
    # Set the means
    class_1_mean = np.full(dimension, -dist)
    class_2_mean = np.full(dimension, dist)

    # Set the covariances
    class_1_cov = np.eye(len(class_1_mean))
    class_2_cov = np.eye(len(class_2_mean))

    # Create multivariate normal distributions
    true_mvn_1 = multivariate_normal(class_1_mean, class_1_cov)
    true_mvn_2 = multivariate_normal(class_2_mean, class_2_cov)

    return class_1_mean, class_1_cov, class_2_mean, class_2_cov, true_mvn_1, true_mvn_2


def sample_distributions(n, true_mvn_1, true_mvn_2, prob_1=0.5):
    # Draw actual number from binomial distribution
    # Ensure that there are at least 2 points per class,
    # otherwise the standard deviation is undefined
    c1_n = max(2, np.random.binomial(n, prob_1))
    c2_n = max(2, n - c1_n)

    # Sample from gaussians
    class_1_samples = true_mvn_1.rvs(c1_n)
    class_2_samples = true_mvn_2.rvs(c2_n)

    # Set hard label weights
    class_1_weights = np.ones(len(class_1_samples))
    class_2_weights = np.ones(len(class_2_samples))

    return class_1_samples, class_2_samples, class_1_weights, class_2_weights


def generate_soft_labels(
    class_1_samples,
    class_2_samples,
    true_mvn_1,
    true_mvn_2,
    prob_1=0.5,
    rebalance=True,
    noise_std=0.1,
):
    # Obtain all probabilities
    true_mvn_1_class_1 = true_mvn_1.pdf(class_1_samples)
    true_mvn_2_class_1 = true_mvn_2.pdf(class_1_samples)
    true_mvn_1_class_2 = true_mvn_1.pdf(class_2_samples)
    true_mvn_2_class_2 = true_mvn_2.pdf(class_2_samples)

    # At low numbers priors may not match samples
    if rebalance:
        prob_1 = len(class_1_samples) / (len(class_1_samples) + len(class_2_samples))

    prob_2 = 1 - prob_1

    # For each sample, calculate the probability of
    # belonging to class 1 or 2 respectively

    prob_class_1_class_1_samples = (
        prob_1
        * true_mvn_1_class_1
        / (prob_1 * true_mvn_1_class_1 + prob_2 * true_mvn_2_class_1)
    )
    prob_class_1_class_2_samples = (
        prob_1
        * true_mvn_1_class_2
        / (prob_1 * true_mvn_1_class_2 + prob_2 * true_mvn_2_class_2)
    )
    prob_class_2_class_1_samples = (
        prob_2
        * true_mvn_2_class_1
        / (prob_1 * true_mvn_1_class_1 + prob_2 * true_mvn_2_class_1)
    )
    prob_class_2_class_2_samples = (
        prob_2
        * true_mvn_2_class_2
        / (prob_1 * true_mvn_1_class_2 + prob_2 * true_mvn_2_class_2)
    )

    # Combine the samples
    total_samples = np.concatenate((class_1_samples, class_2_samples))
    class_1_soft_labels = np.concatenate(
        (prob_class_1_class_1_samples, prob_class_1_class_2_samples)
    )
    class_2_soft_labels = np.concatenate(
        (prob_class_2_class_1_samples, prob_class_2_class_2_samples)
    )

    # Add gaussian noise to the probabilities - mirrored on the other class
    if noise_std > 0:
        noise = np.random.normal(0, noise_std, len(total_samples))
        class_1_soft_labels += noise
        class_2_soft_labels -= noise
        class_1_soft_labels[class_1_soft_labels < 0] = 0
        class_2_soft_labels[class_2_soft_labels < 0] = 0
        class_1_soft_labels[class_1_soft_labels > 1] = 1
        class_2_soft_labels[class_2_soft_labels > 1] = 1

    return total_samples, class_1_soft_labels, class_2_soft_labels


def run_gaussian_experiment(n, repeats, dist=0.5, dimension=2, prob_1=0.5, noise_std=0):
    soft_mean_mse_array_1 = np.zeros(repeats)
    hard_mean_mse_array_1 = np.zeros(repeats)
    soft_mean_mse_array_2 = np.zeros(repeats)
    hard_mean_mse_array_2 = np.zeros(repeats)

    for i in range(repeats):
        (
            class_1_mean,
            class_1_cov,
            class_2_mean,
            class_2_cov,
            true_mvn_1,
            true_mvn_2,
        ) = generate_distributions(dist, dimension)

        (
            class_1_samples,
            class_2_samples,
            class_1_weights,
            class_2_weights,
        ) = sample_distributions(n, true_mvn_1, true_mvn_2, prob_1)

        # Fit a normal distribution to the hard labels
        hard_gaussian_1 = fit_gaussian(class_1_samples, class_1_weights)
        hard_gaussian_2 = fit_gaussian(class_2_samples, class_2_weights)

        total_samples, class_1_soft_labels, class_2_soft_labels = generate_soft_labels(
            class_1_samples,
            class_2_samples,
            true_mvn_1,
            true_mvn_2,
            prob_1,
            noise_std=noise_std,
        )

        # Fit a normal distribution to the soft labels
        soft_gaussian_1 = fit_gaussian(total_samples, class_1_soft_labels)
        soft_gaussian_2 = fit_gaussian(total_samples, class_2_soft_labels)

        # Calculate the error
        soft_mean_mse_1 = np.linalg.norm(class_1_mean - soft_gaussian_1[0], 2)
        hard_mean_mse_1 = np.linalg.norm(class_1_mean - hard_gaussian_1[0], 2)

        soft_mean_mse_2 = np.linalg.norm(class_2_mean - soft_gaussian_2[0], 2)
        hard_mean_mse_2 = np.linalg.norm(class_2_mean - hard_gaussian_2[0], 2)

        soft_mean_mse_array_1[i] = soft_mean_mse_1
        hard_mean_mse_array_1[i] = hard_mean_mse_1
        soft_mean_mse_array_2[i] = soft_mean_mse_2
        hard_mean_mse_array_2[i] = hard_mean_mse_2

    return (
        soft_mean_mse_array_1,
        hard_mean_mse_array_1,
        soft_mean_mse_array_2,
        hard_mean_mse_array_2,
    )


def parallel_experiment(n_samples_list, setting, repeats):
    exp_results = {}

    exp_results[setting, "soft_1"] = []
    exp_results[setting, "hard_1"] = []
    exp_results[setting, "soft_2"] = []
    exp_results[setting, "hard_2"] = []

    for n_samples in n_samples_list:
        temp_result = run_gaussian_experiment(
            n_samples, repeats, setting[0], setting[1], setting[2], setting[3]
        )

        exp_results[setting, "soft_1"].append(np.mean(temp_result[0]))
        exp_results[setting, "hard_1"].append(np.mean(temp_result[1]))
        exp_results[setting, "soft_2"].append(np.mean(temp_result[2]))
        exp_results[setting, "hard_2"].append(np.mean(temp_result[3]))

    return exp_results


def plot_results(x_range, result_list_soft, result_list_hard):
    # Define the x-axis range
    x = x_range

    # Define the y-axis values for the first list
    y1 = result_list_soft

    # Define the y-axis values for the second list
    y2 = result_list_hard

    # Plot the first list
    plt.plot(x, y1, label="Soft List")

    # Plot the second list
    plt.plot(x, y2, label="Hard List")

    # Change the x-axis to be logarithmic
    plt.xscale("log")

    # Add labels and title
    plt.xlabel("Range")
    plt.ylabel("Values")
    plt.title("Plot of Two Lists")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_differences(x_range, result_list_soft, result_list_hard):
    # Define the x-axis range
    x = x_range

    # Define the y-axis values for the first list
    y1 = np.array(result_list_soft) - np.array(result_list_hard)

    # Plot the first list
    plt.plot(x, y1, label="differences")

    # Change the x-axis to be logarithmic
    plt.xscale("log")

    # Add labels and title
    plt.xlabel("Range")
    plt.ylabel("Values")
    plt.title("Plot of Two Lists")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_multiple_lines(x_range, result_lists, labels):
    # Define the x-axis range
    x = x_range

    # set figure for saving
    fig = plt.figure()

    for list, label in zip(result_lists, labels, strict=False):
        # Plot the first list

        plt.plot(x, list, label=label)

    plt.plot(x, np.zeros(len(x)), dashes=[4, 3], color="gray")

    # Change the x-axis to be logarithmic
    plt.xscale("log")

    # Add labels and title
    plt.xlabel("Number of samples")
    plt.ylabel(r"$\Delta~ \overline{MSE}$ (soft,hard)")
    # plt.title("Plot of Two Lists")

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()

    return fig
