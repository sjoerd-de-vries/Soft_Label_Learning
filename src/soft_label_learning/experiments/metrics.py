import numpy as np


def mean_total_variation_distance(D_1, D_2):
    """The mean Total Variation Distance between two distributions

    Parameters
    ----------
    D_1 : array-like
        the first distribution
    D_2 : array-like
        the second distribution

    Returns
    -------
    mean_total_var_dist
        the mean Total Variation Distance
    """
    assert D_1.shape[0] == D_2.shape[0]

    n = D_1.shape[0]
    total_var_dist = np.sum(np.linalg.norm(D_1 - D_2, 1, axis=1))
    mean_total_var_dist = total_var_dist / (2 * n)

    return mean_total_var_dist


def mean_squared_error(D_1, D_2):
    """The Mean Squared Error between two distributions

    Parameters
    ----------
    D_1 : array-like
        the first distribution
    D_2 : array-like
        the second distribution

    Returns
    -------
    mean_squared_error
        the Mean Squared Error
    """
    assert D_1.shape[0] == D_2.shape[0]

    n = D_1.shape[0]

    total_squared_error = np.sum(np.linalg.norm(D_1 - D_2, 2, axis=1))
    mean_squared_error = total_squared_error / n

    return mean_squared_error


# Not all that useful, hugely effected by the log 0s
def mean_log_loss(D_1, D_2, error=1e-15):
    """The mean log loss, or cross-entropy loss between the two distributions

    Parameters
    ----------
    D_1 : array-like
        the first distribution
    D_2 : array-like
        the second distribution

    Returns
    -------
    mean_total_var_dist
        the mean Total Variation Distance
    """
    assert D_1.shape[0] == D_2.shape[0]

    n = D_1.shape[0]
    D_2 = np.clip(D_2, error, 1)

    D_2_log = np.log(D_2)
    multiplied = D_1 * D_2_log
    total_log_loss = -np.sum(multiplied)
    mean_log_loss = total_log_loss / n

    return mean_log_loss
