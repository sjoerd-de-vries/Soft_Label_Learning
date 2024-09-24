# As implemented by the STAC library
import numpy as np
import scipy.stats as st


def friedman_aligned_ranks_test(*args):
    """
    Performs a Friedman aligned ranks ranking test.
    Tests the hypothesis that in a set of k dependent samples groups (where k >= 2) at
    least two of the groups represent populations with different median values.
    The difference with a friedman test is that it uses the median of each group to
    construct the ranking, which is useful when the number of samples is low.

    Parameters
    ----------
    sample1, sample2, ... : array_like
        The sample measurements for each group.

    Returns
    -------
    Chi2-value : float
        The computed Chi2-value of the test.
    p-value : float
        The associated p-value from the Chi2-distribution.
    rankings : array_like
        The ranking for each group.
    pivots : array_like
        The pivotal quantities for each group.

    References
    ----------
     J.L. Hodges, E.L. Lehmann, Ranks methods for combination of independent experiments
     in analysis of variance, Annals of Mathematical Statistics 33 (1962) 482–497.
    """
    k = len(args)
    if k < 2:
        raise ValueError("Less than 2 levels")
    n = len(args[0])
    if len(set([len(v) for v in args])) != 1:
        raise ValueError("Unequal number of samples")

    aligned_observations = []
    for i in range(n):
        loc = np.mean([col[i] for col in args])
        aligned_observations.extend([col[i] - loc for col in args])

    aligned_observations_sort = sorted(aligned_observations)

    aligned_ranks = []
    for i in range(n):
        row = []
        for j in range(k):
            v = aligned_observations[i * k + j]
            row.append(
                aligned_observations_sort.index(v)
                + 1
                + (aligned_observations_sort.count(v) - 1) / 2.0
            )
        aligned_ranks.append(row)

    rankings_avg = [np.mean([case[j] for case in aligned_ranks]) for j in range(k)]
    rankings_cmp = [r / np.sqrt(k * (n * k + 1) / 6.0) for r in rankings_avg]

    r_i = [np.sum(case) for case in aligned_ranks]
    r_j = [np.sum([case[j] for case in aligned_ranks]) for j in range(k)]
    T = (
        (k - 1)
        * (np.sum(v**2 for v in r_j) - (k * n**2 / 4.0) * (k * n + 1) ** 2)
        / float(
            ((k * n * (k * n + 1) * (2 * k * n + 1)) / 6.0)
            - (1.0 / float(k)) * np.sum(v**2 for v in r_i)
        )
    )

    p_value = 1 - st.chi2.cdf(T, k - 1)

    return T, p_value, rankings_avg, rankings_cmp


def finner_test(ranks, control=None):
    """
    Performs a Finner post-hoc test using the pivot quantities obtained by
    a ranking test. Tests the hypothesis that the ranking of the control method
    is different to each of the other methods.

    Parameters
    ----------
    pivots : dictionary_like
        A dictionary with format 'groupname':'pivotal quantity'
    control : string optional
        The name of the control method,  default the group with minimum ranking

    Returns
    ----------
    Comparisons : array-like
        Strings identifier of each comparison with format 'group_i vs group_j'
    Z-values : array-like
        The computed Z-value statistic for each comparison.
    p-values : array-like
        The associated p-value from the Z-distribution which depends on
        the index of the comparison
    Adjusted p-values : array-like
        The associated adjusted p-values which can be compared with a significance level

    References
    ----------
    H. Finner, On a monotonicity problem in step-down multiple test procedures,
    Journal of the American Statistical Association 88 (1993) 920–923.
    """
    k = len(ranks)
    values = ranks.values()
    keys = ranks.keys()

    if not control:
        control_i = list(values).index(min(values))
    else:
        control_i = list(keys).index(control)

    comparisons = [
        list(keys)[control_i] + " vs " + list(keys)[i]
        for i in range(k)
        if i != control_i
    ]
    z_values = [
        abs(list(values)[control_i] - list(values)[i])
        for i in range(k)
        if i != control_i
    ]
    p_values = [2 * (1 - st.norm.cdf(abs(z))) for z in z_values]
    # Sort values by p_value so that p_0 < p_1
    p_values, z_values, comparisons = map(
        list,
        zip(
            *sorted(
                zip(p_values, z_values, comparisons, strict=False), key=lambda t: t[0]
            ),
            strict=False,
        ),
    )
    adj_p_values = [
        min(
            max(
                1 - (1 - p_values[j]) ** ((k - 1) / float(j + 1)) for j in range(i + 1)
            ),
            1,
        )
        for i in range(k - 1)
    ]

    return comparisons, z_values, p_values, adj_p_values
