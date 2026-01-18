import numpy as np
from sklearn.model_selection import train_test_split


# TODO: refactor so that it takes multiple columns and multiple conditions for trimming.
def _trim_probabilities(preds, trimming_threshold=None, method=None, conditions=None):
    if preds.ndim == 1:
        if conditions is not None:
            return np.array(preds)[conditions]
        if method == "both":
            lower_bound = trimming_threshold
            upper_bound = 1 - trimming_threshold
            preds = preds[(preds >= lower_bound) & (preds <= upper_bound)]
        elif method == "higher":
            preds = preds[(preds <= trimming_threshold)]
        elif method == "lower":
            preds = preds[preds >= trimming_threshold]
    else:
        return np.array(preds)[:, conditions]  # selects columns based on conditions

    return preds


def _normalize_propensity_med(
    normalize_ipw,
    score_function,
    outcome,
    treatment_indicator,
    propensity_score=None,
    conditional_pot_med_prob=None,
    conditional_counter_med_prob=None,
    propensity_score_med=None,
):
    if normalize_ipw:
        if outcome == "potential":
            propensity_coef = _normalize_potential(treatment_indicator, propensity_score)
        elif outcome == "counterfactual":
            if score_function == "efficient":
                propensity_coef = _normalize_counterfactual()
            elif score_function == "efficient-alt":
                propensity_coef = _normalize_counterfactual_alt()
    else:
        propensity_coef = propensity_score

    return propensity_coef


def _normalize_potential(treatment_indicator, propensity_score):
    mean_treat = np.mean(np.divide(treatment_indicator, propensity_score))
    propensity_coef = np.divide(np.divide(treatment_indicator, propensity_score), mean_treat)
    return [propensity_coef]


def _normalize_counterfactual(treatment_indicator, propensity_score, conditional_pot_med_prob, conditional_counter_med_prob):
    mean_treat1 = np.mean(
        np.multiply(
            np.divide(treatment_indicator, propensity_score), np.divide(conditional_counter_med_prob, conditional_pot_med_prob)
        )
    )
    mean_treat2 = np.mean(np.divide(1.0 - treatment_indicator, 1.0 - propensity_score))

    propensity_coef1 = np.multiply(
        np.multiply(
            np.divide(treatment_indicator, propensity_score), np.divide(conditional_counter_med_prob, conditional_pot_med_prob)
        ),
        mean_treat1,
    )
    propensity_coef2 = np.multiply(np.divide(1.0 - treatment_indicator, 1.0 - propensity_score), mean_treat2)
    return [propensity_coef1, propensity_coef2]


def _normalize_counterfactual_alt(treatment_indicator, propensity_score, propensity_score_med):
    mean_treat1 = np.mean(
        np.multiply(
            np.divide(treatment_indicator, 1.0 - propensity_score), np.divide(1.0 - propensity_score_med, propensity_score_med)
        )
    )
    mean_treat2 = np.mean(np.divide(1.0 - treatment_indicator, 1.0 - propensity_score))

    propensity_coef1 = np.multiply(
        np.multiply(
            np.divide(treatment_indicator, 1.0 - propensity_score), np.divide(1.0 - propensity_score_med, propensity_score_med)
        ),
        mean_treat1,
    )
    propensity_coef2 = np.multiply(np.divide(1.0 - treatment_indicator, 1.0 - propensity_score), mean_treat2)
    return [propensity_coef1, propensity_coef2]


def extract_sets_from_smpls(
    smpls,
):
    """
    Separates the train and test indices from smpls and returns them
    """
    train_indices = tuple([train_index for (train_index, _) in smpls])
    test_indices = tuple([test_index for (_, test_index) in smpls])
    return train_indices, test_indices


def split_smpls(
    smpls,
    smpls_ratio=0.5,
):
    """
    Splits sample into two subsamples.
    Parameters
    ----------
    smpls_ratio : float
        Describes the ratio of observations in the first subsample compared to those in the second subsample

    Returns
    -------
    results : a list of tuples of ndarrays
        Contains the indexes of the subsamples (subsample1, subsample2, train and test)
    """
    if (smpls is None) or (not smpls):
        raise ValueError("the smpls array is empty")
    if smpls_ratio is None:
        raise ValueError("smpls_ratio must be a float between 0.0 and 1.0")

    subsample1 = []
    subsample2 = []
    for smpl in smpls:
        subsample1_idx, subsample2_idx = train_test_split(smpl, test_size=smpls_ratio)
        subsample1.append(subsample1_idx)
        subsample2.append(subsample2_idx)
    return subsample1, subsample2


def recombine_samples(
    subsmpls1,
    subsmpls2,
):
    """
    Recombines two subsamples into a single smpl-like structure.

    Parameters
    ----------
    subsmpls1: list of tuples of ndarrays
    subsmpls2: list of tuples of ndarrays

    :return:
    list of the pairwise combined inputs.
    """
    result = []
    for s1, s2 in zip(subsmpls1, subsmpls2):
        result.append((s1, s2))
    return result


def _check_inner_sample_splitting(all_inner_smpls, smpls):
    if not isinstance(all_inner_smpls, list):
        raise TypeError(
            f"all_inner_smpls must be a list type. {str(all_inner_smpls)} of type {str(type(all_inner_smpls))} was passed."
        )
    if not len(all_inner_smpls) == 1:
        raise ValueError("all_inner_smpls must consist of exactly one element.")
    if not isinstance(all_inner_smpls[0], list):
        raise TypeError("all_inner_smpls must be a list of lists.")
    all_lists = all([isinstance(lst, list) for lst in all_inner_smpls[0]])
    if not all_lists:
        raise TypeError("all_inner_smpls must be a list of lists of lists.")
    for all_inner_reps, smpls_rep in zip(all_inner_smpls, smpls):
        n_folds_inner = _check_inner_sample_reps(all_inner_reps, smpls_rep)
    # n_reps_inner = all_inner_smpls[0]
    return all_inner_smpls, n_folds_inner  # , n_reps_inner


# TODO: Check if n_reps_inner needs to be same as n_reps


def _check_inner_sample_reps(all_inner_reps, smpls_rep):
    """
    Check that each repetition is a list of lists of tuples.
    :param inner_smpls:
    :param smpls:
    :param n_obs:
    :return:
    """
    if not isinstance(all_inner_reps, list):
        raise TypeError("inner_smpls must be a list.")

    all_is_partition_inner = all(
        [_check_is_inner_partition(inner_reps, smpls_fold) for inner_reps, smpls_fold in zip(all_inner_reps, smpls_rep)]
    )
    if not all_is_partition_inner:
        raise ValueError("Some of the inner smpls do not partition the training samples")
    n_folds_inner = [len(inner_rep) for inner_rep in all_inner_reps]
    if len(np.unique(n_folds_inner)) != 1:
        raise ValueError("Some of the repetitions contain different number of folds.")
    return n_folds_inner[0]


def _check_inner_sample_fold(fold):
    if not isinstance(fold, tuple):
        raise TypeError("fold must be a tuple.")
    if not len(fold) == 2:
        raise ValueError("fold must be a pair of train and test indices.")


def _check_is_inner_partition(inner_smpls, train_smpls):
    """
    Checks whether the inner smpls are correctly partitioned
    :param inner_smpls:
    :param train_smpls:
    :return:
    """
    test_set = set()
    for _, test_index in inner_smpls:
        temp = set(test_index)
        if not test_set.isdisjoint(temp):
            return False
        test_set |= temp
    return test_set == set(train_smpls[0])
