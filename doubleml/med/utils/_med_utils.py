import numpy as np
from sklearn.model_selection import train_test_split


# TODO: refactor so that it takes multiple columns and multiple conditions for trimming.
def _trim_probabilities(preds, trimming_threshold=None, method=None, conditions=None):
    if preds.ndim == 1:
        if conditions!=None:
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

def extract_sets_from_smpls(smpls,):
    """
    Separates the train and test indices from smpls and returns them
    """
    train_smpls=[]
    test_smpls=[]
    for _, (train_idx, test_idx) in enumerate(smpls):
        train_smpls.append(train_idx)
        test_smpls.append(test_idx)
    return [train_smpls, test_smpls]

def split_smpls(smpls, smpls_ratio, ):
    """
    Splits sample into two subsamples for the estimation of the nested estimator used with the efficient-alt scoring function.

    Parameters
    ----------
    smpls_ratio : float
        Describes the ratio of observations in the musample

    Returns
    -------
    results : a list of tuples of ndarrays
        Contains the indexes of the subsamples (mu, delta, train and test)
    """
    if ((smpls is None) or (not smpls)):
        raise ValueError("the smpls array is empty")
    if smpls_ratio == None:
        raise ValueError("smpls_ratio must be a float between 0.0 and 1.0")

    results = []
    subsample1=[]
    subsample2=[]
    for smpl in smpls:
        subsample1_idx, subsample2_idx = train_test_split(smpl, test_size=smpls_ratio)
        subsample1.append(subsample1_idx)
        subsample2.append(subsample2_idx)
    results.append((subsample1, subsample2))
    return results

def recombine_samples(subsmpls1, subsmpls2,):
    # Take only the samples of interest and recombine them.
        # Need indexes to know which sample to recombine
        # Loop through each smpls to get the targeted sample
        # Create new samples made up of subsamples
    result=[]
    for s1, s2 in zip(subsmpls1, subsmpls2):
        result.append((s1, s2))
    return result