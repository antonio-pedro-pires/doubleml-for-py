import numpy as np
from joblib import Parallel, delayed
from sklearn import clone
from sklearn.model_selection import train_test_split, cross_val_predict


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
    train_indices=tuple([train_index for (train_index, _) in smpls])
    test_indices=tuple([test_index for (_, test_index) in smpls])
    return train_indices, test_indices

def split_smpls(smpls, smpls_ratio=0.5, ):
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
    return subsample1, subsample2

def recombine_samples(subsmpls1, subsmpls2,):
    # Take only the samples of interest and recombine them.
        # Need indexes to know which sample to recombine
        # Loop through each smpls to get the targeted sample
        # Create new samples made up of subsamples
    result=[]
    for s1, s2 in zip(subsmpls1, subsmpls2):
        result.append((s1, s2))
    return result

def _fit(estimator, x, y, train_index, idx=None):
    estimator.fit(x[train_index, :], y[train_index])
    return estimator, idx

def fit_predict_efficient_alt(estimator, x, y, smpls=None, n_jobs=None, est_params=None, method="predict", return_train_preds=False, return_models=False):
    res = {"models": None}

    y_list = [y] * len(smpls)
    n_obs = x.shape[0]

    parallel = Parallel(n_jobs=n_jobs, verbose=0, pre_dispatch="2*n_jobs")

    fitted_models= parallel(
        delayed(_fit)(clone(estimator), x, y_list[idx], train_index, idx)
                                         for idx, (train_index, test_index) in enumerate(smpls)
    )

    preds = np.full(n_obs, np.nan)
    targets = np.full(n_obs, np.nan)
    train_preds = list()
    train_targets = list()
    for idx, (train_index, test_index) in enumerate(smpls):
        assert idx == fitted_models[idx][1]
        pred_fun = getattr(fitted_models[idx][0], method)
        if method == "predict_proba":
            preds[test_index] = pred_fun(x[test_index, :])[:, 1]
        else:
            preds[test_index] = pred_fun(x[test_index, :])

        targets[test_index] = y[test_index]

        if return_train_preds:
            train_preds.append(pred_fun(x[train_index, :]))
            train_targets.append(y[train_index])

    res["preds"] = preds
    res["targets"] = targets


    return res
