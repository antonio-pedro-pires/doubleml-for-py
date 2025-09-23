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

def separate_samples_for_nested_estimator(smpls, smpls_ratio,):
    if ((smpls is None) or (not smpls)):
        raise ValueError("the smpls array is empty")
    if smpls_ratio == None:
        raise ValueError("smpls_ratio must be a float between 0.0 and 1.0")

    results = []
    for idx, (train_index, test_index) in enumerate(smpls):
        deltasample, musample = train_test_split(train_index, test_size=smpls_ratio)
        results.append((deltasample, musample, train_index, test_index))
    return results

def fit_predict_nested_estimator(estimator, x, y, smpls=None, est_params=None):
    # Separate smpls into delta, mu, test, train samples.
    # Fit estimators using necessary smpls (may reuse _dml_cv_predict function for that)
    # Predict using necessary smpls
    # Return with the given structure
    pass