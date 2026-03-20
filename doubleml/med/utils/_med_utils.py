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
    outcome,
    treatment_indicator,
    px_preds,
    pmx_preds = None,
):
    n_obs = len(treatment_indicator)
    if normalize_ipw:
        if outcome == "potential":
            #TODO: Have to check if only px_preds must be taken.
            sumscore = np.sum(np.divide(treatment_indicator,px_preds))
            result = np.multiply(np.divide(n_obs,sumscore),np.divide(treatment_indicator, px_preds))        
        elif outcome == "counterfactual":
            sumscore1 = np.sum(np.divide(np.multiply(treatment_indicator, 1.0-pmx_preds), np.multiply(pmx_preds, 1.0-px_preds)))
            sumscore2 = np.sum(np.divide(1.0-treatment_indicator, 1.0-px_preds))
            ps1 = np.multiply(np.divide(n_obs, sumscore1), np.divide(np.multiply(treatment_indicator, 1.0-pmx_preds), np.multiply(pmx_preds, 1.0-px_preds)))
            ps2 = np.multiply(np.divide(n_obs, sumscore2), np.divide(1.0-treatment_indicator,(1.0-px_preds)))
            result=(ps1, ps2)
    else:
        if outcome=="potential":
            result = np.divide(treatment_indicator, px_preds)
        elif outcome=="counterfactual":
            ps1 = np.divide(np.multiply(treatment_indicator, 1.0-pmx_preds), np.multiply(pmx_preds, 1.0-px_preds))
            ps2 = np.divide(1.0-treatment_indicator,1.0-px_preds)
            result=(ps1, ps2)
    return result


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

def _check_inner_sample_splitting(smpls_inner, smpls):
    """
    Check that each repetition is a list of lists of tuples.
    :param inner_smpls:
    :param smpls:
    :param n_obs:
    :return:
    """

    if not isinstance(smpls_inner, list):
        raise TypeError("inner_smpls must be a list.")
    if not isinstance(smpls, list):
        raise TypeError("smpls must be a list.")
    
    if len(smpls_inner)!=len(smpls):
        raise ValueError("Incompatible smpls_inner. smpls_inner must have the same number of repetitions as smpls."
                         +f"smpls has {str(len(smpls))} repetitions while smpls_inner has {str(len(smpls_inner))} repetitions.")

    n_inner_folds_per_rep =[_check_reps_smpls_inner(index, rep_inner, rep) for index, (rep_inner, rep) in enumerate(zip(smpls_inner, smpls))]
    if len(np.unique(n_inner_folds_per_rep))!=1:
        raise ValueError("Some reps in smpls_inner contain different amounts of inner_folds.")
    
    n_folds_per_rep=[len(rep) for rep in smpls_inner]
    if len(np.unique(n_folds_per_rep))!=1:
        raise ValueError("Some reps in smpls_inner contain different amounts of folds.")
    
    n_folds_inner = n_inner_folds_per_rep[0]
    return smpls_inner, n_folds_inner

def _check_reps_smpls_inner(index, rep_inner, rep):
    if not isinstance(rep_inner, list):
        raise TypeError("smpls_inner must be a list of lists." +f"At least an element of smpls_inner is of type {str(type(rep))}")
    
    if len(rep_inner)!=len(rep):
        raise ValueError("Incompatible smpls. Each smpls_inner rep must have the same number of folds as smpls."
                         +f" smpls_inner[{str(index)}] has {str(len(rep_inner))} folds while smpls has {str(len(rep))} folds.")
    
    n_inner_folds = [len(fold) for fold in rep_inner]
    if len(np.unique(n_inner_folds))!=1:
        raise ValueError("Some of the folds contain different numbers of inner_folds.")

    for fold_smpls_inner, fold_smpls in zip(rep_inner, rep):
        _check_is_inner_partition(fold_smpls_inner, fold_smpls) 
        _check_fold_smpls_inner(fold_smpls_inner)
    return n_inner_folds[0]
  
def _check_fold_smpls_inner(fold_smpls_inner):
    if not isinstance(fold_smpls_inner, list):
        raise TypeError("smpls_inner must be a list of lists of lists." 
                        +f"At least an element of a list of smpls_inner is of type {str(type(fold_smpls_inner))}")        

    for inner_fold in fold_smpls_inner:
        if len(inner_fold)!=2:
            raise ValueError("The smpls_inner inner_fold must contain exactly 2 elements."+f"inner_fold with {str(len(fold_smpls_inner))} elements found.")
        if not isinstance(inner_fold, tuple):
            raise TypeError("smpls_inner inner_folds must be a tuple")
        
def _check_is_inner_partition(fold, train_smpls):
    """
    Checks whether the inner smpls are correctly partitioned
    :param inner_smpls:
    :param train_smpls:
    :return:
    """
    test_set = set()
    for (_, test_index) in fold:
        temp = set(test_index)
        if not test_set.isdisjoint(temp):
            return False
        test_set |= temp

    if not test_set == set(train_smpls[0]):
        raise ValueError("Some of the smpls_inner_fold do not partition the training samples")
