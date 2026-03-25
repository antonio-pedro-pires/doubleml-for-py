import numpy as np
from sklearn.model_selection import train_test_split

def _normalize_propensity_med(
    normalize_ipw,
    outcome,
    treatment_indicator,
    px_preds,
    pmx_preds = None,
):
    """
        Normalizes propensity scores for causal mediation analysis. Normalizes both potential and counterfactual scores.
    """
    n_obs = len(treatment_indicator)
    if outcome=="potential":
        result = _normalize_potential(normalize_ipw, treatment_indicator, px_preds, n_obs)
    elif outcome == "counterfactual":
        result = _normalize_counterfactual(normalize_ipw, treatment_indicator, px_preds, pmx_preds, n_obs)    
    return result


def _normalize_potential(normalize_ipw, treatment_indicator, px_preds, n_obs):
    """
        Normalizes potential scores in the context of causal mediation analysis.
    """
    if normalize_ipw:
        #TODO: Have to check if only px_preds must be taken.
        sumscore = np.sum(np.divide(treatment_indicator,px_preds))
        result = (np.multiply(np.divide(n_obs,sumscore),np.divide(treatment_indicator, px_preds)))        
    else:
        result = np.divide(treatment_indicator, px_preds)
    return result


def _normalize_counterfactual(normalize_ipw, treatment_indicator, px_preds, pmx_preds, n_obs):
    """
        Normalizes counterfactual scores in the context of causal mediation analysis.
    """

    if normalize_ipw:
        sumscore1 = np.sum(np.divide(np.multiply(treatment_indicator, 1.0-pmx_preds), np.multiply(pmx_preds, 1.0-px_preds)))
        sumscore2 = np.sum(np.divide(1.0-treatment_indicator, 1.0-px_preds))

        ps1 = np.multiply(np.divide(n_obs, sumscore1), np.divide(np.multiply(treatment_indicator, 1.0-pmx_preds), np.multiply(pmx_preds, 1.0-px_preds)))

        ps2 = np.multiply(np.divide(n_obs, sumscore2), np.divide(1.0-treatment_indicator,(1.0-px_preds)))
        result=(ps1, ps2)
    else:
        ps1 = np.divide(np.multiply(treatment_indicator, 1.0-pmx_preds), np.multiply(pmx_preds, 1.0-px_preds))
        ps2 = np.divide(1.0-treatment_indicator,1.0-px_preds)
        result=(ps1, ps2)
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
