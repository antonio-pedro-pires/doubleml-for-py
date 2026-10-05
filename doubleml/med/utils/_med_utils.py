import numpy as np

from doubleml.data import DoubleMLMEDData
from doubleml.utils._checks import _check_smpl_split_tpl


def _normalize_propensity_med(
    normalize_ipw,
    outcome,
    d,
    treatment_level,
    m_preds,
    M_preds=None,
):
    """
    Normalizes propensity scores for causal mediation analysis. Normalizes both factual and counterfactual scores.
    """
    n_obs = len(d)
    if outcome == "factual":
        result = _normalize_factual(normalize_ipw, d, treatment_level, m_preds, n_obs)
    elif outcome == "counterfactual":
        result = _normalize_counterfactual(normalize_ipw, d, treatment_level, m_preds, M_preds, n_obs)
    return result


def _normalize_factual(normalize_ipw, d, treatment_level, m_preds, n_obs):
    """
    Normalizes factual scores in the context of causal mediation analysis.
    """
    if normalize_ipw:
        if treatment_level == 0:
            sumscore = np.sum(np.divide(1.0 - d, 1.0 - m_preds))
            result = np.multiply(np.divide(1.0 - d, 1.0 - m_preds), np.divide(n_obs, sumscore))
        elif treatment_level == 1:
            sumscore = np.sum(np.divide(d, m_preds))
            result = np.multiply(np.divide(d, m_preds), np.divide(n_obs, sumscore))
    else:
        if treatment_level == 0:
            result = np.divide(1.0 - d, 1.0 - m_preds)
        if treatment_level == 1:
            result = np.divide(d, m_preds)
    return result


def _normalize_counterfactual(normalize_ipw, d, treatment_level, m_preds, M_preds, n_obs):
    """
    Normalizes counterfactual scores in the context of causal mediation analysis.
    """

    if normalize_ipw:
        if treatment_level == 0:
            sumscore1 = np.sum(np.divide(np.multiply(1.0 - d, M_preds), np.multiply(1.0 - M_preds, m_preds)))
            sumscore2 = np.sum(np.divide(d, m_preds))

            ps1 = np.multiply(
                np.divide(np.multiply(1.0 - d, M_preds), np.multiply(1.0 - M_preds, m_preds)), np.divide(n_obs, sumscore1)
            )

            ps2 = np.multiply(np.divide(d, m_preds), np.divide(n_obs, sumscore2))

            result = (ps1, ps2)

        elif treatment_level == 1:
            sumscore1 = np.sum(np.divide(np.multiply(d, 1.0 - M_preds), np.multiply(M_preds, 1.0 - m_preds)))
            sumscore2 = np.sum(np.divide(1.0 - d, 1.0 - m_preds))

            ps1 = np.multiply(
                np.divide(np.multiply(d, 1.0 - M_preds), np.multiply(M_preds, 1.0 - m_preds)), np.divide(n_obs, sumscore1)
            )

            ps2 = np.multiply(np.divide(1.0 - d, 1.0 - m_preds), np.divide(n_obs, sumscore2))

            result = (ps1, ps2)

    else:
        if treatment_level == 0:
            ps1 = np.divide(np.multiply(1.0 - d, M_preds), np.multiply(1.0 - M_preds, m_preds))
            ps2 = np.divide(d, m_preds)

            result = (ps1, ps2)
        elif treatment_level == 1:
            ps1 = np.divide(np.multiply(d, 1.0 - M_preds), np.multiply(M_preds, 1.0 - m_preds))
            ps2 = np.divide(1.0 - d, 1.0 - m_preds)

            result = (ps1, ps2)

    return result


def _check_inner_sample_splitting(smpls_inner, smpls, n_obs):
    """
    Check that each repetition is a list of lists of tuples.
    :param smpls_inner:
    :param smpls:
    :param n_obs: total number of observations, used for index-level validation.
    :return:
    """

    if not isinstance(smpls_inner, list):
        raise TypeError("inner_smpls must be a list.")
    if not isinstance(smpls, list):
        raise TypeError("smpls must be a list.")

    if len(smpls_inner) != len(smpls):
        raise ValueError(
            "Incompatible smpls_inner. smpls_inner must have the same number of repetitions as smpls."
            + f"smpls has {str(len(smpls))} repetitions while smpls_inner has {str(len(smpls_inner))} repetitions."
        )

    n_inner_folds_per_rep = [
        _check_reps_smpls_inner(index, rep_inner, rep, n_obs) for index, (rep_inner, rep) in enumerate(zip(smpls_inner, smpls))
    ]
    if len(np.unique(n_inner_folds_per_rep)) != 1:
        raise ValueError("Some reps in smpls_inner contain different amounts of inner_folds.")

    n_folds_per_rep = [len(rep) for rep in smpls_inner]
    if len(np.unique(n_folds_per_rep)) != 1:
        raise ValueError("Some reps in smpls_inner contain different amounts of folds.")

    n_folds_inner = n_inner_folds_per_rep[0]
    return smpls_inner, n_folds_inner


def _check_reps_smpls_inner(index, rep_inner, rep, n_obs):
    if not isinstance(rep_inner, list):
        raise TypeError(
            "smpls_inner must be a list of lists." + f"At least an element of smpls_inner is of type {str(type(rep_inner))}"
        )

    if len(rep_inner) != len(rep):
        raise ValueError(
            "Incompatible smpls. Each smpls_inner rep must have the same number of folds as smpls."
            + f" smpls_inner[{str(index)}] has {str(len(rep_inner))} folds while smpls has {str(len(rep))} folds."
        )

    n_inner_folds = [len(fold) for fold in rep_inner]
    if len(np.unique(n_inner_folds)) != 1:
        raise ValueError("Some of the folds contain different numbers of inner_folds.")

    for fold_smpls_inner, fold_smpls in zip(rep_inner, rep):
        # validate the structure and the individual index arrays before checking the partition
        _check_fold_smpls_inner(fold_smpls_inner, n_obs)
        _check_is_inner_partition(fold_smpls_inner, fold_smpls)
    return n_inner_folds[0]


def _check_fold_smpls_inner(fold_smpls_inner, n_obs):
    if not isinstance(fold_smpls_inner, list):
        raise TypeError(
            "smpls_inner must be a list of lists of lists."
            + f"At least an element of a list of smpls_inner is of type {str(type(fold_smpls_inner))}"
        )

    for inner_fold in fold_smpls_inner:
        if not isinstance(inner_fold, tuple):
            raise TypeError("smpls_inner inner_folds must be a tuple")
        if len(inner_fold) != 2:
            raise ValueError(
                "The smpls_inner inner_fold must contain exactly 2 elements."
                + f"inner_fold with {str(len(inner_fold))} elements found."
            )
        # delegate index-level validation (integer dtype, uniqueness, range, disjoint train/test) to the base validator
        _check_smpl_split_tpl(inner_fold, n_obs, check_intersect=True)


def _check_is_inner_partition(fold, train_smpls):
    """
    Checks whether the inner smpls form a valid partition of the outer training fold.

    The inner folds are obtained by re-sampling across the training fold, so their
    indices must be drawn from the outer training fold (``train_smpls[0]``) and the
    inner test folds must cover it exactly without overlapping.
    :param fold: list of inner (train, test) index tuples.
    :param train_smpls: the outer fold (train, test) tuple it is nested in.
    :return:
    """
    parent_train = set(train_smpls[0])
    test_set = set()
    for inner_train, test_index in fold:
        if not set(inner_train).issubset(parent_train):
            raise ValueError("Invalid smpls_inner. Inner train indices must be a subset of the outer training fold.")
        if not set(test_index).issubset(parent_train):
            raise ValueError("Invalid smpls_inner. Inner test indices must be a subset of the outer training fold.")

        temp = set(test_index)
        if not test_set.isdisjoint(temp):
            raise ValueError(
                "Invalid smpls_inner. Inner test folds must not overlap; they must partition the outer training fold."
            )
        test_set |= temp

    if not test_set == parent_train:
        raise ValueError("Some of the smpls_inner_fold do not partition the training samples")


def _check_med_data(dml_data):
    if not isinstance(dml_data, DoubleMLMEDData):
        raise TypeError(
            "Mediation analysis requires data of type DoubleMLMEDData."
            f" Data of type {str(type(dml_data))} was provided instead."
        )
    if dml_data.n_treat != 1:
        raise ValueError(
            f"Mediation analysis supports exactly one treatment variable, got {dml_data.n_treat}: {dml_data.d_cols}."
        )
    if not all(dml_data.binary_treats):
        raise ValueError("Treatment variables for mediation analysis must be binary and take values 1 or 0.")
    if dml_data.z_cols is not None:
        raise NotImplementedError("instrumental variables for mediation analysis is not yet implemented.")
    return dml_data


def _inner_g_learner_name(i_fold):
    """Name of the entry holding the inner predictions of ml_G for the outer fold ``i_fold``."""
    return f"ml_G_inner_{i_fold}"


def _inner_g_learner_names(n_folds):
    """Names of the entries holding the inner predictions of ml_G, one for each outer fold."""
    return [_inner_g_learner_name(i_fold) for i_fold in range(n_folds)]


def _check_inner_g_external_predictions(external_predictions, n_folds):
    missing = [
        i
        for i in range(n_folds)
        if _inner_g_learner_name(i) not in external_predictions.keys()
        or external_predictions[_inner_g_learner_name(i)] is None
    ]
    if len(missing) > 0:
        raise ValueError(
            "When providing external predictions for ml_G, also inner predictions for all inner folds "
            f"have to be provided (missing: {', '.join([str(i) for i in missing])})."
        )


def _predict_with_method(model, x, method):
    if method == "predict_proba":
        return model.predict_proba(x)[:, 1]
    return getattr(model, method)(x)
