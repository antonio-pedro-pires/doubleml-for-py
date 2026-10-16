"""Helpers for the mediation estimators: propensity normalization, inner sample splitting and external predictions."""

from typing import Any, Optional, Union, cast

import numpy as np

from doubleml.data import DoubleMLMEDData
from doubleml.utils._checks import _check_smpl_split_tpl
from doubleml.utils._estimation import _double_dml_cv_predict

VALID_OUTCOMES = ("factual", "counterfactual")
VALID_SCORES = ("efficient-alt",)


def _normalize_propensity_med(
    normalize_ipw: bool,
    outcome: str,
    d: np.ndarray,
    treatment_level: int,
    m_preds: np.ndarray,
    M_preds: Optional[np.ndarray] = None,
) -> Union[np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """
    Compute the (optionally normalized) inverse probability weights of the factual and the counterfactual score.

    Parameters
    ----------
    normalize_ipw : bool
        Indicates whether the weights are normalized to add up to the number of observations.
    outcome : str
        ``'factual'`` or ``'counterfactual'``.
    d : np.ndarray
        Treatment indicator of shape ``(n_obs,)``.
    treatment_level : int
        The treatment level :math:`d`, 0 or 1.
    m_preds : np.ndarray
        Predictions of :math:`P(D=1|X)` of shape ``(n_obs,)``.
    M_preds : np.ndarray or None
        Predictions of :math:`P(D=1|M,X)` of shape ``(n_obs,)``. Only used if ``outcome`` is ``'counterfactual'``.

    Returns
    -------
    np.ndarray or tuple of np.ndarray
        The weights of shape ``(n_obs,)`` for ``'factual'``, the two weights of the counterfactual score for
        ``'counterfactual'``.
    """
    n_obs = len(d)
    if outcome == "factual":
        return _normalize_factual(normalize_ipw, d, treatment_level, m_preds, n_obs)
    # M_preds is required for the counterfactual score, which is guaranteed by DoubleMLMED
    return _normalize_counterfactual(normalize_ipw, d, treatment_level, m_preds, cast(np.ndarray, M_preds), n_obs)


def _scale_to_n_obs(weights: np.ndarray, n_obs: int) -> np.ndarray:
    """Scale the weights by ``n_obs / sum(weights)``, so that they add up to the number of observations."""
    return np.multiply(weights, np.divide(n_obs, np.sum(weights)))


def _normalize_factual(
    normalize_ipw: bool, d: np.ndarray, treatment_level: int, m_preds: np.ndarray, n_obs: int
) -> np.ndarray:
    """
    Compute the weight of the factual score, :math:`D / P(D=1|X)` or :math:`(1-D) / (1-P(D=1|X))`.

    Parameters
    ----------
    normalize_ipw : bool
        Indicates whether the weights are normalized to add up to ``n_obs``.
    d : np.ndarray
        Treatment indicator of shape ``(n_obs,)``.
    treatment_level : int
        The treatment level, 0 or 1.
    m_preds : np.ndarray
        Predictions of :math:`P(D=1|X)` of shape ``(n_obs,)``.
    n_obs : int
        The number of observations.

    Returns
    -------
    np.ndarray
        The weights of shape ``(n_obs,)``.
    """
    if treatment_level == 0:
        weights = np.divide(1.0 - d, 1.0 - m_preds)
    else:
        weights = np.divide(d, m_preds)
    return _scale_to_n_obs(weights, n_obs) if normalize_ipw else weights


def _normalize_counterfactual(
    normalize_ipw: bool, d: np.ndarray, treatment_level: int, m_preds: np.ndarray, M_preds: np.ndarray, n_obs: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the two weights of the counterfactual score.

    The first weight multiplies the residual of ``ml_G``, the second weight the residual of ``ml_nested_g``. Each of the
    weights is normalized with its own sum.

    Parameters
    ----------
    normalize_ipw : bool
        Indicates whether the weights are normalized to add up to ``n_obs``.
    d : np.ndarray
        Treatment indicator of shape ``(n_obs,)``.
    treatment_level : int
        The treatment level, 0 or 1.
    m_preds : np.ndarray
        Predictions of :math:`P(D=1|X)` of shape ``(n_obs,)``.
    M_preds : np.ndarray
        Predictions of :math:`P(D=1|M,X)` of shape ``(n_obs,)``.
    n_obs : int
        The number of observations.

    Returns
    -------
    tuple of np.ndarray
        The two weights, each of shape ``(n_obs,)``.
    """
    if treatment_level == 0:
        ps1 = np.divide(np.multiply(1.0 - d, M_preds), np.multiply(1.0 - M_preds, m_preds))
        ps2 = np.divide(d, m_preds)
    else:
        ps1 = np.divide(np.multiply(d, 1.0 - M_preds), np.multiply(M_preds, 1.0 - m_preds))
        ps2 = np.divide(1.0 - d, 1.0 - m_preds)

    if normalize_ipw:
        return _scale_to_n_obs(ps1, n_obs), _scale_to_n_obs(ps2, n_obs)
    return ps1, ps2


def _check_inner_sample_splitting(smpls_inner: list, smpls: list, n_obs: int) -> tuple[list, int]:
    """
    Check that each repetition is a list of lists of tuples.

    Parameters
    ----------
    smpls_inner : list
        The inner sample splitting, one list of inner folds for each outer fold of each repetition.
    smpls : list
        The outer sample splitting.
    n_obs : int
        Total number of observations, used for index-level validation.

    Returns
    -------
    tuple of (list, int)
        The validated ``smpls_inner`` and the number of inner folds.
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


def _check_reps_smpls_inner(index: int, rep_inner: list, rep: list, n_obs: int) -> int:
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


def _check_fold_smpls_inner(fold_smpls_inner: list, n_obs: int) -> None:
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


def _check_is_inner_partition(fold: list, train_smpls: tuple) -> None:
    """
    Checks whether the inner smpls form a valid partition of the outer training fold.

    The inner folds are obtained by re-sampling across the training fold, so their
    indices must be drawn from the outer training fold (``train_smpls[0]``) and the
    inner test folds must cover it exactly without overlapping. A single inner tuple is the pair of
    subsamples ``(A, B)``, which must together cover the outer training fold.
    :param fold: list of inner (train, test) index tuples.
    :param train_smpls: the outer fold (train, test) tuple it is nested in.
    :return:
    """
    parent_train = set(train_smpls[0])
    if len(fold) == 1:
        inner_train, test_index = fold[0]
        if not (set(inner_train) | set(test_index)) == parent_train:
            raise ValueError(
                "Invalid smpls_inner. With a single inner tuple (A, B), the subsamples A and B must together cover "
                "the outer training fold."
            )
        return
    test_set: set[int] = set()
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


_SUBSAMPLE_ARM_HINT = " Redraw the sample splitting, use fewer folds or pass a stratified smpls_inner to set_sample_splitting."


def _check_subsample_arms(smpls_inner_rep: list, treated: np.ndarray, treatment_level: int) -> None:
    """Checks that ml_G finds treated units in subsample A and ml_nested_g untreated units in subsample B."""
    for i_fold, inner_fold in enumerate(smpls_inner_rep):
        subsample_a, subsample_b = inner_fold[0]
        n_treated_a = int(np.sum(treated[subsample_a]))
        if n_treated_a == 0:
            raise ValueError(
                f"ml_G requires at least one observation with D={treatment_level} in subsample A of outer fold {i_fold}."
                f" Found 0 of {len(subsample_a)} observations." + _SUBSAMPLE_ARM_HINT
            )
        n_untreated_b = int(np.sum(~treated[subsample_b]))
        if n_untreated_b == 0:
            raise ValueError(
                f"ml_nested_g requires at least one observation with D!={treatment_level} in subsample B of outer fold"
                f" {i_fold}. Found 0 of {len(subsample_b)} observations." + _SUBSAMPLE_ARM_HINT
            )


def _double_dml_cv_predict_by_fold(
    estimator: Any,
    estimator_name: str,
    x: np.ndarray,
    y: np.ndarray,
    smpls: list,
    smpls_inner: list,
    n_jobs: Optional[int],
    est_params: Optional[list],
    method: str,
) -> dict[str, Any]:
    """
    Calls ``_double_dml_cv_predict`` with the parameters of each outer fold.

    The parameters of a learner are stored with one entry for each outer fold, whereas ``_double_dml_cv_predict`` hands
    its ``est_params`` to the inner fits of every outer fold, which expect one entry for each inner fold.
    """
    if est_params is None:
        return _double_dml_cv_predict(
            estimator, estimator_name, x, y, smpls=smpls, smpls_inner=smpls_inner, n_jobs=n_jobs, method=method
        )

    res: dict[str, Any] = {
        "preds": np.zeros(y.shape, dtype=float),
        "preds_inner": [],
        "targets_inner": [],
        "models": [],
        "targets": np.copy(y),
    }
    for fold_smpls, fold_smpls_inner, fold_params in zip(smpls, smpls_inner, est_params):
        res_fold = _double_dml_cv_predict(
            estimator,
            estimator_name,
            x,
            y,
            smpls=[fold_smpls],
            smpls_inner=[fold_smpls_inner],
            n_jobs=n_jobs,
            est_params=fold_params,
            method=method,
        )
        # the predictions of a fold are zero outside of the observations of the fold
        res["preds"] += res_fold["preds"]
        for key in ["preds_inner", "targets_inner", "models"]:
            res[key].extend(res_fold[key])
    return res


def _check_med_data(dml_data: DoubleMLMEDData) -> DoubleMLMEDData:
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


def _first_inner_split(smpls_inner: list) -> list:
    """Keep only the first inner tuple, the (A, B) pair, of every outer fold."""
    return [[[inner_fold[0]] for inner_fold in rep] for rep in smpls_inner]


def _inner_g_learner_name(i_fold: int) -> str:
    """Name of the entry holding the inner predictions of ml_G for the outer fold ``i_fold``."""
    return f"ml_G_inner_{i_fold}"


def _inner_g_learner_names(n_folds: int) -> list[str]:
    """Names of the entries holding the inner predictions of ml_G, one for each outer fold."""
    return [_inner_g_learner_name(i_fold) for i_fold in range(n_folds)]


def _check_inner_g_external_predictions(external_predictions: dict[str, Any], n_folds: int) -> None:
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


def _check_inner_g_external_values(inner_predictions: list, nested_smpls: list) -> None:
    """Checks that the external entries ml_G_inner_i are finite on the observations where ml_nested_g is fitted."""
    for i_fold, (preds, (nested_train, _)) in enumerate(zip(inner_predictions, nested_smpls)):
        n_not_finite = int(np.sum(~np.isfinite(preds[nested_train])))
        if n_not_finite > 0:
            raise ValueError(
                f"The external predictions {_inner_g_learner_name(i_fold)} have to be finite for the observations on which "
                f"ml_nested_g is fitted in the outer fold {i_fold}, but {n_not_finite} of the {len(nested_train)} "
                "observations are not finite."
            )


def _predict_with_method(model: Any, x: np.ndarray, method: str) -> np.ndarray:
    """Predict with ``method``, returning the probability of the positive class for ``predict_proba``."""
    if method == "predict_proba":
        return model.predict_proba(x)[:, 1]
    return getattr(model, method)(x)
