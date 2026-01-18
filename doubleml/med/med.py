import warnings

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y

from doubleml import DoubleMLMEDData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import _check_inner_sample_splitting
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import (
    _cond_targets,
    _dml_cv_predict,
    _double_dml_cv_predict,
    _get_cond_smpls,
)
from doubleml.utils._tune_optuna import _dml_tune_optuna


# TODO: remove yx_learner in counterfactual nuisance estimation dependent on how trimming is applied
class DoubleMLMED(LinearScoreMixin, DoubleML):
    """Double machine learning for causal mediation analysis.

    Parameters
    ----------
    med_data : :class:`DoubleMLMediationData` object
        The :class:`DoubleMLMediationData` object providing the data and specifying the variables for the causal model.

    ml_yx : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`E[Y|D,X]`.

    ml_px : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`P(D=d|X)`.

    ml_ymx : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods for the nuisance function :math:`E[Y|D,M,X]`.
        Only required if ``target`` is 'counterfactual'.

    ml_pmx : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods for the nuisance function :math:`P(D=d|M,X)`.
        Only required if ``target`` is 'counterfactual'.

    ml_nested : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods for the nested outcome.
        Only required if ``target`` is 'counterfactual'.

    target : str
        The target parameter to estimate.
        - 'potential': Estimate the potential outcome :math:`E[Y(d, M(d))]`.
        - 'counterfactual': Estimate the counterfactual outcome :math:`E[Y(d, M(d'))]`.
        Default is 'potential'.

    treatment_level : int
        The treatment level :math:`d` for the potential outcome.

    mediation_level : int
        The treatment level :math:`d'` for the mediator.
        Only required if ``target`` is 'counterfactual'.

    score : str
        The score function to use.
        Default is 'MED'.

    score_function : str
        The specific score type.
        Default is 'efficient-alt'.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``False``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.
    """

    def __init__(
        self,
        med_data,
        ml_px,
        ml_yx=None,
        ml_ymx=None,
        ml_pmx=None,
        ml_nested=None,
        target="potential",
        treatment_level=1,
        mediation_level=1,
        score="MED",
        score_function="efficient-alt",
        n_folds=5,
        n_rep=1,
        n_folds_inner=5,
        normalize_ipw=False,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        draw_sample_splitting=True,
    ):
        self._med_data = med_data
        if not isinstance(med_data, DoubleMLMEDData):
            raise TypeError(
                "Mediation analysis requires data of type DoubleMLMediationData."
                + f"data of type {str(type(med_data))} was provided instead."
            )

        if target == "potential" and ml_yx is None:
            raise ValueError("ml_yx is required when target is 'potential'")
        self.n_folds_inner = n_folds_inner
        super().__init__(med_data, n_folds, n_rep, score, draw_sample_splitting, double_sample_splitting=True)

        valid_targets = ["potential", "counterfactual"]
        if target not in valid_targets:
            raise ValueError(f"Invalid target {target}. " + "Valid targets " + " or ".join(valid_targets) + ".")
        self._target = target

        valid_score = ["MED"]
        self._score = score
        _check_score(score, valid_score)

        self._score_function = score_function
        valid_scores_types = ["efficient", "efficient-alt"]
        _check_score(score_function, valid_scores_types)

        self._treatment_level = treatment_level
        self._mediation_level = mediation_level

        self._treated = self._med_data.d == treatment_level
        self._mediated = self._med_data.m == mediation_level

        if self._target == "potential":
            self._learner = {"ml_yx": ml_yx, "ml_px": ml_px}
        else:
            self._learner = {"ml_px": ml_px, "ml_ymx": ml_ymx, "ml_pmx": ml_pmx, "ml_nested": ml_nested}
        self._check_learners()

        self._initialize_ml_nuisance_params()

        self._normalize_ipw = normalize_ipw
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        self._external_predictions_implemented = True

    @property
    def target(self):
        """
        The target parameter to estimate.
        """
        return self._target

    @property
    def normalize_ipw(self):
        """
        indicates whether the  inverse probability weights are normalized
        """
        return self._normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Indicates the trimming threshold.
        """
        return self._trimming_threshold

    @property
    def treatment_level(self):
        """
        Chosen treatment level.
        """
        return self._treatment_level

    @property
    def mediation_level(self):
        """
        Chosen mediation level.
        """
        return self._mediation_level

    @property
    def treated(self):
        """
        Indicator for observations with chosen treatment level.
        """
        return self._treated

    @property
    def mediated(self):
        """
        Indicator for observations with chosen mediation level.
        """
        return self._mediated

    @property
    def _score_element_names(self):
        return ["psi_a", "psi_b"]

    def _check_learners(self):
        if self._target == "potential":
            required_learners = ["ml_yx", "ml_px"]
        else:
            required_learners = ["ml_px", "ml_ymx", "ml_pmx", "ml_nested"]

        for learner in required_learners:
            if self._learner[learner] is None:
                raise ValueError(f"Learner {learner} is required for target {self._target}.")

        self._predict_method = {}
        for key, learner in self._learner.items():
            if learner is not None:
                if key in ["ml_px", "ml_pmx"]:
                    _ = self._check_learner(learner, key, regressor=False, classifier=True)
                    self._predict_method[key] = "predict_proba"
                else:
                    is_classifier_ = self._check_learner(learner, key, regressor=True, classifier=True)
                    if self._med_data.binary_outcome:
                        if is_classifier_:
                            self._predict_method[key] = "predict_proba"
                        else:
                            raise ValueError(f"Learner {learner} must be classifier.")
                    else:
                        if is_classifier_:
                            raise ValueError(f"Learner {learner} must be regressor.")
                        else:
                            self._predict_method[key] = "predict"

    def _initialize_ml_nuisance_params(self):
        if self._target == "potential":
            learners = ["ml_yx", "ml_px"]
            params_names = learners
        else:
            learners = ["ml_px", "ml_ymx", "ml_pmx", "ml_nested"]
            inner_ymx_names = [f"ml_ymx_inner_{i}" for i in range(self.n_folds)]
            params_names = learners + inner_ymx_names

        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in params_names}

    def _nuisance_est(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models=False,
    ):
        x, y = check_X_y(self._med_data.x, self._med_data.y, ensure_all_finite=True)
        x, d = check_X_y(x, self._med_data.d, ensure_all_finite=True)

        if self._target == "potential":
            # Check whether there are external predictions for each parameter.
            px_external = external_predictions["ml_px"] is not None
            yx_external = external_predictions["ml_yx"] is not None

            # Prepare the samples
            _, smpls_d1 = _get_cond_smpls(smpls, self.treated)

            if px_external:
                px_hat = {"preds": external_predictions["ml_px"], "targets": None, "models": None}
            else:
                px_hat = _dml_cv_predict(
                    self._learner["ml_px"],
                    x,
                    self.treated,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_px"),
                    method=self._predict_method["ml_px"],
                    return_models=return_models,
                )

            if yx_external:
                yx_hat = {
                    "preds": external_predictions["ml_yx"],
                    "targets": _cond_targets(y, cond_sample=(self.treated == 1)),
                    "models": None,
                }
            else:
                yx_hat = _dml_cv_predict(
                    self._learner["ml_yx"],
                    x,
                    y,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_yx"),
                    method=self._predict_method["ml_yx"],
                    return_models=return_models,
                )
            _check_finite_predictions(yx_hat["preds"], self._learner["ml_yx"], "ml_yx", smpls)
            # adjust target values to consider only compatible subsamples
            yx_hat["targets"] = _cond_targets(yx_hat["targets"], cond_sample=(self.treated == 1))

            preds = {
                "predictions": {
                    "ml_px": px_hat["preds"],
                    "ml_yx": yx_hat["preds"],
                },
                "targets": {
                    "ml_px": px_hat["targets"],
                    "ml_yx": yx_hat["targets"],
                },
                "models": {
                    "ml_px": px_hat["models"],
                    "ml_yx": yx_hat["models"],
                },
            }

            psi_a, psi_b = self._score_elements(y, px_hat["preds"], yx_hat["preds"])
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        else:  # target == "counterfactual"
            x, m = check_X_y(x, self._med_data.m, ensure_all_finite=True)
            xm = np.column_stack((x, m))

            # Check whether there are external predictions for each parameter.
            px_external = external_predictions["ml_px"] is not None
            pmx_external = external_predictions["ml_pmx"] is not None
            ymx_external = external_predictions["ml_ymx"] is not None
            nested_external = external_predictions["ml_nested"] is not None

            # Prepare the samples
            smpls_d0, smpls_d1 = _get_cond_smpls(smpls, self.treated)

            # Estimate the probability of treatment conditional on the covariates.
            if px_external:
                px_hat = {"preds": external_predictions["ml_px"], "targets": None, "models": None}
            else:
                px_hat = _dml_cv_predict(
                    self._learner["ml_px"],
                    x,
                    d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_px"),
                    method=self._predict_method["ml_px"],
                    return_models=return_models,
                )

            # Estimate the probability of treatment conditional on the mediator and covariates.
            if pmx_external:
                pmx_hat = {"preds": external_predictions["ml_pmx"], "targets": None, "models": None}
            else:
                pmx_hat = _dml_cv_predict(
                    self._learner["ml_pmx"],
                    xm,
                    d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_pmx"),
                    method=self._predict_method["ml_pmx"],
                    return_models=return_models,
                )

            if ymx_external:
                # expect per-inner-fold keys ml_ymx_inner_i
                missing = [
                    i
                    for i in range(self.n_folds_inner)
                    if (f"ml_ymx_inner_{i}") not in external_predictions.keys()
                    or external_predictions[f"ml_ymx_inner_{i}"] is None
                ]
                if len(missing) > 0:
                    raise ValueError(
                        "When providing external predictions for ml_ymx, also inner predictions for all inner folds "
                        f"have to be provided (missing: {', '.join([str(i) for i in missing])})."
                    )
                ymx_hat_inner = [external_predictions[f"ml_ymx_inner_{i}"] for i in range(self.n_folds_inner)]
                ymx_hat = {
                    "preds": external_predictions["ml_ymx"],
                    "preds_inner": ymx_hat_inner,
                    "targets": self._dml_data.y,
                    "models": None,
                }
            else:
                # TODO: Have to filter and only have observations where d=1
                ymx_hat = _double_dml_cv_predict(
                    estimator=self._learner["ml_ymx"],
                    estimator_name=self._learner["ml_ymx"],
                    x=xm,
                    y=y,
                    smpls=smpls_d1,
                    smpls_inner=self._DoubleML__smpls__inner,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_ymx"),
                    method=self._predict_method["ml_ymx"],
                )

            if nested_external:
                nested_hat = {"preds": external_predictions["ml_nested"], "targets": None, "models": None}
            else:
                ymx_inner_preds = np.zeros_like(y)
                for pred, (train, test) in zip(ymx_hat["preds_inner"], smpls):
                    ymx_inner_preds[train] += pred[train]
                ymx_inner_preds /= len(ymx_hat["preds_inner"])

                nested_hat = _dml_cv_predict(
                    self._learner["ml_nested"],
                    xm,
                    ymx_inner_preds,
                    smpls=smpls_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_nested"),
                    method=self._predict_method["ml_nested"],
                    return_models=return_models,
                )

            preds = {
                "predictions": {
                    "ml_px": px_hat["preds"],
                    "ml_pmx": pmx_hat["preds"],
                    "ml_ymx": ymx_hat["preds"],
                    "ml_nested": nested_hat["preds"],
                    # store inner predictions as separate keys per inner fold
                    # ml_M inner
                    **{f"ml_ymx_inner_{i}": ymx_hat["preds_inner"][i] for i in range(len(ymx_hat["preds_inner"]))},
                },
                "targets": {
                    "ml_px": px_hat["targets"],
                    "ml_pmx": pmx_hat["targets"],
                    "ml_ymx": ymx_hat["targets"],
                    "ml_nested": nested_hat["targets"],
                    **(
                        {
                            f"ml_ymx_inner_{i}": (
                                ymx_hat.get("targets_inner")[i]
                                if ymx_hat.get("targets_inner") is not None and i < len(ymx_hat["targets_inner"])
                                else None
                            )
                            for i in range(len(ymx_hat.get("preds_inner", [])))
                        }
                    ),
                },
                "models": {
                    "ml_px": px_hat["models"],
                    "ml_pmx": pmx_hat["models"],
                    "ml_ymx": ymx_hat["models"],
                    "ml_nested": nested_hat["models"],
                },
            }

            psi_a, psi_b = self._score_elements(
                y, px_hat=px_hat["preds"], pmx_hat=pmx_hat["preds"], ymx_hat=ymx_hat["preds"], nested_hat=nested_hat["preds"]
            )
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_elements(self, y, px_hat, yx_hat=None, pmx_hat=None, ymx_hat=None, nested_hat=None):
        if self._target == "potential":
            u_hat = y - yx_hat
            psi_a = -1.0
            psi_b = np.multiply(np.divide(self.treated, px_hat), u_hat) + yx_hat
        else:
            u_hat = y - ymx_hat
            w_hat = ymx_hat - nested_hat
            psi_a = -1.0

            t1 = np.multiply(
                np.multiply(np.divide(self.treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)),
                u_hat,
            )
            t2 = np.multiply(np.divide(1.0 - self.treated, 1.0 - px_hat), w_hat)
            psi_b = t1 + t2 + nested_hat

        return psi_a, psi_b

    def _nuisance_tuning(
        self,
        smpls,
        param_grids,
        scoring_methods,
        n_folds_tune,
        n_jobs_cv,
        search_mode,
        n_iter_randomized_search,
    ):
        raise NotImplementedError(
            "Nuisance tuning is not implemented for DoubleMLMediation. Please use the method _nuisance_tuning_optuna instead."
        )

    def _nuisance_tuning_optuna(self, optuna_params, scoring_methods, cv, optuna_settings):
        x, y = check_X_y(self._med_data.x, self._med_data.y, ensure_all_finite=True)
        x, d = check_X_y(x, self._med_data.d, ensure_all_finite=True)

        if scoring_methods is None:
            scoring_methods = {"ml_yx": None, "ml_px": None, "ml_ymx": None, "ml_pmx": None, "ml_nested": None}

        treated_mask = self.treated
        not_treated_mask = np.logical_not(treated_mask)

        px_tune_res = _dml_tune_optuna(
            y=d,
            x=x,
            learner=self._learner["ml_px"],
            param_grid_func=optuna_params["ml_px"],
            scoring_method=scoring_methods["ml_px"],
            cv=cv,
            optuna_settings=optuna_settings,
            learner_name="ml_px",
            params_name="ml_px",
        )

        if self.target == "potential":
            yx_tune_res = _dml_tune_optuna(
                y=y[treated_mask],
                x=x[treated_mask],
                learner=self._learner["ml_yx"],
                param_grid_func=optuna_params["ml_yx"],
                scoring_method=scoring_methods["ml_yx"],
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_yx",
                params_name="ml_yx",
            )

            results = {
                "ml_px": px_tune_res,
                "ml_yx": yx_tune_res,
            }

        else:
            x, m = check_X_y(x, self._med_data.m, ensure_all_finite=False)
            xm = np.column_stack((x, m))

            pmx_tune_res = _dml_tune_optuna(
                y=d,
                x=xm,
                learner=self._learner["ml_pmx"],
                param_grid_func=optuna_params["ml_pmx"],
                scoring_method=scoring_methods["ml_pmx"],
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_pmx",
                params_name="ml_pmx",
            )

            ymx_tune_res = _dml_tune_optuna(
                y=y[treated_mask],
                x=xm[treated_mask],
                learner=self._learner["ml_ymx"],
                param_grid_func=optuna_params["ml_ymx"],
                scoring_method=scoring_methods.get("ml_ymx"),
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_ymx",
                params_name="ml_ymx",
            )

            # Prepare targets for the nested estimator
            ymx_hat = cross_val_predict(
                estimator=self._learner["ml_nested"],
                X=xm,
                y=y,
                cv=cv,
                method=self._predict_method["ml_nested"],
            )

            nested_tune_res = _dml_tune_optuna(
                y=ymx_hat[not_treated_mask],
                x=xm[not_treated_mask],
                learner=self._learner["ml_nested"],
                param_grid_func=optuna_params["ml_nested"],
                scoring_method=scoring_methods["ml_nested"],
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_nested",
                params_name="ml_nested",
            )
            results = {
                "ml_px": px_tune_res,
                "ml_ymx": ymx_tune_res,
                "ml_pmx": pmx_tune_res,
                "ml_nested": nested_tune_res,
            }

        return results

    def _sensitivity_element_est(self, preds):
        pass

    def _set_smpls_inner_splitting(self, all_inner_smpls):
        self._smpls_inner, self.n_folds_inner = _check_inner_sample_splitting(all_inner_smpls, self.smpls)


# TODO: Transplant methods into utils documents.
# TODO: Apply threshold
class DoubleMLMEDP(DoubleMLMED):
    """
    Double machine learning for the estimation of the potential outcome in causal mediation analysis.

    .. deprecated:: 0.10.0
        The class `DoubleMLMEDP` is deprecated and will be removed in a future version.
        Please use `DoubleMLMediation` with `target='potential'` instead.
    """

    def __init__(
        self,
        med_data,
        treatment_level,
        ml_g,
        ml_m,
        score="MED",
        score_function="efficient-alt",
        n_folds=5,
        n_rep=1,
        normalize_ipw=False,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        order=1,
        fewsplits=False,
        draw_sample_splitting=True,
    ):
        warnings.warn(
            "The class `DoubleMLMEDP` is deprecated and will be removed in a future version. "
            "Please use `DoubleMLMediation` with `target='potential'` instead.",
            FutureWarning,
            stacklevel=2,
        )

        super().__init__(
            med_data=med_data,
            target="potential",
            treatment_level=treatment_level,
            mediation_level=treatment_level,  # Implied for potential outcome
            ml_yx=ml_g,
            ml_px=ml_m,
            score=score,
            score_function=score_function,
            n_folds=n_folds,
            n_rep=n_rep,
            normalize_ipw=normalize_ipw,
            trimming_rule=trimming_rule,
            trimming_threshold=trimming_threshold,
            draw_sample_splitting=draw_sample_splitting,
        )


class DoubleMLMEDC(DoubleMLMED):
    """
    Double machine learning for the estimation of the counterfactual outcome in causal mediation analysis.

    .. deprecated:: 0.10.0
        The class `DoubleMLMEDC` is deprecated and will be removed in a future version.
        Please use `DoubleMLMediation` with `target='counterfactual'` instead.
    """

    def __init__(
        self,
        med_data,
        treatment_level,
        mediation_level,
        ml_yx,
        ml_ymx,
        ml_px,
        ml_pmx,
        ml_nested=None,
        score="MED",
        score_function="efficient-alt",
        n_folds=5,
        n_rep=1,
        normalize_ipw=False,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        order=1,
        fewsplits=False,
        draw_sample_splitting=True,
    ):
        warnings.warn(
            "The class `DoubleMLMEDC` is deprecated and will be removed in a future version. "
            "Please use `DoubleMLMediation` with `target='counterfactual'` instead.",
            FutureWarning,
            stacklevel=2,
        )

        super().__init__(
            med_data=med_data,
            target="counterfactual",
            treatment_level=treatment_level,
            mediation_level=mediation_level,
            ml_yx=ml_yx,
            ml_px=ml_px,
            ml_ymx=ml_ymx,
            ml_pmx=ml_pmx,
            ml_nested=ml_nested,
            score=score,
            score_function=score_function,
            n_folds=n_folds,
            n_rep=n_rep,
            normalize_ipw=normalize_ipw,
            trimming_rule=trimming_rule,
            trimming_threshold=trimming_threshold,
            draw_sample_splitting=draw_sample_splitting,
        )
