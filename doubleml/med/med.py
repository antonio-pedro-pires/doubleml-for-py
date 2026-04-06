import numbers
import warnings
from typing import Optional

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y

from doubleml import DoubleMLMEDData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import _check_inner_sample_splitting, _normalize_propensity_med
from doubleml.utils._checks import _check_finite_predictions, _check_sample_splitting
from doubleml.utils._estimation import (
    _cond_targets,
    _dml_cv_predict,
    _double_dml_cv_predict,
    _get_cond_smpls,
)
from doubleml.utils._tune_optuna import _dml_tune_optuna
from doubleml.utils.propensity_score_processing import PSProcessorConfig, init_ps_processor


# TODO: remove yx_learner in counterfactual nuisance estimation dependent on how trimming is applied
class DoubleMLMED(LinearScoreMixin, DoubleML):
    """Double machine learning for causal mediation analysis.

    Parameters
    ----------
    dml_data : :class:`DoubleMLMediationData` object
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
        dml_data,
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
        double_sample_splitting=True,
        ps_processor_config: Optional[PSProcessorConfig] = None,
    ):
        self._dml_data = self._check_dml_data(dml_data)

        self._double_sample_splitting = (
            double_sample_splitting if double_sample_splitting and target == "counterfactual" else False
        )
        self.n_folds_inner = n_folds_inner

        super().__init__(
            dml_data, n_folds, n_rep, score, draw_sample_splitting, double_sample_splitting=self.double_sample_splitting
        )

        self._target = self._check_target(target)
        self._treatment_level, self._mediation_level = self._check_levels(treatment_level, mediation_level)

        self._treated = self._dml_data.d == treatment_level
        self._mediated = self._dml_data.m == mediation_level

        self._predict_method = {}
        self._check_learners(ml_yx=ml_yx, ml_px=ml_px, ml_ymx=ml_ymx, ml_pmx=ml_pmx, ml_nested=ml_nested)

        self._normalize_ipw = normalize_ipw
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        self._external_predictions_implemented = True
        self._sensitivity_implemented = False

        self._ps_processor_config, self._ps_processor = init_ps_processor(
            ps_processor_config, trimming_rule, trimming_threshold
        )

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
    def double_sample_splitting(self):
        """
        Indicates whether the training data is split for estimating the nested models of the nuisance parameter .
        """
        return self._double_sample_splitting

    @property
    def _score_element_names(self):
        return ["psi_a", "psi_b"]

    @property
    def ps_processor_config(self):
        """
        Configuration for propensity score processing (clipping, calibration, etc.).
        """
        return self._ps_processor_config

    @property
    def ps_processor(self):
        """
        Propensity score processor.
        """
        return self._ps_processor

    def _initialize_ml_nuisance_params(self):
        if self._target == "potential":
            learners = ["ml_yx", "ml_px"]
        else:
            learners = ["ml_px", "ml_ymx", "ml_pmx", "ml_nested"]
            if self.double_sample_splitting:
                inner_ymx_learners = [f"ml_ymx_inner_{i}" for i in range(self.n_folds)]
                learners += inner_ymx_learners

        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in learners}

    def _nuisance_est(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models=False,
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=True)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=True)

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
            px_hat["preds"] = self._ps_processor.adjust_ps(px_hat["preds"], self.treated)

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

            psi_a, psi_b = self._score_elements(y, px_hat_preds=px_hat["preds"], yx_hat_preds=yx_hat["preds"])
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        else:  # target == "counterfactual"
            x, m = check_X_y(x, self._dml_data.m, ensure_all_finite=True, multi_output=True)
            xm = np.column_stack((x, m))

            # Check whether there are external predictions for each parameter.
            px_external = external_predictions["ml_px"] is not None
            pmx_external = external_predictions["ml_pmx"] is not None
            ymx_external = external_predictions["ml_ymx"] is not None
            nested_external = external_predictions["ml_nested"] is not None

            # Get samples conditional on treatment:
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
            px_hat["preds"] = self._ps_processor.adjust_ps(px_hat["preds"], self.treated)

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
            pmx_hat["preds"] = self._ps_processor.adjust_ps(pmx_hat["preds"], self.treated)

            inner_predictions = {}
            inner_targets = {}
            if self.double_sample_splitting:
                # Get inner samples conditional on treatment:
                smpls_inner_d1 = []
                for fold in self._smpls_inner[self._i_rep]:
                    _, inner_smpls_d1 = _get_cond_smpls(fold, self.treated)
                    smpls_inner_d1.append(inner_smpls_d1)

                if ymx_external:
                    # expect per-inner-fold keys ml_ymx_inner_i
                    missing = [
                        i
                        for i in range(self.n_folds)
                        if (f"ml_ymx_inner_{i}") not in external_predictions.keys()
                        or external_predictions[f"ml_ymx_inner_{i}"] is None
                    ]
                    if len(missing) > 0:
                        raise ValueError(
                            "When providing external predictions for ml_ymx, also inner predictions for all inner folds "
                            f"have to be provided (missing: {', '.join([str(i) for i in missing])})."
                        )
                    ymx_hat_inner = [external_predictions[f"ml_ymx_inner_{i}"] for i in range(self.n_folds)]
                    ymx_hat = {
                        "preds": external_predictions["ml_ymx"],
                        "preds_inner": ymx_hat_inner,
                        "targets": self._dml_data.y,
                        "models": None,
                    }
                else:

                    ymx_hat = _double_dml_cv_predict(
                        estimator=self._learner["ml_ymx"],
                        estimator_name=self._learner["ml_ymx"],
                        x=xm,
                        y=y,
                        smpls=smpls_d1,
                        smpls_inner=smpls_inner_d1,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_ymx"),
                        method=self._predict_method["ml_ymx"],
                    )

                if nested_external:
                    nested_hat = {"preds": external_predictions["ml_nested"], "targets": None, "models": None}
                else:

                    # _dml_cv_predict perceives the fitting of the nested estimator as a case where there are multiple
                    # fold specific targets because of the shape of ymx_hat["preds"]. The method throws a
                    # 'ValueError: shape mismatch' when setting xx[train_index]=y[idx]
                    # (line 120 in doubleml/utils/_estimation.py). In order to avoid this error, it's necessary to
                    # feed the method a 'y' parameter whose subarrays match the 'smpls' parameter's subarrays shapes,
                    # which is exactly what the following line does.
                    ymx_hat_inner_preds = [
                        ymx_hat_preds[train] for (train, _), ymx_hat_preds in zip(smpls_d0, ymx_hat["preds_inner"])
                    ]

                    nested_hat = _dml_cv_predict(
                        self._learner["ml_nested"],
                        xm,
                        ymx_hat_inner_preds,
                        smpls=smpls_d0,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_nested"),
                        method=self._predict_method["ml_nested"],
                        return_models=return_models,
                    )
                # store inner predictions as separate keys per inner fold
                inner_predictions = {
                    f"ml_ymx_inner_{i}": ymx_hat["preds_inner"][i] for i in range(len(ymx_hat["preds_inner"]))
                }
                inner_targets = {
                    f"ml_ymx_inner_{i}": (
                        ymx_hat.get("targets_inner")[i]
                        if ymx_hat.get("targets_inner") is not None and i < len(ymx_hat["targets_inner"])
                        else None
                    )
                    for i in range(len(ymx_hat.get("preds_inner", [])))
                }

            else:
                if ymx_external:
                    ymx_hat = {
                        "preds": external_predictions["ml_ymx"],
                        "targets": None,
                        "models": None,
                    }
                else:
                    ymx_hat = _dml_cv_predict(
                        self.learner["ml_ymx"],
                        x=xm,
                        y=y,
                        smpls=smpls_d1,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_ymx"),
                        method=self._predict_method["ml_ymx"],
                    )
                if nested_external:
                    nested_hat = {"preds": external_predictions["ml_nested"], "targets": None, "models": None}
                else:
                    nested_hat = _dml_cv_predict(
                        self.learner["ml_nested"],
                        x=xm,
                        y=ymx_hat["preds"],
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
                    **inner_predictions,
                },
                "targets": {
                    "ml_px": px_hat["targets"],
                    "ml_pmx": pmx_hat["targets"],
                    "ml_ymx": ymx_hat["targets"],
                    "ml_nested": nested_hat["targets"],
                    **inner_targets,
                },
                "models": {
                    "ml_px": px_hat["models"],
                    "ml_pmx": pmx_hat["models"],
                    "ml_ymx": ymx_hat["models"],
                    "ml_nested": nested_hat["models"],
                },
            }

            psi_a, psi_b = self._score_elements(
                y,
                px_hat_preds=px_hat["preds"],
                pmx_hat_preds=pmx_hat["preds"],
                ymx_hat_preds=ymx_hat["preds"],
                nested_hat_preds=nested_hat["preds"],
            )
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_elements(
        self, y, px_hat_preds, yx_hat_preds=None, pmx_hat_preds=None, ymx_hat_preds=None, nested_hat_preds=None
    ):
        if self._target == "potential":
            u_hat = y - yx_hat_preds
            propensity_score = _normalize_propensity_med(
                self.normalize_ipw,
                outcome=self._target,
                d=self._dml_data.d,
                treatment_level=self.treatment_level,
                px_preds=px_hat_preds,
            )
            psi_a = -1.0
            psi_b = np.multiply(propensity_score, u_hat) + yx_hat_preds
        else:
            u_hat = y - ymx_hat_preds
            w_hat = ymx_hat_preds - nested_hat_preds
            psi_a = -1.0
            ps1, ps2 = _normalize_propensity_med(
                self.normalize_ipw,
                outcome=self._target,
                d=self._dml_data.d,
                treatment_level=self._treatment_level,
                px_preds=px_hat_preds,
                pmx_preds=pmx_hat_preds,
            )
            psi_b = np.multiply(ps1, u_hat) + np.multiply(ps2, w_hat) + nested_hat_preds

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
            "Nuisance tuning using the 'tune' method is not implemented for DoubleMLMediation. "
            + "Please use the 'tune_ml_models' method instead."
        )

    def _nuisance_tuning_optuna(self, optuna_params, scoring_methods, cv, optuna_settings):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=True)
        x, d = check_X_y(x, self._dml_data.d, ensure_all_finite=True)

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
            x, m = check_X_y(x, self._dml_data.m, ensure_all_finite=False, multi_output=True)
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

    def _set_smpls_sampling(self, smpls, all_smpls_cluster=None, is_cluster_data=False, smpls_inner=None):
        if self.double_sample_splitting:
            if smpls_inner is None:
                raise ValueError("smpls_inner is required")
            if all_smpls_cluster is not None or is_cluster_data:
                raise NotImplementedError("sample setting with cluster data and inner samples not supported.")
            self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
                all_smpls=smpls,
                all_smpls_cluster=all_smpls_cluster,
                dml_data=self._dml_data,
                is_cluster_data=is_cluster_data,
                n_obs=None,
            )
            self._smpls_inner, self._n_folds_inner = _check_inner_sample_splitting(
                smpls_inner,
                self._smpls,
            )
        else:
            if all_smpls_cluster is not None or is_cluster_data:
                raise NotImplementedError("sample setting with cluster data and inner samples not supported.")
            self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
                all_smpls=smpls,
                all_smpls_cluster=all_smpls_cluster,
                dml_data=self._dml_data,
                is_cluster_data=is_cluster_data,
                n_obs=None,
            )

    def _check_learners(self, ml_yx, ml_px, ml_ymx, ml_pmx, ml_nested):
        if self._target == "potential":
            self._learner = {"ml_yx": ml_yx, "ml_px": ml_px}
        else:
            self._learner = {"ml_px": ml_px, "ml_ymx": ml_ymx, "ml_pmx": ml_pmx, "ml_nested": ml_nested}

        for learner_name, learner in self._learner.items():
            if self._learner[learner_name] is None:
                raise ValueError(f"Learner {learner_name} is required when the target is {self._target}.")

            if learner_name in ["ml_px", "ml_pmx"]:
                is_classifier_ = self._check_learner(learner, learner_name, regressor=True, classifier=True)
                if is_classifier_:
                    self._predict_method[learner_name] = "predict_proba"
                else:
                    self._predict_method
            else:
                is_classifier_ = self._check_learner(learner, learner_name, regressor=True, classifier=True)
                if self._dml_data.binary_outcome and not is_classifier_:
                    raise ValueError(
                        f"The learner {learner_name} must be a classifier"
                        + "since the outcome variable is binary with values 0 and 1."
                    )
                elif not self._dml_data.binary_outcome and is_classifier_:
                    raise ValueError(
                        f"The learner {learner_name} was identified as a classifier "
                        + "but the outcome variable is not binary with values 0 and 1."
                    )
                else:
                    if is_classifier_:
                        self._predict_method[learner_name] = "predict_proba"
                    else:
                        self._predict_method[learner_name] = "predict"
        self._initialize_ml_nuisance_params()

    def _check_dml_data(self, dml_data):
        if not isinstance(dml_data, DoubleMLMEDData):
            raise TypeError(
                "Mediation analysis requires data of type DoubleMLMediationData."
                + f" Data of type {str(type(dml_data))} was provided instead."
            )
        if not all(dml_data.binary_treats):
            raise ValueError(
                f"Treatment data {dml_data.d} must be a binary variable with values either 0 or 1."
                + f" Treatment data contains levels {np.unique(dml_data.d)}."
            )
        if dml_data.z_cols is not None:
            warnings.warn(
                "The current framework for causal mediation analysis does not perform analysis with instrumental variables."
                + " The results will not take into account the instrumental variables."
            )
        return dml_data

    def _check_target(self, target):
        if not isinstance(target, str):
            raise TypeError("Target must be a string." + f"{str(target)} of type {str(type(target))} provided instead.")

        valid_targets = ["potential", "counterfactual"]
        if target not in valid_targets:
            raise ValueError(f"Invalid target {target}. " + "Valid targets " + " or ".join(valid_targets) + ".")

        return target

    def _check_levels(self, treatment_level, mediation_level):
        if not isinstance(treatment_level, int):
            raise TypeError(
                "Treatment level must be an integer."
                + f" Treatment level {str(treatment_level)} of type {str(type(treatment_level))} provided."
            )
        if not isinstance(mediation_level, numbers.Number):
            raise TypeError(
                "Mediation level must be a number."
                + f" Mediation level {str(mediation_level)} of type {str(type(mediation_level))} provided."
            )
        if not 0 <= treatment_level <= 1:
            raise ValueError(
                "Treatment level must be either 0 or 1" + f" Treatment level provided was {str(treatment_level)}."
            )
        return treatment_level, mediation_level


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
        dml_data,
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
        double_sample_splitting=False,
        draw_sample_splitting=True,
    ):
        warnings.warn(
            "The class `DoubleMLMEDP` is deprecated and will be removed in a future version. "
            "Please use `DoubleMLMediation` with `target='potential'` instead.",
            FutureWarning,
            stacklevel=2,
        )

        super().__init__(
            dml_data=dml_data,
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
        dml_data,
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
            dml_data=dml_data,
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
