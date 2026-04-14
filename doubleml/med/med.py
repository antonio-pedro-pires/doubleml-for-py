import warnings
from typing import Optional

import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y

from doubleml import DoubleMLMEDData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import _check_inner_sample_splitting, _normalize_propensity_med
from doubleml.utils._checks import _check_finite_predictions, _check_sample_splitting, _check_score
from doubleml.utils._estimation import (
    _cond_targets,
    _dml_cv_predict,
    _double_dml_cv_predict,
    _get_cond_smpls,
)
from doubleml.utils._tune_optuna import _dml_tune_optuna
from doubleml.utils.propensity_score_processing import PSProcessorConfig, init_ps_processor


class DoubleMLMED(LinearScoreMixin, DoubleML):
    """Double machine learning for causal mediation analysis with binary treatment.

    Parameters
    ----------
    dml_data : :class:`DoubleMLMediationData` object
        The :class:`DoubleMLMediationData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`E[Y|D,X]`.

    ml_G : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`E[Y|D,M,X]`.
        Only required if ``outcome`` is 'counterfactual'.

    ml_nested_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestRegressor`) for the nuisance function :math:`E[E[Y|D=d,M,X]|D=1-d, X]`
        Only required if ``outcome`` is 'counterfactual'.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`P(D=d|X)`.

    ml_M : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods for the nuisance function :math:`P(D=d|M,X)`.
        Only required if ``outcome`` is 'counterfactual'.

    outcome : str
        The outcome parameter to estimate.
        - 'potential': Estimate the potential outcome :math:`E[Y(d, M(d))]`.
        - 'counterfactual': Estimate the counterfactual outcome :math:`E[Y(d, M(d'))]`.
        where :math:`d' \neq d

    treatment_level : int
        The treatment level :math:`d`.

    score : str
        A str (``'efficient-alt'``)  specifying the score function to use.
        Default is 'efficient-alt'.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    trimming_rule : str
        A str (``'truncate'`` is the only choice) specifying the trimming approach.
        Default is ``'truncate'``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``1e-2``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    double_sample_splitting : bool
        Indicates whether the data is resampled for the estimation of the nested parameter.
        Default is ``True``.
    """

    def __init__(
        self,
        dml_data,
        treatment_level,
        outcome,
        ml_m,
        ml_g=None,
        ml_G=None,
        ml_M=None,
        ml_nested_g=None,
        score="efficient-alt",
        n_folds=5,
        n_rep=1,
        n_folds_inner=5,
        normalize_ipw=True,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        draw_sample_splitting=True,
        double_sample_splitting=True,
        ps_processor_config: Optional[PSProcessorConfig] = None,
    ):
        self._dml_data = self._check_dml_data(dml_data)

        self._double_sample_splitting = (
            double_sample_splitting if double_sample_splitting and outcome == "counterfactual" else False
        )
        self.n_folds_inner = n_folds_inner

        super().__init__(
            dml_data, n_folds, n_rep, score, draw_sample_splitting, double_sample_splitting=self.double_sample_splitting
        )

        valid_scores = ["efficient-alt"]
        _check_score(self.score, valid_scores, allow_callable=False)

        self._outcome = self._check_outcome(outcome)
        self._treatment_level = self._check_levels(treatment_level)

        self._treated = self._dml_data.d == treatment_level

        self._predict_method = {}
        self._check_learners(ml_g=ml_g, ml_m=ml_m, ml_G=ml_G, ml_M=ml_M, ml_nested_g=ml_nested_g)

        self._normalize_ipw = normalize_ipw
        self._trimming_rule = trimming_rule
        self._trimming_threshold = trimming_threshold
        self._external_predictions_implemented = True
        self._sensitivity_implemented = False

        self._ps_processor_config, self._ps_processor = init_ps_processor(
            ps_processor_config, trimming_rule, trimming_threshold
        )

    @property
    def outcome(self):
        """
        The outcome parameter to estimate.
        """
        return self._outcome

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
    def treated(self):
        """
        Indicator for observations with chosen treatment level.
        """
        return self._treated

    @property
    def double_sample_splitting(self):
        """
        Indicates whether the data is resampled for the estimation of the nested parameter.
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
        if self._outcome == "potential":
            learners = ["ml_g", "ml_m"]
        else:
            learners = ["ml_m", "ml_G", "ml_M", "ml_nested_g"]
            if self.double_sample_splitting:
                inner_G_learners = [f"ml_G_inner_{i}" for i in range(self.n_folds)]
                learners += inner_G_learners

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

        if self._outcome == "potential":
            # Check whether there are external predictions for each parameter.
            m_external = external_predictions["ml_m"] is not None
            g_external = external_predictions["ml_g"] is not None

            # Prepare the samples
            _, smpls_d1 = _get_cond_smpls(smpls, self.treated)

            if m_external:
                m_hat = {"preds": external_predictions["ml_m"], "targets": None, "models": None}
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    self.treated,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
            m_hat["preds"] = self._ps_processor.adjust_ps(m_hat["preds"], self.treated)

            if g_external:
                g_hat = {
                    "preds": external_predictions["ml_g"],
                    "targets": _cond_targets(y, cond_sample=(self.treated == 1)),
                    "models": None,
                }
            else:
                g_hat = _dml_cv_predict(
                    self._learner["ml_g"],
                    x,
                    y,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g"),
                    method=self._predict_method["ml_g"],
                    return_models=return_models,
                )
            _check_finite_predictions(g_hat["preds"], self._learner["ml_g"], "ml_g", smpls)
            # adjust target values to consider only compatible subsamples
            g_hat["targets"] = _cond_targets(g_hat["targets"], cond_sample=(self.treated == 1))

            preds = {
                "predictions": {
                    "ml_m": m_hat["preds"],
                    "ml_g": g_hat["preds"],
                },
                "targets": {
                    "ml_m": m_hat["targets"],
                    "ml_g": g_hat["targets"],
                },
                "models": {
                    "ml_m": m_hat["models"],
                    "ml_g": g_hat["models"],
                },
            }

            psi_a, psi_b = self._score_elements(y, m_hat_preds=m_hat["preds"], g_hat_preds=g_hat["preds"])
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        else:  # outcome == "counterfactual"
            x, m = check_X_y(x, self._dml_data.m, ensure_all_finite=True, multi_output=True)
            xm = np.column_stack((x, m))

            # Check whether there are external predictions for each parameter.
            m_external = external_predictions["ml_m"] is not None
            M_external = external_predictions["ml_M"] is not None
            G_external = external_predictions["ml_G"] is not None
            nested_g_external = external_predictions["ml_nested_g"] is not None

            # Get samples conditional on treatment:
            smpls_d0, smpls_d1 = _get_cond_smpls(smpls, self.treated)

            # Estimate the probability of treatment conditional on the covariates.
            if m_external:
                m_hat = {"preds": external_predictions["ml_m"], "targets": None, "models": None}
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
            m_hat["preds"] = self._ps_processor.adjust_ps(m_hat["preds"], self.treated)

            # Estimate the probability of treatment conditional on the mediator and covariates.
            if M_external:
                M_hat = {"preds": external_predictions["ml_M"], "targets": None, "models": None}
            else:
                M_hat = _dml_cv_predict(
                    self._learner["ml_M"],
                    xm,
                    d,
                    smpls=smpls,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_M"),
                    method=self._predict_method["ml_M"],
                    return_models=return_models,
                )
            M_hat["preds"] = self._ps_processor.adjust_ps(M_hat["preds"], self.treated)

            inner_predictions = {}
            inner_targets = {}
            if self.double_sample_splitting:
                # Get inner samples conditional on treatment:
                smpls_inner_d1 = []
                for fold in self._smpls_inner[self._i_rep]:
                    _, inner_smpls_d1 = _get_cond_smpls(fold, self.treated)
                    smpls_inner_d1.append(inner_smpls_d1)

                if G_external:
                    # expect per-inner-fold keys ml_G_inner_i
                    missing = [
                        i
                        for i in range(self.n_folds)
                        if (f"ml_G_inner_{i}") not in external_predictions.keys()
                        or external_predictions[f"ml_G_inner_{i}"] is None
                    ]
                    if len(missing) > 0:
                        raise ValueError(
                            "When providing external predictions for ml_G, also inner predictions for all inner folds "
                            f"have to be provided (missing: {', '.join([str(i) for i in missing])})."
                        )
                    G_hat_inner = [external_predictions[f"ml_G_inner_{i}"] for i in range(self.n_folds)]
                    G_hat = {
                        "preds": external_predictions["ml_G"],
                        "preds_inner": G_hat_inner,
                        "targets": self._dml_data.y,
                        "models": None,
                    }
                else:

                    G_hat = _double_dml_cv_predict(
                        estimator=self._learner["ml_G"],
                        estimator_name=self._learner["ml_G"],
                        x=xm,
                        y=y,
                        smpls=smpls_d1,
                        smpls_inner=smpls_inner_d1,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_G"),
                        method=self._predict_method["ml_G"],
                    )

                if nested_g_external:
                    nested_g_hat = {"preds": external_predictions["ml_nested_g"], "targets": None, "models": None}
                else:

                    # _dml_cv_predict perceives the fitting of the nested estimator as a case where there are multiple
                    # fold specific targets because of the shape of G_hat["preds"]. The method throws a
                    # 'ValueError: shape mismatch' when setting xx[train_index]=y[idx]
                    # (line 120 in doubleml/utils/_estimation.py). In order to avoid this error, it's necessary to
                    # feed the method a 'y' parameter whose subarrays match the 'smpls' parameter's subarrays shapes,
                    # which is exactly what the following line does.
                    G_hat_inner_preds = [G_hat_preds[train] for (train, _), G_hat_preds in zip(smpls_d0, G_hat["preds_inner"])]

                    nested_g_hat = _dml_cv_predict(
                        self._learner["ml_nested_g"],
                        xm,
                        G_hat_inner_preds,
                        smpls=smpls_d0,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_nested_g"),
                        method=self._predict_method["ml_nested_g"],
                        return_models=return_models,
                    )
                # store inner predictions as separate keys per inner fold
                inner_predictions = {f"ml_G_inner_{i}": G_hat["preds_inner"][i] for i in range(len(G_hat["preds_inner"]))}
                inner_targets = {
                    f"ml_G_inner_{i}": (
                        G_hat.get("targets_inner")[i]
                        if G_hat.get("targets_inner") is not None and i < len(G_hat["targets_inner"])
                        else None
                    )
                    for i in range(len(G_hat.get("preds_inner", [])))
                }

            else:
                if G_external:
                    G_hat = {
                        "preds": external_predictions["ml_G"],
                        "targets": None,
                        "models": None,
                    }
                else:
                    G_hat = _dml_cv_predict(
                        self.learner["ml_G"],
                        x=xm,
                        y=y,
                        smpls=smpls_d1,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_G"),
                        method=self._predict_method["ml_G"],
                    )
                if nested_g_external:
                    nested_g_hat = {"preds": external_predictions["ml_nested_g"], "targets": None, "models": None}
                else:
                    nested_g_hat = _dml_cv_predict(
                        self.learner["ml_nested_g"],
                        x=xm,
                        y=G_hat["preds"],
                        smpls=smpls_d0,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_nested_g"),
                        method=self._predict_method["ml_nested_g"],
                        return_models=return_models,
                    )
            preds = {
                "predictions": {
                    "ml_m": m_hat["preds"],
                    "ml_M": M_hat["preds"],
                    "ml_G": G_hat["preds"],
                    "ml_nested_g": nested_g_hat["preds"],
                    **inner_predictions,
                },
                "targets": {
                    "ml_m": m_hat["targets"],
                    "ml_M": M_hat["targets"],
                    "ml_G": G_hat["targets"],
                    "ml_nested_g": nested_g_hat["targets"],
                    **inner_targets,
                },
                "models": {
                    "ml_m": m_hat["models"],
                    "ml_M": M_hat["models"],
                    "ml_G": G_hat["models"],
                    "ml_nested_g": nested_g_hat["models"],
                },
            }

            psi_a, psi_b = self._score_elements(
                y,
                m_hat_preds=m_hat["preds"],
                M_hat_preds=M_hat["preds"],
                G_hat_preds=G_hat["preds"],
                nested_g_hat_preds=nested_g_hat["preds"],
            )
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_elements(self, y, m_hat_preds, g_hat_preds=None, M_hat_preds=None, G_hat_preds=None, nested_g_hat_preds=None):
        if self._outcome == "potential":
            u_hat = y - g_hat_preds
            propensity_score = _normalize_propensity_med(
                self.normalize_ipw,
                outcome=self._outcome,
                d=self._dml_data.d,
                treatment_level=self.treatment_level,
                m_preds=m_hat_preds,
            )
            psi_a = -1.0
            psi_b = np.multiply(propensity_score, u_hat) + g_hat_preds
        else:
            u_hat = y - G_hat_preds
            w_hat = G_hat_preds - nested_g_hat_preds
            psi_a = -1.0
            ps1, ps2 = _normalize_propensity_med(
                self.normalize_ipw,
                outcome=self._outcome,
                d=self._dml_data.d,
                treatment_level=self._treatment_level,
                m_preds=m_hat_preds,
                M_preds=M_hat_preds,
            )
            psi_b = np.multiply(ps1, u_hat) + np.multiply(ps2, w_hat) + nested_g_hat_preds

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
            scoring_methods = {"ml_g": None, "ml_m": None, "ml_G": None, "ml_M": None, "ml_nested_g": None}

        treated_mask = self.treated
        not_treated_mask = np.logical_not(treated_mask)

        m_tune_res = _dml_tune_optuna(
            y=d,
            x=x,
            learner=self._learner["ml_m"],
            param_grid_func=optuna_params["ml_m"],
            scoring_method=scoring_methods["ml_m"],
            cv=cv,
            optuna_settings=optuna_settings,
            learner_name="ml_m",
            params_name="ml_m",
        )

        if self.outcome == "potential":
            g_tune_res = _dml_tune_optuna(
                y=y[treated_mask],
                x=x[treated_mask],
                learner=self._learner["ml_g"],
                param_grid_func=optuna_params["ml_g"],
                scoring_method=scoring_methods["ml_g"],
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_g",
                params_name="ml_g",
            )

            results = {
                "ml_m": m_tune_res,
                "ml_g": g_tune_res,
            }

        else:
            x, m = check_X_y(x, self._dml_data.m, ensure_all_finite=False, multi_output=True)
            xm = np.column_stack((x, m))

            M_tune_res = _dml_tune_optuna(
                y=d,
                x=xm,
                learner=self._learner["ml_M"],
                param_grid_func=optuna_params["ml_M"],
                scoring_method=scoring_methods["ml_M"],
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_M",
                params_name="ml_M",
            )

            G_tune_res = _dml_tune_optuna(
                y=y[treated_mask],
                x=xm[treated_mask],
                learner=self._learner["ml_G"],
                param_grid_func=optuna_params["ml_G"],
                scoring_method=scoring_methods.get("ml_G"),
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_G",
                params_name="ml_G",
            )

            # Prepare targets for the nested estimator
            G_hat = cross_val_predict(
                estimator=self._learner["ml_nested_g"],
                X=xm,
                y=y,
                cv=cv,
                method=self._predict_method["ml_nested_g"],
            )

            nested_g_tune_res = _dml_tune_optuna(
                y=G_hat[not_treated_mask],
                x=xm[not_treated_mask],
                learner=self._learner["ml_nested_g"],
                param_grid_func=optuna_params["ml_nested_g"],
                scoring_method=scoring_methods["ml_nested_g"],
                cv=cv,
                optuna_settings=optuna_settings,
                learner_name="ml_nested_g",
                params_name="ml_nested_g",
            )
            results = {
                "ml_m": m_tune_res,
                "ml_G": G_tune_res,
                "ml_M": M_tune_res,
                "ml_nested_g": nested_g_tune_res,
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

    def _check_learners(self, ml_g, ml_m, ml_G, ml_M, ml_nested_g):
        if self._outcome == "potential":
            self._learner = {"ml_g": ml_g, "ml_m": ml_m}
        else:
            self._learner = {"ml_m": ml_m, "ml_G": ml_G, "ml_M": ml_M, "ml_nested_g": ml_nested_g}

        for learner_name, learner in self._learner.items():
            if self._learner[learner_name] is None:
                raise ValueError(f"Learner {learner_name} is required when the outcome is {self._outcome}.")

            if learner_name in ["ml_m", "ml_M"]:
                is_classifier_ = self._check_learner(learner, learner_name, regressor=False, classifier=True)
                if is_classifier_:
                    self._predict_method[learner_name] = "predict_proba"
                else:
                    raise ValueError(
                        f"The learner {learner_name} must be a classifier. " f"{learner_name} identified as a regressor."
                    )
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

    def _check_outcome(self, outcome):
        if not isinstance(outcome, str):
            raise TypeError("Outcome must be a string." + f"{str(outcome)} of type {str(type(outcome))} provided instead.")

        valid_outcomes = ["potential", "counterfactual"]
        if outcome not in valid_outcomes:
            raise ValueError(f"Invalid outcome {outcome}. " + "Valid outcomes " + " or ".join(valid_outcomes) + ".")

        return outcome

    def _check_levels(self, treatment_level):
        if not isinstance(treatment_level, int):
            raise TypeError(
                "Treatment level must be an integer."
                + f" Treatment level {str(treatment_level)} of type {str(type(treatment_level))} provided."
            )

        if not 0 <= treatment_level <= 1:
            raise ValueError(
                "Treatment level must be either 0 or 1" + f" Treatment level provided was {str(treatment_level)}."
            )
        return treatment_level
