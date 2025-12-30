import warnings

import numpy as np
from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.utils import check_X_y

from doubleml import DoubleMLMediationData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import (
    _cond_targets,
    _dml_cv_predict,
    _dml_tune,
    _get_cond_smpls,
)
from doubleml.utils._tune_optuna import _dml_tune_optuna


# TODO: remove yx_learner in counterfactual nuisance estimation dependent on how trimming is applied
class DoubleMLMediation(LinearScoreMixin, DoubleML):
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
        ml_yx,
        ml_px,
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
        if not isinstance(med_data, DoubleMLMediationData):
            raise TypeError(
                "Mediation analysis requires data of type DoubleMLMediationData."
                + f"data of type {str(type(med_data))} was provided instead."
            )

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
            self._learner = {"ml_yx": ml_yx, "ml_px": ml_px, "ml_ymx": ml_ymx, "ml_pmx": ml_pmx, "ml_nested": ml_nested}
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
            required_learners = ["ml_yx", "ml_px", "ml_ymx", "ml_pmx", "ml_nested"]

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
            learners = ["ml_yx", "ml_px", "ml_ymx", "ml_pmx", "ml_nested"]
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
            yx_external = external_predictions["ml_yx"] is not None
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

            # Estimate the conditional expectation of outcome Y given D, M, and X.
            if yx_external:
                yx_hat = {"preds": external_predictions["ml_yx"], "targets": None, "models": None}
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

            ymx_inner_preds = None
            if ymx_external:
                missing = [
                    i
                    for i in range(self.n_folds_inner)
                    if f"ml_ymx_inner_{i}" not in external_predictions.keys()
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
                ymx_inner_hat = {}
                ymx_inner_hat["preds_inner"] = []
                ymx_inner_hat["targets_inner"] = []
                ymx_inner_hat["models"] = []
                for smpls, smpls_inner in zip(smpls, self._DoubleML__smpls__inner):
                    inner_smpls_d = [(np.intersect1d(train, np.where(self.treated == 1)), test) for train, test in smpls_inner]
                    ymx_inner = _dml_cv_predict(
                        self._learner["ml_ymx"],
                        xm,
                        y,
                        smpls=inner_smpls_d,
                        n_jobs=n_jobs_cv,
                        est_params=self._get_params("ml_ymx"),
                        method=self._predict_method["ml_ymx"],
                        return_models=return_models,
                    )
                    _check_finite_predictions(ymx_inner["preds"], self._learner["ml_ymx"], "ml_ymx", smpls_inner)

                    ymx_inner_hat["preds_inner"].append(ymx_inner["preds"])
                    ymx_inner_hat["targets_inner"].append(ymx_inner["targets"])

                ymx_inner_preds = np.array(
                    [
                        (
                            ymx_inner_hat["preds_inner"][1][i]
                            if not np.isnan(ymx_inner_hat["preds_inner"][1][i])
                            else ymx_inner_hat["preds_inner"][0][i]
                        )
                        for i in range(len(y))
                    ]
                )

                ymx_hat = _dml_cv_predict(
                    self._learner["ml_ymx"],
                    xm,
                    y,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_ymx"),
                    method=self._predict_method["ml_ymx"],
                    return_models=return_models,
                )

            if nested_external:
                nested_hat = {"preds": external_predictions["ml_nested"], "targets": None, "models": None}
            else:
                nested_hat = _dml_cv_predict(
                    self._learner["ml_nested"],
                    x,
                    ymx_inner_preds,
                    smpls=smpls_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_nested"),
                    method=self._predict_method["ml_nested"],
                )

            preds = {
                "predictions": {
                    "ml_px": px_hat["preds"],
                    "ml_pmx": pmx_hat["preds"],
                    "ml_yx": yx_hat["preds"],
                    "ml_ymx": ymx_hat["preds"],
                    "ml_nested": nested_hat["preds"],
                    **{f"ml_ymx_inner_{i}": ymx_inner_hat["preds_inner"][i] for i in range(len(ymx_inner_hat["preds_inner"]))},
                },
                "targets": {
                    "ml_px": px_hat["targets"],
                    "ml_pmx": pmx_hat["targets"],
                    "ml_yx": yx_hat["targets"],
                    "ml_ymx": ymx_hat["targets"],
                    "ml_nested": nested_hat["targets"],
                    **{
                        f"ml_ymx_inner_{i}": ymx_inner_hat["targets_inner"][i]
                        for i in range(len(ymx_inner_hat["targets_inner"]))
                    },
                },
                "models": {
                    "ml_px": px_hat["models"],
                    "ml_pmx": pmx_hat["models"],
                    "ml_yx": yx_hat["models"],
                    "ml_ymx": ymx_hat["models"],
                    "ml_nested": nested_hat["models"],
                },
            }

            psi_a, psi_b = self._score_elements(
                y, px_hat["preds"], yx_hat["preds"], pmx_hat["preds"], ymx_hat["preds"], nested_hat["preds"]
            )
            psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_elements(self, y, px_hat, yx_hat, pmx_hat=None, ymx_hat=None, nested_hat=None):
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
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._med_data.x, self._med_data.y, ensure_all_finite=True)
        x, d = check_X_y(x, self._med_data.d, ensure_all_finite=True)

        if scoring_methods is None:
            scoring_methods = {}

        train_inds = [train_index for (train_index, _) in smpls]
        _, smpls_d1 = _get_cond_smpls(smpls, self.treated)
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]

        res = {"params": {}, "tune_res": {}}

        # Tune ml_px
        if "ml_px" in param_grids:
            res["tune_res"]["ml_px"] = _dml_tune(
                d,
                x,
                train_inds,
                self._learner["ml_px"],
                param_grids["ml_px"],
                scoring_methods.get("ml_px"),
                n_folds_tune,
                n_jobs_cv,
                search_mode,
                n_iter_randomized_search,
            )
            res["params"]["ml_px"] = [xx.best_params_ for xx in res["tune_res"]["ml_px"]]

        # Tune ml_yx
        if "ml_yx" in param_grids:
            res["tune_res"]["ml_yx"] = _dml_tune(
                y,
                x,
                train_inds_d1,
                self._learner["ml_yx"],
                param_grids["ml_yx"],
                scoring_methods.get("ml_yx"),
                n_folds_tune,
                n_jobs_cv,
                search_mode,
                n_iter_randomized_search,
            )
            res["params"]["ml_yx"] = [xx.best_params_ for xx in res["tune_res"]["ml_yx"]]

        if self._target == "counterfactual":
            x, m = check_X_y(x, self._med_data.m, ensure_all_finite=True)
            xm = np.column_stack((x, m))

            # Tune ml_pmx
            if "ml_pmx" in param_grids:
                res["tune_res"]["ml_pmx"] = _dml_tune(
                    d,
                    xm,
                    train_inds,
                    self._learner["ml_pmx"],
                    param_grids["ml_pmx"],
                    scoring_methods.get("ml_pmx"),
                    n_folds_tune,
                    n_jobs_cv,
                    search_mode,
                    n_iter_randomized_search,
                )
                res["params"]["ml_pmx"] = [xx.best_params_ for xx in res["tune_res"]["ml_pmx"]]

            # Tune ml_ymx
            if "ml_ymx" in param_grids:
                res["tune_res"]["ml_ymx"] = _dml_tune(
                    y,
                    xm,
                    train_inds_d1,
                    self._learner["ml_ymx"],
                    param_grids["ml_ymx"],
                    scoring_methods.get("ml_ymx"),
                    n_folds_tune,
                    n_jobs_cv,
                    search_mode,
                    n_iter_randomized_search,
                )
                res["params"]["ml_ymx"] = [xx.best_params_ for xx in res["tune_res"]["ml_ymx"]]

        return res

    def _nuisance_tuning_optuna(self, optuna_params, scoring_methods, cv, optuna_settings):
        x, y = check_X_y(self._med_data.x, self._med_data.y, ensure_all_finite=True)
        x, d = check_X_y(x, self._med_data.d, ensure_all_finite=True)

        if scoring_methods is None:
            scoring_methods = {"ml_yx": None, "ml_px": None, "ml_ymx": None, "ml_pmx": None}

        res = {"params": {}, "tune_res": {}}

        if "ml_px" in optuna_params:
            res["tune_res"]["ml_px"] = _dml_tune_optuna(
                d,
                x,
                self._learner["ml_px"],
                optuna_params["ml_px"],
                scoring_methods.get("ml_px"),
                cv,
                optuna_settings,
                learner_name="ml_px",
                params_name="ml_px",
            )
            res["params"]["ml_px"] = [xx.best_params_ for xx in res["tune_res"]["ml_px"]]

        if "ml_yx" in optuna_params:
            res["tune_res"]["ml_yx"] = _dml_tune_optuna(
                y[d == 1],
                x[d == 1],
                self._learner["ml_yx"],
                optuna_params["ml_yx"],
                scoring_methods.get("ml_yx"),
                cv,
                optuna_settings,
                learner_name="ml_yx",
                params_name="ml_yx",
            )
            res["params"]["ml_yx"] = [xx.best_params_ for xx in res["tune_res"]["ml_yx"]]

        if self._target == "counterfactual":
            x, m = check_X_y(x, self._med_data.m, force_all_finite=False)
            xm = np.column_stack((x, m))

            if "ml_pmx" in optuna_params:
                res["tune_res"]["ml_pmx"] = _dml_tune_optuna(
                    d,
                    xm,
                    self._learner["ml_pmx"],
                    optuna_params["ml_pmx"],
                    scoring_methods.get("ml_pmx"),
                    cv,
                    optuna_settings,
                    learner_name="ml_pmx",
                    params_name="ml_pmx",
                )
                res["params"]["ml_pmx"] = [xx.best_params_ for xx in res["tune_res"]["ml_pmx"]]

            if "ml_ymx" in optuna_params:
                # ml_ymx is tuned on the subsample with D=1
                xm_d1 = xm[self.treated]
                y_d1 = y[self.treated]
                res["tune_res"]["ml_ymx"] = _dml_tune_optuna(
                    y_d1,
                    xm_d1,
                    self._learner["ml_ymx"],
                    optuna_params["ml_ymx"],
                    scoring_methods.get("ml_ymx"),
                    cv,
                    optuna_settings,
                    learner_name="ml_ymx",
                    params_name="ml_ymx",
                )
                res["params"]["ml_ymx"] = [xx.best_params_ for xx in res["tune_res"]["ml_ymx"]]

            # TODO: Add logic to tune the nested parameter
            if "ml_nested" in optuna_params:
                # Logic: Use ml_ymx to predict targets for ml_nested on D=1
                # If ml_ymx was tuned, use the tuned estimator; otherwise use the base learner
                if "ml_ymx" in res["tune_res"]:
                    ymx_best_est = res["tune_res"]["ml_ymx"][0].best_estimator_
                else:
                    ymx_best_est = self._learner["ml_ymx"]

                # Generate targets: E[Y|D=1, M, X]
                ymx_hat = cross_val_predict(clone(ymx_best_est), xm_d1, y_d1, cv=cv, method=self._predict_method["ml_ymx"])

                # Tune ml_nested: x -> ymx_hat
                res["tune_res"]["ml_nested"] = _dml_tune_optuna(
                    ymx_hat,
                    x_d1,  # Target, Features
                    self._learner["ml_nested"],
                    optuna_params["ml_nested"],
                    scoring_methods.get("ml_nested"),
                    cv,
                    optuna_settings,
                    learner_name="ml_nested",
                    params_name="ml_nested",
                )
                res["params"]["ml_nested"] = [xx.best_params_ for xx in res["tune_res"]["ml_nested"]]

    def _sensitivity_element_est(self, preds):
        pass


# TODO: Transplant methods into utils documents.
# TODO: Apply threshold
class DoubleMLMEDP(DoubleMLMediation):
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


class DoubleMLMEDC(DoubleMLMediation):
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
