import warnings
from copy import deepcopy

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y

from doubleml import DoubleMLMediationData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import _cond_targets, _dml_cv_predict, _dml_tune, _get_cond_smpls, _get_cond_smpls_2d


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

        super().__init__(med_data, n_folds, n_rep, score, draw_sample_splitting)

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
        else:
            learners = ["ml_yx", "ml_px", "ml_ymx", "ml_pmx", "ml_nested"]

        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in learners}

    def _nuisance_est(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models=False,
    ):
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)

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
            x, m = check_X_y(x, self._med_data.m, force_all_finite=False)
            xm = np.column_stack((x, m))

            # Check whether there are external predictions for each parameter.
            px_external = external_predictions["ml_px"] is not None
            pmx_external = external_predictions["ml_pmx"] is not None
            yx_external = external_predictions["ml_yx"] is not None
            ymx_external = external_predictions["ml_ymx"] is not None
            nested_external = external_predictions["ml_nested"] is not None

            # Prepare the samples
            smpls_d0, smpls_d1 = _get_cond_smpls(smpls, self.treated)
            _, _, smpls_d1_m0, smpls_d1_m1 = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

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

            if ymx_external:
                ymx_hat = {"preds": external_predictions["ml_ymx"], "targets": None, "models": None}
            else:
                mu_test_smpls = np.full([len(smpls), 2], object)
                for idx, (train, test) in enumerate(smpls):
                    mu, delta = train_test_split(train, test_size=0.5)
                    mu_test_smpls[idx][0] = mu
                    mu_test_smpls[idx][1] = test
                mu_test_d0, mu_test_d1 = _get_cond_smpls(mu_test_smpls, self.treated)
                ymx_hat = _dml_cv_predict(
                    self._learner["ml_ymx"],
                    xm,
                    y,
                    smpls=mu_test_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_ymx"),
                    method=self._predict_method["ml_ymx"],
                    return_models=return_models,
                )

            if nested_external:
                nested_hat = {"preds": external_predictions["ml_nested"], "targets": None, "models": None}
            else:
                mu_test_smpls = np.full([len(smpls), 2], object)
                mu_delta_smpls = deepcopy(mu_test_smpls)
                delta_test_smpls = deepcopy(mu_test_smpls)

                # TODO: Probably will need to copy the ml_ymx learner to have a ml_ymx_delta learner.
                for idx, (train, test) in enumerate(smpls):
                    mu, delta = train_test_split(train, test_size=0.5)
                    mu_test_smpls[idx][0] = mu
                    mu_test_smpls[idx][1] = test
                    mu_delta_smpls[idx][0] = mu
                    mu_delta_smpls[idx][1] = delta
                    delta_test_smpls[idx][0] = delta
                    delta_test_smpls[idx][1] = test
                mu_delta_d0, mu_delta_d1 = _get_cond_smpls(mu_delta_smpls, self.treated)
                delta_test_d0, delta_test_d1 = _get_cond_smpls(delta_test_smpls, self.treated)

                ymx_delta_hat = _dml_cv_predict(
                    self._learner["ml_ymx"],
                    xm,
                    y,
                    smpls=mu_delta_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_ymx"),
                    method=self._predict_method["ml_ymx"],
                    return_models=return_models,
                )
                nested_hat = _dml_cv_predict(
                    self._learner["ml_nested"],
                    x,
                    ymx_delta_hat["preds"],
                    smpls=delta_test_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_nested"),
                    method=self._predict_method["ml_nested"],
                    return_models=return_models,
                )

            preds = {
                "predictions": {
                    "ml_px": px_hat["preds"],
                    "ml_pmx": pmx_hat["preds"],
                    "ml_yx": yx_hat["preds"],
                    "ml_ymx": ymx_hat["preds"],
                    "ml_nested": nested_hat["preds"],
                },
                "targets": {
                    "ml_px": px_hat["targets"],
                    "ml_pmx": pmx_hat["targets"],
                    "ml_yx": yx_hat["targets"],
                    "ml_ymx": ymx_hat["targets"],
                    "ml_nested": nested_hat["targets"],
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
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)

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
            x, m = check_X_y(x, self._med_data.m, force_all_finite=False)
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
