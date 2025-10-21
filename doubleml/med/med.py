import numpy as np
from sklearn.utils import check_consistent_length, check_X_y

from doubleml import DoubleMLMediationData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import _normalize_propensity_med, split_smpls, recombine_samples, \
    extract_sets_from_smpls
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import _cond_targets, _dml_cv_predict, _dml_tune, _get_cond_smpls, _get_cond_smpls_2d
from doubleml.utils._propensity_score import _trimm


# TODO: Transplant methods into utils documents.
# TODO: Apply threshold
class DoubleMLMEDP(LinearScoreMixin, DoubleML):
    """Double machine learning for the estimation of the potential outcome in causal mediation analysis.

    Parameters
    ----------
    med_data : :class:`DoubleMLMediationData` object
        The :class:`DoubleMLMediationData` object providing the data and specifying the variables for the causal model.

    n_folds : int
        Number of folds.
        Default is ``5``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``5e-2``.

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
        self._med_data = med_data

        if not isinstance(med_data, DoubleMLMediationData):
            raise TypeError(
                "Mediation analysis requires data of type DoubleMLMediationData."
                + f"data of type {str(type(med_data))} was provided instead."
            )

        super().__init__(med_data, n_folds, n_rep, score, draw_sample_splitting)
        valid_score = ["MED"]
        self._score = score
        _check_score(score, valid_score)

        self._score_function = score_function
        valid_scores_types = ["efficient", "efficient-alt"]
        _check_score(score_function, valid_scores_types)

        self._treatment_level = treatment_level
        self._treated = self._med_data.d == treatment_level
        self._mediated = self._med_data.m == treatment_level

        self._learner = {"ml_m": ml_m, "ml_g": ml_g}
        self._predict_method = {"ml_m": "predict_proba"}
        self._check_learner(learner=ml_m, learner_name="ml_m", regressor=False, classifier=True)
        is_classifier_ml_g = self._check_learner(learner=ml_g, learner_name="ml_g", regressor=True, classifier=True)
        if is_classifier_ml_g:
            self._predict_method["ml_g"] = "predict_proba"
        else:
            self._predict_method["ml_g"] = "predict"
        self._initialize_ml_nuisance_params()

        self._normalize_ipw = normalize_ipw
        self._external_predictions_implemented = True

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
        return self.trimming_rule

    @property
    def trimming_threshold(self):
        """
        Indicates the trimming threshold.
        """
        return self.trimming_threshold

    @property
    def order(self):
        """
        Indicates the order of the polynomials used to estimate any conditional probability or conditional mean outcome.
        """
        return self.order

    @property
    def few_splits(self):
        """
        Indicates whether the same training data is used for estimating nested models of nuisance parameters.
        """
        return self.few_splits

    @property
    def score_function(self):
        """
        Indicates the type of the score function.
        """
        return self._score_function

    @property
    def treated(self):
        """
        Indicator for observations with chosen treatment level.
        """
        return self._treated

    @property
    def treatment_level(self):
        """
        Chosen treatment level.
        """
        return self._treatment_level

    @property
    def mediated(self):
        """
        Indicator for observations with chosen mediation level.
        """
        return self._mediated

    @property
    def mediation_level(self):
        """
        Chosen mediation level.
        """
        return self._mediation_level

    @property
    def is_potential_outcome(self):
        """
        Indicates whether the current score function computes the potential outcome: Y(d, m(d)).
        """
        return self.treatment_level == self.mediation_level

    def _initialize_ml_nuisance_params(self):
        valid_learner = ["ml_m", "ml_g_1"]
        self._params = {learner: {key: [None] * self.n_rep for key in self._med_data.d_cols} for learner in valid_learner}

    def _nuisance_est(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models=True,
    ):

        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)

        # Check whether there are external predictions for each parameter.
        m_external = external_predictions["ml_m"] is not None
        g_1_external = external_predictions["ml_g_1"] is not None

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

        # trimm external predictions
        # m_hat["preds"] = _trimm(m_hat["preds"], self.trimming_rule, self.trimming_threshold)

        if g_1_external:
            g_1_hat = {
                "preds": external_predictions["ml_g_1"],
                "targets": _cond_targets(y, cond_sample=(self.treated == 1)),
                "models": None,
            }
        else:
            g_1_hat = _dml_cv_predict(
                self._learner["ml_g"],
                x,
                y,
                smpls=smpls_d1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_1"),
                method=self._predict_method["ml_g"],
                return_models=return_models,
            )
        _check_finite_predictions(g_1_hat["preds"], self._learner["ml_g"], "ml_g", smpls)
        # adjust target values to consider only compatible subsamples
        g_1_hat["targets"] = _cond_targets(g_1_hat["targets"], cond_sample=(self.treated == 1))

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                "ml_g_1": g_1_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                "ml_g_1": g_1_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_g_1": g_1_hat["models"],
            },
        }

        psi_a, psi_b = self._score_elements(y, m_hat["preds"], g_1_hat["preds"])
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_elements(self, y, m_hat, g_1_hat):
        u_hat = y - g_1_hat
        # adjusted_propensity = _normalize_propensity_med(
        #    self.normalize_ipw,
        #    self.score,
        #    self.score_function,
        #    self.treated,
        # )
        psi_a = -1.0
        psi_b = np.multiply(np.divide(self.treated, m_hat), u_hat) + g_1_hat
        return psi_a, psi_b

    # TODO: Refactor tuning to take away all of the mentions to d0, d1 and others.
    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        # TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(
            x, self._med_data["z"]
        )  # Check that the mediators have the same number of samples as X and

        dx = np.column_stack((d, x))

        treated = self.treated
        _, smpls_d1 = _get_cond_smpls(smpls, treated)

        train_inds = [train_index for (train_index, _) in smpls]
        # train_inds_1 = [train_index for (train_index, _) in smpls_d1]
        train_inds_g = None

        # TODO: Check what this does
        if scoring_methods is None:
            scoring_methods = {"ml_g_1": None, "ml_m": None}

        g_1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_g,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        m_tune_res = _dml_tune(
            treated,
            x,
            train_inds,
            self._learner["ml_m"],
            param_grids["ml_m"],
            scoring_methods["ml_m"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        g_1_best_params = [xx.best_params_ for xx in g_1_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {"ml_g_1": g_1_best_params, "ml_m": m_best_params}
        tune_res = {"ml_g_1": g_1_tune_res, "ml_m": m_tune_res}

        res = {"params": params, "tune_res": tune_res}
        return res

    def _sensitivity_element_est(self, preds):
        pass

    def _check_data(self, med_data):
        if not isinstance(med_data, DoubleMLMediationData):
            raise TypeError(
                f"The data must be of DoubleMLMediationData type. {str(med_data)} "
                f"of type {str(type(med_data))} was passed."
            )

    def _check_score_functions(self):
        valid_score_function = ["efficient", "efficient-alt"]
        if self.score_function == "efficient":
            if self._med_data.n_meds > 1:
                raise ValueError(
                    f"score_function defined as {self.score_function}. "
                    + f"Mediation analysis based on {self.score_function} scores assumes only one mediation variable. "
                    + f"Data contains {self._med_data.n_meds} mediation variables. "
                    + "Please choose another score_function for mediation analysis."
                )
            if not self._med_data.binary_meds.all():
                raise ValueError(
                    "Mediation analysis based on efficient scores requires a binary mediation variable"
                    + "with integer values equal to 0 or 1 and no missing values."
                    + f"Actual data contains {np.unique(self._med_data.data.m)}"
                    + "unique values and/or may contain missing values."
                )
        if self.score_function in valid_score_function and not self._med_data.force_all_m_finite:
            raise ValueError(
                f"Mediation analysis based on {str(valid_score_function)} "
                f"requires finite mediation variables with no missing values."
            )
        # TODO: Probably want to check that elements of mediation variables are floats or ints.


class DoubleMLMEDC(LinearScoreMixin, DoubleML):
    """Double machine learning for the estimation of the counterfactual outcome in causal mediation analysis.

    Parameters
    ----------
    med_data : :class:`DoubleMLMediationData` object
        The :class:`DoubleMLMediationData` object providing the data and specifying the variables for the causal model.

    n_folds : int
        Number of folds.
        Default is ``5``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``5e-2``.

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
        smpls_ratio=0.5,
        normalize_ipw=False,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        order=1,
        fewsplits=False,
        draw_sample_splitting=True,
    ):
        self._med_data = med_data

        if not isinstance(med_data, DoubleMLMediationData):
            raise TypeError(
                "Mediation analysis requires data of type DoubleMLMediationData."
                + f"data of type {str(type(med_data))} was provided instead."
            )

        super().__init__(med_data, n_folds, n_rep, score, draw_sample_splitting)
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
        self.smpls_ratio=smpls_ratio

        self._learner = {"ml_px":ml_px, "ml_pmx":ml_pmx, "ml_yx":ml_yx, "ml_ymx":ml_ymx, "ml_nested":ml_nested}
        self._check_learner(learner=ml_px, learner_name="ml_px", regressor=False, classifier=True)
        self._check_learner(learner=ml_pmx, learner_name="ml_pmx", regressor=False, classifier=True)

        is_classifier_ml_yx = self._check_learner(learner=ml_yx, learner_name="ml_yx", regressor=True, classifier=True)
        is_classifier_ml_ymx = self._check_learner(learner=ml_ymx, learner_name="ml_ymx", regressor=True, classifier=True)
        is_classifier_ml_nested = self._check_learner(learner=ml_nested, learner_name="ml_nested", regressor=True, classifier=True)

        self._predict_method = {"ml_px": "predict_proba", "ml_pmx": "predict_proba"}
        if is_classifier_ml_yx:
            self._predict_method["ml_yx"] = "predict_proba"
        else:
            self._predict_method["ml_yx"] = "predict"
        if is_classifier_ml_ymx:
            self._predict_method["ml_ymx"] = "predict_proba"
        else:
            self._predict_method["ml_ymx"] = "predict"
        if is_classifier_ml_nested:
            self._predict_method["ml_nested"] = "predict_proba"
        else:
            self._predict_method["ml_nested"] = "predict"
        self._initialize_ml_nuisance_params()

    @property
    def normalize_ipw(self):
        """
        indicates whether the  inverse probability weights are normalized
        """
        return self.normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the used trimming rule.
        """
        return self.trimming_rule

    @property
    def trimming_threshold(self):
        """
        Indicates the trimming threshold.
        """
        return self.trimming_threshold

    @property
    def order(self):
        """
        Indicates the order of the polynomials used to estimate any conditional probability or conditional mean outcome.
        """
        return self.order

    @property
    def few_splits(self):
        """
        Indicates whether the same training data is used for estimating nested models of nuisance parameters.
        """
        return self.few_splits

    @property
    def score_function(self):
        """
        Indicates the type of the score function.
        """
        return self._score_function

    @property
    def treated(self):
        """
        Indicator for observations with chosen treatment level.
        """
        return self._treated

    @property
    def treatment_level(self):
        """
        Chosen treatment level.
        """
        return self._treatment_level

    @property
    def mediated(self):
        """
        Indicator for observations with chosen mediation level.
        """
        return self._mediated

    @property
    def mediation_level(self):
        """
        Chosen mediation level.
        """
        return self._mediation_level

    def _initialize_ml_nuisance_params(self):
        valid_learner = ["ml_px", "ml_pmx", "ml_yx", "ml_ymx", "ml_nested"]
        self._params = {learner: {key: [None] * self.n_rep for key in self._med_data.d_cols} for learner in valid_learner}

    def _nuisance_est(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models=False,
    ):
        # TODO: For each nuisance_est function, don't forget to trim predictions.
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        x, m = check_X_y(x, self._med_data.m, force_all_finite=False)
        xm = np.column_stack((x, m))

        # Check whether there are external predictions for each parameter.
        px_external = external_predictions["ml_px"] is not None
        pmx_external = external_predictions["ml_pmx"] is not None
        yx_external = external_predictions["ml_yx"] is not None
        ymx_external = external_predictions["ml_ymx"] is not None
        nested_external = external_predictions["ml_nested"] is not None

        # TODO: Samples have to be split into musample and deltasample.
        # Prepare the samples
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, self.treated)
        _, _, smpls_d1_m0, smpls_d1_m1 = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

        # TODO: Often required to fit learners on both samples.
        # TODO: Maybe will need to use other strategy than _dml_cv_predict()
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

        ymx_hat, nested_hat = self._estimate_nested_outcomes(y, x, xm, smpls, external_predictions, n_jobs_cv, return_models, ymx_external, nested_external)
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

        psi_a, psi_b = self._score_elements(y, px_hat, pmx_hat, yx_hat, ymx_hat, nested_hat, smpls)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds


    def _estimate_nested_outcomes(self, y, x, xm, smpls, external_predictions, n_jobs_cv, return_models, ymx_external, nested_external):
        # Separate the training set into two disjointed sets: mu and delta.
        train_idx, test_idx = extract_sets_from_smpls(smpls, index=0)
        mu_idx, delta_idx = split_smpls(train_idx, self.smpls_ratio)

        # Recombine the disjointed sets into a smpls like structure.
        mu_delta_smpls = recombine_samples(mu_idx, delta_idx)
        mu_test_smpls = recombine_samples(mu_idx, test_idx)
        delta_test_smpls = recombine_samples(delta_idx, test_idx)

        # Get samples conditional on treatment for estimation.
        _, mu_test_smpls_d1 = _get_cond_smpls(mu_test_smpls, self.treated)
        _, mu_delta_smpls_d1 = _get_cond_smpls(mu_delta_smpls, self.treated)
        smpls_delta_0, _ = _get_cond_smpls(delta_test_smpls, self.treated)

        # TODO: maybe the estimator entring in g_nested is g_d1_m_hat_list or something different. Check
        if ymx_external:
            ymx_hat = {"preds": external_predictions["ml_ymx"], "targets": None, "models": None}
        else:
            ymx_hat = _dml_cv_predict(
                self._learner["ml_ymx"],
                xm,
                y,
                smpls=mu_test_smpls_d1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_ymx"),
                method=self._predict_method["ml_ymx"],
                return_models=return_models,
            )

        if nested_external:
            nested_hat = {"preds": external_predictions["ml_nested"], "targets": None, "models": None}
        else:
            ymx_delta_hat = _dml_cv_predict(
                self._learner["ml_ymx"],
                xm,
                y,
                smpls=mu_delta_smpls_d1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_ymx"),
                method=self._predict_method["ml_ymx"],
                return_models=return_models,
            )
            nested_hat = _dml_cv_predict(
                self._learner["ml_nested"],
                x,
                ymx_delta_hat,
                smpls=delta_test_smpls,  # TODO: Change this sample
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_nested"),
                method=self._predict_method["ml_nested"],
                return_models=return_models,
            )

        return ymx_hat, nested_hat

    def _score_elements(self, y, px_hat, pmx_hat, yx_hat, ymx_hat, nested_hat, smpls):
        if self.mediation_level == self.treatment_level:
            u_hat = y-ymx_hat
            psi_a=-1.0
            psi_b=ymx_hat+ np.divide(np.multiply(self.treated, u_hat), px_hat)
        else:

            u_hat = y - ymx_hat
            w_hat = ymx_hat - nested_hat

            propensity1 = np.divide(np.multiply(self.treated, 1.0 - pmx_hat), np.multiply(1.0 - px_hat, pmx_hat))
            propensity2 = np.divide(1.0 - self.treated, 1.0 - px_hat)

            adjusted_propensity1, adjusted_propensity2 = _normalize_propensity_med(
                normalize_ipw=self.normalize_ipw,
                score_function="efficient-alt",
                outcome="counterfactual",
                treatment_indicator=self.treated,
                propensity_score=px_hat,
                propensity_score_med=pmx_hat,
            )
            adjusted_propensity1 = np.multiply(propensity1, np.multiply(adjusted_propensity1))
            adjusted_propensity2 = np.multiply(propensity2, np.multiply(adjusted_propensity2))

            psi_a = -1.0
            # psi_b = np.multiply(adjusted_propensity1, u_hat) + np.multiply(adjusted_propensity2, w_hat) + g_nested_hat
            psi_b = (
                np.multiply(np.multiply(np.divide(self.treated, 1.0 - px_hat), np.divide(1.0 - pmx_hat, pmx_hat)), u_hat)
                + np.multiply(np.divide(1.0 - self.treated, 1.0 - px_hat), w_hat)
                + nested_hat
            )
        return psi_a, psi_b

    # TODO: Refactor tuning to take away all of the mentions to d0, d1 and others.
    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        pass

    def _sensitivity_element_est(self, preds):
        pass

    def _check_data(self, med_data):
        if not isinstance(med_data, DoubleMLMediationData):
            raise TypeError(
                f"The data must be of DoubleMLMediationData type. {str(med_data)} "
                f"of type {str(type(med_data))} was passed."
            )

    #TODO: Change this for efficient-alt only.
    def _check_score_functions(self):
        valid_score_function = ["efficient", "efficient-alt"]
        if self.score_function == "efficient":
            if self._med_data.n_meds > 1:
                raise ValueError(
                    f"score_function defined as {self.score_function}. "
                    + f"Mediation analysis based on {self.score_function} scores assumes only one mediation variable. "
                    + f"Data contains {self._med_data.n_meds} mediation variables. "
                    + "Please choose another score_function for mediation analysis."
                )
            if not self._med_data.binary_meds.all():
                raise ValueError(
                    "Mediation analysis based on efficient scores requires a binary mediation variable"
                    + "with integer values equal to 0 or 1 and no missing values."
                    + f"Actual data contains {np.unique(self._med_data.data.m)}"
                    + "unique values and/or may contain missing values."
                )
        if self.score_function in valid_score_function and not self._med_data.force_all_m_finite:
            raise ValueError(
                f"Mediation analysis based on {str(valid_score_function)} "
                f"requires finite mediation variables with no missing values."
            )
        # TODO: Probably want to check that elements of mediation variables are floats or ints.
