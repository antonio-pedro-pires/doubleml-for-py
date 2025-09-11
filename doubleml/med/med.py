import numpy as np
from sklearn.utils import check_consistent_length, check_X_y

from doubleml import DoubleMLMediationData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import _normalize_propensity_med
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import _cond_targets, _dml_cv_predict, _dml_tune, _get_cond_smpls, _get_cond_smpls_2d


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
        mediation_level,
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
        ml_g,
        ml_m,
        ml_med=None,
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

        self._learner = ["ml_m", "ml_g"]
        self._check_learner(learner=ml_m, learner_name="ml_m", regressor=False, classifier=True)
        is_classifier_ml_g = self._check_learner(learner=ml_g, learner_name="ml_g", regressor=True, classifier=True)
        self._predict_method = {"ml_m": "predict_proba"}
        if is_classifier_ml_g:
            self._predict_method["ml_g"] = "predict_proba"
        else:
            self._predict_method["ml_g"] = "predict"
        if score_function == "efficient":
            self._learner.append("ml_med")
            is_ml_med_classifier = self._check_learner(learner=ml_med, learner_name="ml_med", regressor=True, classifier=True)
            if is_ml_med_classifier:
                self._predict_method["ml_med"] = "predict_proba"
            else:
                self._predict_method["ml_med"] = "predict"
        elif score_function == "efficient-alt":
            self._learner.append("ml_nested")
            is_ml_nested_classifier = self._check_learner(
                learner=ml_nested, learner_name="ml_nested", regressor=True, classifier=True
            )
            if is_ml_nested_classifier:
                self._predict_method["is_ml_nested_classifier"] = "predict_proba"
            else:
                self._predict_method["is_ml_nested_classifier"] = "predict"
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
        if self.score_function == "efficient":
            valid_learner = ["ml_g_d1_m1", "ml_g_d1_m0", "ml_m", "ml_med_d1", "ml_med_d0"]
        elif self.score_function == "efficient-alt":
            valid_learner = ["ml_g_1", "ml_g_nested", "ml_m", "ml_m_med"]

        self._params = {learner: {key: [None] * self.n_rep for key in self._med_data.d_cols} for learner in valid_learner}

    def _nuisance_est(
        self,
        smpls,
        external_predictions,
        n_jobs_cv,
        return_models=False,
    ):
        # TODO: For each nuisance_est function, don't forget to trim predictions.
        if self.score_function == "efficient":
            return self._nuisance_est_counterfactual_efficient(smpls, external_predictions, n_jobs_cv, return_models)
        elif self.score_function == "efficient-alt":
            return self._nuisance_est_counterfactual_efficient_alt(smpls, external_predictions, n_jobs_cv, return_models)

    def _nuisance_est_counterfactual_efficient(self, smpls, external_predictions, n_jobs_cv, return_models):
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        _, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        _, m = check_X_y(x, self._med_data.m, force_all_finite=False)

        # Check whether there are external predictions for each parameter.
        m_external = external_predictions["ml_m"] is not None
        g_d1_m1_external = external_predictions["ml_g_d1_m1"] is not None
        g_d1_m0_external = external_predictions["ml_g_d1_m0"] is not None
        med_d1_external = external_predictions["ml_med_d1"] is not None
        med_d0_external = external_predictions["ml_med_d0"] is not None

        # TODO: Redo samples so that m is defined based on d
        # smpls_d0 points to the non-treated observations, smpls_d1 points to the treated observations.
        # smpls_d1_m0 points to the treated but not mediated observations,
        # the other variations (d1_m1, d0_m0, d0_m1) point.
        # to observations with different combinations of treatment and mediation present.
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, self.treated)
        _, _, smpls_d1_m0, smpls_d1_m1 = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

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
        if g_d1_m1_external:
            g_d1_m1_hat = {"preds": external_predictions["ml_g_d1_m1"], "targets": None, "models": None}
        else:
            g_d1_m1_hat = _dml_cv_predict(
                self._learner["ml_g_d1_m1"],
                x,
                y,
                smpls=smpls_d1_m1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_d1_m1"),
                method=self._predict_method["ml_g_d1_m1"],
                return_models=return_models,
            )
        if g_d1_m0_external:
            g_d1_m0_hat = {"preds": external_predictions["ml_g_d1_m0"], "targets": None, "models": None}
        else:
            g_d1_m0_hat = _dml_cv_predict(
                self._learner["ml_g_d1_m0"],
                x,
                d,
                smpls=smpls_d1_m0,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_d1_m0"),
                method=self._predict_method["ml_g_d1_m0"],
                return_models=return_models,
            )
        if med_d1_external:
            med_d1_hat = {"preds": external_predictions["ml_med_d1"], "targets": None, "models": None}
        else:
            med_d1_hat = _dml_cv_predict(
                self._learner["ml_med_d1"],
                x,
                y,
                smpls=smpls_d1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_med_d1"),
                method=self._predict_method["ml_med_d1"],
                return_models=return_models,
            )
        if med_d0_external:
            med_d0_hat = {"preds": external_predictions["ml_med_d0"], "targets": None, "models": None}
        else:
            med_d0_hat = _dml_cv_predict(
                self._learner["ml_med_d0"],
                x,
                y,
                smpls=smpls_d0,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_med_d0"),
                method=self._predict_method["ml_med_d0"],
                return_models=return_models,
            )

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                "ml_g_d1_m1": g_d1_m1_hat["preds"],
                "ml_g_d1_m0": g_d1_m0_hat["preds"],
                "ml_med_d1": med_d1_hat["preds"],
                "ml_med_d0": med_d0_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                "ml_g_d1_m1": g_d1_m1_hat["targets"],
                "ml_g_d1_m0": g_d1_m0_hat["targets"],
                "ml_med_d1": med_d1_hat["targets"],
                "ml_med_d0": med_d0_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_g_d1_m1": g_d1_m1_hat["models"],
                "ml_g_d1_m0": g_d1_m0_hat["models"],
                "ml_med_d1": med_d1_hat["models"],
                "ml_med_d0": med_d0_hat["models"],
            },
        }

        psi_a, psi_b = self._score_counterfactual_outcome(y, m_hat, g_d1_m1_hat, g_d1_m0_hat, med_d1_hat, med_d0_hat, smpls)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _nuisance_est_counterfactual_efficient_alt(self, smpls, external_predictions, n_jobs_cv, return_models):
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        x, m = check_X_y(x, self._med_data.m, force_all_finite=False)
        xm = np.column_stack(x, m)

        # Check whether there are external predictions for each parameter.
        m_external = external_predictions["ml_m"] is not None
        g_1_external = external_predictions["ml_g_1"] is not None
        g_nested_external = external_predictions["ml_g_nested"] is not None
        m_med_external = external_predictions["ml_m_med"] is not None

        # TODO: Samples have to be split into musample and deltasample.
        # Prepare the samples
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, self.treated)
        _, _, smpls_d1_m0, smpls_d1_m1 = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

        # TODO: Often required to fit learners on both samples.
        # TODO: Maybe will need to use other strategy than _dml_cv_predict()
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
        if g_1_external:
            g_1_hat = {"preds": external_predictions["ml_g_1"], "targets": None, "models": None}
        else:
            g_1_hat = _dml_cv_predict(
                self._learner["ml_g"],
                xm,
                y,
                smpls=smpls_d1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_1"),
                method=self._predict_method["ml_g"],
                return_models=return_models,
            )
        if g_nested_external:
            g_nested_hat = {"preds": external_predictions["ml_g_nested"], "targets": None, "models": None}
        else:
            g_nested_hat = _dml_cv_predict(
                self._learner["ml_g_nested"],
                x,
                g_1_hat,
                smpls=smpls_d1_m0,  # TODO: Change this sample
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_nested"),
                method=self._predict_method["ml_g_nested"],
                return_models=return_models,
            )
        if m_med_external:
            m_med_hat = {"preds": external_predictions["ml_m_med"], "targets": None, "models": None}
        else:
            m_med_hat = _dml_cv_predict(
                self._learner["ml_m_med"],
                xm,
                d,
                smpls=smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_m_med"),
                method=self._predict_method["ml_m_med"],
                return_models=return_models,
            )

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                "ml_g_1": g_1_hat["preds"],
                "ml_g_nested": g_nested_hat["preds"],
                "ml_m_med": m_med_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                "ml_g_1": g_1_hat["targets"],
                "ml_g_nested": g_nested_hat["targets"],
                "ml_m_med": m_med_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_g_1": g_1_hat["models"],
                "ml_g_nested": g_nested_hat["models"],
                "ml_m_med": m_med_hat["models"],
            },
        }

        psi_a, psi_b = self._score_counterfactual_alt_outcome(y, m_hat, g_1_hat, g_nested_hat, m_med_hat, smpls)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_counterfactual_outcome(self, y, m_hat, g_d1_m1_hat, g_d1_m0_hat, med_d1_hat, med_d0_hat, smpls):
        propensity1 = np.divide(np.multiply(self.treated, med_d0_hat), np.multiply(m_hat, med_d1_hat))
        propensity2 = np.divide(1.0 - self.treated, 1.0 - m_hat)
        adjusted_propensity1, adjusted_propensity2 = _normalize_propensity_med(
            normalize_ipw=self.normalize_ipw,
            score_function="efficient",
            outcome="counterfactual",
            treatment_indicator=self.treated,
            propensity_score=m_hat,
            conditional_pot_med_prob=med_d1_hat,
            conditional_counter_med_prob=med_d0_hat,
        )

        adjusted_propensity1 = np.multiply(propensity1, np.multiply(adjusted_propensity1))
        adjusted_propensity2 = np.multiply(propensity2, np.multiply(adjusted_propensity2))
        mu_hat = np.multiply(self.mediated, g_d1_m1_hat) + np.multiply((1.0 - self.mediated), g_d1_m0_hat)
        g_mu_hat = np.multiply(g_d1_m1_hat, med_d0_hat) + np.multiply(g_d1_m0_hat, (1.0 - med_d0_hat))

        u_hat = y - mu_hat
        w_hat = mu_hat - g_mu_hat

        # TODO: Continue here next time
        adjusted_propensity_score1, adjusted_propensity_score2 = _normalize_propensity_med(
            self.normalize_ipw,
            score=self.score,
            outcome="counterfactual",
            treatment_indicator=self.treated,
            propensity_score=m_hat,
            conditional_pot_med_prob=med_d1_hat,
            conditional_counter_med_prob=med_d0_hat,
        )

        psi_a = -1.0
        # psi_b = np.multiply(adjusted_propensity1, u_hat) + np.multiply(adjusted_propensity2, w_hat) + w_hat
        psi_b = (
            np.multiply(np.multiply(np.divide(self.treated, m_hat), np.divide(med_d0_hat, med_d1_hat)), u_hat)
            + np.multiply(np.divide(1.0 - self.treated, 1.0 - m_hat), w_hat)
            + w_hat
        )
        return psi_a, psi_b

    def _score_counterfactual_alt_outcome(self, y, m_hat, g_1_hat, g_nested_hat, m_med_hat, smpls):
        u_hat = y - g_1_hat
        w_hat = g_1_hat - g_nested_hat

        propensity1 = np.divide(np.multiply(self.treated, 1.0 - m_med_hat), np.multiply(1.0 - m_hat, m_med_hat))
        propensity2 = np.divide(1.0 - self.treated, 1.0 - m_hat)

        adjusted_propensity1, adjusted_propensity2 = _normalize_propensity_med(
            normalize_ipw=self.normalize_ipw,
            score_function="efficient-alt",
            outcome="counterfactual",
            treatment_indicator=self.treated,
            propensity_score=m_hat,
            propensity_score_med=m_med_hat,
        )
        adjusted_propensity1 = np.multiply(propensity1, np.multiply(adjusted_propensity1))
        adjusted_propensity2 = np.multiply(propensity2, np.multiply(adjusted_propensity2))

        psi_a = -1.0
        # psi_b = np.multiply(adjusted_propensity1, u_hat) + np.multiply(adjusted_propensity2, w_hat) + g_nested_hat
        psi_b = (
            np.multiply(np.multiply(np.divide(self.treated, 1.0 - m_hat), np.divide(1.0 - m_med_hat, m_med_hat)), u_hat)
            + np.multiply(np.divide(1.0 - self.treated, 1.0 - m_hat), w_hat)
            + w_hat
        )
        return psi_a, psi_b

    # TODO: Refactor tuning to take away all of the mentions to d0, d1 and others.
    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        if self.score_function == "efficient":
            res = self._counterfactual_tuning()
        elif self.score_function == "efficient-alt":
            res = self._counterfactual_alt_tuning()
        return res

    def _counterfactual_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        # TODO: Apply counterfactual_tuning for score_function == "efficient_alt".
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        # TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(
            x, self._med_data["z"]
        )  # Check that the mediators have the same number of samples as X and

        treated = self.treated
        mediated = self.mediated
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        smpls_d0_m0, smpls_d0_m1, smpls_d1_m0, smpls_d1_m1 = _get_cond_smpls_2d(smpls, treated, mediated)

        dx = np.column_stack((d, x))

        # Learner for E(Y|D=d, M=d, X)
        ml_g_d1_m1 = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        ml_g_d1_m0 = self.params_names[1]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d_lvl0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d_lvl1 = [train_index for (train_index, _) in smpls_d1]

        # TODO: Check which ml_g_1_m to use
        if ml_g_d1_m1 == "ml_g_d0_m0":
            # smpls_d1_m1 = smpls_d0_m0
            # smpls_d1_m0 = smpls_d0_m1
            train_inds_pot = [train_index for (train_index, _) in smpls_d0_m0]
            train_inds_counter = [train_index for (train_index, _) in smpls_d0_m1]
        else:
            # smpls_d1_m1 = smpls_d1_m1
            # smpls_d1_m0 = smpls_d1_m0
            train_inds_pot = [train_index for (train_index, _) in smpls_d1_m1]
            train_inds_counter = [train_index for (train_index, _) in smpls_d1_m0]

        g_d1_m1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_pot,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_d1_m0_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_counter,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        m_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds,
            self._learner["ml_m"],
            param_grids["ml_m"],
            scoring_methods["ml_m"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        med_d0_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_d_lvl0,
            self._learner["ml_med"],
            param_grids["ml_med"],
            scoring_methods["ml_med"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        med_d1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_d_lvl1,
            self._learner["ml_med"],
            param_grids["ml_med"],
            scoring_methods["ml_med"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        g_d1_m1_best_params = [xx.best_params_ for xx in g_d1_m1_tune_res]
        g_d1_m0_best_params = [xx.best_params_ for xx in g_d1_m0_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        med_d0_best_params = [xx.best_params_ for xx in med_d0_tune_res]
        med_d1_best_params = [xx.best_params_ for xx in med_d1_tune_res]

        params = {
            "ml_g_d1_m1": g_d1_m1_best_params,
            "ml_g_d1_m0": g_d1_m0_best_params,
            "ml_m": m_best_params,
            "ml_med_d0": med_d0_best_params,
            "ml_med_d1": med_d1_best_params,
        }
        tune_res = {
            "ml_g_d1_m1": g_d1_m1_tune_res,
            "ml_g_d1_m0": g_d1_m0_tune_res,
            "ml_m": m_tune_res,
            "ml_med_d0": med_d0_tune_res,
            "ml_med_d1": med_d1_tune_res,
        }

        res = {"params": params, "tune_res": tune_res}

        return res

    # TODO: Modify this method (taken from the efficient scoring) to perform the efficient-alt tuning.
    def _counterfactual_alt_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        # TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(
            x, self._med_data["z"]
        )  # Check that the mediators have the same number of samples as X and

        treated = self.treated
        mediated = self.mediated
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        smpls_d0_m0, smpls_d0_m1, smpls_d1_m0, smpls_d1_m1 = _get_cond_smpls_2d(smpls, treated, mediated)

        dx = np.column_stack((d, x))

        # Learner for E(Y|D=d, M=d, X)
        ml_g_d1_m1 = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        ml_g_d1_m0 = self.params_names[1]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d_lvl0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d_lvl1 = [train_index for (train_index, _) in smpls_d1]

        # TODO: Check which ml_g_d_m to use
        if ml_g_d1_m1 == "ml_g_d0_m0":
            # smpls_d1_m1 = smpls_d0_m0
            # smpls_d1_m0 = smpls_d0_m1
            train_inds_pot = [train_index for (train_index, _) in smpls_d0_m0]
            train_inds_counter = [train_index for (train_index, _) in smpls_d0_m1]
        else:
            # smpls_d1_m1 = smpls_d1_m1
            # smpls_d1_m0 = smpls_d1_m0
            train_inds_pot = [train_index for (train_index, _) in smpls_d1_m1]
            train_inds_counter = [train_index for (train_index, _) in smpls_d1_m0]

        g_d1_m1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_pot,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_d1_m0_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_counter,
            self._learner["ml_g"],
            param_grids["ml_g"],
            scoring_methods["ml_g"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        m_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds,
            self._learner["ml_m"],
            param_grids["ml_m"],
            scoring_methods["ml_m"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        med_d0_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_d_lvl0,
            self._learner["ml_med"],
            param_grids["ml_med"],
            scoring_methods["ml_med"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        med_d1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_d_lvl1,
            self._learner["ml_med"],
            param_grids["ml_med"],
            scoring_methods["ml_med1"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        g_d1_m1_best_params = [xx.best_params_ for xx in g_d1_m1_tune_res]
        g_d1_m0_best_params = [xx.best_params_ for xx in g_d1_m0_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        med_d0_best_params = [xx.best_params_ for xx in med_d0_tune_res]
        med_d1_best_params = [xx.best_params_ for xx in med_d1_tune_res]

        params = {
            "ml_g_d1_m1": g_d1_m1_best_params,
            "ml_g_d1_m0": g_d1_m0_best_params,
            "ml_m": m_best_params,
            "ml_med_d0": med_d0_best_params,
            "ml_med_d1": med_d1_best_params,
        }
        tune_res = {
            "ml_g_d1_m1": g_d1_m1_tune_res,
            "ml_g_d1_m0": g_d1_m0_tune_res,
            "ml_m": m_tune_res,
            "ml_med_d0": med_d0_tune_res,
            "ml_med_d1": med_d1_tune_res,
        }

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
