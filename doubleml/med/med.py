import numpy as np
from sklearn.utils import check_consistent_length, check_X_y

from doubleml import DoubleMLMediationData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import _normalize_propensity_med
from doubleml.utils._checks import _check_score
from doubleml.utils._estimation import _dml_cv_predict, _dml_tune, _get_cond_smpls, _get_cond_smpls_2d


# TODO: Transplant methods into utils documents.
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
        self._mediation_level = mediation_level
        self._treated = self._med_data.d == treatment_level
        self._mediated = self._med_data.m == mediation_level

        self._learner = ["ml_m", "ml_g"]
        self._predict_method = {"ml_m": "predict_proba"}
        self._check_learner(learner=ml_m, learner_name="ml_m", regressor=False, classifier=True)
        is_classifier_ml_g = self._check_learner(learner=ml_g, learner_name="ml_g", regressor=True, classifier=True)
        if is_classifier_ml_g:
            self._predict_method["ml_g"] = "predict_proba"
        else:
            self._predict_method["ml_g"] = "predict"
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

    @property
    def is_potential_outcome(self):
        """
        Indicates whether the current score function computes the potential outcome: Y(d, m(d)).
        """
        return self.treatment_level == self.mediation_level

    def _initialize_ml_nuisance_params(self):
        valid_learner = ["ml_m", "ml_g_d"]
        self._params = {learner: {key: [None] * self.n_rep for key in self._med_data.d_cols} for learner in valid_learner}

    def _nuisance_est(
        self,
        smpls,
        external_predictions,
        n_jobs_cv,
        return_models=False,
    ):

        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)

        # Check whether there are external predictions for each parameter.
        m_external = external_predictions["ml_m"] is not None
        g_d_external = external_predictions["ml_g_d"] is not None

        # Prepare the samples
        _, smpls_d = _get_cond_smpls(smpls, self.treated)

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

        if g_d_external:
            g_d_hat = {"preds": external_predictions["ml_g_d"], "targets": None, "models": None}
        else:
            g_d_hat = _dml_cv_predict(
                self._learner["ml_g_d"],
                x,
                y,
                smpls=smpls_d,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_d"),
                method=self._predict_method["ml_g_d"],
                return_models=return_models,
            )

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                "ml_g_d": g_d_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                "ml_g_d": g_d_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_g_d": g_d_hat["models"],
            },
        }

        psi_a, psi_b = self._score_potential_outcome(y, x, d, m_hat["preds"], g_d_hat["preds"])
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_elements(self, y, m_hat, g_d_hat):
        u_hat = y - g_d_hat
        propensity_score = np.multiply(
            np.divide(self.treated, m_hat),
            _normalize_propensity_med(
                self.normalize_ipw,
                self.score,
                self.score_function,
                self.treated,
            ),
        )

        psi_a = -1.0
        psi_b = propensity_score * (u_hat) + g_d_hat
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
        smpls_1md, smpls_d = _get_cond_smpls(smpls, treated)

        train_inds = [train_index for (train_index, _) in smpls]
        # train_inds_d = [train_index for (train_index, _) in smpls_d]
        train_inds_g = None

        # TODO: Check what this does
        if scoring_methods is None:
            scoring_methods = {"ml_g_d": None, "ml_m": None}

        g_d_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_g,
            self._learner["ml_g_d"],
            param_grids["ml_g_d"],
            scoring_methods["ml_g_d"],
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

        g_d_best_params = [xx.best_params_ for xx in g_d_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]

        params = {"ml_g_d": g_d_best_params, "ml_m": m_best_params}
        tune_res = {"ml_g_d": g_d_tune_res, "ml_m": m_tune_res}

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
            valid_learner = ["ml_g_d_med_pot", "ml_g_d_med_counter", "ml_m", "ml_med_pot", "ml_med_counter"]
        elif self.score_function == "efficient-alt":
            valid_learner = ["ml_g_d", "ml_g_nested", "ml_m", "ml_m_med"]

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
        g_d_med_pot_external = external_predictions["ml_g_d_med_pot"] is not None
        g_d_med_counter_external = external_predictions["ml_g_d_med_counter"] is not None
        med_pot_external = external_predictions["ml_med_pot"] is not None
        med_counter_external = external_predictions["ml_med_counter"] is not None

        # Prepare the samples
        smpls_1minusd, smpls_d = _get_cond_smpls(smpls, self.treated)
        _, _, smpls_d_1minusd, smpls_d_d = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

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
        if g_d_med_pot_external:
            g_d_med_pot_hat = {"preds": external_predictions["ml_g_d_med_pot"], "targets": None, "models": None}
        else:
            g_d_med_pot_hat = _dml_cv_predict(
                self._learner["ml_g_d_med_pot"],
                x,
                y,
                smpls=smpls_d_d,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_d_med_pot"),
                method=self._predict_method["ml_g_d_med_pot"],
                return_models=return_models,
            )
        if g_d_med_counter_external:
            g_d_med_counter_hat = {"preds": external_predictions["ml_g_d_med_counter"], "targets": None, "models": None}
        else:
            g_d_med_counter_hat = _dml_cv_predict(
                self._learner["ml_g_d_med_counter"],
                x,
                d,
                smpls=smpls_d_1minusd,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_d_med_counter"),
                method=self._predict_method["ml_g_d_med_counter"],
                return_models=return_models,
            )
        if med_pot_external:
            med_pot_hat = {"preds": external_predictions["ml_med_pot"], "targets": None, "models": None}
        else:
            med_pot_hat = _dml_cv_predict(
                self._learner["ml_med_pot"],
                x,
                y,
                smpls=smpls_d,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_med_pot"),
                method=self._predict_method["ml_med_pot"],
                return_models=return_models,
            )
        if med_counter_external:
            med_counter_hat = {"preds": external_predictions["ml_med_counter"], "targets": None, "models": None}
        else:
            med_counter_hat = _dml_cv_predict(
                self._learner["ml_med_counter"],
                x,
                y,
                smpls=smpls_1minusd,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_med_counter"),
                method=self._predict_method["ml_med_counter"],
                return_models=return_models,
            )

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                "ml_g_d_med_pot": g_d_med_pot_hat["preds"],
                "ml_g_d_med_counter": g_d_med_counter_hat["preds"],
                "ml_med_pot": med_pot_hat["preds"],
                "ml_med_counter": med_counter_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                "ml_g_d_med_pot": g_d_med_pot_hat["targets"],
                "ml_g_d_med_counter": g_d_med_counter_hat["targets"],
                "ml_med_pot": med_pot_hat["targets"],
                "ml_med_counter": med_counter_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_g_d_med_pot": g_d_med_pot_hat["models"],
                "ml_g_d_med_counter": g_d_med_counter_hat["models"],
                "ml_med_pot": med_pot_hat["models"],
                "ml_med_counter": med_counter_hat["models"],
            },
        }

        psi_a, psi_b = self._score_counterfactual_outcome(
            y, m_hat, g_d_med_pot_hat, g_d_med_counter_hat, med_pot_hat, med_counter_hat, smpls
        )
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _nuisance_est_counterfactual_efficient_alt(self, smpls, external_predictions, n_jobs_cv, return_models):
        x, y = check_X_y(self._med_data.x, self._med_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._med_data.d, force_all_finite=False)
        x, m = check_X_y(x, self._med_data.m, force_all_finite=False)
        xm = np.column_stack(x, m)

        # Check whether there are external predictions for each parameter.
        m_external = external_predictions["ml_m"] is not None
        g_d_external = external_predictions["ml_g_d"] is not None
        g_nested_external = external_predictions["ml_g_nested"] is not None
        m_med_external = external_predictions["ml_m_med"] is not None

        # TODO: Samples have to be split into musample and deltasample.
        # Prepare the samples
        smpls_1minusd, smpls_d = _get_cond_smpls(smpls, self.treated)
        _, _, smpls_d_1minusd, smpls_d_d = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

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
        if g_d_external:
            g_d_hat = {"preds": external_predictions["ml_g_d"], "targets": None, "models": None}
        else:
            g_d_hat = _dml_cv_predict(
                self._learner["ml_g_d"],
                xm,
                y,
                smpls=smpls_d,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_g_d"),
                method=self._predict_method["ml_g_d"],
                return_models=return_models,
            )
        if g_nested_external:
            g_nested_hat = {"preds": external_predictions["ml_g_nested"], "targets": None, "models": None}
        else:
            g_nested_hat = _dml_cv_predict(
                self._learner["ml_g_nested"],
                x,
                g_d_hat,
                smpls=smpls_d_1minusd,  # TODO: Change this sample
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
                "ml_g_d": g_d_hat["preds"],
                "ml_g_nested": g_nested_hat["preds"],
                "ml_m_med": m_med_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                "ml_g_d": g_d_hat["targets"],
                "ml_g_nested": g_nested_hat["targets"],
                "ml_m_med": m_med_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_g_d": g_d_hat["models"],
                "ml_g_nested": g_nested_hat["models"],
                "ml_m_med": m_med_hat["models"],
            },
        }

        psi_a, psi_b = self._score_counterfactual_outcome(y, m_hat, g_d_hat, g_nested_hat, m_med_hat, smpls)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        return psi_elements, preds

    def _score_counterfactual_outcome(
        self, y, m_hat, g_d_med_pot_hat, g_d_med_counter_hat, med_pot_hat, med_counter_hat, smpls
    ):
        mu_hat = np.multiply(self.mediated, g_d_med_pot_hat) + np.multiply((1.0 - self.mediated), g_d_med_counter_hat)
        g_mu_hat = np.multiply(g_d_med_pot_hat, med_counter_hat) + np.multiply(g_d_med_counter_hat, (1.0 - med_counter_hat))

        u_hat = y - mu_hat
        w_hat = mu_hat - g_mu_hat

        psi_a = -1.0
        term1, term2, term3 = self._normalize_counterfactual(med_counter_hat, m_hat, med_pot_hat, g_mu_hat, u_hat, w_hat)
        psi_b = term1 + term2 + term3

        return psi_a, psi_b

    def _score_counterfactual_alt_outcome(self, y, m_hat, g_d_hat, g_nested_hat, m_med_hat, smpls):
        u_hat = y - g_d_hat
        w_hat = g_d_hat - g_nested_hat

        psi_a = -1.0
        term1, term2, term3 = self._normalize_counterfactual(m_hat, g_d_hat, g_nested_hat, m_med_hat, u_hat, w_hat)
        psi_b = term1 + term2 + term3

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
        smpls_1md, smpls_d = _get_cond_smpls(smpls, treated)
        smpls_1md_1md, smpls_1md_d, smpls_d_1md, smpls_d_d = _get_cond_smpls_2d(smpls, treated, mediated)

        dx = np.column_stack((d, x))

        # Learner for E(Y|D=d, M=d, X)
        ml_g_d_med_pot = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        ml_g_d_med_counter = self.params_names[1]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d_lvl0 = [train_index for (train_index, _) in smpls_1md]
        train_inds_d_lvl1 = [train_index for (train_index, _) in smpls_d]

        # TODO: Check which ml_g_d_m to use
        if ml_g_d_med_pot == "ml_g_d0_m0":
            # smpls_d_d = smpls_1md_1md
            # smpls_d_1md = smpls_1md_d
            train_inds_pot = [train_index for (train_index, _) in smpls_1md_1md]
            train_inds_counter = [train_index for (train_index, _) in smpls_1md_d]
        else:
            # smpls_d_d = smpls_d_d
            # smpls_d_1md = smpls_d_1md
            train_inds_pot = [train_index for (train_index, _) in smpls_d_d]
            train_inds_counter = [train_index for (train_index, _) in smpls_d_1md]

        g_d_med_pot_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_pot,
            self._learner[ml_g_d_med_pot],
            param_grids[ml_g_d_med_pot],
            scoring_methods[ml_g_d_med_pot],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_d_med_counter_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_counter,
            self._learner[ml_g_d_med_counter],
            param_grids[ml_g_d_med_counter],
            scoring_methods[ml_g_d_med_counter],
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
            self._learner["ml_med_d0"],
            param_grids["ml_med_d0"],
            scoring_methods["ml_med_d0"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        med_d1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_d_lvl1,
            self._learner["ml_med_d1"],
            param_grids["ml_med_d1"],
            scoring_methods["ml_med_d1"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        g_d_med_pot_best_params = [xx.best_params_ for xx in g_d_med_pot_tune_res]
        g_d_med_counter_best_params = [xx.best_params_ for xx in g_d_med_counter_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        med_d0_best_params = [xx.best_params_ for xx in med_d0_tune_res]
        med_d1_best_params = [xx.best_params_ for xx in med_d1_tune_res]

        params = {
            "ml_g_d_med_pot": g_d_med_pot_best_params,
            "ml_g_d_med_counter": g_d_med_counter_best_params,
            "ml_m": m_best_params,
            "ml_med_d0": med_d0_best_params,
            "ml_med_d1": med_d1_best_params,
        }
        tune_res = {
            "ml_g_d_med_pot": g_d_med_pot_tune_res,
            "ml_g_d_med_counter": g_d_med_counter_tune_res,
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
        smpls_1md, smpls_d = _get_cond_smpls(smpls, treated)
        smpls_1md_1md, smpls_1md_d, smpls_d_1md, smpls_d_d = _get_cond_smpls_2d(smpls, treated, mediated)

        dx = np.column_stack((d, x))

        # Learner for E(Y|D=d, M=d, X)
        ml_g_d_med_pot = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        ml_g_d_med_counter = self.params_names[1]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d_lvl0 = [train_index for (train_index, _) in smpls_1md]
        train_inds_d_lvl1 = [train_index for (train_index, _) in smpls_d]

        # TODO: Check which ml_g_d_m to use
        if ml_g_d_med_pot == "ml_g_d0_m0":
            # smpls_d_d = smpls_1md_1md
            # smpls_d_1md = smpls_1md_d
            train_inds_pot = [train_index for (train_index, _) in smpls_1md_1md]
            train_inds_counter = [train_index for (train_index, _) in smpls_1md_d]
        else:
            # smpls_d_d = smpls_d_d
            # smpls_d_1md = smpls_d_1md
            train_inds_pot = [train_index for (train_index, _) in smpls_d_d]
            train_inds_counter = [train_index for (train_index, _) in smpls_d_1md]

        g_d_med_pot_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_pot,
            self._learner[ml_g_d_med_pot],
            param_grids[ml_g_d_med_pot],
            scoring_methods[ml_g_d_med_pot],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_d_med_counter_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_counter,
            self._learner[ml_g_d_med_counter],
            param_grids[ml_g_d_med_counter],
            scoring_methods[ml_g_d_med_counter],
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
            self._learner["ml_med_d0"],
            param_grids["ml_med_d0"],
            scoring_methods["ml_med_d0"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        med_d1_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_d_lvl1,
            self._learner["ml_med_d1"],
            param_grids["ml_med_d1"],
            scoring_methods["ml_med_d1"],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )

        g_d_med_pot_best_params = [xx.best_params_ for xx in g_d_med_pot_tune_res]
        g_d_med_counter_best_params = [xx.best_params_ for xx in g_d_med_counter_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        med_d0_best_params = [xx.best_params_ for xx in med_d0_tune_res]
        med_d1_best_params = [xx.best_params_ for xx in med_d1_tune_res]

        params = {
            "ml_g_d_med_pot": g_d_med_pot_best_params,
            "ml_g_d_med_counter": g_d_med_counter_best_params,
            "ml_m": m_best_params,
            "ml_med_d0": med_d0_best_params,
            "ml_med_d1": med_d1_best_params,
        }
        tune_res = {
            "ml_g_d_med_pot": g_d_med_pot_tune_res,
            "ml_g_d_med_counter": g_d_med_counter_tune_res,
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
