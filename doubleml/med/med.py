import numpy as np
from sklearn.utils import check_consistent_length, check_X_y

from doubleml import DoubleMLMediationData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import _cond_targets, _dml_cv_predict, _dml_tune, _get_cond_smpls, _get_cond_smpls_2d


class DoubleMLMED(LinearScoreMixin, DoubleML):
    """Double machine learning for causal mediation analysis.

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
        ml_med,
        ml_nested=None,
        score="MED",
        score_type="efficient_alt",
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

        self._score_type = score_type
        valid_scores_types = ["efficient", "efficient-alt"]
        _check_score(score_type, valid_scores_types)

        self._treatment_level = treatment_level
        self._mediation_level = mediation_level
        self._treated = self._med_data.d == treatment_level
        self._mediated = self._med_data.m == mediation_level
        self.score_obj = None

        self._learner = ["ml_m", "ml_g"]
        self._predict_method = {"ml_m": "predict_proba", "ml_g": "predict"}
        self._check_learner(learner=ml_m, learner_name="ml_m", regressor=False, classifier=True)
        self._check_learner(learner=ml_g, learner_name="ml_g", regressor=True, classifier=True)
        if score_type == "efficient":
            self._learner.append("ml_med")
            is_ml_med_classifier = self._check_learner(learner=ml_med, learner_name="ml_med", regressor=True, classifier=True)
            if is_ml_med_classifier:
                self._predict_method["ml_med"] = "predict_proba"
            else:
                self._predict_method["ml_med"] = "predict"
        elif score_type == "efficient-alt":
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
    def score_type(self):
        """
        Indicates the type of the score function.
        """
        return self._score_type

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

    def _nuisance_est(
        self,
        score,
        smpls,
        external_predictions,
        n_jobs_cv,
        return_models=False,
    ):

        preds = self._initialize_preds_dict()
        training_params = self._get_training_params(n_jobs_cv, return_models, external_predictions, smpls)

        for learner, params in training_params:
            temp = self._extract_predictions(n_jobs_cv, return_models, **params)
            preds["predictions"][learner] = temp["preds"]
            preds["targets"][learner] = temp["targets"]
            preds["models"][learner] = temp["models"]

        # TODO: instead of feeding it predictions one by one, feed it a list.
        psi_a, psi_b = self._score_elements(**preds["predictions"])
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}
        return psi_elements, preds

    def _score_elements(self, **predictions):
        #    y = self._med_data.y
        #    d = self._med_data.d
        #    x = self._med_data.x

        #    psi_a = -1.0
        #    psi_b = 0
        if self.is_potential_outcome:
            pass
        else:
            if self.score_type == "efficient":
                pass
            elif self.score_type == "efficient_alt":
                pass
        pass

    # def _score_elements_alt(self, smpls, y, d, x, px, m_med=None, g_d=None, g_nested=None):
    #    psi_a = -1.0
    #    psi_b = None

    # m = self._med_data.m
    #    if self.normalize_ipw:
    #        n_obs = self._med_data.n_obs
    #        sumscore1 = np.sum((1 - d) * m_med / ((1 - m_med) * px))
    #        sumscore2 = np.sum(d / px)
    #        sumscore3 = sum((1 - d / (1 - px)))
    #        sumscore4 = sum(d * (1 - m_med) / (m_med * (1 - px)))
    #        if self._score == "Y(0, M(0))":
    #            psi_b = g_d + (n_obs * (1 - d) * (y - g_d) / (1 - px)) / sumscore3
    #        elif self._score == "Y(0, M(1))":
    #            psi_b = (
    #                (n_obs * (1 - d) * m_med / ((1 - m_med) * px) * (y - g_d)) / sumscore1
    #                + (n_obs * d / px * (g_d - g_nested)) / sumscore2
    #                + g_nested
    #            )
    #        elif self._score == "Y(1, M(0))":
    #            psi_b = (
    #                (n_obs * d * (1 - m_med) / (m_med * (1 - px)) * (y - g_d)) / sumscore4
    #                + (n_obs * (1 - d) / (1 - px) * (g_d - g_nested)) / sumscore3
    #                + g_nested
    #            )
    #        elif self._score == "Y(1, M(1))":
    #            psi_b = g_d + (n_obs * d * (y - g_d) / px) / sumscore2
    #    else:
    #        if self._score == "Y(0, M(0))":
    #            psi_b = g_d + (1 - d) * (y - g_d) / (1 - px)
    #        elif self._score == "Y(0, M(1))":
    #            psi_b = (1 - d) * m_med / ((1 - m_med) * px) * (y - g_d) + d / px * (g_d - g_nested) + g_nested
    #        elif self._score == "Y(1, M(0))":
    #            psi_b = d * (1 - m_med) / (m_med * (1 - px)) * (y - g_d) + (1 - d) / (1 - px) * (g_d - g_nested) + g_nested
    #        elif self._score == "Y(1, M(1))":
    #            psi_b = g_d + d * (y - g_d) / px
    #    return psi_a, psi_b

    def _normalize(self, propensity_score):
        #    mean_treat_d = np.mean(np.multiply(self.treated, propensity_score))
        #    mean_treat_1md = np.mean(np.multiply(1 - self.treated, 1 - propensity_score))

        #    if self.score == "Y(d, M(d))":
        #        np.divide(np.multiply(self.treated, propensity_score)
        #        else:
        #        pass

        #       if self.score_type == "efficient":

        #      elif self.score_type == "efficient_alt":
        pass

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        if self._score == "Y(d, M(d))":
            res = self._potential_tuning()
        elif self._score == "Y(d, M(1-d))":
            res = self._counterfactual_tuning()
        return res

    # Check tuning for samples, since everything changed so much
    def _potential_tuning(
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

    def _counterfactual_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        # TODO: Apply counterfactual_tuning for score_type == "efficient_alt".
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
        ml_g_d_med_d = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        ml_g_d_med_1md = self.params_names[1]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d_lvl0 = [train_index for (train_index, _) in smpls_1md]
        train_inds_d_lvl1 = [train_index for (train_index, _) in smpls_d]

        if ml_g_d_med_d == "ml_g_d0_m0":
            # smpls_d_d = smpls_1md_1md
            # smpls_d_1md = smpls_1md_d
            train_inds_pot = [train_index for (train_index, _) in smpls_1md_1md]
            train_inds_counter = [train_index for (train_index, _) in smpls_1md_d]
        else:
            # smpls_d_d = smpls_d_d
            # smpls_d_1md = smpls_d_1md
            train_inds_pot = [train_index for (train_index, _) in smpls_d_d]
            train_inds_counter = [train_index for (train_index, _) in smpls_d_1md]

        g_d_med_d_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_pot,
            self._learner[ml_g_d_med_d],
            param_grids[ml_g_d_med_d],
            scoring_methods[ml_g_d_med_d],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_d_med_1md_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_counter,
            self._learner[ml_g_d_med_1md],
            param_grids[ml_g_d_med_1md],
            scoring_methods[ml_g_d_med_1md],
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

        g_d_med_d_best_params = [xx.best_params_ for xx in g_d_med_d_tune_res]
        g_d_med_1md_best_params = [xx.best_params_ for xx in g_d_med_1md_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        med_d0_best_params = [xx.best_params_ for xx in med_d0_tune_res]
        med_d1_best_params = [xx.best_params_ for xx in med_d1_tune_res]

        params = {
            "ml_g_d_med_d": g_d_med_d_best_params,
            "ml_g_d_med_1md": g_d_med_1md_best_params,
            "ml_m": m_best_params,
            "ml_med_d0": med_d0_best_params,
            "ml_med_d1": med_d1_best_params,
        }
        tune_res = {
            "ml_g_d_med_d": g_d_med_d_tune_res,
            "ml_g_d_med_1md": g_d_med_1md_tune_res,
            "ml_m": m_tune_res,
            "ml_med_d0": med_d0_tune_res,
            "ml_med_d1": med_d1_tune_res,
        }

        res = {"params": params, "tune_res": tune_res}

        return res

    def _sensitivity_element_est(self, preds):
        pass

    def _extract_predictions(
        self,
        n_jobs_cv,
        return_models,
        learner,
        external_prediction,
        x,
        y,
        smpls,
    ):
        if external_prediction is not None:
            return {"preds": external_prediction, "targets": _cond_targets(y, cond_sample=smpls), "models": None}
        else:
            temp = _dml_cv_predict(
                self._learner[learner],
                x,
                y,
                smpls,
                n_jobs_cv,
                est_params=self._get_params(learner),
                method=self._predict_method[learner],
                return_models=return_models,
            )
            _check_finite_predictions(temp["preds"], self._learner[learner], learner, smpls)
            return temp

    def _check_data(self, med_data):
        if not isinstance(med_data, DoubleMLMediationData):
            raise TypeError(
                f"The data must be of DoubleMLMediationData type. {str(med_data)} "
                f"of type {str(type(med_data))} was passed."
            )

    def _check_score_types(self):
        valid_score_type = ["efficient", "efficient-alt"]
        if self.score_type == "efficient":
            if self._med_data.n_meds > 1:
                raise ValueError(
                    f"score_type defined as {self.score_type}. "
                    + f"Mediation analysis based on {self.score_type} scores assumes only one mediation variable. "
                    + f"Data contains {self._med_data.n_meds} mediation variables. "
                    + "Please choose another score_type for mediation analysis."
                )
            if not self._med_data.binary_meds.all():
                raise ValueError(
                    "Mediation analysis based on efficient scores requires a binary mediation variable"
                    + "with integer values equal to 0 or 1 and no missing values."
                    + f"Actual data contains {np.unique(self._med_data.data.m)}"
                    + "unique values and/or may contain missing values."
                )
        if self.score_type in valid_score_type and not self._med_data.force_all_m_finite:
            raise ValueError(
                f"Mediation analysis based on {str(valid_score_type)} "
                f"requires finite mediation variables with no missing values."
            )
        # TODO: Probably want to check that elements of mediation variables are floats or ints.

    def _initialize_ml_nuisance_params(self):
        valid_learner = ["ml_m"]
        if self.is_potential_outcome:
            valid_learner.append("ml_g_d")
        else:
            if self.score_type == "efficient":
                valid_learner.extend(["ml_g_d_med_d", "ml_g_d_med_1md", "ml_m", "ml_med_d", "ml_med_1md"])
            elif self.score_type == "efficient-alt":
                valid_learner.extend(["ml_g_d", "ml_g_d_1md", "ml_m", "ml_m_med_x"])

        self._params = {learner: {key: [None] * self.n_rep for key in self._med_data.d_cols} for learner in valid_learner}

    def _initialize_preds_dict(self):
        learner_dict = {learner: None for learner in self.params.keys()}
        preds = {"predictions": learner_dict, "targets": learner_dict, "models": learner_dict}
        return preds

    def _get_training_params(self, n_jobs_cv, return_models, external_predictions, smpls):
        # Label data for readability
        x = self._med_data.x
        y = self._med_data.y
        d = self._med_data.d
        m = self._med_data.m

        # TODO: Verify that the samples gotten for score_type == "efficient_alt" are musample, deltasample, psample, etc.
        # Get the conditional samples
        smpls_1md, smpls_d = _get_cond_smpls(smpls, self.treated)
        smpls_1md_1md, smpls_1md_d, smpls_d_1md, smpls_d_d = _get_cond_smpls_2d(smpls, self.treated, self.mediated)

        training_params = dict()
        if self.is_potential_outcome:
            training_params = {
                "ml_m": {
                    "external_predictions": external_predictions["ml_m"],
                    "x": x,
                    "y": d,
                    "smpls": smpls,
                },
                "ml_g_d": {
                    "external_predictions": external_predictions["ml_g_d"],
                    "x": x,
                    "y": y,
                    "smpls": smpls_d,
                },
            }
        else:
            if self.score_type == "efficient":

                training_params = {
                    "ml_m": {
                        "external_predictions": external_predictions["ml_m"],
                        "x": x,
                        "y": d,
                        "smpls": smpls,
                    },
                    "ml_g_d_med_d": {
                        "external_predictions": external_predictions["ml_g_d_med_d"],
                        "x": x,
                        "y": y,
                        "smpls": smpls_d_d,
                    },
                    "ml_g_d_med_1md": {
                        "external_predictions": external_predictions["ml_g_d_med_1md"],
                        "x": x,
                        "y": y,
                        "smpls": smpls_d_1md,
                        # This has to be changed, has to take previous result
                    },
                    "ml_med_d_hat": {
                        "external_predictions": external_predictions["ml_med_d_hat"],
                        "x": x,
                        "y": m,
                        "smpls": smpls_d,
                    },
                    "ml_med_1md_hat": {
                        "external_predictions": external_predictions["ml_med_1md_hat"],
                        "x": x,
                        "y": m,
                        "smpls": smpls_1md,
                    },
                }
            elif self.score_type == "efficient_alt":
                # TODO: Change the samples

                training_params = {
                    "ml_m": {
                        "external_predictions": external_predictions["ml_m"],
                        "x": x,
                        "y": d,
                        "smpls": smpls,
                    },
                    "ml_g_d": {
                        "external_predictions": external_predictions["ml_g_d"],
                        "x": x,
                        "y": y,
                        "smpls": smpls_d,
                    },
                    "ml_g_d_1md": {
                        "external_predictions": external_predictions["ml_g_d_1md"],
                        "x": x,
                        "y": self._extract_predictions(n_jobs_cv, return_models, **training_params["ml_g_d"])["preds"],
                        "smpls": smpls_d,  # This has to be changed, has to take previous result
                    },
                    "ml_m_med_x": {
                        "external_predictions": external_predictions["ml_m_med_x"],
                        "x": x,
                        "y": m,
                        "smpls": smpls,
                    },
                }

        return training_params
