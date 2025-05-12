import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils import check_X_y, check_consistent_length

from doubleml import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.utils._checks import _check_finite_predictions, _check_score
from doubleml.utils._estimation import _get_cond_smpls, _get_cond_smpls_2d, _dml_cv_predict, _cond_targets


class DoubleMLMED(LinearScoreMixin, DoubleML):
    """ Double machine learning for causal mediation analysis.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    n_folds : int
        Number of folds.
        Default is ``5``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``5e-2``.

    """

    def __init__(self,
        obj_dml_data,
        ml_g,
        ml_m,
        ml_med,
        ml_nested = None,
        n_folds=5,
        n_rep=1,
        score = None,
        normalize_ipw=False,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        order=1,
        multmed=True,
        fewsplits=False,
        draw_sample_splitting=True):

            super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting)

            valid_scores = ["Y(0, M(0))", "Y(0, M(1))", "Y(1, M(0))", "Y(1, M(1))"]
            _check_score(self.score, valid_scores, allow_callable=False)
            self._multmed = multmed

            #TODO: Check if methods can be classifiers or only regressors.
            self._learner = {
                "ml_g": ml_g,
                "ml_m": clone(ml_m),
                "ml_med": clone(ml_med),
            }
            self._predict_method = {
                "ml_g": "predict",
                "ml_m": "predict",
                "ml_med": "predict",
            }

            if self._multmed:
                self._learner["ml_nested"] = ml_nested
                self._predict_method["ml_nested"] = "predict"
            else:
                self._learner["ml_m"] = clone(ml_m)
                self._predict_method["ml_m"] = "predict"
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
    def multmed(self):
        """
        Indicates whether the mediators are multiple and/or continuous, or whether they are binary.
        """
        return self.multmed

    @property
    def few_splits(self):
        """
        Indicates whether the same training data is used for estimating nested models of nuisance parameters.
        """
        return self.few_splits

    def _fit_nuisance_and_score_elements(self, n_jobs_cv, store_predictions, external_predictions, store_models):
        return super()._fit_nuisance_and_score_elements(n_jobs_cv, store_predictions, external_predictions,
                                                        store_models)

    def _solve_score_and_estimate_se(self):
        super()._solve_score_and_estimate_se()

    def _initialize_ml_nuisance_params(self):
        #TODO: See if I can estimate ml_m, ml_med_d0 and ml_med_d1 only once for meds.py
        if self._score == "Y(0, M(0))":
            valid_learner = ["ml_g_d0", "ml_m"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                            valid_learner}
        elif self._score == "Y(0, M(1))":
            valid_learner = ["ml_g_d0_med0", "ml_g_d0_med1", "ml_m", "ml_med_d0", "ml_med_d1"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                            valid_learner}
        elif self._score == "Y(1, M(0))":
            valid_learner = ["ml_g_d1_med1", "ml_g_d1_med0", "ml_m", "ml_med_d0", "ml_med_d1"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                            valid_learner}
        elif self._score == "Y(1, M(1))":
            valid_learner = ["ml_g_d1", "ml_m"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in
                            valid_learner}

    def _nuisance_est(self, smpls, n_jobs_cv, return_models, external_predictions):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        #TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(x, self._dml_data[
            "z"])  # Check that the mediators have the same number of samples as X and

        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, d)
        smpls_d0_med0, smpls_d0_med1, smpls_d1_med0, smpls_d1_med1 = _get_cond_smpls_2d(smpls, d, m)
        dx = np.column_stack((d, x))
        mdx = np.column_stack((m, dx))

        #TODO: Maybe create a function for the outcomes estimation.
        #TODO: Create options for multmed.
        #TODO: Idea, create functions that make outcomes and deal with multmed. This way it is easier to understand.
        if self._score == "Y(0, M(0))":
            m_external = external_predictions["ml_m"] is not None
            g_d0_external = external_predictions["ml_g_d0"] is not None

            # Compute the probability of treatment given the cofounders. Pr(D=1|X)
            if m_external:
                m_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(d, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
                _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls_d1)

            # Compute the conditional expectation of outcome Y given non-treatment D=0 and co-founders X. E(Y|D=0,X)
            if g_d0_external:
                g_d0_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(y, cond_sample=smpls_d0),
                    "models": None
                }
            else:
                g_d0_hat = _dml_cv_predict(
                    self._learner["ml_g_d0"],
                    x,
                    y,
                    smpls=smpls_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g_d0"),
                    method=self._predict_method["ml_g_d0"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_d0_hat["preds"], self._learner["ml_g_d0"], "ml_g_d0", smpls_d0)

                preds = {
                    "predictions": {
                        "ml_m": m_hat["preds"],
                        "ml_g_d0": g_d0_hat["preds"],
                    },
                    "targets": {
                        "ml_m": m_hat["targets"],
                        "ml_g_d0": g_d0_hat["targets"],
                    },
                    "models": {
                        "ml_m": m_hat["models"],
                        "ml_g_d0": g_d0_hat["models"],
                    },
                }
        elif self._score == "Y(0, M(1))":
            m_external = external_predictions["ml_m"] is not None
            g_d0_med0_external = external_predictions["ml_g_d0_m0"] is not None
            g_d0_med1_external = external_predictions["ml_g_d0_m1"] is not None
            med_d0_external = external_predictions["ml_med_d0"] is not None
            med_d1_external = external_predictions["ml_med_d1"] is not None

            # Compute the probability of treatment given the cofounders. Pr(D=1|X)
            if m_external:
                m_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(d, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
                _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls_d1)

            # Compute the conditional expectation of outcome Y given non-treatment D=0, non-mediation M=0 and co-founders X. E(Y|D=0,M=0,X)
            if g_d0_med0_external:
                med_d0_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(y, cond_sample=smpls_d0_med0),
                    "models": None
                }
            else:
                g_d0_med0_hat = _dml_cv_predict(
                    self._learner["ml_g_d0_med0"],
                    x,
                    y,
                    smpls=smpls_d0_med0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g_d0_med0"),
                    method=self._predict_method["ml_g_d0_med0"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_d0_med0_hat["preds"], self._learner["g_d0_med0_hat"], "g_d0_med0_hat",
                                          smpls_d0_med0)

            # Compute the conditional expectation of outcome Y given non-treatment D=0, mediation M=1 and co-founders X. E(Y|D=0,M=1,X)
            if g_d0_med1_external:
                g_d0_med1_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(y, cond_sample=smpls_d0_med1),
                    "models": None
                }
            else:
                g_d0_med1_hat = _dml_cv_predict(
                    self._learner["ml_g_d0_med1"],
                    x,
                    y,
                    smpls=smpls_d0_med1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g_d0_med1"),
                    method=self._predict_method["ml_g_d0_med1"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_d0_med1_hat["preds"], self._learner["ml_g_d0_med1"], "ml_g_d0_med1",
                                          smpls_d0_med1)

            # Compute the mediator mean conditional on the treatment and cofounders. E[M|D=1, X]
            if med_d1_external:
                med_d1_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(m, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                med_d1_hat = _dml_cv_predict(
                    self._learner["ml_med_d1"],
                    x,
                    m,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_med_d1_hat"),
                    method=self._predict_method["ml_med_d1_hat"],
                    return_models=return_models,
                )
                _check_finite_predictions(med_d1_hat["preds"], self._learner["ml_med_d1"], "ml_med_d1",
                                          smpls_d1)

            # Compute the mediator mean conditional on  non-treatment and cofounders. E[M|D=0, X]
            if med_d0_external:
                med_d0_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(m, cond_sample=smpls_d0),
                    "models": None
                }
            else:
                med_d0_hat = _dml_cv_predict(
                    self._learner["ml_med_d0"],
                    x,
                    m,
                    smpls=smpls_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_med_d0"),
                    method=self._predict_method["ml_med_d0"],
                    return_models=return_models,
                )
                _check_finite_predictions(med_d0_hat["preds"], self._learner["ml_med_d0"], "ml_med_d0",
                                          smpls_d0)

                preds = {
                    "predictions": {
                        "ml_m": m_hat["preds"],
                        "ml_g_d0_med0": g_d0_med0_hat["preds"],
                        "ml_g_d0_med1": g_d0_med1_hat["preds"],
                        "ml_med_d0": med_d0_hat["preds"],
                        "ml_med_d1": med_d1_hat["preds"],
                    },
                    "targets": {
                        "ml_m": m_hat["targets"],
                        "ml_g_d0_med0": g_d0_med0_hat["targets"],
                        "ml_g_d0_med1": g_d0_med1_hat["targets"],
                        "ml_med_d0": med_d0_hat["targets"],
                        "ml_med_d1": med_d1_hat["targets"],
                    },
                    "models": {
                        "ml_m": m_hat["models"],
                        "ml_g_d0_med0": g_d0_med0_hat["models"],
                        "ml_g_d0_med1": g_d0_med1_hat["models"],
                        "ml_med_d0": med_d0_hat["models"],
                        "ml_med_d1": med_d1_hat["models"],
                    },
                }
        elif self._score == "Y(1, M(0))":
            m_external = external_predictions["ml_m"] is not None
            g_d1_med0_external = external_predictions["ml_g_d1_m0"] is not None
            g_d1_med1_external = external_predictions["ml_g_d1_m1"] is not None
            med_d0_external = external_predictions["ml_med_d0"] is not None
            med_d1_external = external_predictions["ml_med_d1"] is not None

            # Compute the probability of treatment given the cofounders. Pr(D=1|X)
            if m_external:
                m_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(d, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
                _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls_d1)

                # Compute the conditional expectation of outcome Y given treatment D=1, non-mediation M=0 and co-founders X. E(Y|D=1,M=0,X)
            if g_d1_med0_external:
                g_d1_med0_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(y, cond_sample=smpls_d1_med0),
                    "models": None
                }
            else:
                g_d1_med0_hat = _dml_cv_predict(
                    self._learner["ml_g_d1_med0"],
                    x,
                    y,
                    smpls=smpls_d1_med0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g_d1_med0"),
                    method=self._predict_method["ml_g_d1_med0"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_d1_med0_hat["preds"], self._learner["ml_g_d1_med0"], "ml_g_d1_med0",
                                          smpls_d1_med0)

            # Compute the conditional expectation of outcome Y given treatment D=1, mediation M=1 and co-founders X. E(Y|D=1,M=1,X)
            if g_d1_med1_external:
                g_d1_med1_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(y, cond_sample=smpls_d1_med1),
                    "models": None
                }
            else:
                g_d1_med1_hat = _dml_cv_predict(
                    self._learner["ml_g_d1_med1"],
                    mdx,
                    y,
                    smpls=smpls_d1_med1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g_d1_med1"),
                    method=self._predict_method["ml_g_d1_med1"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_d1_med1_hat["preds"], self._learner["ml_g_d1_med1"], "ml_g_d1_med1",
                                          smpls_d1_med1)

            # Compute the mediator mean conditional on the treatment and cofounders. E[M|D=1, X]
            if med_d1_external:
                med_d1_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(m, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                med_d1_hat = _dml_cv_predict(
                    self._learner["ml_med_d1"],
                    x,
                    m,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_med_d1_hat"),
                    method=self._predict_method["ml_med_d1_hat"],
                    return_models=return_models,
                )
                _check_finite_predictions(med_d1_hat["preds"], self._learner["ml_med_d1"], "ml_med_d1", smpls_d1)

            # Compute the mediator mean conditional on  non-treatment and cofounders. E[M|D=0, X]
            if med_d0_external:
                med_d0_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(m, cond_sample=smpls_d0),
                    "models": None
                }
            else:
                med_d0_hat = _dml_cv_predict(
                    self._learner["ml_med_d0"],
                    x,
                    m,
                    smpls=smpls_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_med_d0"),
                    method=self._predict_method["ml_med_d0"],
                    return_models=return_models,
                )
                _check_finite_predictions(med_d0_hat["preds"], self._learner["ml_med_d0"], "ml_med_d0", smpls_d0)

                preds = {
                    "predictions": {
                        "ml_m": m_hat["preds"],
                        "ml_g_d0_med0": g_d1_med0_hat["preds"],
                        "ml_g_d0_med1": g_d1_med1_hat["preds"],
                        "ml_med_d0": med_d0_hat["preds"],
                        "ml_med_d1": med_d1_hat["preds"],
                    },
                    "targets": {
                        "ml_m": m_hat["targets"],
                        "ml_g_d0_med0": g_d1_med0_hat["targets"],
                        "ml_g_d0_med1": g_d1_med1_hat["targets"],
                        "ml_med_d0": med_d0_hat["targets"],
                        "ml_med_d1": med_d1_hat["targets"],
                    },
                    "models": {
                        "ml_m": m_hat["models"],
                        "ml_g_d0_med0": g_d1_med0_hat["models"],
                        "ml_g_d0_med1": g_d1_med1_hat["models"],
                        "ml_med_d0": med_d0_hat["models"],
                        "ml_med_d1": med_d1_hat["models"],
                    },
                }
        elif self._score == "Y(1, M(1))":
            m_external = external_predictions["ml_m"] is not None
            g_d1_external = external_predictions["ml_g_d1"] is not None

            # Compute the conditional expectation of outcome Y given treatment D=1 and co-founders X. E(Y|D=1,X)
            if g_d1_external:
                g_d1_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(y, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                g_d1_hat = _dml_cv_predict(
                    self._learner["ml_g_d1"],
                    x,
                    y,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_g_d1"),
                    method=self._predict_method["ml_g_d1"],
                    return_models=return_models,
                )
                _check_finite_predictions(g_d1_hat["preds"], self._learner["ml_g_d1"], "ml_g_d1", smpls_d1)

            # Compute the probability of treatment given the cofounders. Pr(D=1|X)
            if m_external:
                m_hat = {
                    "preds": external_predictions["preds"],
                    "targets": _cond_targets(d, cond_sample=smpls_d1),
                    "models": None
                }
            else:
                m_hat = _dml_cv_predict(
                    self._learner["ml_m"],
                    x,
                    d,
                    smpls=smpls_d1,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_m"),
                    method=self._predict_method["ml_m"],
                    return_models=return_models,
                )
                _check_finite_predictions(m_hat["preds"], self._learner["ml_m"], "ml_m", smpls_d1)

            preds = {
                "predictions": {
                    "ml_m": m_hat["preds"],
                    "ml_g_d1": g_d1_hat["preds"],
                },
                "targets": {
                    "ml_m": m_hat["targets"],
                    "ml_g_d1": g_d1_hat["targets"],
                },
                "models": {
                    "ml_m": m_hat["models"],
                    "ml_g_d1": g_d1_hat["models"],
                },
            }
        # TODO: Check for how to initialize external predictions.

        return preds

    #TODO: Check if these functions are needed or not.
    def _est_counterfactual(self):
        pass

    def _est_counterfactual_alt(self):
        pass

    def _est_potential(self):
        pass

    def _nuisance_tuning(self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode,
                         n_iter_randomized_search):
        pass

    def _sensitivity_element_est(self, preds):
        pass


    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                f"The data must be of DoubleMLData type. {str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if self.multmed==False:
            if len(obj_dml_data.m_cols) > 1:
                raise ValueError(
                    " mediation variable is supposed to be unique but the data contains multiple  \
                                                 mediation variables. To estimate treatment effects with multiple mediation variables, \
                                                 specify multmed to be True."
                )
            # TODO: Add warning for continuous mediation variable
        if  self.multmed==True:
            if len(obj_dml_data.m_cols) == 1:
                raise ValueError(
                    "mediation variable was specified to be multiple but the data contains only one \
                                                 mediation variable. To estimate treatment effects with a sole mediation variable, \
                                                 specify multmed to be False."
                )
        # TODO: Add warning for missing values and/or nans in mediation variables.

