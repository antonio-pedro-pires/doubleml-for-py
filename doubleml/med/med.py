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
    obj_dml_data : :class:`DoubleMLData` object
        The :class:`DoubleMLData` object providing the data and specifying the variables for the causal model.

    n_folds : int
        Number of folds.
        Default is ``5``.

    trimming_threshold : float
        The threshold used for trimming.
        Default is ``5e-2``.

    """

    def __init__(
        self,
        obj_dml_data,
        ml_g,
        ml_m,
        ml_med,
        ml_nested=None,
        n_folds=5,
        n_rep=1,
        score=None,
        normalize_ipw=False,
        trimming_rule="truncate",
        trimming_threshold=1e-2,
        order=1,
        fewsplits=False,
        draw_sample_splitting=True,
        score_type="ipw",
    ):

        if not isinstance(obj_dml_data, DoubleMLMediationData):
            raise TypeError(
                "Mediation analysis requires data of type DoubleMLMediationData."
                + f"data of type {str(type(obj_dml_data))} was provided instead."
            )

        super().__init__(obj_dml_data, n_folds, n_rep, score, draw_sample_splitting)

        self.score_type = score_type
        valid_scores = ["Y(0, M(0))", "Y(0, M(1))", "Y(1, M(0))", "Y(1, M(1))"]
        valid_score_types = ["efficient", "ipw"]
        _check_score(self.score, valid_scores, allow_callable=False)
        _check_score(self.score_type, valid_score_types, allow_callable=False)
        self._check_score_type()

        self._mediated = self._dml_data.m == 1

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
        return self.score_type

    # TODO: Do I have to initialize the learners somewhere else(at init???)
    def _initialize_ml_nuisance_params(self):
        # TODO: See if I can estimate ml_m, ml_med_d0 and ml_med_d1 only once for meds.py
        if self._score == "Y(0, M(0))":
            valid_learner = ["ml_g_d0", "ml_m"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in valid_learner}
        elif self._score == "Y(0, M(1))":
            if self.score_type == "ipw":
                valid_learner = ["ml_g_d0", "ml_g_d0_d1", "ml_m", "ml_m_med"]
            else:
                valid_learner = ["ml_g_d0_med0", "ml_g_d0_med1", "ml_m", "ml_med_d0", "ml_med_d1"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in valid_learner}
        elif self._score == "Y(1, M(0))":
            if self.score_type == "ipw":
                valid_learner = ["ml_g_d1", "ml_g_d1_d0", "ml_m", "ml_m_med"]
            else:
                valid_learner = ["ml_g_d1_med1", "ml_g_d1_med0", "ml_m", "ml_med_d0", "ml_med_d1"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in valid_learner}
        elif self._score == "Y(1, M(1))":
            valid_learner = ["ml_g_d1", "ml_m"]
            self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in valid_learner}

    # TODO: Check which learners take which samples
    # TODO: DON'T FORGET TO ADD NORMALISE!!!
    def _nuisance_est(self, smpls, n_jobs_cv, return_models, external_predictions):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        # TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(
            x, self._dml_data[self._dml_data.m_cols]
        )  # Check that the mediators have the same number of samples as X and

        # dx = np.column_stack((d, x))
        # mdx = np.column_stack((m, dx))

        if self._score == "Y(0, M(0))" or self._score == "Y(1, M(1))":
            if self.score_type == "ipw":
                psi_elements, preds = self._est_potential_alt(
                    smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions
                )
            else:
                psi_elements, preds = self._est_potential(smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions)
        elif self._score == "Y(0, M(1))" or self._score == "Y(1, M(0))":
            if self.score_type == "ipw":
                psi_elements, preds = self._est_counterfactual_alt(
                    smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions
                )
            else:
                psi_elements, preds = self._est_counterfactual_alt(
                    smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions
                )

        # TODO: Check for how to initialize external predictions.

        return psi_elements, preds

    # TODO: Check if these functions are needed or not.
    # TODO: Check whether all parameters are useful or not.
    def _est_counterfactual(self, smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions):
        # TODO: These functions might be wrong.
        treated = self.treated
        mediated = self.mediated

        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        smpls_d0_med0, smpls_d0_med1, smpls_d1_med0, smpls_d1_med1 = _get_cond_smpls_2d(smpls, treated, mediated)

        m_external = external_predictions["ml_m"] is not None
        med_d0_external = external_predictions["ml_med_d0"] is not None
        med_d1_external = external_predictions["ml_med_d1"] is not None
        # smpls_pot = None
        smpls_counter = None

        # Learner for E(Y|D=d, M=d, X)
        g_pot_learner_name = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        g_counter_learner_name = self.params_names[1]

        if g_pot_learner_name == "ml_g_d0_m0":
            g_pot_external = external_predictions["ml_g_d0_m0"] is not None
            g_counter_external = external_predictions["ml_g_d0_med1"] is not None
            smpls_pot = smpls_d0_med0
            smpls_counter = smpls_d0_med1
        else:
            g_pot_external = external_predictions["ml_g_d1_med1"] is not None
            g_counter_external = external_predictions["ml_g_d1_med0"] is not None
            smpls_pot = smpls_d1_med1
            smpls_counter = smpls_d1_med0

        # Compute the probability of treatment given the cofounders. Pr(D=1|X)
        if m_external:
            m_hat = {"preds": external_predictions["preds"], "targets": _cond_targets(d, cond_sample=smpls_d1), "models": None}
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

        # Compute the conditional expectation of outcome Y given non-treatment D=0,
        # non-mediation M=0 and co-founders X. E(Y|D=0,M=0,X)
        if g_pot_external:
            g_pot_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(y, cond_sample=smpls_pot),
                "models": None,
            }
        else:
            g_pot_hat = _dml_cv_predict(
                self._learner[g_pot_learner_name],
                x,
                y,
                smpls=smpls_pot,
                n_jobs=n_jobs_cv,
                est_params=self._get_params(g_pot_learner_name),
                method=self._predict_method[g_pot_learner_name],
                return_models=return_models,
            )
            _check_finite_predictions(g_pot_hat["preds"], self._learner[g_pot_learner_name], g_pot_learner_name, smpls_pot)

        # Compute the conditional expectation of outcome Y given non-treatment D=0,
        # mediation M=1 and co-founders X. E(Y|D=0,M=1,X)
        if g_counter_external:
            g_counter_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(y, cond_sample=smpls_counter),
                "models": None,
            }
        else:
            g_counter_hat = _dml_cv_predict(
                self._learner[g_counter_learner_name],
                x,
                y,
                smpls=smpls_d0_med1,
                n_jobs=n_jobs_cv,
                est_params=self._get_params(g_counter_learner_name),
                method=self._predict_method[g_counter_learner_name],
                return_models=return_models,
            )
            _check_finite_predictions(
                g_counter_hat["preds"], self._learner[g_counter_learner_name], g_counter_learner_name, smpls_counter
            )

        # Compute the mediator mean conditional on the treatment and cofounders. E[M|D=1, X]
        if med_d1_external:
            med_d1_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(m, cond_sample=smpls_d1),
                "models": None,
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
                "models": None,
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

            psi_a, psi_b = self._score_elements(
                y,
                d,
                m,
                x,
                m_hat=m_hat["preds"],
                g_pot_hat=g_pot_hat["preds"],
                g_counter_hat=g_counter_hat["preds"],
                med_d0_hat=med_d0_hat["preds"],
                med_d1_hat=med_d1_hat["preds"],
            )
            preds = {
                "predictions": {
                    "ml_m": m_hat["preds"],
                    g_pot_learner_name: g_pot_hat["preds"],
                    g_counter_learner_name: g_counter_hat["preds"],
                    "ml_med_d0": med_d0_hat["preds"],
                    "ml_med_d1": med_d1_hat["preds"],
                },
                "targets": {
                    "ml_m": m_hat["targets"],
                    g_pot_learner_name: g_pot_hat["targets"],
                    g_counter_learner_name: g_counter_hat["targets"],
                    "ml_med_d0": med_d0_hat["targets"],
                    "ml_med_d1": med_d1_hat["targets"],
                },
                "models": {
                    "ml_m": m_hat["models"],
                    g_pot_learner_name: g_pot_hat["models"],
                    g_counter_learner_name: g_counter_hat["models"],
                    "ml_med_d0": med_d0_hat["models"],
                    "ml_med_d1": med_d1_hat["models"],
                },
            }

        psi_a, psi_b = self._score_elements(
            m_hat=m_hat, g_pot=g_pot_hat, g_counter=g_counter_hat, med_d0=med_d0_hat, med_d1=med_d1_hat
        )
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                g_pot_learner_name: g_pot_hat["preds"],
                g_counter_learner_name: g_counter_hat["preds"],
                "ml_med_d0": med_d0_hat["preds"],
                "ml_med_d1": med_d1_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                g_pot_learner_name: g_pot_hat["targets"],
                g_counter_learner_name: g_counter_hat["targets"],
                "ml_med_d0": med_d0_hat["targets"],
                "ml_med_d1": med_d1_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                g_pot_learner_name: g_pot_hat["models"],
                g_counter_learner_name: g_counter_hat["models"],
                "ml_med_d0": med_d0_hat["models"],
                "ml_med_d1": med_d1_hat["models"],
            },
        }
        return psi_elements, preds

    def _est_counterfactual_alt(self, smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions):
        # TODO: These functions might be wrong.
        # TODO: Also, check if samples and regressors are right.
        # TODO: Verify that the samples gotten for score_type == "ipw" are musample, deltasample, psample, etc.
        treated = self.treated
        # mediated = self.mediated
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)

        m_external = external_predictions["ml_m"] is not None
        m_med_external = external_predictions["ml_m_med"] is not None

        # Learner for E(Y|D=d, M=d, X)
        g_d_learner_name = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        g_nested_name = self.params_names[1]
        # Prepare samples to be used
        smpls_gd = None
        smpls_nested = None

        # xm = np.concat((x, m))
        # TODO: Erase these comments
        #   y0m1=((1-dte)*pmxte/((1-pmxte)*pxte)*(yte-eymx0te)+dte/pxte*(eymx0te-regweymx0te)+regweymx0te
        #   y1m0=(dte*(1-pmxte)/(pmxte*(1-pxte))*(yte-eymx1te)+(1-dte)/(1-pxte)*(eymx1te-regweymx1te)+regweymx1te)
        #   dte, pmxte, pxte, yte, eymx0te, regweymx0te, eyx0te, eymx1te, regweymx1te, eyx1te
        #    1     2     3     4     5         6            7       8          9         10
        # pmxte = Pr(D=1|M,X), pxte = Pr(D=1|X)
        # eyx0te = E(Y|D=0, X), eyx1te = E(Y|D=1, X)
        # eymx1te = E(Y|D=1, M, X), eymx0te = E(Y|D=0, M, X)
        # regweymx1te = E[E(Y|M,X,D=1)|D=0,X], regweymx0te = E[E(Y|M,X,D=0)|D=1,X]
        if g_d_learner_name == "ml_g_d0":
            g_d_external = external_predictions["ml_g_d0"] is not None
            g_nested_external = external_predictions["ml_g_d0_d1"] is not None
            smpls_gd = smpls_d0
            smpls_nested = smpls_d1
        else:
            g_d_external = external_predictions["ml_g_d1"] is not None
            g_nested_external = external_predictions["ml_g_d1_d0"] is not None
            smpls_gd = smpls_d1
            smpls_nested = smpls_d0

        # Compute the probability of treatment given the cofounders. Pr(D=1|X)
        if m_external:
            m_hat = {"preds": external_predictions["preds"], "targets": _cond_targets(d, cond_sample=smpls_d1), "models": None}
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

        # Compute the conditional expectation of outcome Y given treatment level D=d,
        # mediators M and co-founders X. E(Y|D=0,M,X)
        if g_d_external:
            g_d_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(y, cond_sample=smpls_gd),
                "models": None,
            }
        else:
            g_d_hat = _dml_cv_predict(
                self._learner[g_d_learner_name],
                x,
                y,
                smpls=smpls_gd,
                n_jobs=n_jobs_cv,
                est_params=self._get_params(g_d_learner_name),
                method=self._predict_method[g_d_learner_name],
                return_models=return_models,
            )
            _check_finite_predictions(g_d_hat["preds"], self._learner[g_d_learner_name], g_d_learner_name, smpls_gd)

        # Compute the conditional expectation of outcome Y given non-treatment D=0,
        # mediation M=1 and co-founders X. E(Y|D=0,M=1,X)
        if g_nested_external:
            g_nested_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(y, cond_sample=smpls_nested),
                "models": None,
            }
        else:
            g_nested_hat = _dml_cv_predict(
                self._learner[g_nested_name],
                x,
                g_d_hat["preds"],
                smpls=smpls_nested,
                n_jobs=n_jobs_cv,
                est_params=self._get_params(g_nested_name),
                method=self._predict_method[g_nested_name],
                return_models=return_models,
            )
            _check_finite_predictions(g_nested_hat["preds"], self._learner[g_nested_name], g_nested_name, smpls_nested)

        # Compute the mediator mean conditional on the treatment and cofounders. E[M|D=1, X]
        if m_med_external:
            m_med_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(m, cond_sample=smpls),
                "models": None,
            }
        else:
            m_med_hat = _dml_cv_predict(
                self._learner["ml_m_med"],
                x,
                m,
                smpls=smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_m_med"),
                method=self._predict_method["ml_m_med"],
                return_models=return_models,
            )
            _check_finite_predictions(m_med_hat["preds"], self._learner["ml_m_med"], "ml_m_med", smpls)

        # TODO: See to change name of m_hat or parameter in _score_elements_alt
        psi_a, psi_b = self._score_elements_alt(self, smpls, y, d, x, m_hat, m_med_hat, g_d_hat, g_nested_hat)
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                g_d_learner_name: g_d_hat["preds"],
                g_nested_name: g_nested_hat["preds"],
                "ml_m_med": m_med_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                g_d_learner_name: g_d_hat["targets"],
                g_nested_name: g_nested_hat["targets"],
                "ml_m_med": m_med_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                g_d_learner_name: g_d_hat["models"],
                g_nested_name: g_nested_hat["models"],
                "ml_m_med": m_med_hat["targets"],
            },
        }
        return psi_elements, preds

    def _est_potential(self, smpls, x, y, d, m, n_jobs_cv, return_models, external_predictions):
        treated = self.treated
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        # TODO: Erase this comment.
        # Just so that we know the type of smpls_g
        smpls_g = None
        g_learner_name = self.params_names()[0]

        g_d_external = external_predictions[g_learner_name] is not None
        m_external = external_predictions["ml_m"] is not None

        if g_learner_name == "ml_g_d0":
            smpls_g = smpls_d0
        else:
            smpls_g = smpls_d1

        # Compute the conditional expectation of outcome Y given non-treatment D=0 and co-founders X. E(Y|D=0,X)
        if g_d_external:
            g_d_hat = {
                "preds": external_predictions["preds"],
                "targets": _cond_targets(y, cond_sample=smpls_g),
                "models": None,
            }
        else:
            g_d_hat = _dml_cv_predict(
                self._learner[g_learner_name],
                x,
                y,
                smpls=smpls_g,
                n_jobs=n_jobs_cv,
                est_params=self._get_params(g_learner_name),
                method=self._predict_method[g_learner_name],
                return_models=return_models,
            )
            _check_finite_predictions(g_d_hat["preds"], self._learner[g_learner_name], g_learner_name, smpls_g)

        # Compute the probability of treatment given the cofounders. Pr(D=1|X)
        if m_external:
            m_hat = {"preds": external_predictions["preds"], "targets": _cond_targets(d, cond_sample=smpls_d1), "models": None}
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

        psi_a, psi_b = self._score_elements(y, d, m, x, g_d_hat["preds"], m_hat["preds"])
        psi_elements = {"psi_a": psi_a, "psi_b": psi_b}

        preds = {
            "predictions": {
                "ml_m": m_hat["preds"],
                g_learner_name: g_d_hat["preds"],
            },
            "targets": {
                "ml_m": m_hat["targets"],
                g_learner_name: g_d_hat["targets"],
            },
            "models": {
                "ml_m": m_hat["models"],
                g_learner_name: g_d_hat["models"],
            },
        }
        return psi_elements, preds

    def _est_potential_alt(self):
        #   y1m1=(eyx1te + dte *(yte-eyx1te)/pxte)
        #   y0m0=(eyx0te + (1-dte)*(yte-eyx0te)/(1-pxte))

        pass

    def _score_elements(self, g_d_hat, m_hat, g_pot, g_counter, med_d0, med_d1):
        # TODO: Make sure that y, d, m, x are from the test parts of the samples
        psi_a = -1.0
        psi_b = None
        d = self._dml_data.d
        y = self._dml_data.y
        # TODO: Create data class for mediator.
        m = self._dml_data.m

        if self.normalize_ipw:
            n_obs = self._obj_dml_data.n_obs
            sumscores1 = np.sum((d * med_d0) / (m_hat * med_d1))
            sumscores2 = np.sum((1 - d) / 1 - m_hat)
            sumscores3 = np.sum(d / m_hat)
            sumscores4 = np.sum((1 - d) * med_d1 / ((1 - m_hat) * med_d0))
            if self._score == "Y(0, M(0))":
                psi_b = g_d_hat + (n_obs * (1 - d) * (y - g_d_hat) / (1 - m_hat)) / sumscores2
            elif self._score == "Y(1, M(1))":
                psi_b = g_d_hat + (n_obs * d * (y - g_d_hat) / m_hat) / sumscores3
            elif self._score == "Y(0, M(1))":
                eymx0te = m * g_counter + (1 - m) * g_pot
                eta01 = g_counter * med_d1 + g_pot * (1 - med_d1)
                psi_b = (
                    (n_obs * (1 - d) * med_d1 / ((1 - m_hat) * med_d0) * (y - eymx0te)) / sumscores4
                    + (n_obs * d / m_hat * (eymx0te - eta01)) / sumscores3
                    + eta01
                )
            elif self._score == "Y(1, M(0))":
                eymx1te = m * g_pot + (1 - m) * g_counter
                eta10 = g_pot * med_d0 + g_counter * (1 - med_d0)
                psi_b = (
                    (n_obs * d * med_d0 / (m_hat * med_d0) * (y - eymx1te)) / sumscores1
                    + (n_obs * (1 - d) / (1 - m_hat) * (eymx1te - eta10)) / sumscores2
                    + eta10
                )
        else:
            # TODO: test that this test_index method works, that d gives the d columns in the test set
            # Probably won't work. Need to see how it's all averaged.
            if self._score == "Y(0, M(0))":
                psi_b = g_d_hat + (1 - d) * (y - g_d_hat) / (1 - m_hat)
            elif self._score == "Y(1, M(1))":
                psi_b = g_d_hat + d * (y - g_d_hat) / (m_hat)
            elif self._score == "Y(0, M(1))":
                eymx0te = m * g_counter + (1 - m) * g_pot
                eta01 = g_counter * med_d1 + g_pot * (1 - med_d1)
                psi_b = (1 - d) * med_d1 / ((1 - m_hat) * med_d0) * (y - eymx0te) + d / m_hat * (eymx0te - eta01) + eta01
            elif self._score == "Y(1, M(0))":
                eymx1te = m * g_pot + (1 - m) * g_counter
                eta10 = g_pot * med_d0 + g_counter * (1 - med_d0)
                psi_b = d * med_d0 / (m_hat * med_d0) * (y - eymx1te) + (1 - d) / (1 - m_hat) * (eymx1te - eta10) + eta10
        return psi_a, psi_b

    def _score_elements_alt(self, smpls, y, d, x, px, m_med=None, g_d=None, g_nested=None):
        psi_a = -1.0
        psi_b = None

        # m = self._dml_data.m
        if self.normalize_ipw:
            n_obs = self._dml_data.n_obs
            sumscore1 = np.sum((1 - d) * m_med / ((1 - m_med) * px))
            sumscore2 = np.sum(d / px)
            sumscore3 = sum((1 - d / (1 - px)))
            sumscore4 = sum(d * (1 - m_med) / (m_med * (1 - px)))
            if self._score == "Y(0, M(0))":
                psi_b = g_d + (n_obs * (1 - d) * (y - g_d) / (1 - px)) / sumscore3
            elif self._score == "Y(0, M(1))":
                psi_b = (
                    (n_obs * (1 - d) * m_med / ((1 - m_med) * px) * (y - g_d)) / sumscore1
                    + (n_obs * d / px * (g_d - g_nested)) / sumscore2
                    + g_nested
                )
            elif self._score == "Y(1, M(0))":
                psi_b = (
                    (n_obs * d * (1 - m_med) / (m_med * (1 - px)) * (y - g_d)) / sumscore4
                    + (n_obs * (1 - d) / (1 - px) * (g_d - g_nested)) / sumscore3
                    + g_nested
                )
            elif self._score == "Y(1, M(1))":
                psi_b = g_d + (n_obs * d * (y - g_d) / px) / sumscore2
        else:
            if self._score == "Y(0, M(0))":
                psi_b = g_d + (1 - d) * (y - g_d) / (1 - px)
            elif self._score == "Y(0, M(1))":
                psi_b = (1 - d) * m_med / ((1 - m_med) * px) * (y - g_d) + d / px * (g_d - g_nested) + g_nested
            elif self._score == "Y(1, M(0))":
                psi_b = d * (1 - m_med) / (m_med * (1 - px)) * (y - g_d) + (1 - d) / (1 - px) * (g_d - g_nested) + g_nested
            elif self._score == "Y(1, M(1))":
                psi_b = g_d + d * (y - g_d) / px
        return psi_a, psi_b

    def _nuisance_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        if self._score == "Y(0, M(0))" or self._score == "Y(1, M(1))":
            res = self._potential_tuning()
        if self._score == "Y(0, M(1))" or self._score == "Y(1, M(0))":
            res = self._counterfactual_tuning()
        return res

    def _potential_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        # TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(
            x, self._dml_data["z"]
        )  # Check that the mediators have the same number of samples as X and

        dx = np.column_stack((d, x))

        treated = self.treated
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        ml_g_learner_name = self.params_names[0]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d1 = [train_index for (train_index, _) in smpls_d1]
        train_inds_g = None
        if ml_g_learner_name == "ml_g_d0":
            train_inds_g = [train_index for (train_index, _) in smpls_d0]
        else:
            train_inds_g = train_inds_d1

        # TODO: Check what this does
        if scoring_methods is None:
            scoring_methods = {ml_g_learner_name: None, "ml_m": None}

        g_d_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_g,
            self._learner[ml_g_learner_name],
            param_grids[ml_g_learner_name],
            scoring_methods[ml_g_learner_name],
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

        params = {ml_g_learner_name: g_d_best_params, "ml_m": m_best_params}
        tune_res = {ml_g_learner_name: g_d_tune_res, "ml_m": m_tune_res}

        res = {"params": params, "tune_res": tune_res}
        return res

    def _counterfactual_tuning(
        self, smpls, param_grids, scoring_methods, n_folds_tune, n_jobs_cv, search_mode, n_iter_randomized_search
    ):
        # TODO: Apply counterfactual_tuning for score_type == "ipw".
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, force_all_finite=False)
        x, d = check_X_y(x, self._dml_data.d, force_all_finite=False)
        # TODO: Create new data class for mediation. Do not use z column for this.
        _, m = check_consistent_length(
            x, self._dml_data["z"]
        )  # Check that the mediators have the same number of samples as X and

        treated = self.treated
        mediated = self.mediated
        smpls_d0, smpls_d1 = _get_cond_smpls(smpls, treated)
        smpls_d0_med0, smpls_d0_med1, smpls_d1_med0, smpls_d1_med1 = _get_cond_smpls_2d(smpls, treated, mediated)

        dx = np.column_stack((d, x))

        # Learner for E(Y|D=d, M=d, X)
        g_pot_learner_name = self.params_names[0]
        # Learner for E(Y|D=d, M=1-d, X)
        g_counter_learner_name = self.params_names[1]

        train_inds = [train_index for (train_index, _) in smpls]
        train_inds_d_lvl0 = [train_index for (train_index, _) in smpls_d0]
        train_inds_d_lvl1 = [train_index for (train_index, _) in smpls_d1]

        if g_pot_learner_name == "ml_g_d0_m0":
            # smpls_pot = smpls_d0_med0
            # smpls_counter = smpls_d0_med1
            train_inds_pot = [train_index for (train_index, _) in smpls_d0_med0]
            train_inds_counter = [train_index for (train_index, _) in smpls_d0_med1]
        else:
            # smpls_pot = smpls_d1_med1
            # smpls_counter = smpls_d1_med0
            train_inds_pot = [train_index for (train_index, _) in smpls_d1_med1]
            train_inds_counter = [train_index for (train_index, _) in smpls_d1_med0]

        g_pot_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_pot,
            self._learner[g_pot_learner_name],
            param_grids[g_pot_learner_name],
            scoring_methods[g_pot_learner_name],
            n_folds_tune,
            n_jobs_cv,
            search_mode,
            n_iter_randomized_search,
        )
        g_counter_tune_res = _dml_tune(
            y,
            dx,  # used to obtain an estimation over several treatment levels (reduced variance in sensitivity)
            train_inds_counter,
            self._learner[g_counter_learner_name],
            param_grids[g_counter_learner_name],
            scoring_methods[g_counter_learner_name],
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

        g_pot_best_params = [xx.best_params_ for xx in g_pot_tune_res]
        g_counter_best_params = [xx.best_params_ for xx in g_counter_tune_res]
        m_best_params = [xx.best_params_ for xx in m_tune_res]
        med_d0_best_params = [xx.best_params_ for xx in med_d0_tune_res]
        med_d1_best_params = [xx.best_params_ for xx in med_d1_tune_res]

        params = {
            g_pot_learner_name: g_pot_best_params,
            g_counter_learner_name: g_counter_best_params,
            "ml_m": m_best_params,
            "ml_med_d0": med_d0_best_params,
            "ml_med_d1": med_d1_best_params,
        }
        tune_res = {
            g_pot_learner_name: g_pot_tune_res,
            g_counter_learner_name: g_counter_tune_res,
            "ml_m": m_tune_res,
            "ml_med_d0": med_d0_tune_res,
            "ml_med_d1": med_d1_tune_res,
        }

        res = {"params": params, "tune_res": tune_res}

        return res

    def _sensitivity_element_est(self, preds):
        pass

    def _check_data(self, obj_dml_data):
        if not isinstance(obj_dml_data, DoubleMLMediationData):
            raise TypeError(
                f"The data must be of DoubleMLMediationData type. {str(obj_dml_data)} "
                f"of type {str(type(obj_dml_data))} was passed."
            )

    def _check_score_types(self):
        finite_scores = ["efficient", "ipw"]
        if self.score_type == "efficient":
            if self._dml_data.n_meds > 1:
                raise ValueError(
                    f"score_type defined as {self.score_type}. "
                    + f"Mediation analysis based on {self.score_type} scores assumes only one mediation variable. "
                    + f"Data contains {self._dml_data.n_meds} mediation variables. "
                    + "Please choose another score_type for mediation analysis."
                )
            if not self._dml_data.binary_meds.all():
                raise ValueError(
                    "Mediation analysis based on efficient scores requires a binary mediation variable"
                    + "with integer values equal to 0 or 1 and no missing values."
                    + f"Actual data contains {np.unique(self._dml_data.data.m)}"
                    + "unique values and/or may contain missing values."
                )
        if self.score_type in finite_scores and not self._dml_data.force_all_m_finite:
            raise ValueError(
                f"Mediation analysis based on {str(finite_scores)} requires finite mediation variables with no missing values."
            )
        # TODO: Probably want to check that elements of mediation variables are floats or ints.
