from numbers import Integral
from typing import Optional

import numpy as np
from sklearn.utils import check_array, check_consistent_length, check_X_y

from doubleml.double_ml import DoubleML
from doubleml.double_ml_score_mixins import LinearScoreMixin
from doubleml.med.utils._med_utils import (
    _check_inner_g_external_predictions,
    _check_inner_sample_splitting,
    _check_med_data,
    _check_subsample_arms,
    _double_dml_cv_predict_by_fold,
    _first_inner_split,
    _inner_g_learner_name,
    _inner_g_learner_names,
    _normalize_propensity_med,
    _predict_with_method,
)
from doubleml.utils._checks import _check_finite_predictions, _check_sample_splitting, _check_score
from doubleml.utils._estimation import (
    _cond_targets,
    _dml_cv_predict,
    _get_cond_smpls,
)
from doubleml.utils._tune_optuna import _dml_tune_optuna, resolve_optuna_cv
from doubleml.utils.propensity_score_processing import PSProcessor, PSProcessorConfig


class DoubleMLMED(LinearScoreMixin, DoubleML):
    """Double machine learning for causal mediation analysis with binary treatment.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLMEDData` object
        The :class:`DoubleMLMEDData` object providing the data and specifying the variables for the causal model.

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
        Only required if ``outcome`` is 'counterfactual'. Its target is not an observed outcome but the predictions
        of ``ml_G``, which also applies to :meth:`tune_ml_models`, see the notes.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.ensemble.RandomForestClassifier`) for the nuisance function :math:`P(D=d|X)`.

    ml_M : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods for the nuisance function :math:`P(D=d|M,X)`.
        Only required if ``outcome`` is 'counterfactual'.

    outcome : str
        The outcome parameter to estimate.
        - 'factual': Estimate the factual outcome :math:`E[Y(d, M(d))]`.
        - 'counterfactual': Estimate the counterfactual outcome :math:`E[Y(d, M(d'))]`.
        where :math:`d' \\neq d`.

    treatment_level : int
        The treatment level :math:`d`.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'efficient-alt'``)  specifying the score function to use.
        The only option currently available is 'efficient-alt'.

    n_folds_inner : int
        Number of inner folds used for the estimation of the nested parameter if ``double_sample_splitting=True``.
        The training sample of each of the ``n_folds`` folds is split into ``n_folds_inner`` inner folds,
        see the notes. If ``n_folds_inner=1``, the training sample is instead split once into two subsamples, as in
        Algorithm 2 of Farbmacher et al. (2022), see the notes.
        Default is ``5``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    double_sample_splitting : bool
        Indicates whether the data is resampled for the estimation of the nested parameter
        :math:`E[E[Y|D=d,M,X]|D=1-d,X]`, see the notes. Only used for the estimation if ``outcome`` is 'counterfactual'.
        Default is ``True``.

    ps_processor_config : PSProcessorConfig
        configuration file for the propensity score processor (PSProcessor) object.
        When None, the PSProcessor is initialized with its default values.
        Default is ``None``

        As in the other DoubleML models, propensity scores are clipped to
        ``[clipping_threshold, 1 - clipping_threshold]`` (default ``clipping_threshold=0.01``) and no observations
        are dropped. The original paper and the R function ``medDML`` of the package ``causalweight``
        (argument ``trim``) instead drop observations with extreme propensity scores, so estimates can differ
        slightly if propensity scores are close to 0 or 1. Use ``PSProcessorConfig(clipping_threshold=...)`` to
        change the threshold.

    Notes
    -----
    The counterfactual outcome requires the nested parameter :math:`E[E[Y|D=d,M,X]|D=1-d,X]`, which is estimated in two
    stages: ``ml_G`` estimates :math:`E[Y|D=d,M,X]` and ``ml_nested_g`` regresses the predictions of ``ml_G`` on
    :math:`X`. To avoid overfitting, the two stages are not estimated on the same data.

    If ``double_sample_splitting=True`` and ``n_folds_inner=1``, the training sample of each of the ``n_folds`` folds is
    split once at random into two disjoint subsamples A and B. ``ml_G`` is fitted on the observations of A with ``D = d``
    and predicts the observations of B and of the fold itself. Its predictions for B are the targets of ``ml_nested_g``,
    which is fitted on the observations of B with ``D = 1-d``, and its predictions for the observations of the fold itself
    are used in the score. Each fold has one ``ml_G`` model and the predictions are not averaged. This is Algorithm 2 of
    Farbmacher et al. (2022), for which Theorem 4.2 of the paper is stated, and it corresponds to ``fewsplits=FALSE`` of
    the R function ``medDML`` of the package ``causalweight`` (the default). With the same folds and subsamples and linear
    learners, the predictions of the nuisance functions are identical (tested). Both ``A`` with ``D = d`` and ``B`` with
    ``D = 1-d`` have to contain observations; otherwise an error is raised and the sample splitting can be redrawn.

    If ``double_sample_splitting=True`` and ``n_folds_inner >= 2``, the training sample of each fold is split into
    ``n_folds_inner`` inner folds. ``ml_G`` is fitted on the inner training folds, its out-of-fold predictions within
    the training sample are the targets of ``ml_nested_g``, which is fitted on all observations of the training sample with
    ``D = 1-d``, and its predictions for the observations of the fold itself are the average of the predictions of the
    inner models. This cross-fitted variant is not part of the paper. With two inner folds it uses both directions of
    the split of Algorithm 2 and averages them.

    If ``double_sample_splitting=False``, ``n_folds_inner`` is not used and both stages use the training sample of the
    fold. ``ml_G`` is fitted once on the training sample, its predictions for the observations of the training sample
    with ``D = 1-d`` are the targets of ``ml_nested_g``, and its predictions for the observations of the fold itself are
    used in the score. This corresponds to ``fewsplits=TRUE`` of ``medDML``. The model of a fold is never fitted on the
    observations of that fold, so the nested parameter of a fold does not depend on the outcomes of that fold. In all
    cases, the entries ``ml_G_inner_i`` of the predictions hold the predictions of ``ml_G`` that serve as targets of
    ``ml_nested_g`` for the fold ``i``. With ``n_folds_inner=1``, they are missing outside of subsample B.

    The score ``'efficient-alt'`` is the modified score function of expression (3.2) of Farbmacher et al. (2022)
    (Algorithm 2 and Theorem 4.2), which avoids conditional mediator densities, for the counterfactual outcome
    :math:`E[Y(d,M(1-d))]`. The factual outcome :math:`E[Y(d,M(d))]` uses the doubly robust score of the mean potential
    outcome. If ``normalize_ipw=True``, every inverse probability weight is multiplied by the number of observations
    and divided by the sum of the weights, so that the weights of a treatment group sum to the number of observations,
    as proposed in the paper. The sum runs over all observations and not fold by fold, as in ``medDML``.

    The paper uses four-fold cross-fitting in its simulations and ``medDML`` uses three folds by default, while
    ``n_folds`` is ``5`` by default here.

    The class estimates the mean potential outcome :math:`E[Y(d,M(d'))]` of a binary treatment, :class:`DoubleMLMEDS`
    combines these into the total, direct and indirect effects. The following parts of the paper and of ``medDML`` are
    not implemented: the controlled direct effect, the average treatment effect on the treated, the score based on
    conditional mediator densities (expression (3.1), Algorithm 1 and Theorem 4.1, ``multmed=FALSE`` in ``medDML``),
    the polynomial expansion of the covariates (``order`` in ``medDML``), cluster data, instrumental variables and
    sensitivity analysis.

    The learner ``ml_nested_g`` has no observed outcome. Its targets are predictions of ``ml_G`` for the observations with
    ``D = 1-d``, so :meth:`tune_ml_models` tunes it after ``ml_G``, using the covariates :math:`X` only. The tuning
    criterion is the cross-validated error of ``ml_nested_g`` against out-of-fold predictions of the tuned ``ml_G``. It
    therefore measures how well ``ml_nested_g`` reproduces the predictions of ``ml_G``, not the distance to an observed
    outcome, and the quality of this tuning depends on how well ``ml_G`` fits. The targets are an approximation of the
    ones used for estimation, because tuning uses one cross-fitted pass of the tuned ``ml_G`` and neither the inner
    sample splitting nor the subsamples A and B (a similar compromise is made for ``ml_t`` in :class:`DoubleMLLPLR`). If
    no parameter space is passed for ``ml_nested_g``, it is not tuned. To tune ``ml_nested_g`` against the tuned
    ``ml_G``, pass both in the same ``ml_param_space``. A later call with ``set_as_params=True`` that leaves ``ml_G`` out
    sets its parameters back to those of the learner passed to the constructor, which is the general behavior of
    :meth:`tune_ml_models`.

    :meth:`evaluate_learners` compares ``ml_G`` with the outcome of the observations with ``D = d``. It compares
    ``ml_nested_g`` with the predictions of ``ml_G`` for the observations with ``D = 1-d``, because it has no observed
    outcome. As in other models, observations without a target are marked by ``nan``, which the ``metric`` has to ignore.

    References
    ----------
    Helmut Farbmacher, Martin Huber, Lukáš Lafférs, Henrika Langen, Martin Spindler (2022).
    Causal mediation analysis with double machine learning. The Econometrics Journal, 25(2), 277-300.
    https://doi.org/10.1093/ectj/utac003

    Hugo Bodory, Martin Huber, Jannis Kueck. causalweight: Estimation Methods for Causal Inference Based on Inverse
    Probability Weighting and Doubly Robust Estimation. R package (MIT license), https://CRAN.R-project.org/package=causalweight.
    The estimator follows the function ``medDML`` of this package, which implements the procedure of the paper.

    Examples
    --------
    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.med.datasets import make_med_data
    >>> from sklearn.linear_model import Lasso, LogisticRegression
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> ml_g = Lasso()
    >>> ml_G = Lasso()
    >>> ml_nested_g = Lasso()
    >>> ml_m = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
    >>> ml_M = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
    >>> obj_dml_data = make_med_data()
    >>> dml_med_obj = dml.DoubleMLMED(obj_dml_data, 1, "counterfactual", ml_m, ml_g, ml_G, ml_M, ml_nested_g)
    >>> dml_med_obj.fit().summary  # doctest: +SKIP
           coef   std err         t         P>|t|     2.5 %   97.5 %
    d  0.995196  0.059411  16.75092  5.575699e-63  0.878752  1.11164
    """

    def __init__(
        self,
        obj_dml_data,
        treatment_level,
        outcome,
        ml_m,
        ml_g=None,
        ml_G=None,
        ml_M=None,
        ml_nested_g=None,
        n_folds=5,
        n_rep=1,
        score="efficient-alt",
        n_folds_inner=5,
        normalize_ipw=True,
        draw_sample_splitting=True,
        double_sample_splitting=True,
        ps_processor_config: Optional[PSProcessorConfig] = None,
    ):
        self._dml_data = _check_med_data(obj_dml_data)

        self._double_sample_splitting = double_sample_splitting
        self.n_folds_inner = n_folds_inner

        super().__init__(
            obj_dml_data, n_folds, n_rep, score, draw_sample_splitting, double_sample_splitting=self.double_sample_splitting
        )

        valid_scores = ["efficient-alt"]
        _check_score(self.score, valid_scores, allow_callable=False)

        self._outcome = self._check_outcome(outcome)
        self._treatment_level = self._check_levels(treatment_level)

        self._treated = self._dml_data.d == treatment_level

        self._predict_method = {}
        self._check_learners(ml_g=ml_g, ml_m=ml_m, ml_G=ml_G, ml_M=ml_M, ml_nested_g=ml_nested_g)

        self._normalize_ipw = normalize_ipw
        self._external_predictions_implemented = True
        self._sensitivity_implemented = False

        self._ps_processor_config = ps_processor_config if ps_processor_config is not None else PSProcessorConfig()
        self._ps_processor = PSProcessor.from_config(self._ps_processor_config)

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

    def draw_sample_splitting(self):
        """
        Draw sample splitting for the mediation model.

        If ``double_sample_splitting=True`` and ``n_folds_inner=1``, the training sample of each fold is split once into
        two disjoint subsamples ``(A, B)``.

        Returns
        -------
        self : object
        """
        if not (self.double_sample_splitting and self.n_folds_inner == 1):
            return super().draw_sample_splitting()

        # the inner resampling needs at least two folds, the first inner tuple is the (A, B) pair
        self.n_folds_inner = 2
        try:
            super().draw_sample_splitting()
        finally:
            self.n_folds_inner = 1
        self._smpls_inner = _first_inner_split(self._smpls_inner)
        return self

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
        if self._outcome == "factual":
            learners = ["ml_g", "ml_m"]
        else:
            learners = ["ml_m", "ml_G", "ml_M", "ml_nested_g"] + _inner_g_learner_names(self.n_folds)

        self._params = {learner: {key: [None] * self.n_rep for key in self._dml_data.d_cols} for learner in learners}

    def _nuisance_est(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models,
    ):
        if self._outcome == "factual":
            return self._nuisance_est_factual(smpls, n_jobs_cv, external_predictions, return_models)
        else:
            return self._nuisance_est_counterfactual(smpls, n_jobs_cv, external_predictions, return_models)

    def _nuisance_est_factual(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models,
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        # check_X_y's ensure_all_finite is only ever applied to its first (X) argument, never to
        # the second (y) argument (see sklearn.utils.validation._check_y); since d sits in that
        # slot, validate it with check_array instead to actually honor ensure_all_finite=False.
        d = check_array(self._dml_data.d, ensure_2d=False, allow_nd=False, ensure_all_finite=False, input_name="d")
        check_consistent_length(x, d)

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
                d,
                smpls=smpls,
                n_jobs=n_jobs_cv,
                est_params=self._get_params("ml_m"),
                method=self._predict_method["ml_m"],
                return_models=return_models,
            )
        m_hat["preds"] = self._ps_processor.adjust_ps(m_hat["preds"], d, cv=smpls, learner_name="ml_m")

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

        return psi_elements, preds

    def _nuisance_est_counterfactual(
        self,
        smpls,
        n_jobs_cv,
        external_predictions,
        return_models=False,
    ):
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        # see _nuisance_est: check_array avoids check_X_y's ensure_all_finite being a no-op for
        # the y-argument slot that d occupies here.
        d = check_array(self._dml_data.d, ensure_2d=False, allow_nd=False, ensure_all_finite=False, input_name="d")
        check_consistent_length(x, d)

        # check_X_y(..., multi_output=True) hardcodes ensure_all_finite=True for the y-argument
        # internally (sklearn's _check_y), ignoring the caller's setting. m is not really a
        # "y" target here, so validate it with check_array instead to honor ensure_all_finite.
        m = check_array(self._dml_data.m, ensure_2d=False, allow_nd=False, ensure_all_finite=False, input_name="m")
        check_consistent_length(x, m)
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
        m_hat["preds"] = self._ps_processor.adjust_ps(m_hat["preds"], d, cv=smpls, learner_name="ml_m")

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
        M_hat["preds"] = self._ps_processor.adjust_ps(M_hat["preds"], d, cv=smpls, learner_name="ml_M")

        inner_predictions = {}
        inner_targets = {}
        inner_models = {}
        if self.double_sample_splitting:
            if self.n_folds_inner == 1:
                _check_subsample_arms(self._smpls_inner[self._i_rep], self.treated, self.treatment_level)
            # Get inner samples conditional on treatment:
            smpls_inner_d1 = []
            for fold in self._smpls_inner[self._i_rep]:
                _, inner_smpls_d1 = _get_cond_smpls(fold, self.treated)
                smpls_inner_d1.append(inner_smpls_d1)

            if G_external:
                # expect per-inner-fold keys ml_G_inner_i
                _check_inner_g_external_predictions(external_predictions, self.n_folds)
                G_hat_inner = [external_predictions[name] for name in _inner_g_learner_names(self.n_folds)]
                G_hat = {
                    "preds": external_predictions["ml_G"],
                    "preds_inner": G_hat_inner,
                    "targets": self._dml_data.y,
                    "models": None,
                }
            else:

                G_hat = _double_dml_cv_predict_by_fold(
                    estimator=self._learner["ml_G"],
                    estimator_name="ml_G",
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
                if self.n_folds_inner == 1:
                    # the nested estimator is fitted on subsample B only, the units ml_G has predictions for
                    smpls_nested = [
                        (np.intersect1d(train_d0, inner_fold[0][1]), test)
                        for (train_d0, test), inner_fold in zip(smpls_d0, self._smpls_inner[self._i_rep])
                    ]
                else:
                    smpls_nested = smpls_d0
                G_hat_inner_preds = [G_hat_preds[train] for (train, _), G_hat_preds in zip(smpls_nested, G_hat["preds_inner"])]

                nested_g_hat = _dml_cv_predict(
                    self._learner["ml_nested_g"],
                    x,
                    G_hat_inner_preds,
                    smpls=smpls_nested,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_nested_g"),
                    method=self._predict_method["ml_nested_g"],
                    return_models=return_models,
                )
            # store inner predictions as separate keys per inner fold
            inner_predictions = {_inner_g_learner_name(i): G_hat["preds_inner"][i] for i in range(len(G_hat["preds_inner"]))}
            inner_targets = {
                _inner_g_learner_name(i): (
                    G_hat.get("targets_inner")[i]
                    if G_hat.get("targets_inner") is not None and i < len(G_hat["targets_inner"])
                    else None
                )
                for i in range(len(G_hat.get("preds_inner", [])))
            }
            # the flat list of inner models is ordered by outer fold, with one model per inner fold
            n_inner = len(G_hat["models"]) // self.n_folds if G_hat["models"] is not None else None
            inner_models = {
                _inner_g_learner_name(i): G_hat["models"][i * n_inner : (i + 1) * n_inner] if n_inner is not None else None
                for i in range(len(G_hat.get("preds_inner", [])))
            }

        else:
            # The model of each fold is fitted on the training sample of the fold only. It predicts the observations of the
            # fold (used in the score) and the observations of its training sample with D = 1-d (the targets of
            # ml_nested_g). Cross-fitted predictions of ml_G cannot serve as targets, because the target of an observation
            # of another fold is predicted by a model that has been fitted on the observations of this fold.
            if G_external:
                _check_inner_g_external_predictions(external_predictions, self.n_folds)
                G_hat = {
                    "preds": external_predictions["ml_G"],
                    "preds_inner": [external_predictions[name] for name in _inner_g_learner_names(self.n_folds)],
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
                    return_models=True,
                )
                n_obs = x.shape[0]
                G_hat["preds_inner"] = []
                G_hat["targets_inner"] = []
                for (train_d0, _), fold_model in zip(smpls_d0, G_hat["models"]):
                    preds_fold = np.full(n_obs, np.nan)
                    preds_fold[train_d0] = _predict_with_method(fold_model, xm[train_d0], self._predict_method["ml_G"])
                    targets_fold = np.full(n_obs, np.nan)
                    targets_fold[train_d0] = y[train_d0]
                    G_hat["preds_inner"].append(preds_fold)
                    G_hat["targets_inner"].append(targets_fold)
                if not return_models:
                    G_hat["models"] = None

            if nested_g_external:
                nested_g_hat = {"preds": external_predictions["ml_nested_g"], "targets": None, "models": None}
            else:
                # fold specific targets, see the comment of the double sample splitting
                G_hat_inner_preds = [G_hat["preds_inner"][i][train] for i, (train, _) in enumerate(smpls_d0)]
                nested_g_hat = _dml_cv_predict(
                    self.learner["ml_nested_g"],
                    x=x,
                    y=G_hat_inner_preds,
                    smpls=smpls_d0,
                    n_jobs=n_jobs_cv,
                    est_params=self._get_params("ml_nested_g"),
                    method=self._predict_method["ml_nested_g"],
                    return_models=return_models,
                )
            # the entries ml_G_inner_i hold the predictions of the model of the fold i for its training sample
            inner_predictions = {_inner_g_learner_name(i): G_hat["preds_inner"][i] for i in range(self.n_folds)}
            inner_targets = {
                _inner_g_learner_name(i): G_hat["targets_inner"][i] if "targets_inner" in G_hat else None
                for i in range(self.n_folds)
            }
            inner_models = {
                _inner_g_learner_name(i): [G_hat["models"][i]] if G_hat["models"] is not None else None
                for i in range(self.n_folds)
            }
        # ml_G estimates E[Y|D=d,M,X], so its predictions are compared with the outcome of the observations with D = d. The
        # nested estimator has no observed outcome: its predictions are compared with the predictions of ml_G for the
        # observations with D = 1-d, which are the targets of the nested estimation.
        ml_G_targets = _cond_targets(y, cond_sample=self.treated)
        inner_targets = {
            name: None if targets is None else _cond_targets(targets, cond_sample=self.treated)
            for name, targets in inner_targets.items()
        }
        nested_g_targets = np.where(self.treated, np.nan, G_hat["preds"])

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
                "ml_G": ml_G_targets,
                "ml_nested_g": nested_g_targets,
                **inner_targets,
            },
            "models": {
                "ml_m": m_hat["models"],
                "ml_M": M_hat["models"],
                "ml_G": G_hat["models"],
                "ml_nested_g": nested_g_hat["models"],
                **inner_models,
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
        if self._outcome == "factual":
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
        x, y = check_X_y(self._dml_data.x, self._dml_data.y, ensure_all_finite=False)
        # see _nuisance_est: check_array avoids check_X_y's ensure_all_finite being a no-op for
        # the y-argument slot that d occupies here.
        d = check_array(self._dml_data.d, ensure_2d=False, allow_nd=False, ensure_all_finite=False, input_name="d")
        check_consistent_length(x, d)

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

        if self.outcome == "factual":
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
            # see _nuisance_est_counterfactual: check_array avoids check_X_y(multi_output=True)
            # hardcoding ensure_all_finite=True for m.
            m = check_array(self._dml_data.m, ensure_2d=False, allow_nd=False, ensure_all_finite=False, input_name="m")
            check_consistent_length(x, m)
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

            # Targets of the nested estimator: ml_G has to be tuned first, since the nested estimator regresses its
            # predictions on x. We use out-of-fold predictions of the tuned ml_G (fitted on the treated, using x and m)
            # for all observations. Unlike in _nuisance_est_counterfactual, no inner cross-fitting is used, so the targets
            # are an approximation of the ones used for estimation (a similar compromise is made for ml_t in
            # DoubleMLLPLR._nuisance_tuning).
            cv_resolved = resolve_optuna_cv(cv)
            tuning_smpls = cv_resolved if isinstance(cv_resolved, list) else list(cv_resolved.split(xm))
            _, tuning_smpls_treated = _get_cond_smpls(tuning_smpls, treated_mask)
            G_hat = _dml_cv_predict(
                self._learner["ml_G"],
                xm,
                y,
                smpls=tuning_smpls_treated,
                est_params=G_tune_res.best_params,
                method=self._predict_method["ml_G"],
            )["preds"]

            # the nested estimator regresses these predictions on x among the non-treated
            nested_g_tune_res = _dml_tune_optuna(
                y=G_hat[not_treated_mask],
                x=x[not_treated_mask],
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
        raise NotImplementedError()

    def set_sample_splitting(self, smpls, all_smpls_cluster=None, is_cluster_data=False, smpls_inner=None):
        """
        Set the sample splitting for the model.

        The attributes ``n_folds`` and ``n_rep`` are derived from the provided partition ``smpls``. If
        ``double_sample_splitting=True``, ``smpls_inner`` has to be provided and ``n_folds_inner`` is derived from it.

        Parameters
        ----------
        smpls : list or tuple
            The sample splitting, in the format described in :meth:`DoubleML.set_sample_splitting`.

        all_smpls_cluster : list or None
            Cluster sample splitting. Not supported together with inner samples.

        is_cluster_data : bool
            Indicates cluster data. Not supported together with inner samples.

        smpls_inner : list or None
            The inner sample splitting used for the nested estimation. For each repetition and each outer fold, a list
            of ``n_folds_inner`` tuples ``(train_ind, test_ind)`` of indices within the outer training sample. A single
            tuple ``(A, B)`` selects the two-subsample scheme of Algorithm 2 of Farbmacher et al. (2022).
            Required if ``double_sample_splitting=True``.

        Returns
        -------
        self : object

        Notes
        -----
        The inner samples have to partition the outer training sample of each fold: inner training and test indices must
        lie within the outer training sample, and the inner test folds must be disjoint and cover it. A single inner
        tuple ``(A, B)`` is valid if A and B are disjoint and together cover the outer training sample. As for the outer
        sample splitting, the splits should be drawn independently of the data, in particular of the outcome.
        :class:`DoubleMLMEDS` uses this method to share one sample splitting between its models.
        """
        if self.double_sample_splitting:
            if smpls_inner is None:
                raise ValueError("smpls_inner is required")
            if all_smpls_cluster is not None or is_cluster_data:
                raise NotImplementedError("sample setting with cluster data and inner samples not supported.")
            new_smpls, new_smpls_cluster, new_n_rep, new_n_folds = _check_sample_splitting(
                all_smpls=smpls,
                all_smpls_cluster=all_smpls_cluster,
                dml_data=self._dml_data,
                is_cluster_data=is_cluster_data,
                n_obs=None,
            )
            new_smpls_inner, new_n_folds_inner = _check_inner_sample_splitting(
                smpls_inner,
                new_smpls,
                self._dml_data.n_obs,
            )

            self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = new_smpls, new_smpls_cluster, new_n_rep, new_n_folds
            self._smpls_inner, self.n_folds_inner = new_smpls_inner, new_n_folds_inner
            self._initialize_dml_model()
            return self

        return super().set_sample_splitting(all_smpls=smpls, all_smpls_cluster=all_smpls_cluster)

    def _check_learners(self, ml_g, ml_m, ml_G, ml_M, ml_nested_g):
        if self._outcome == "factual":
            self._learner = {"ml_g": ml_g, "ml_m": ml_m}
        else:
            self._learner = {"ml_m": ml_m, "ml_G": ml_G, "ml_M": ml_M, "ml_nested_g": ml_nested_g}

        # Standard requirements mapping
        # (name): (must_be_classifier, must_be_regressor)
        # None means it depends on data.binary_outcome
        learner_requirements = {
            "ml_m": (True, False),
            "ml_M": (True, False),
            "ml_nested_g": (False, True),
            "ml_g": (None, None),
            "ml_G": (None, None),
        }

        for learner_name, learner in self._learner.items():
            if learner is None:
                raise ValueError(f"Learner {learner_name} is required when the outcome is {self._outcome}.")

            req_class, req_reg = learner_requirements[learner_name]
            if req_class is None:  # Type of learning task is data dependent
                req_class = self._dml_data.binary_outcome
                req_reg = not self._dml_data.binary_outcome

            is_classifier = self._check_learner(learner, learner_name, regressor=True, classifier=True)

            if req_class and not is_classifier:
                raise ValueError(f"Learner '{learner_name}' must be a classifier.")
            if req_reg and is_classifier:
                raise ValueError(f"Learner '{learner_name}' must be a regressor.")

            self._predict_method[learner_name] = "predict_proba" if is_classifier else "predict"
        self._initialize_ml_nuisance_params()

    def _check_outcome(self, outcome):
        if not isinstance(outcome, str):
            raise TypeError("Outcome must be a string." + f"{str(outcome)} of type {str(type(outcome))} provided instead.")

        valid_outcomes = ["factual", "counterfactual"]
        if outcome not in valid_outcomes:
            raise ValueError(f"Invalid outcome {outcome}. " + "Valid outcomes " + " or ".join(valid_outcomes) + ".")

        return outcome

    def _check_levels(self, treatment_level):
        if not isinstance(treatment_level, Integral):
            raise TypeError(
                "Treatment level must be an integer."
                + f" Treatment level {str(treatment_level)} of type {str(type(treatment_level))} provided."
            )

        if not 0 <= treatment_level <= 1:
            raise ValueError(
                "Treatment level must be either 0 or 1" + f" Treatment level provided was {str(treatment_level)}."
            )
        return int(treatment_level)
