"""causal mediation analysis binary treatment levels and continuous or discrete mediation variables
computing total, direct and indirect effects."""

import itertools
from typing import Any, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import clone

from doubleml.data import DoubleMLMEDData
from doubleml.double_ml_framework import concat
from doubleml.double_ml_sampling_mixins import SampleSplittingMixin
from doubleml.med.med import DoubleMLMED
from doubleml.med.utils._med_utils import (
    VALID_OUTCOMES,
    VALID_SCORES,
    _check_med_data,
    _first_inner_split,
    _inner_g_learner_names,
)
from doubleml.med.utils._meds_utils import generate_effects_summary
from doubleml.utils import PSProcessorConfig
from doubleml.utils._checks import _check_score
from doubleml.utils._descriptive import generate_summary
from doubleml.utils._tune_optuna import TUNE_ML_MODELS_DOC


class DoubleMLMEDS(SampleSplittingMixin):
    """Double machine learning for causal mediation analysis with binary treatment.
    Computes the average treatment effect (ATE), direct effects
    and indirects effect.

    Parameters
    ----------
    obj_dml_data : :class:`DoubleMLMEDData` object
        The :class:`DoubleMLMEDData` object providing the data and specifying the variables for the causal model.

    ml_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.linear_model.Lasso`) for the nuisance function :math:`E[Y|D,X]`.

    ml_G : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.linear_model.Lasso`) for the nuisance function :math:`E[Y|D,M,X]`.
        Only required if ``outcome`` is 'counterfactual'.

    ml_nested_g : estimator implementing ``fit()`` and ``predict()``
        A machine learner implementing ``fit()`` and ``predict()`` methods (e.g.
        :py:class:`sklearn.linear_model.Lasso`) for the nuisance function :math:`E[E[Y|D=d,M,X]|D=1-d, X]`
        Only required if ``outcome`` is 'counterfactual'. Its target is not an observed outcome but the predictions
        of ``ml_G``, which also applies to :meth:`tune_ml_models`, see the notes.

    ml_m : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.linear_model.LogisticRegression`) for the nuisance function :math:`P(D=d|X)`.

    ml_M : classifier implementing ``fit()`` and ``predict_proba()``
        A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
        :py:class:`sklearn.linear_model.LogisticRegression`) for the nuisance function :math:`P(D=d|M,X)`.
        Only required if ``outcome`` is 'counterfactual'.

    n_folds : int
        Number of folds.
        Default is ``5``.

    n_rep : int
        Number of repetitions for the sample splitting.
        Default is ``1``.

    score : str
        A str (``'efficient-alt'`` is the only choice)  specifying the score function to use.
        Default is 'efficient-alt'.

    n_folds_inner : int
        Number of inner folds used for the estimation of the nested parameter if ``double_sample_splitting=True``.
        The training sample of each of the ``n_folds`` folds is split into ``n_folds_inner`` inner folds,
        see the notes. If ``n_folds_inner=1``, the training sample is instead split once into two subsamples, as in
        Algorithm 2 of Farbmacher et al. (2022).
        Default is ``5``.

    normalize_ipw : bool
        Indicates whether the inverse probability weights are normalized.
        Default is ``True``.

    draw_sample_splitting : bool
        Indicates whether the sample splitting should be drawn during initialization of the object.
        Default is ``True``.

    double_sample_splitting : bool
        Indicates whether the data is resampled for the estimation of the nested parameter
        :math:`E[E[Y|D=d,M,X]|D=1-d,X]`, see the notes. Only used for the estimation of the counterfactual outcomes.
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
    The sample splitting for the estimation of the nested parameter, which is controlled by
    ``double_sample_splitting`` and ``n_folds_inner``, is described in the notes of :class:`DoubleMLMED`, as is the
    tuning of ``ml_nested_g`` against the predictions of ``ml_G``.

    :meth:`tune_ml_models` tunes the nuisance learners of each model separately. Its ``ml_param_space`` can contain any
    of the learners ``ml_g``, ``ml_m``, ``ml_G``, ``ml_M`` and ``ml_nested_g``. A model tunes the learners it has, and a
    learner that is missing or ``None``, e.g. ``{"ml_nested_g": None}``, is left untuned. As in :class:`DoubleMLMED`, tuning
    ``ml_nested_g`` against the tuned ``ml_G`` requires both in the same ``ml_param_space``.

    The scope of the estimators, the score and the normalization of the weights are described in the notes of
    :class:`DoubleMLMED`. This class does not offer ``set_ml_nuisance_params``, ``evaluate_learners`` and ``p_adjust``, which
    :class:`DoubleMLMED` provides, and sensitivity analysis is not implemented for either class. The hyperparameters are
    set on the learners passed to the constructor or by :meth:`tune_ml_models`, and the models in ``modeldict`` provide
    the methods of :class:`DoubleMLMED`. Joint confidence intervals are not offered, because the asymptotic theory of the
    paper covers each estimator on its own.

    References
    ----------
    Helmut Farbmacher, Martin Huber, Lukáš Lafférs, Henrika Langen, Martin Spindler (2022).
    Causal mediation analysis with double machine learning. The Econometrics Journal, 25(2), 277-300.
    https://doi.org/10.1093/ectj/utac003

    Hugo Bodory, Martin Huber, Jannis Kueck. causalweight: Estimation Methods for Causal Inference Based on Inverse
    Probability Weighting and Doubly Robust Estimation. R package (MIT license), https://CRAN.R-project.org/package=causalweight.

    Examples
    --------

    >>> import numpy as np
    >>> import doubleml as dml
    >>> from doubleml.med.datasets import make_med_data
    >>> from sklearn.linear_model import LogisticRegression, Lasso
    >>> from sklearn.base import clone
    >>> np.random.seed(3141)
    >>> ml_g = Lasso()
    >>> ml_G = Lasso()
    >>> ml_nested_g = Lasso()
    >>> ml_m = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
    >>> ml_M = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
    >>> obj_dml_data = make_med_data()
    >>> dml_med_obj = dml.DoubleMLMEDS(obj_dml_data, ml_m, ml_g, ml_G, ml_M, ml_nested_g)
    >>> print(dml_med_obj.effects_summary) # doctest: +SKIP

                       coef   std err          t     P>|t|     2.5 %    97.5 %
    ATE            1.007247  0.071861  14.016534  0.000000  0.866402  1.148093
    DIR_TREAT      0.840616  0.076604  10.973519  0.000000  0.690475  0.990757
    DIR_CONTROL    0.824458  0.077925  10.580134  0.000000  0.671728  0.977189
    INDIR_TREAT    0.182789  0.020347   8.983643  0.000000  0.142910  0.222668
    INDIR_CONTROL  0.166631  0.041467   4.018419  0.000059  0.085358  0.247905
    """

    _dml_data: DoubleMLMEDData

    def __init__(
        self,
        obj_dml_data: DoubleMLMEDData,
        ml_m: Any,
        ml_g: Any,
        ml_G: Any,
        ml_M: Any,
        ml_nested_g: Any,
        n_folds: int = 5,
        n_rep: int = 1,
        score: str = "efficient-alt",
        n_folds_inner: int = 5,
        normalize_ipw: bool = True,
        draw_sample_splitting: bool = True,
        double_sample_splitting: bool = True,
        ps_processor_config: Optional[PSProcessorConfig] = None,
    ) -> None:

        self._dml_data = _check_med_data(obj_dml_data)
        self._is_cluster_data = self._dml_data.is_cluster_data

        # Check score
        self._score = score
        _check_score(self._score, list(VALID_SCORES), allow_callable=False)

        self._normalize_ipw = normalize_ipw
        self._double_sample_splitting = double_sample_splitting
        self._n_folds = n_folds
        self._n_rep = n_rep
        self._n_folds_inner = n_folds_inner

        self._id_pairs = self._valid_id_pairs()
        self._models_ids = self._initialize_models_ids()
        # initialize learners and parameters which are set model specific
        self._learner = {
            "ml_m": clone(ml_m),
            "ml_g": clone(ml_g),
            "ml_G": clone(ml_G),
            "ml_M": clone(ml_M),
            "ml_nested_g": clone(ml_nested_g),
        }
        self._params: Any = None

        # Initialize framework constructed after the fit method is called.
        self._framework: Any = None

        # Initialize all properties to None
        self._se: Any = None
        self._pvalues: Any = None
        self._coef: Any = None
        self._ci: Any = None

        # perform sample splitting
        self._smpls: Any = None
        self._modeldict: Any = None
        self._smpls_inner: Any = None
        self._n_obs_sample_splitting = self._dml_data.n_obs
        self._strata: Any = None

        self._ps_processor_config = ps_processor_config if ps_processor_config is not None else PSProcessorConfig()

        if draw_sample_splitting:
            self.draw_sample_splitting()

        self._effects: Any = None

    def __str__(self) -> str:
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        fit_summary = str(self.summary)
        res = header + "\n------------------ Fit summary       ------------------\n" + fit_summary
        effects_summary = str(self.effects_summary)
        res = res + "\n------------------ Effects summary       ------------------\n" + effects_summary
        return res

    @property
    def score(self) -> str:
        """
        The score function.
        """
        return self._score

    @property
    def smpls_inner(self) -> Optional[list]:
        """
        The inner sample splitting used for the estimation of the nested parameter.
        """
        return self._smpls_inner

    @property
    def n_folds(self) -> int:
        """
        Specifies the number of folds to be used for cross-validation
        """
        return self._n_folds

    @property
    def n_rep(self) -> int:
        """
        Number of repetitions for the sample splitting.
        """
        return self._n_rep

    @property
    def n_folds_inner(self) -> int:
        """
        Number of folds to be used for cross-validation of the inner models.
        """
        return self._n_folds_inner

    @property
    def normalize_ipw(self) -> bool:
        """
        Indicates whether the inverse probability weights are normalised.
        """
        return self._normalize_ipw

    @property
    def models_ids(self) -> list[str]:
        """
        The identifiers ``'<outcome>_<treatment level>'`` of the models, e.g. ``'factual_1'``.
        """
        return self._models_ids

    @property
    def double_sample_splitting(self) -> bool:
        """
        Indicates whether the data is resampled for the estimation of the nested parameter.
        """
        return self._double_sample_splitting

    @property
    def treatments(self) -> np.ndarray:
        """
        Describes the different treatment levels
        """
        return np.unique(self._dml_data.d)

    @property
    def ps_processor_config(self) -> PSProcessorConfig:
        """
        Configuration for propensity score processing (clipping, calibration, etc.).
        """
        return self._ps_processor_config

    @property
    def effects(self) -> Optional[dict[str, Any]]:
        """
        Estimates for the ATE and the combination of the direct/indirect, treatment/control groups effects after calling
        :meth:`evaluate-effects`
        """
        return self._effects

    @property
    def coef(self) -> Optional[np.ndarray]:
        """
        Estimates for the causal parameter(s) after calling :meth:`fit` (shape(4,)).
        """
        if self._framework is None:
            coef = None
        else:
            coef = self.framework.thetas
        return coef

    @property
    def all_coef(self) -> Optional[np.ndarray]:
        """
        Estimates of the causal parameter(s) for the ``n_rep`` different sample splits after calling :meth: `fit`
        (shape (4, ``n_rep``))
        """
        if self._framework is None:
            all_coef = None
        else:
            all_coef = self.framework.all_thetas
        return all_coef

    @property
    def se(self) -> Optional[np.ndarray]:
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit` (shape (4,)).
        """
        if self._framework is None:
            se = None
        else:
            se = self.framework.ses
        return se

    @property
    def all_se(self) -> Optional[np.ndarray]:
        """
        Standard errors for the causal parameter(s) of interest :meth:`fit` (shape (4,)).
        """
        if self._framework is None:
            all_se = None
        else:
            all_se = self.framework.all_ses
        return all_se

    @property
    def t_stat(self) -> Optional[np.ndarray]:
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            t_stats = None
        else:
            t_stats = self.framework.t_stats
        return t_stats

    @property
    def pval(self) -> Optional[np.ndarray]:
        """
        p-values for the causal parameter(s) (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            pvals = None
        else:
            pvals = self.framework.pvals
        return pvals

    @property
    def smpls(self) -> list:
        """
        The partition used for cross-fitting.
        """
        if self._smpls is None:
            err_msg = (
                "Sample splitting not specified. Draw samples via .draw_sample splitting(). "
                + "External samples not implemented yet."
            )
            raise ValueError(err_msg)
        return self._smpls

    @property
    def framework(self) -> Any:
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

    @property
    def modeldict(self) -> dict[str, DoubleMLMED]:
        """
        The list of models for each level.
        """
        if self._modeldict is None:
            raise ValueError(
                "Sample splitting not specified. Draw samples via .draw_sample splitting(). "
                + "External samples not implemented yet."
            )
        return self._modeldict

    @property
    def summary(self) -> pd.DataFrame:
        """
        A summary for the estimated potential outcomes after calling :meth:`fit`.
        """
        if self.framework is None:
            col_names = ["coef", "std err", "t", "P>|t|"]
            df_summary = pd.DataFrame(columns=col_names)
        else:
            if self._effects is None:
                self.evaluate_effects()
            ci = self.confint()
            df_summary = generate_summary(
                coef=self.coef, se=self.se, t_stat=self.t_stat, pval=self.pval, ci=ci, index_names=self.models_ids
            )
        return df_summary

    @property
    def effects_summary(self) -> pd.DataFrame:
        """
        A summary for the estimated effects. The effects are computed automatically
        (via :meth:`evaluate_effects`) on first access once :meth:`fit` has been called.
        """
        if self.framework is not None and self._effects is None:
            self.evaluate_effects()
        if self._effects is None:
            col_names = ["coef", "std err", "t", "P>|t|"]
            df_effects_summary = pd.DataFrame(columns=col_names)
        else:
            df_effects_summary = generate_effects_summary(self._effects)
        return df_effects_summary

    def fit(
        self,
        n_jobs_models: Optional[int] = None,
        n_jobs_cv: Optional[int] = None,
        store_predictions: bool = True,
        store_models: bool = False,
        external_predictions: Optional[dict[str, Any]] = None,
    ) -> "DoubleMLMEDS":
        """
        Estimates DoubleMLMEDS objects.

        Parameters
        ----------
        n_jobs_models : int, optional
            Number of jobs to run in parallel for the models.
            Default is ``None``.
        n_jobs_cv : int, optional
            Number of jobs to run in parallel for the cross-fitting.
            Default is ``None``.
        store_predictions : bool, optional
            Whether to store the predictions.
            Default is ``True``.
        store_models : bool, optional
            Whether to store the models.
            Default is ``False``.
        external_predictions : dict, optional
            External predictions to use for the estimation.
            Default is ``None``.

        Examples
        --------
        >>> import numpy as np
        >>> import doubleml as dml
        >>> from doubleml.med.datasets import make_med_data
        >>> from sklearn.linear_model import LogisticRegression, Lasso
        >>> np.random.seed(3141)
        >>> ml_g = Lasso()
        >>> ml_G = Lasso()
        >>> ml_nested_g = Lasso()
        >>> ml_m = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
        >>> ml_M = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
        >>> obj_dml_data = make_med_data()
        >>> dml_meds_obj = dml.DoubleMLMEDS(obj_dml_data, ml_m, ml_g, ml_G, ml_M, ml_nested_g)
        >>> dml_meds_obj.fit().summary  # doctest: +SKIP
        """
        if external_predictions is not None:
            self._check_external_predictions(external_predictions)

        # parallel estimation of the models
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")
        fitted_models = parallel(
            delayed(self._fit_model)(model_id, n_jobs_cv, store_predictions, store_models, external_predictions)
            for model_id in self.models_ids
        )

        # combine the estimates and model_ids
        framework_list = [None] * len(self.models_ids)

        for idx, model_id in enumerate(self.models_ids):
            self._modeldict[model_id] = fitted_models[idx]
            framework_list[idx] = self._modeldict[model_id].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)

        # the effects are derived from the frameworks of the models, so they are recomputed on the next access
        self._effects = None

        return self

    def _fit_model(
        self,
        score: str,
        n_jobs_cv: Optional[int] = None,
        store_predictions: bool = True,
        store_models: bool = False,
        external_predictions_dict: Optional[dict[str, Any]] = None,
    ) -> DoubleMLMED:
        model = self.modeldict[score]
        if external_predictions_dict is not None:
            # models without entry in external_predictions are estimated as usual
            external_predictions = external_predictions_dict.get(score)
        else:
            external_predictions = None

        model.fit(
            n_jobs_cv=n_jobs_cv,
            store_predictions=store_predictions,
            store_models=store_models,
            external_predictions=external_predictions,
        )

        return model

    def confint(self, level: float = 0.95) -> pd.DataFrame:
        """
        Confidence intervals for DoubleMLMEDS objects.

        The intervals are computed for each model separately. Joint confidence intervals are not offered, because the
        asymptotic theory of Farbmacher et al. (2022) covers each estimator on its own.

        Parameters
        ----------
        level : float
            The confidence level.
            Default is ``0.95``.

        Returns
        -------
        df_ci : pd.DataFrame
            A data frame with the confidence interval(s), one row for each model.

        Examples
        --------
        >>> import numpy as np
        >>> import doubleml as dml
        >>> from doubleml.med.datasets import make_med_data
        >>> from sklearn.linear_model import LogisticRegression, Lasso
        >>> np.random.seed(3141)
        >>> ml_g = Lasso()
        >>> ml_G = Lasso()
        >>> ml_nested_g = Lasso()
        >>> ml_m = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
        >>> ml_M = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
        >>> obj_dml_data = make_med_data()
        >>> dml_meds_obj = dml.DoubleMLMEDS(obj_dml_data, ml_m, ml_g, ml_G, ml_M, ml_nested_g)
        >>> dml_meds_obj.fit().confint()  # doctest: +SKIP
        """
        if self.framework is None:
            raise ValueError("Apply fit() before confint().")
        df_ci = self.framework.confint(joint=False, level=level)
        df_ci.set_index(pd.Index(self.models_ids), inplace=True)
        return df_ci

    def evaluate_effects(self) -> "DoubleMLMEDS":
        """
        Evaluate the effects of the DoubleMLMEDS object after calling :meth:`fit`.

        The following effects are evaluated:
        - Average Treatment Effect (ATE)
        - Direct treatment effect (DIR_TREAT)
        - Direct control effect (DIR_CONTROL)
        - Indirect treatment effect (INDIR_TREAT)
        - Indirect control effect (INDIR_CONTROL)

        Returns
        -------
        self : object

        Examples
        --------
        >>> import numpy as np
        >>> import doubleml as dml
        >>> from doubleml.med.datasets import make_med_data
        >>> from sklearn.linear_model import LogisticRegression, Lasso
        >>> np.random.seed(3141)
        >>> ml_g = Lasso()
        >>> ml_G = Lasso()
        >>> ml_nested_g = Lasso()
        >>> ml_m = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
        >>> ml_M = LogisticRegression(l1_ratio=1, solver="liblinear", max_iter=1000)
        >>> obj_dml_data = make_med_data()
        >>> dml_meds_obj = dml.DoubleMLMEDS(obj_dml_data, ml_m, ml_g, ml_G, ml_M, ml_nested_g)
        >>> dml_meds_obj.fit().evaluate_effects().effects_summary  # doctest: +SKIP
        """
        if self.framework is None:
            raise ValueError("Apply fit() before calling evaluate_effects()")
        ate = self._modeldict["factual_1"].framework - self._modeldict["factual_0"].framework
        dir_treat = self._modeldict["factual_1"].framework - self._modeldict["counterfactual_0"].framework
        dir_control = self._modeldict["counterfactual_1"].framework - self._modeldict["factual_0"].framework
        indir_treat = self._modeldict["factual_1"].framework - self._modeldict["counterfactual_1"].framework
        indir_control = self._modeldict["counterfactual_0"].framework - self._modeldict["factual_0"].framework

        self._effects = {
            "ATE": ate,
            "DIR_TREAT": dir_treat,
            "DIR_CONTROL": dir_control,
            "INDIR_TREAT": indir_treat,
            "INDIR_CONTROL": indir_control,
        }

        return self

    def draw_sample_splitting(self) -> "DoubleMLMEDS":
        """
        Draw sample splitting for DoubleMLMEDS models.

        The samples are drawn according to the attributes ``n_folds``, ``n_rep`` and, for double sample splitting,
        ``n_folds_inner``. If ``n_folds_inner=1``, the training sample of each fold is split once into two disjoint
        subsamples ``(A, B)``. The models are initialized with the new sample splitting, so results of a previous
        :meth:`fit` are discarded.

        Returns
        -------
        self : object
        """
        if self._double_sample_splitting and self._n_folds_inner == 1:
            # the inner resampling needs at least two folds, the first inner tuple is the (A, B) pair
            self._n_folds_inner = 2
            try:
                super().draw_sample_splitting()
            finally:
                self._n_folds_inner = 1
            self._smpls_inner = _first_inner_split(self._smpls_inner)
        else:
            super().draw_sample_splitting()
        self._initialize_dml_model()
        return self

    def _initialize_dml_model(self) -> "DoubleMLMEDS":
        self._modeldict = self._initialize_models()
        # results of a previous fit belong to the models that have just been replaced
        self._framework = None
        self._effects = None
        return self

    def _initialize_models(self) -> dict[str, DoubleMLMED]:
        modeldict: dict[str, Any] = {score: None for score in self.models_ids}

        pot_learners = {
            "ml_m": self._learner["ml_m"],
            "ml_g": self._learner["ml_g"],
        }
        counter_learners = {
            "ml_m": self._learner["ml_m"],
            "ml_G": self._learner["ml_G"],
            "ml_M": self._learner["ml_M"],
            "ml_nested_g": self._learner["ml_nested_g"],
        }

        for score, (outcome, treatment) in zip(self.models_ids, self._id_pairs):
            learners = pot_learners if outcome == "factual" else counter_learners

            model = DoubleMLMED(
                obj_dml_data=self._dml_data,
                outcome=outcome,
                treatment_level=treatment,
                n_rep=self._n_rep,
                n_folds=self._n_folds,
                n_folds_inner=self._n_folds_inner,
                normalize_ipw=self._normalize_ipw,
                double_sample_splitting=self._double_sample_splitting,
                draw_sample_splitting=False,
                ps_processor_config=self._ps_processor_config,
                **learners,
            )

            model.set_sample_splitting(smpls=self._smpls, smpls_inner=self.smpls_inner)

            modeldict[f"{outcome}_{treatment}"] = model

        return modeldict

    def _initialize_models_ids(self) -> list[str]:
        return [f"{outcome}_{treatment}" for outcome, treatment in self._id_pairs]

    def _valid_id_pairs(self) -> list[tuple[str, int]]:
        # the treatments are binary, which is guaranteed by _check_med_data
        self.valid_outcomes = list(VALID_OUTCOMES)
        return list(itertools.product(self.valid_outcomes, map(int, self.treatments)))

    def _check_external_predictions(self, external_predictions_dict: dict[str, Any]) -> None:
        if not isinstance(external_predictions_dict, dict):
            raise TypeError(
                "external_predictions must be a dictionary. " + f"Object of type {type(external_predictions_dict)} passed."
            )
        external_predictions_keys = external_predictions_dict.keys()
        if not set(external_predictions_keys).issubset(set(self.models_ids)):
            raise ValueError(
                "external_predictions must be a subset of all scores. "
                + f"Expected keys: {set(self.models_ids)}. "
                + f"Passed keys: {set(external_predictions_keys)}."
            )

        expected_learners_keys = [
            "ml_m",
            "ml_M",
            "ml_g",
            "ml_G",
            "ml_nested_g",
        ]
        expected_learners_keys += _inner_g_learner_names(self.n_folds)

        for model_id, model_predictions in external_predictions_dict.items():
            if not isinstance(model_predictions, dict):
                raise TypeError(
                    f"external_predictions[{model_id}] must hold a dictionary. "
                    + f"Current value in external_predictions[{model_id}] is of type {type(model_predictions)}"
                )
            if not set(model_predictions.keys()).issubset(set(self._dml_data.d_cols)):
                raise ValueError(
                    f"external_predictions[{model_id}] must be keyed by the treatment variables. "
                    + f"Expected keys: {set(self._dml_data.d_cols)}. "
                    + f"Passed keys: {set(model_predictions.keys())}."
                )
            for d_col in model_predictions.keys():
                if not isinstance(model_predictions[d_col], dict):
                    raise TypeError(
                        f"external_predictions[{model_id}][{d_col}] must hold a dictionary. "
                        + f"Current value is of type {type(model_predictions[d_col])}"
                    )
                if not set(model_predictions[d_col].keys()).issubset(expected_learners_keys):
                    raise ValueError(
                        f"external_predictions[{d_col}] must hold a dictionnary whose keys are a "
                        + f"subset of {set(expected_learners_keys)}. "
                        + f"Passed keys: {set(model_predictions[d_col].keys())}."
                    )

    def tune_ml_models(
        self,
        ml_param_space: dict[str, Any],
        scoring_methods: Optional[dict[str, Any]] = None,
        cv: Any = 5,
        set_as_params: bool = True,
        return_tune_res: bool = False,
        optuna_settings: Optional[dict[str, Any]] = None,
    ) -> Any:
        """Hyperparameter-tuning for DoubleML models using Optuna."""
        tuning_kwargs = {
            "scoring_methods": scoring_methods,
            "cv": cv,
            "set_as_params": set_as_params,
            "return_tune_res": return_tune_res,
            "optuna_settings": optuna_settings,
        }
        if not isinstance(ml_param_space, dict) or len(ml_param_space) == 0:
            raise ValueError("ml_param_space must be a non-empty dictionary.")
        valid_learners = ["ml_g", "ml_m", "ml_G", "ml_M", "ml_nested_g"]
        invalid_keys = [name for name in ml_param_space if name not in valid_learners]
        if invalid_keys:
            raise ValueError(
                f"Invalid ml_param_space keys for DoubleMLMEDS: {', '.join(map(str, invalid_keys))}. "
                f"Valid keys are: {', '.join(valid_learners)}."
            )

        tune_res: Optional[dict[str, Any]] = {} if return_tune_res else None
        for key, model in self.modeldict.items():
            # each model tunes the learners it has, a model without a learner in ml_param_space is skipped
            model_param_space = {name: ml_param_space[name] for name in model.learner_names if name in ml_param_space}
            if len(model_param_space) == 0:
                continue
            res = model.tune_ml_models(ml_param_space=model_param_space, **tuning_kwargs)
            if tune_res is not None:
                tune_res[key] = res
        return tune_res if return_tune_res else None

    tune_ml_models.__doc__ = TUNE_ML_MODELS_DOC
