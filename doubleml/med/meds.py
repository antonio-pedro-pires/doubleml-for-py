import itertools
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import clone

from doubleml import DoubleMLMEDData
from doubleml.double_ml_framework import concat
from doubleml.double_ml_sampling_mixins import SampleSplittingMixin
from doubleml.med.med import DoubleMLMED
from doubleml.med.utils._meds_utils import generate_effects_summary
from doubleml.utils import PSProcessorConfig
from doubleml.utils._checks import _check_score
from doubleml.utils._descriptive import generate_summary


# TODO Checklist:
# Add possibility to not perform nested sampling
# Add ipw_normalization (with ps module?)
# Add/update data/learner checks
# Add confint
# Check bootstrap logic
# Add sensitivity analysis
class DoubleMLMEDS(SampleSplittingMixin):
    """Double machine learning for causal mediation analysis with binary treatment.

        Parameters
        ----------
        dml_data : :class:`DoubleMLMediationData` object
            The :class:`DoubleMLMediationData` object providing the data and specifying the variables for the causal model.

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
            Only required if ``outcome`` is 'counterfactual'.

        ml_m : classifier implementing ``fit()`` and ``predict_proba()``
            A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
            :py:class:`sklearn.linear_model.LogisticRegression`) for the nuisance function :math:`P(D=d|X)`.

        ml_M : classifier implementing ``fit()`` and ``predict_proba()``
            A machine learner implementing ``fit()`` and ``predict_proba()`` methods (e.g.
            :py:class:`sklearn.linear_model.LogisticRegression`) for the nuisance function :math:`P(D=d|M,X)`.
            Only required if ``outcome`` is 'counterfactual'.

        score : str
            A str (``'efficient-alt'`` is the only choice)  specifying the score function to use.
            Default is 'efficient-alt'.

        n_folds : int
            Number of folds.
            Default is ``5``.

        n_rep : int
            Number of repetitions for the sample splitting.
            Default is ``1``.

        normalize_ipw : bool
            Indicates whether the inverse probability weights are normalized.
            Default is ``True``.

        draw_sample_splitting : bool
            Indicates whether the sample splitting should be drawn during initialization of the object.
            Default is ``True``.

        double_sample_splitting : bool
            Indicates whether the data is resampled for the estimation of the nested parameter.
            Default is ``True``.

        ps_processor_config : PSProcessorConfig
            configuration file for the propensity score processor (PSProcessor) object.
            When None, the PSProcessor is initialized with its default values.
            Default is ``None``

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
    >>> dml_med_obj = dml.DoubleMLMED(obj_dml_data, ml_m, ml_g, ml_G, ml_M, ml_nested_g)
    >>> dml_med_obj.fit().summary  # doctest: +SKIP
    """

    def __init__(
        self,
        dml_data,
        ml_m,
        ml_g,
        ml_G,
        ml_M,
        ml_nested_g,
        n_folds=5,
        n_rep=1,
        n_folds_inner=5,
        score="efficient-alt",
        normalize_ipw=True,
        draw_sample_splitting=True,
        double_sample_splitting=True,
        ps_processor_config: Optional[PSProcessorConfig] = None,
    ):

        self._check_data(dml_data)
        self._dml_data = dml_data
        self._is_cluster_data = self._dml_data.is_cluster_data

        # Check score
        self._score = score
        valid_scores = ["efficient-alt"]
        _check_score(self._score, valid_scores, allow_callable=False)

        self._normalize_ipw = normalize_ipw
        self._double_sample_splitting = double_sample_splitting
        # _check_resampling_specifications(n_folds, n_rep)
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
        self._params = None

        # Initialize framework constructed after the fit method is called.
        self._framework = None

        # Set labels for returns
        self._results_labels = ["ATE", "dir.treat", "dir.control", "indir.treat", "indir.control", "Y(0, M(0))"]

        # Initialize all properties to None
        self._se = None
        self._pvalues = None
        self._coef = None
        self._ci = None
        self.n_trimmed = None

        # perform sample splitting
        self._smpls = None
        self._smpls_inner = None
        self._n_obs_sample_splitting = self._dml_data.n_obs
        self._strata = None

        self._ps_processor_config = ps_processor_config if ps_processor_config is not None else PSProcessorConfig()

        if draw_sample_splitting:
            self.draw_sample_splitting()
            self._initialize_dml_model()

        self._effects = None

    def __str__(self):
        class_name = self.__class__.__name__
        header = f"================== {class_name} Object ==================\n"
        fit_summary = str(self.summary)
        res = header + "\n------------------ Fit summary       ------------------\n" + fit_summary
        effects_summary = str(self.effects_summary)
        res = res + "\n------------------ Effects summary       ------------------\n" + effects_summary
        return res

    @property
    def score(self):
        """
        The score function.
        """
        return self._score

    @property
    def smpls_inner(self):
        return self._smpls_inner

    @property
    def n_folds(self):
        """
        Specifies the number of folds to be used for cross-validation
        """
        return self._n_folds

    @property
    def n_rep(self):
        """
        Number of repetitions for the sample splitting.
        """
        return self._n_rep

    @property
    def n_folds_inner(self):
        """
        Number of folds to be used for cross-validation of the inner models.
        """
        return self._n_folds_inner

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalised.
        """
        return self._normalize_ipw

    @property
    def models_ids(self):
        return self._models_ids

    @property
    def double_sample_splitting(self):
        """
        Indicates whether the data is resampled for the estimation of the nested parameter.
        """
        return self._double_sample_splitting

    @property
    def treatments(self):
        """
        Describes the different treatment levels
        """
        return np.unique(self._dml_data.d)

    @property
    def ps_processor_config(self):
        """
        Configuration for propensity score processing (clipping, calibration, etc.).
        """
        return self._ps_processor_config

    @property
    def effects(self):
        """
        Estimates for the ATE and the combination of the direct/indirect, treatment/control groups effects after calling
        :meth:`evaluate-effects`
        """
        if self._effects is None:
            effects = None
        else:
            effects = self._effects
        return effects

    @property
    def coef(self):
        """
        Estimates for the causal parameter(s) after calling :meth:`fit` (shape(4,)).
        """
        if self._framework is None:
            coef = None
        else:
            coef = self.framework.thetas
        return coef

    @property
    def all_coef(self):
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
    def se(self):
        """
        Standard errors for the causal parameter(s) after calling :meth:`fit` (shape (4,)).
        """
        if self._framework is None:
            se = None
        else:
            se = self.framework.ses
        return se

    @property
    def all_se(self):
        """
        Standard errors for the causal parameter(s) of interest :meth:`fit` (shape (4,)).
        """
        if self._framework is None:
            all_se = None
        else:
            all_se = self.framework.all_ses
        return all_se

    @property
    def t_stat(self):
        """
        t-statistics for the causal parameter(s) after calling :meth:`fit` (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            t_stats = None
        else:
            t_stats = self.framework.t_stats
        return t_stats

    @property
    def pval(self):
        """
        p-values for the causal parameter(s) (shape (``n_treatment_levels``,)).
        """
        if self._framework is None:
            pvals = None
        else:
            pvals = self.framework.pvals
        return pvals

    @property
    def smpls(self):
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
    def framework(self):
        """
        The corresponding :class:`doubleml.DoubleMLFramework` object.
        """
        return self._framework

    # TODO: Add bootstrap or remove it
    @property
    def boot_t_stat(self):
        """
        Bootstrapped t-statistics for the causal parameter(s) after calling :meth:`fit` and :meth:`bootstrap`
         (shape (``n_rep_boot``, ``n_treatment_levels``, ``n_rep``)).
        """
        if self._framework is None:
            boot_t_stat = None
        else:
            boot_t_stat = self._framework.boot_t_stat
        return boot_t_stat

    @property
    def modeldict(self):
        """
        The list of models for each level.
        """
        return self._modeldict

    @property
    def summary(self):
        """
        A summary for the estimated causal effect after calling :meth:`fit`.
        """
        if self.framework is None:
            col_names = ["coef", "std err", "t", "P>|t|"]
            df_summary = pd.DataFrame(columns=col_names)
        else:
            ci = self.confint()
            df_summary = generate_summary(
                coef=self.coef, se=self.se, t_stat=self.t_stat, pval=self.pval, ci=ci, index_names=self.models_ids
            )
        return df_summary

    @property
    def effects_summary(self):
        """
        A summary for the estimated effects after calling :meth:`evaluate_effects`.
        """
        if self._effects is None:
            col_names = ["coef", "std err", "t", "P>|t|"]
            df_effects_summary = pd.DataFrame(columns=col_names)
        else:
            df_effects_summary = generate_effects_summary(self._effects)
        return df_effects_summary

    def fit(self, n_jobs_models=None, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions=None):
        if external_predictions is not None:
            self._check_external_predictions(external_predictions)
            ext_pred_dict = external_predictions
        else:
            ext_pred_dict = None

        # parallel estimation of the models
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")
        fitted_models = parallel(
            delayed(self._fit_model)(model_id, n_jobs_cv, store_predictions, store_models, ext_pred_dict)
            for model_id in self.models_ids
        )

        # combine the estimates and model_ids
        framework_list = [None] * len(self.models_ids)

        for idx, (model_id, (outcome, treatment)) in enumerate(zip(self.models_ids, self._id_pairs)):
            assert fitted_models[idx].outcome == outcome
            assert fitted_models[idx].treatment_level == treatment

            self._modeldict[model_id] = fitted_models[idx]
            framework_list[idx] = self._modeldict[model_id].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)

        return self

    def _fit_model(self, score, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions_dict=None):
        model = self.modeldict[score]
        if external_predictions_dict is not None:
            external_predictions = external_predictions_dict[score]
        else:
            external_predictions = None

        model.fit(
            n_jobs_cv=n_jobs_cv,
            store_predictions=store_predictions,
            store_models=store_models,
            external_predictions=external_predictions,
        )

        return model

    def confint(self, joint=False, level=0.95):
        if self.framework is None:
            raise ValueError("Apply fit() before confint().")
        df_ci = self.framework.confint(joint=joint, level=level)
        df_ci.set_index(pd.Index(self.models_ids), inplace=True)
        return df_ci

    def evaluate_effects(self):
        if self.framework is None:
            raise ValueError("Apply fit() before calling evaluate_effects()")
        ate = self._modeldict["potential_1"].framework - self._modeldict["potential_0"].framework
        dir_treat = self._modeldict["potential_1"].framework - self._modeldict["counterfactual_0"].framework
        dir_control = self._modeldict["counterfactual_1"].framework - self._modeldict["potential_0"].framework
        indir_treat = self._modeldict["potential_1"].framework - self._modeldict["counterfactual_1"].framework
        indir_control = self._modeldict["counterfactual_0"].framework - self._modeldict["potential_0"].framework

        self._effects = {
            "ATE": ate,
            "DIR_TREAT": dir_treat,
            "DIR_CONTROL": dir_control,
            "INDIR_TREAT": indir_treat,
            "INDIR_CONTROL": indir_control,
        }

    def _check_data(self, meds_data):
        if not isinstance(meds_data, DoubleMLMEDData):
            raise TypeError(
                f"The data must be of DoubleMLMediationData type. {str(meds_data)} of type {str(type(meds_data))} was passed."
            )
        if not all(meds_data.binary_treats):
            raise ValueError("Treatment variables for mediation analysis must be binary and take values 1 or 0.")
        if meds_data.z_cols is not None:
            raise NotImplementedError("instrumental variables for mediation analysis is not yet implemented.")

    def _initialize_dml_model(self):
        self._modeldict = self._initialize_models()
        return self

    def _initialize_models(self):
        modeldict = {score: None for score in self.models_ids}

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
            assert f"{outcome}_{treatment}" == score
            learners = pot_learners if outcome == "potential" else counter_learners

            model = DoubleMLMED(
                dml_data=self._dml_data,
                outcome=outcome,
                treatment_level=treatment,
                n_rep=self._n_rep,
                n_folds=self._n_folds,
                normalize_ipw=self._normalize_ipw,
                double_sample_splitting=self._double_sample_splitting,
                draw_sample_splitting=False,
                ps_processor_config=self._ps_processor_config,
                **learners,
            )

            model.set_sample_splitting(smpls=self._smpls, smpls_inner=self.smpls_inner)

            modeldict[f"{outcome}_{treatment}"] = model

        return modeldict

    def _initialize_models_ids(self):
        return [f"{outcome}_{treatment}" for outcome, treatment in self._id_pairs]

    def _valid_id_pairs(self):
        if all(self._dml_data.binary_treats):
            treatment_levels = self.treatments
            self.valid_outcomes = ["potential", "counterfactual"]
            valid_id_pairs = list(itertools.product(self.valid_outcomes, map(int, treatment_levels)))
        return valid_id_pairs

    def _check_external_predictions(self, external_predictions_dict):
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
        if self.double_sample_splitting:
            expected_inner_learners_keys = [f"ml_G_inner_{i}" for i in range(self.n_folds)]
            expected_learners_keys += expected_inner_learners_keys

        for key, value in external_predictions_dict.items():
            if not isinstance(value, dict):
                raise TypeError(
                    f"external_predictions[{key}] must hold a dictionary. "
                    + f"Current value in external_predictions[{key}] is of type {type(value)}"
                )
            for d_col in value.keys():
                assert d_col in set(self._dml_data.d_cols)
                if not set(value[d_col].keys()).issubset(expected_learners_keys):
                    raise ValueError(
                        f"external_predictions[{d_col}] must hold a dictionnary whose keys are a "
                        + f"subset of {set(expected_learners_keys)}. "
                        + f"Passed keys: {set(value[self._dml_data.d_cols].keys())}."
                    )
            return

    def tune_ml_models(
        self,
        ml_param_space,
        scoring_methods=None,
        cv=5,
        set_as_params=True,
        return_tune_res=False,
        optuna_settings=None,
    ):

        tuning_kwargs = {
            "scoring_methods": scoring_methods,
            "cv": cv,
            "set_as_params": set_as_params,
            "return_tune_res": return_tune_res,
            "optuna_settings": optuna_settings,
        }
        tune_res = {} if return_tune_res else None

        for key, model in self.modeldict.items():
            if model.outcome == "potential":
                res = model.tune_ml_models(
                    ml_param_space={
                        "ml_g": ml_param_space["ml_g"],
                        "ml_m": ml_param_space["ml_m"],
                    },
                    **tuning_kwargs,
                )
            elif model.outcome == "counterfactual":
                res = model.tune_ml_models(
                    ml_param_space={
                        "ml_m": ml_param_space["ml_m"],
                        "ml_G": ml_param_space["ml_G"],
                        "ml_M": ml_param_space["ml_M"],
                        "ml_nested_g": ml_param_space["ml_nested_g"],
                    },
                    **tuning_kwargs,
                )

            if return_tune_res:
                tune_res[key] = res
        return tune_res if return_tune_res else None
