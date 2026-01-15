import itertools

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLMEDData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_framework import concat
from doubleml.med.med import DoubleMLMED
from doubleml.utils._checks import _check_external_predictions, _check_sample_splitting
from doubleml.utils._descriptive import generate_summary
from sklearn import clone

from doubleml.double_ml_sampling_mixins import SampleSplittingMixin


# TODO: Add new data class for mediation analysis
# TODO: Learn how sampling works.
class DoubleMLMEDS(SampleSplittingMixin):
    """Mediation analysis with double machine learning."""

    def __init__(
        self,
        meds_data,
        ml_px,
        ml_yx,
        ml_ymx,
        ml_pmx,
        ml_nested,
        n_folds=5,
        n_rep=1,
        n_folds_inner=5,
        score=None,
        normalize_ipw=False,
        trimming_threshold=1e-2,
        order=1,
        multmed=True,
        fewsplits=False,
        draw_sample_splitting=True,
    ):

        self._check_data(meds_data, trimming_threshold)
        self._dml_data = meds_data
        self._is_cluster_data = self._dml_data.is_cluster_data

        self._trimming_threshold = trimming_threshold
        self._order = order
        self._multmed = multmed
        self._fewsplits = fewsplits
        self._normalize_ipw = normalize_ipw

        # _check_resampling_specifications(n_folds, n_rep)
        self._n_folds = n_folds
        self._n_rep = n_rep
        self._n_folds_inner = n_folds_inner

        self._multmed = multmed

        # initialize learners and parameters which are set model specific
        self._learner = {"ml_px": clone(ml_px), "ml_yx": clone(ml_yx), "ml_ymx": clone(ml_ymx), "ml_pmx": clone(ml_pmx), "ml_nested": clone(ml_nested)}
        self._params = None

        # Initialize framework constructed after the fit method is called.
        self._framework = None

        # TODO: Add functionality to check if learners are good.
        if multmed:
            if fewsplits:
                pass
            else:
                pass
            pass
        else:
            pass

        # Set labels for returns
        self._results_labels = ["ATE", "dir.treat", "dir.control", "indir.treat", "indir.control", "Y(0, M(0))"]

        self._learner = {"ml_px": clone(ml_px),
                         "ml_yx": clone(ml_yx),
                         "ml_ymx": clone(ml_ymx),
                         "ml_pmx": clone(ml_pmx),
                         "ml_nested": clone(ml_nested)}

        # Initialize all properties to None
        self._se = None
        self._pvalues = None
        self._coef = None
        self._ci = None
        self.n_trimmed = None

        # perform sample splitting
        self._smpls = None
        self._n_obs_sample_splitting = self._dml_data.n_obs
        self._strata = None

        if draw_sample_splitting:
            self.draw_sample_splitting()

            self._initialize_med_models()

        pass

    def __str__(self):
        pass

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
    def trimming_rule(self):
        """
        Specifies the trimming rule used.
        """
        return self._trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the trimming threshold.
        """
        return self._trimming_threshold

    # TODO: Check if the definition is true
    @property
    def order(self):
        """
        Specifies the order of the terms (interactions)
        """
        return self._order

    @property
    def multmed(self):
        """
        Indicates if the mediator variable is continuous and/or multiple.
        Determines the score function for the counterfactual E[Y(D=d, M(1-d))].
        """
        return self._multmed

    @property
    def fewsplits(self):
        """
        Indicates whether the same training data split is used for estimating the nested models of the nuisance parameter .
        """
        return self._fewsplits

    # TODO: Add definition
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
    def sensitivity_elements(self):
        """
        Values of the sensitivity components after calling :meth:`fit`;
        If available (e.g., PLR, IRM) a dictionary with entries ``sigma2``, ``nu2``, ``psi_sigma2``, ``psi_nu2``
        and ``riesz_rep``.
        """
        if self._framework is None:
            sensitivity_elements = None
        else:
            sensitivity_elements = self._framework.sensitivity_elements
        return sensitivity_elements

    @property
    def sensitivity_params(self):
        """
        Values of the sensitivity parameters after calling :meth:`sesitivity_analysis`;
        If available (e.g., PLR, IRM) a dictionary with entries ``theta``, ``se``, ``ci``, ``rv``
        and ``rva``.
        """
        if self._framework is None:
            sensitivity_params = None
        else:
            sensitivity_params = self._framework.sensitivity_params
        return sensitivity_params

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
            df_summary = generate_summary(self.coef, self.se, self.t_stat, self.pval, ci, self._treatment_levels)
        return df_summary

    @property
    def sensitivity_summary(self):
        """
        Returns a summary for the sensitivity analysis after calling :meth:`sensitivity_analysis`.

        Returns
        -------
        res : str
            Summary for the sensitivity analysis.
        """
        if self._framework is None:
            raise ValueError("Apply sensitivity_analysis() before sensitivity_summary.")
        else:
            sensitivity_summary = self._framework.sensitivity_summary
        return sensitivity_summary

    def fit(self, n_jobs_models=None, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions=None):
        if external_predictions is not None:
            _check_external_predictions(external_predictions)
            # ext_pred_dict = _rename_external_predictions(external_predictions)
        else:
            ext_pred_dict = None

        # parallel estimation of the models
        parallel = Parallel(n_jobs=n_jobs_models, verbose=0, pre_dispatch="2*n_jobs")
        fitted_models = parallel(
            delayed(self._fit_model)(score, n_jobs_cv, store_predictions, store_models, ext_pred_dict)
            for score in range(self._score_len)
        )

        # combine the estimates and scores
        framework_list = [None] * self._score_len

        for score in range(self._score_len):
            self._modeldict[score] = fitted_models[score]
            framework_list[score] = self._modeldict[score].framework

        # aggregate all frameworks
        self._framework = concat(framework_list)

        return self

    def _fit_model(self, score, n_jobs_cv=None, store_predictions=True, store_models=False, external_predictions_dict=None):
        for (d, k) in self._modeldict.items():
            for (_, value) in k.items():
                model = value

                if external_predictions_dict is not None:
                    external_predictions = external_predictions_dict[model.treatment_level][model.mediation_level]
                else:
                    external_predictions = None

                model.fit(n_jobs_cv=n_jobs_cv, store_predictions=store_predictions, store_models=store_models,
                          external_predictions=external_predictions)

        # TODO: Add external predictions
        return model

    def confint(self, joint=False, level=0.95):
        if self.framework is None:
            raise ValueError("Apply fit() before confint().")
        df_ci = self.framework.confint(joint=joint, level=level)
        # TODO: Add the score function to the index for better readibility.
        # df_ci.set_index(pd.Index(self._treatment_levels), inplace=True)
        return df_ci

    def bootstrap(self, method="normal", n_rep_boot=500):
        if self._framework is None:
            raise ValueError("Apply fit() before calling bootstrap()")
        self._framework.bootstrap(method=method, n_rep_boot=n_rep_boot)
        return self

    def evaluate_effects(self):
        ate = self._modeldict["Y(1, M(1)"].framework - self._modeldict["Y(0, M(0)"].framework
        dir_control = self._modeldict["Y(0, M(1)"].framework - self._modeldict["Y(0, M(0)"].framework
        dir_treatment = self._modeldict["Y(1, M(1)"].framework - self._modeldict["Y(1, M(0)"].framework
        indir_control = self._modeldict["Y(1, M(1)"].framework - self._modeldict["Y(1, M(0)"].framework
        indir_treatment = self._modeldict["Y(0, M(1)"].framework - self._modeldict["Y(0, M(0)"].framework

        return ate, dir_control, dir_treatment, indir_control, indir_treatment

    def _check_data(self, meds_data, threshold):
        if not isinstance(meds_data, DoubleMLMEDData):
            raise TypeError(
                f"The data must be of DoubleMLMediationData type. {str(meds_data)} of type {str(type(meds_data))} was passed."
            )
        if meds_data.z_cols is not None:
            raise NotImplementedError("instrumental variables for mediation analysis is not yet implemented.")
        if not np.all(meds_data.binary_treats):
            raise NotImplementedError("Treatment variables for mediation analysis must be binary" +
                                      "and with values either 0 or 1. Not binary treatment is not yet implemented yet.")

    def _initialize_med_models(self):
        self._modeldict = self._initialize_models()
        return self

    def _initialize_models(self):

        treatment_levels = np.unique(self._dml_data.d)
        mediation_levels = np.unique(self._dml_data.m)

        #TODO: Maybe will have to work this out. How to create dict to contain objects.
        modeldict={d: {m: object for m in mediation_levels} for d in treatment_levels}
        d_m_levels = itertools.product(treatment_levels, mediation_levels)

        for (treatment, mediation) in d_m_levels:
            if treatment == mediation:
                target = "potential"

                kwargs = {
                    "med_data": self._dml_data,
                    "ml_px": self._learner["ml_px"],
                    "ml_yx": self._learner["ml_yx"],
                    "target": target,
                    "treatment_level": treatment,
                    "mediation_level": mediation,
                    "n_folds": self.n_folds,
                    "n_rep": self.n_rep,
                    "n_folds_inner": self.n_folds_inner,
                    "trimming_threshold": self.trimming_threshold,
                    "normalize_ipw": self.normalize_ipw,
                    "draw_sample_splitting": False,
                }

                model = DoubleMLMED(**kwargs)

            else:
                target = "mediation"

                kwargs = {
                    "meds_data": self._dml_data,
                    "ml_px": self._learner["ml_px"],
                    "ml_ymx": self._learner["ml_ymx"],
                    "ml_pmx": self._learner["ml_pmx"],
                    "ml_nested": self._learner["ml_nested"],
                    "target": target,
                    "treatment_level": treatment,
                    "mediation_level": mediation,
                    "n_folds": self.n_folds,
                    "n_rep": self.n_rep,
                    "n_folds_inner": self.n_folds_inner,
                    "trimming_threshold": self.trimming_threshold,
                    "normalize_ipw": self.normalize_ipw,
                    "draw_sample_splitting": False,
                }

                model = DoubleMLMED(**kwargs)

            # synchronize the sample splitting
            #TODO: Probably will need to set samples for the inner samples.
            model.set_sample_splitting(all_smpls=self.smpls)
            modeldict[treatment][mediation] = model

        return modeldict
