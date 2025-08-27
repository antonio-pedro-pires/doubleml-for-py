import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.utils.multiclass import type_of_target

from doubleml import DoubleMLData
from doubleml.double_ml import DoubleML
from doubleml.double_ml_framework import concat
from doubleml.med.med import DoubleMLMEDP, DoubleMLMEDC
from doubleml.utils._checks import _check_external_predictions, _check_sample_splitting
from doubleml.utils._descriptive import generate_summary


# TODO: Add new data class for mediation analysis
# TODO: Learn how sampling works.
class DoubleMLMEDS:
    """Mediation analysis with double machine learning."""

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
        trimming_threshold=1e-2,
        order=1,
        multmed=True,
        fewsplits=False,
        draw_sample_splitting=True,
    ):

        self._check_data(obj_dml_data, trimming_threshold)
        self._dml_data = obj_dml_data

        # _check_resampling_specifications(n_folds, n_rep)
        self._n_folds = n_folds
        self._n_rep = n_rep

        self._multmed = multmed

        # initialize learners and parameters which are set model specific
        self._learner = None
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

        self._score_names = ["Y(0, M(0))", "Y(0, M(1))", "Y(1, M(0))", "Y(1, M(1))"]
        self._score_len = np.unique(self._score_names)
        self._score = score
        self._normalize_ipw = normalize_ipw
        self._order = order
        self._draw_sample_splitting = draw_sample_splitting

        # Set labels for returns
        self._results_labels = ["ATE", "dir.treat", "dir.control", "indir.treat", "indir.control", "Y(0, M(0))"]

        # Initialize all properties to None
        self._se = None
        self._pvalues = None
        self._coef = None
        self._ci = None
        self.n_trimmed = None

        # Check the learners are correctly specified.
        ml_g_is_classifier = DoubleML._check_learner(ml_g, "ml_g", regressor=True, classifier=True)
        _ = DoubleML._check_learner(ml_m, "ml_m", regressor=False, classifier=True)
        ml_med_is_classifier = DoubleML._check_learner(ml_med, "ml_med", regressor=True, classifier=True)

        self._learner = {"ml_g": clone(ml_g), "ml_m": clone(ml_m), "ml_med": clone(ml_med)}

        if obj_dml_data.binary_outcome:
            if ml_g_is_classifier:
                self._predict_method = {"ml_g": "predict_proba", "ml_m": "predict_proba"}
            else:
                raise ValueError(
                    "The outcome variable has been identified as a binary variable with values 0 and 1."
                    f"but The ml_g learner {str(ml_g)} is not a classifier."
                )
        else:
            self._predict_method = {"ml_g": "predict", "ml_m": "predict_proba"}

        if ml_med_is_classifier:
            self._predict_method["ml_med"] = "predict_proba"
        else:
            self._predict_method["ml_med"] = "predict"

        if self._multmed:
            ml_nested_is_classifier = DoubleML._check_learner(ml_nested, "ml_nested", regressor=True, classifier=True)
            self._learner["ml_nested"] = clone(ml_nested)
            if ml_nested_is_classifier:
                self._predict_method["ml_nested"] = "predict_proba"
            else:
                self._predict_method["ml_nested"] = "predict"

        # TODO: erase following comment.
        # Taken from DoubleMLAPOS.
        # Perform sample splitting
        self._smpls = None
        if draw_sample_splitting:
            self.draw_sample_splitting()

            # initialize all models if splits are known
            self._modeldict = self._initialize_models()
        pass

    def __str__(self):
        pass

    @property
    def n_folds(self):
        """
        Specifies the number of folds to be used for cross-validation
        """
        return self.n_folds

    @property
    def n_rep(self):
        """
        Number of repetitions for the sample splitting.
        """
        return self.n_rep

    @property
    def normalize_ipw(self):
        """
        Indicates whether the inverse probability weights are normalised.
        """
        return self.normalize_ipw

    @property
    def trimming_rule(self):
        """
        Specifies the trimming rule used.
        """
        return self.trimming_rule

    @property
    def trimming_threshold(self):
        """
        Specifies the trimming threshold.
        """
        return self.trimming_threshold

    # TODO: Check if the definition is true
    @property
    def order(self):
        """
        Specifies the order of the terms (interactions)
        """
        return self.order

    @property
    def multmed(self):
        """
        Indicates if the mediator variable is continuous and/or multiple.
        Determines the score function for the counterfactual E[Y(D=d, M(1-d))].
        """
        return self.multmed

    @property
    def fewsplits(self):
        """
        Indicates whether the same training data split is used for estimating the nested models of the nuisance parameter .
        """
        return self.fewsplits

    # TODO: Add definition
    @property
    def draw_sample_splitting(self):
        return self.draw_sample_splitting

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
        model = self._modeldict[score]
        # TODO: Add external predictions

        model.fit(
            n_jobs_cv=n_jobs_cv, store_predictions=store_predictions, store_models=store_models, external_predictions=None
        )
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

    # TODO: Check how to perform sensitivity analysis for causal mediation analysis.
    def sensitivity_analysis(self):
        pass

    def sensitivity_plot(self):
        pass

    def sensitivity_benchmark(self):
        pass

    def draw_sample_benchmark(self):
        pass

    def set_sample_splitting(self, all_smpls, all_smpls_cluster=None):
        self._smpls, self._smpls_cluster, self._n_rep, self._n_folds = _check_sample_splitting(
            all_smpls, all_smpls_cluster, self._dml_data, self._is_cluster_data
        )

        self._modeldict = self._initialize_models()

        return self

    def _check_and_set_learner(self, ml_g, ml_m, ml_med, ml_nested):
        _ = DoubleML._check_learner(ml_g, "ml_g", regressor=True, classifier=False)
        _ = DoubleML._check_learner(ml_m, "ml_m", regressor=True, classifier=False)
        _ = DoubleML._check_learner(ml_med, "ml_med", regressor=True, classifier=False)
        self._learner = {
            "ml_g_d0": clone(self._ml_g),
            "ml_g_d1": clone(self._ml_g),
            "ml_g_d0_med0": clone(self._ml_g),
            "ml_g_d0_med1": clone(self._ml_g),
            "ml_g_d1_med0": clone(self._ml_g),
            "ml_g_d1_med1": clone(self._ml_g),
            "ml_m": clone(self._ml_m),
            "ml_med_d1": clone(self._ml_med),
            "ml_med_d0": clone(self._ml_med),
        }
        self._predict_method = {
            "ml_g_d0": "predict",
            "ml_g_d1": "predict",
            "ml_g_d0_med0": "predict",
            "ml_g_d0_med1": "predict",
            "ml_g_d1_med0": "predict",
            "ml_g_d1_med1": "predict",
            "ml_m": "predict",
            "ml_med_d1": "predict",
            "ml_med_d0": "predict",
        }
        if self._multmed:
            if ml_nested is not None:
                _ = DoubleML._check_learner(ml_nested, "ml_nested", regressor=True, classifier=False)
                self._learner["ml_nested_d01"], self._learner["ml_nested_d10"] = clone(self._ml_nested), clone(self._ml_nested)
                self._predict_method["ml_nested_d01"], self._predict_method["ml_nested_d10"] = "predict", "predict"
            else:
                raise ValueError("mediation analysis with continuous or multiple mediators requires a nested lerner.")

        pass

    def _check_data(self, obj_dml_data, threshold):
        if not isinstance(obj_dml_data, DoubleMLData):
            raise TypeError(
                f"The data must be of DoubleMLData type. {str(obj_dml_data)} of type {str(type(obj_dml_data))} was passed."
            )
        if obj_dml_data.z_cols is None:
            raise ValueError("Incompatible data." + "Mediator variable has not been set. ")
        is_continuous = type_of_target(obj_dml_data.s) == "continuous"
        if not self._multmed:
            if is_continuous:
                raise ValueError("Incompatible data. The boolean multmed must be set to True when using a continuous mediator")
            # TODO: Raise error for multidimensional mediators and not multmed.

        if obj_dml_data.z_cols is None:
            raise ValueError("Incompatible data. Mediator analysis requires mediator variables.")

        if not isinstance(threshold, float):
            raise TypeError(f"Cutoff value has to be a float. Object of type {str(type(threshold))} passed.")

        # TODO: Test if one_treat works. Not sure of the __len__() method to see if there is only one treatment variable.
        one_treat = obj_dml_data.d_cols.__len__() > 1
        binary_treat = type_of_target(obj_dml_data.d) != "binary"
        zero_one_treat = np.all((np.power(obj_dml_data.d, 2) - obj_dml_data.d) == 0)
        if not (one_treat and binary_treat and zero_one_treat):
            raise ValueError(
                "Incompatible data. To fit a MedDML model with DML binary treatments"
                "exactly one binary variable with values 0 and 1"
                "needs to be specified as treatment variable."
            )

    def _initialize_models(self):
        # TODO: Instead of using an array with numbers for the scores, I could use a dictionnary. Makes it more readable.
        modeldict = dict.fromkeys(self._score_names)
        kwargs = {
            "obj_dml_data": self._dml_data,
            "ml_g": self._learner["ml_g"],
            "ml_m": self._learner["ml_m"],
            "ml_med": self._learner["ml_med"],
            "ml_nested": self._learner["ml_nested"],
            "n_folds": self.n_folds,
            "n_rep": self.n_rep,
            "trimming_threshold": self.trimming_threshold,
            "normalize_ipw": self.normalize_ipw,
            "draw_sample_splitting": False,
        }
        for score in self._score_names:
            # initialize models for all levels
            model = DoubleMLMED(score=self._score_names[score], **kwargs)

            # synchronize the sample splitting
            model.set_sample_splitting(all_smpls=self.smpls)
            modeldict[score] = model

        return modeldict
